# 다국어 확장 구현 로드맵

**대상**: `finetune/` adaptation toolkit
**범위**: 일본어(1차) → 한국어(2차)를 **일회성 커스텀 훅이 아니라 툴킷의 기능**으로 지원
**작성일**: 2026-08-30 (최종 갱신 2026-08-30)
**상태**: M1 코드 완료(G1 화자 검수 대기). M0·M2 이후는 CUDA 머신에서 진행.

이 문서는 [CONTRACT.md](../CONTRACT.md)의 공개 계약과 [SCOPE.md](SCOPE.md)의 지원 범위를
전제로 한다. 두 문서와 충돌하는 항목은 이 로드맵이 아니라 그쪽이 우선한다.

---

## 0. 왜 "확장"인가

현재 툴킷으로도 일본어 파인튜닝은 **오늘 가능하다**. `--frontend custom
--frontend-hook ./ja.py:create_frontend`로 끝난다. 그러나 그 경로는 언어가 늘어날수록
비용이 선형으로 늘어난다:

- 훅 `.py` 파일을 사용자가 직접 보관하고 `export` 때 다시 넘겨야 한다
  ([CUSTOM_G2P.md](CUSTOM_G2P.md) "Deployment requirements").
- 언어별 정규화·G2P 회귀 테스트가 툴킷 CI에 들어오지 못한다.
- 2단계 학습(언어 베이스 → 목표 화자)에 필요한 다화자 준비를 지원하지 않는다.
- 평가가 신호 진단(peak/RMS/무음/클리핑)뿐이라 언어 품질을 판정할 수 없다.

따라서 확장의 목표는 **"일본어를 되게 하는 것"이 아니라 "언어를 추가하는 비용을 상수로
만드는 것"** 이다. 한국어는 그 구조가 실제로 상수인지 검증하는 두 번째 사례다.

---

## 1. 검증된 기준선 (2026-08-30 실측)

아래는 추정이 아니라 이 저장소에서 실행해 확인한 결과다. 재현 명령은 §6.3에 있다.

> 코드 인용은 **심볼 이름이 정본**이다. 줄 번호는 편집으로 즉시 어긋나므로 쓰지 않는다.

| # | 사실 | 근거 | 함의 |
|---|---|---|---|
| V1 | **일본어 음소 전체가 릴리스 178심볼 인벤토리 안에 있다.** OpenJTalk 음소 집합(모음5·무성화5·`N`·`cl`·`pau` + 자음 30여)을 IPA로 매핑했을 때 신규 심볼 **0개** | `symbols.BASE_SYMBOLS` 대조 | 임베딩 마이그레이션이 전 행 복사. 랜덤 초기화 행 없음 |
| V2 | **한국어도 신규 심볼 0개.** espeak-ng `ko` 출력 중 base 밖 문자는 `-`(U+002D) 하나뿐이며, 이는 경음 표지이므로 `ʼ`(U+02BC, base에 존재)로 매핑하면 해소 | 동일 | JA/KO 모두 178 계약 유지 가능 |
| V3 | **eSpeak `ja`는 사용 불가.** 한자마다 영어 단어 "chinese letter"(`tʃˈaɪniːzlˈe̞tə`)를 리터럴로 방출. 카나 입력에서도 base 밖 문자 `ä`·`̞`(U+031E)·`ᵝ`(U+1D5D) 발생 | espeak-ng 1.52 실행 | 일본어는 커스텀 프론트엔드 **필수** |
| V4 | **eSpeak `ko`는 IPA 변환기로는 쓸 만하나 음운 규칙이 빠진다.** `옵니다`→`ˈopnidˌɐ`(비음화 미적용), `신라면`→`sˈinɾɐmjˌʌn`(유음화 미적용), `밟았다`→`pˈɐlbɐt-t-ˌɐ`(겹받침 미처리) | 동일 | 한국어는 espeak 단독 불가, G2P 전처리 필요 |
| V5 | **`pyopenjtalk-plus` 0.4.1이 prebuilt wheel로 설치된다.** 본가 `pyopenjtalk`는 py3.10 wheel 없음(소스 빌드 필요) | `pip install --only-binary=:all:` | 일본어 의존성이 컴파일러 없이 해결됨 |
| V6 | **`g2pkk` + espeak `ko` 조합이 음운 규칙을 IPA까지 전달한다.** `있습니다`→`읻씀니다`→`ˈid-s-ɯmnˌidɐ`, `국물`→`궁물`→`ɡˈuŋmuɫ`, `신라면`→`실라면`→`silˈɐmjʌn` | 실행 | 한국어 2단 구조(G2P→IPA) 확정 |
| V7 | **pyopenjtalk가 supertonic이 어휘사전으로 고쳐야 했던 항목을 그냥 맞게 읽는다.** `抗うつ剤`→コーウツザイ, `対策`→タイサク, `痛み止め薬`→イタミドメヤク, `2026年8月30日`→ニセンニジューロクネンハチガツサンジューニチ | 실행 | supertonic의 `jf-surgical-v1~v6` 아크는 이식 대상이 아님(§5.1) |
| V8 | **2단계 체이닝이 현재 코드로 동작한다.** `export`가 `config.json`+`model.pth`+`runtime/`을 쓰고 `resolve_base_model()`이 로컬 디렉터리를 받는다 | `exporting.export_checkpoint()`, `modeling.resolve_base_model()` | 언어 베이스 → 화자 적응 가능 |
| V9 | **단, 심볼 수가 정확히 178이 아니면 체이닝이 깨진다.** `load_runtime_components()`가 `len(symbols) != 178`에서 `RuntimeError` | `modeling.load_runtime_components()` | V1/V2 덕분에 오늘은 문제없지만 **단일 실패점**(§2.2) |
| V10 | **다화자 준비가 하드 블록이다.** speaker 값이 2개 이상이면 `prepare`/`audit`이 즉시 실패 | `prepare_dataset()` · `audit_dataset()`의 speaker 가드 | 언어 베이스 단계에 우회 필요 |

### 확인하지 못한 것

- `M:` 드라이브(일본어 데이터셋)가 미마운트 상태라 **실물 데이터는 보지 못했다.**
  구성은 `supertonic-ja-ft/configs/paths.example.yaml` 기준으로만 파악했다.
- 릴리스 `config.json`의 `mel_fmin`/`mel_fmax` 실측값 (오프라인). 고 F0 여성 타깃에서
  확인 필요 — §7 R4.
- 현재 WSL에 **CUDA가 없다** (`torch.cuda.is_available() == False`, `nvidia-smi` 부재).

---

## 2. 설계

### 2.1 F1 — 언어 프론트엔드 레지스트리

새 패키지 `inflect_finetune/frontends/`를 만들고, 언어 프론트엔드를 **기존 훅 계약과
동일한 인터페이스**(`normalize` / `phonemize` / `symbols` / `metadata`)로 등록한다.
계약을 바꾸지 않는 것이 핵심이다 — `frontend.py`의 검증 경로(결정성 2회 호출, 미선언
심볼 거부, 소스 해시, 제어문자 거부)를 그대로 재사용한다.

```
inflect_finetune/frontends/
  __init__.py          # REGISTRY + resolve() + registry_record() + hook_path_for_record()
  ja_openjtalk.py      # pyopenjtalk-plus                                    [구현됨]
  ko_g2pkk.py          # g2pkk + espeak(ko) IPA 단계                          [M5]
```

**구현된 형태(2026-08-30)**: 레지스트리 항목은 새 mode가 아니라 **동봉된 custom
프론트엔드 파일에 대한 이름 별칭**이다. `resolve()`가 이름을
`FrontendOptions(mode="custom", hook="<pkg>/frontends/ja_openjtalk.py:create_frontend")`
로 바꾼다. 그 결과 `exporting.py`와 `frontend.py`를 **한 줄도 바꾸지 않고** 기존 custom
경로의 검증·패키징을 전부 재사용한다. `espeak.py`·`ipa.py`는 불필요해 만들지 않았다 —
espeak은 이미 `frontend.py`가 소유하고, 매핑 헬퍼는 언어 모듈 안에 있으면 충분하다.

CLI 변화:

```bash
# 지금
--frontend custom --frontend-hook ./ja.py:create_frontend
# 확장 후
--frontend ja-openjtalk
```

`custom`/`prephonemized`/`espeak`은 그대로 둔다. `dataset.json`의 `frontend` 블록은
`type: "custom"` + 기존 `hook` 레코드(소스 해시·metadata 해시)를 유지하고, 그 옆에
`registry` 블록으로 이름·언어·**필요한 extra**·툴킷 버전을 기록한다.

> 의존성 선언이 필요한 이유: [CUSTOM_G2P.md](CUSTOM_G2P.md)가 "외부 아티팩트를 요구하면
> self-contained라고 부르지 말 것"을 명시한다. pyopenjtalk 사전과 mecab-ko-dic은 정확히
> 그 외부 아티팩트다. 패키지가 스스로 그 사실을 기록하게 만든다.

### 2.2 F2 — zero-extension 심볼 정책과 178 제약

V1/V2로 JA·KO 모두 신규 심볼 0개가 가능하다. 이걸 **우연이 아니라 계약**으로 만든다.

1. `audit`에 `--require-no-new-symbols` 추가. 프론트엔드 수정이 조용히 심볼을 늘리는
   회귀를 잡는다.
2. `modeling.load_runtime_components()`의 `!= 178` 검사를 **`>= 178` + base prefix 일치**로 완화한다.
   지금은 신규 심볼이 하나라도 생기면 그 체크포인트를 다음 단계의 `--base`로 못 쓴다.
   완화 후에도 `modeling.load_symbols()`의 prefix 검증이 정체성을 지킨다.

이 두 개는 서로를 보완한다. (1)은 "심볼을 늘리지 마라", (2)는 "늘려야만 하는 언어가
나왔을 때 막다른 길이 아니게 하라"이다. (1)은 M1에서 구현됐다. (2)는 M3에 남아 있고,
그때까지는 신규 심볼을 쓰는 선택지가 닫혀 있다 — 일본어 악센트를 base의 `↑`/`↓`로
확정한 이유다(§5.1 D1).

### 2.3 F3 — 2단계 학습 (language-base → voice)

```
stage 1  다화자 일본어 코퍼스 ──> ja-base 체크포인트 (언어·음운·타이밍)
stage 2  단일 화자 (고 F0 여성) ──> 제품 체크포인트 (음색·음역)   [--base = stage1 export]
```

영어 남성 베이스에서 일본어 고 F0 여성으로 **한 번에** 가는 것은 SCOPE.md가 경고하는
"모든 부분을 동시에 움직이라"는 요구다. 언어 이동과 화자 이동을 분리한다.

구현: `prepare`에 `--corpus-role {voice,language-base}` 추가.
- `voice`(기본): 현재 단일 화자 가드 유지.
- `language-base`: 다화자 허용. `dataset.json`에 `corpus_role`과 화자 목록을 기록하고,
  `PREPARATION_REPORT.txt`에 "이 데이터셋은 화자 정체성 학습에 쓸 수 없다"를 명시.

`audit_dataset()`의 동일 가드도 `corpus_role`을 읽도록 한다. 기본 동작은 바뀌지 않는다.

### 2.4 F4 — 평가 확장

현재 `evaluate`는 duration/silence/clipping/peak/RMS/DC/non-finite만 본다. 언어 품질도
음성 정체성도 판정하지 못한다. supertonic 프로젝트의 가장 값비싼 교훈이 여기 적용된다 —
**모든 자동 지표는 스크린이고, 판정자는 청취다** (§5.2).

추가할 것:
1. **언어별 ASR/CER 평가기.** `examples/transcript_evaluator_plugin.py` 훅이 이미 비어
   있다. JA는 kana 정규화 CER, KO는 자모 정규화 CER. 툴킷이 ASR을 자동 다운로드하지
   않는 현재 정책은 유지한다(플러그인으로만).
2. **F0 진단.** `_signal_metrics()`에 f0 median/IQR/유성 프레임 비율 추가. 고 F0 여성
   타깃에서 레지스터 붕괴와 피치 평탄화를 조기에 잡는 유일한 싼 관측치다.
3. **블라인드 A/B 청취 페이지 생성기.** 랜덤 라벨, 실물 앵커 1개 강제 포함, 판정
   JSON export. 고음역 행을 반드시 페이지에 올린다.

### 2.5 F5 — 배포 패키징

`exporting._write_deployment_runtime()`이 레지스트리 프론트엔드를 인식하고,
생성 런타임에 (a) 프론트엔드 모듈, (b) `requirements-frontend.txt`, (c) 사전/의존성이
없을 때 **영어로 조용히 폴백하지 않고 실패**하는 경로를 쓰게 한다. 마지막 항목은
CONTRACT.md의 검증 게이트에 이미 있는 요구사항이다.

---

## 3. 단계별 로드맵

각 단계는 **수락 게이트를 통과해야** 다음으로 간다. 게이트는 착수 전에 고정하고,
데이터를 본 뒤에 옮기지 않는다.

### M0 — 환경 확정 (CUDA 머신)

- CUDA torch 동작, `nvidia-smi`, VRAM 확인
- `M:` 마운트 및 일본어 데이터셋 실물 확인
- `finetune/` editable 설치 + `pytest` 전체 통과
- 릴리스 `config.json`의 `mel_fmin`/`mel_fmax`/`filter_length`/`hop_length` 기록

**게이트 G0**: `inflect-adapt --help`, 기존 테스트 스위트 green, Micro 릴리스 다운로드 성공.

### M1 — 프론트엔드 레지스트리 + 일본어 프론트엔드 — **코드 완료 (2026-08-30)**

> 구현·테스트·문서는 끝났다. **G1은 아직 열려 있다** — 200문장 화자 검수가 남았고
> 그것은 CUDA 머신 인계 후에 마무리한다.

- F1 골격(`frontends/` 패키지, 레지스트리, CLI 배선)
- `ja_openjtalk.py` 구현: 텍스트 정규화 → `extract_fullcontext` → 음소+악센트 → IPA
- 사용자 사전 슬롯(고유명사 오독 대응). supertonic의 어휘사전 **형식**은 참고하되
  내용은 이식하지 않는다(V7).
- 회귀 테스트: 결정성, 심볼 선언 일치, 신규 심볼 0개, 고정 입력 → 고정 출력 스냅샷

**게이트 G1**: 대표 문장 200개에 대해 (a) 신규 심볼 0, (b) 2회 호출 결과 동일,
(c) 일본어 화자가 정규화 텍스트와 카나 표기를 검수해 오독률 기록. **오독은 0을 요구하지
않는다 — 측정하고 기록하는 것이 게이트다.**

**(a)(b) 통과 (2026-08-30)**: `examples/japanese_review_suite.txt` 207문장에 대해
실패 0 · 릴리스 인벤토리 밖 문자 0. 재현:

```bash
python examples/frontend_review_dump.py \
  --sentences examples/japanese_review_suite.txt \
  --frontend ja-openjtalk --language ja --output review/ja.tsv
```

**(c) 미완**: 화자 검수는 CUDA 머신 인계 후. 커버리지 스위트는 프론트엔드 동작을 덮지
코퍼스를 대표하지 않으므로, 실제 전사 무작위 표본에 대해서도 같은 덤프를 돌린다.

**검수·리뷰가 잡아낸 결함 3건(전부 수정 + 테스트 잠금)** — 스위트가 그냥 통과했다면
발견하지 못했다:
1. 소수점이 문장 끝으로 처리돼 `1.5キロ`가 「イチ。ゴキロ」로 읽혔다. 숫자 사이 `.`는
   분할 대상에서 제외했다.
2. `3,000円`이 「サン ゼロゼロゼロ」로 읽혔다. Open JTalk은 자릿수 구분 쉼표를 모른다 —
   정규화에서 제거한다.
3. **(2)의 첫 수정이 과잉이었다.** `(?<=\d),(?=\d)`가 `1,2,3`을 `123`으로 합쳤다 —
   고치려던 것보다 나쁜 조용한 데이터 손상. 자릿수 쉼표는 **정확히 세 자리**가 뒤따를
   때만 제거하고, 살아남은 쉼표는 열거 구분자로 분할한다.

### M2 — 데이터 준비 (일본어)

- 매니페스트 생성. `group_id`에 **source file**을 채운다 (§5.2 필수 항목)
- 클리핑 행 정책 결정: anime 계열은 코퍼스 전역이 0 dBFS 초과 (§5.2)
- stage-1 다화자 세트 / stage-2 단일 화자(고 F0 여성) 세트를 분리해 준비
- `audit` 통과

**게이트 G2**: `phoneme_coverage.json`의 `added_symbol_count == 0`,
group/텍스트 누출 0, 검증셋 음소가 학습셋에 전부 존재.

### M3 — F3 다단계 학습 배선 + 일본어 stage 1

- `--corpus-role` 구현, `modeling.py` 178 제약 완화(F2-2)
- ja-base 학습 (Micro). 디코더 언프리즈는 늦게.
- export → 그 디렉터리를 `--base`로 재로드하는 **체이닝 스모크**

**게이트 G3**: stage-1 export를 stage-2의 `--base`로 로드해 1 step 학습이 돈다.
held-out 합성이 일본어로 들린다(정체성·품질 불문).

### M4 — 평가 확장 + 일본어 stage 2

- F4(CER 플러그인, F0 진단, 블라인드 페이지)
- 고 F0 여성 화자 적응
- 체크포인트 선택 규칙을 **데이터 보기 전에** 선언

**게이트 G4**: 사전 선언한 규칙으로 고른 체크포인트가 블라인드 청취에서 baseline 대비
우세. 고음역 행이 페이지에 포함되어 있을 것. 여기서 처음으로 "일본어 파인튜닝 성공"을
말할 수 있다.

### M5 — 한국어 프론트엔드 (구조 검증)

- `ko_g2pkk.py`: g2pkk → 발음 한글 → espeak(`ko`) → IPA, `-`→`ʼ` 매핑
- **M1에서 만든 레지스트리에 코드 변경 없이 얹히는지**가 진짜 게이트다.
  얹히지 않으면 F1 설계가 틀린 것이므로 M1로 되돌아간다.

**게이트 G5**: `ko_g2pkk.py` 추가 외에 `frontend.py`/`prepare.py`/`exporting.py`
변경 0줄. 신규 심볼 0개. 한국어 화자 검수.

### M6 — 한국어 적응 + 배포 패키징

- F5, 한국어 데이터 확보(§5.3), stage 1/2 반복
- 언어별 릴리스 노트(데이터 출처·동의·프론트엔드·알려진 한계)

**게이트 G6**: 클린 환경에서 export 패키지 로드 + ONNX parity + 프론트엔드 의존성
누락 시 **영어 폴백 없이 실패**.

---

## 4. 코드 변경 목록

| ID | 변경 | 위치 | 단계 | 비고 |
|---|---|---|---|---|
| C1 | `frontends/` 레지스트리 패키지 | `frontends/__init__.py` | M1 | ✅ 완료. 훅 계약 불변 |
| ~~C2~~ | ~~`FrontendOptions.mode`에 레지스트리 이름 허용~~ | — | — | **삭제.** 레지스트리가 `mode="custom"`으로 해석하므로 `frontend.py`는 변경 불필요 |
| C3 | `--frontend` choices 확장 + `prepare` 배선 | `cli.py`, `prepare.py` | M1 | ✅ 완료 |
| C4 | `ja_openjtalk.py` | `frontends/ja_openjtalk.py` | M1 | ✅ 완료. pyopenjtalk-plus |
| C5 | `--require-no-new-symbols` | `audit.py`, `cli.py` | M1 | ✅ 완료 |
| C5b | export의 동봉 훅 자동 해석 | `cli.py` | M1 | ✅ 완료. 없으면 JA 경로가 end-to-end로 닫히지 않는다 |
| C6 | 178 → `>=178 + prefix` 완화 | `modeling.load_runtime_components()` | M3 | **체이닝 단일 실패점** |
| C7 | `--corpus-role` (다화자 허용) | `prepare_dataset()` · `audit_dataset()` | M3 | 기본 동작 불변 |
| C8 | F0 진단 추가 | `evaluation._signal_metrics()` | M4 | |
| C9 | ASR/CER 플러그인 (JA/KO) | `examples/` | M4 | 자동 다운로드 금지 유지 |
| C10 | 블라인드 A/B 페이지 생성기 | 신규 | M4 | |
| C11 | `ko_g2pkk.py` | 신규 | M5 | **다른 파일 변경 0을 목표** |
| C12 | 배포 런타임 프론트엔드 패키징 | `_write_deployment_runtime()` | M6 | |

학습 코어(`training.py`), 임베딩 마이그레이션(`checkpoint.py`), 분할 로직은 **변경
대상이 아니다.**

---

## 5. 부록

### 5.1 일본어

**프론트엔드 파이프라인**
```
원문 → NFKC/공백 정규화 → pyopenjtalk.extract_fullcontext
     → (음소, 악센트구 위치 A:) → IPA 매핑 → 악센트 표기 → phoneme string
```

IPA 매핑(전부 base 178 안): `a i ɯ e o` / `ɴ`(N) `ʔ`(cl) / `k kʲ kʷ ɡ ɡʲ ɡʷ s ɕ z dʑ
t ts tʲ tɕ d dʲ n ɲ h ç ɸ b bʲ p pʲ m mʲ j ɾ ɾʲ w v`.

**장음은 길이 기호가 아니라 모음 반복으로 쓴다**(`koo`, `koː` 아님). 일본어는 모라 박자
언어이고 duration predictor가 심볼 단위로 동작하므로, 모라마다 심볼 하나가 예측 분포를
단봉으로 유지한다. 그래서 `ː`는 사용하지 않는다.

**D1 (결정됨 2026-08-30) — 피치 악센트 표기 = base의 `↑`/`↓`.** 178 인벤토리를 유지해
2단계 체이닝이 안전하고, C6 완화를 기다리지 않아도 된다. 대가는 두 임베딩 행이 영어
학습에서 거의 안 쓰였다는 점 — 실질적으로 신규 행에 가깝고 코퍼스가 가르쳐야 한다.
대안이었던 `ꜜ`(U+A71C)는 C6 선행이 필요해 보류했다.

**D2 (열림) — 악센트구 경계 문자.** 현재 경계는 **공백**이라 어절 공백과 구분되지 않는다
(`pau`는 `,`로 구분된다). base 안에서 `—`가 비어 있어 청취에서 구 분할 문제가 보이면
심볼 수 변경 없이 교체할 수 있다. metadata의 `accent_phrase_boundary` 필드가 이 선택을
기록한다.

**구현 노트 — 모라 경계.** 널리 복사되는 라벨 단위 악센트 규칙은 **자기 자신이 악센트구인
1모라**(예: 조사 `と`)에서 자음과 모음 사이에 경계를 끼워 넣는다. `_MORA_FINAL_PHONES`로
모라 종단에서만 표기하도록 막았다 (`tests/test_ja_frontend.py`가 회귀를 잠근다).

**무성화 모음**(`I`/`U`)은 1차에서 평문 모음으로 접는다. 필요성이 청취로 확인되면 그때
별도 표기를 도입한다 — 추측으로 심볼을 늘리지 않는다.

### 5.2 supertonic-ja-ft에서 반영하는 것

같은 사용자의 일본어 파인튜닝 선행 프로젝트(`~/github/supertonic-ja-ft`, 실험 124건 ·
청취 교훈 81건)에서 **구조가 달라도 유효한 것**만 가져온다.

**반드시 반영**

| 항목 | 출처 | 로드맵 반영 위치 |
|---|---|---|
| 자동 지표는 전부 스크린, 판정자는 청취 | 교훈 1·2·47·72·74·75 | F4, G4 |
| source-file-disjoint split (발화 해시 split은 동일파일 오염 100%였다) | `product_voice_data.md` §4 (NORMATIVE) | M2, G2 |
| 사전 등록 게이트를 데이터 본 뒤 옮기지 않기 | 교훈 5·67 | §3 서두 |
| 수치는 렌더 관례와 함께만 인용 | 교훈 79 | M0 config 기록 |
| anime 코퍼스는 타 코퍼스보다 5.7~16.6 dB 뜨겁고 0 dBFS 초과가 코퍼스 전역 성질 — 행 필터로 못 거른다 | 교훈 70·71 | M2 클리핑 정책 |
| 데이터 권리 게이트 (JSUT/JVS 상업 학습 허가 / つくよみちゃん "다른 캐릭터" 조항 / anime CC0 주장 무효) | `licenses_and_rights.md` decision 9·13 | M2, M6 릴리스 노트 |

**고 F0 여성 타깃에 특히 유효**

- **교훈 14·15** — F0 평균만 맞추는 목적함수는 flat-pitch 퇴화해를 갖는다. 평균 +2.48 st를
  달성했는데 컨투어가 평평해져 "오히려 더 어색"이 나왔다. F4의 F0 진단을 median 단독이
  아니라 **IQR과 함께** 보는 이유다.
- **교훈 73·76** — 540.9 Hz에서 실제 녹음은 깨끗한데 시스템만 갈라졌다. "여성 고음이라
  원래 그렇다"를 미리 차단하는 관측치. G4가 고음역 행 포함을 요구하는 이유다.
- **교훈 57·69** — 최악 tail 행을 직접 청취 페이지에 올린다. arm 귀속(paired)과 절대
  품질(floor)을 **분리해** 사전 등록한다. 공유 결함에 절대 바닥선이 발화한 전례가 있다.

**반영하지 않음**

- ONNX forensic → PyTorch parity → wav-to-latent → adapter 스택 전체. supertonic 전제의
  대부분은 "학습용 체크포인트 미공개"에서 파생됐다. Inflect는 학습 가능한 체크포인트와
  warm-start 경로를 공개한다 — **우회로가 통째로 불필요하다.**
- `jf-surgical-v1~v6` 어휘사전 아크(exp-j#001a~002c). orthography 프론트엔드(모델이
  텍스트를 직접 소비) 제약에서 나온 싸움이며, V7이 그 실패 클래스의 소멸을 보인다.
  다만 **고유명사 오독이라는 새 실패 클래스**가 생기므로 사용자 사전 슬롯은 유지한다(M1).
- `midhigh` 비음 학습 표적 (교훈 81 — 소진됨). 새 학습 목적함수로 제안하지 않는다.
- jp-base v4 / ship candidate `xa080` 등 자산. Supertonic 아키텍처 전용.

### 5.3 한국어

**프론트엔드 파이프라인**
```
원문 → 정규화 → g2pkk (음운 규칙: 비음화·유음화·경음화·겹받침·수사 읽기)
     → 발음 한글 → espeak-ng(ko) → IPA → '-' → 'ʼ' → phoneme string
```

espeak 단독으로는 음운 규칙이 빠지고(V4), g2pkk 단독으로는 IPA가 없다. 조합이 두 문제를
동시에 해결한다(V6). 의존성: `g2pkk`, `python-mecab-ko`, `python-mecab-ko-dic` — 전부
wheel로 설치된다.

**일본어보다 쉬운 점**: 어휘 피치 악센트가 없어 D1에 해당하는 결정이 없다.
**어려운 점**: 어절 경계를 넘는 유음화 과적용이 관찰됐다(`오늘 날씨` → `오늘 랄씨`).
경계 처리 규칙과 회귀 테스트가 필요하다.

**데이터**: 미확보. HF 캐시의 `Bingsu/KSS_Dataset`은 메타데이터 스텁(12K)이고 오디오는
없다. `fsicoli/common_voice_17_0`(2.3G 캐시)은 다화자라 stage-1 후보다.
**한국어 단일 화자 코퍼스 확보는 M6의 선행 조건이며 사용자 결정 사항이다**(§8 Q3).

---

## 6. CUDA 머신 인계

### 6.1 환경 구축

```bash
cd ~/github/Inflect/finetune
python -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[onnx,dev]"
python -m pip install --only-binary=:all: pyopenjtalk-plus   # 일본어
python -m pip install g2pkk                                   # 한국어(M5부터)
pytest
```

**주의 (현 WSL에서 실제로 겪은 것)**
- `python3 -m venv`가 ensurepip 부재로 실패할 수 있다 → `python3 -m pip install --target`
  으로 우회하거나 `python3-venv` 설치.
- 본가 `pyopenjtalk`는 py3.10 wheel이 없다. **반드시 `pyopenjtalk-plus`**.
- `torch.cuda.is_available()`를 먼저 확인할 것. 현 WSL에서는 `False`였다.

### 6.2 데이터셋 마운트

```bash
wsl.exe -d Ubuntu-24.04 -u root -- bash -lc "mkdir -p /mnt/m && mount -t drvfs M: /mnt/m"
```

경로 목록은 `~/github/supertonic-ja-ft/configs/paths.example.yaml`,
환경 절차는 같은 저장소 `docs/environment.md`.

### 6.3 기준선 재현 (V1~V7 재확인)

M0에서 아래를 실행해 이 문서의 표가 그 머신에서도 참인지 확인한다.

```python
# 신규 심볼 0개 확인 (V1, V2)
import sys, unicodedata
sys.path.insert(0, "finetune")
from inflect_finetune.symbols import BASE_SYMBOLS
base = set(BASE_SYMBOLS)
print(len(BASE_SYMBOLS))                      # 178
print(sorted(set("aiɯeoɴʔkɡsɕzdʑtɕɸçɲɾʲʷː") - base))   # []  (JA)
print(sorted(set("ɐʌɯɫŋʰʼqtɕ") - base))                # []  (KO)
```

```python
# 일본어 G2P (V5, V7)
import pyopenjtalk
print(pyopenjtalk.g2p("彼女は2026年8月30日に来ます。"))
print(pyopenjtalk.g2p("抗うつ剤の対策について、痛み止め薬を飲みました。", kana=True))
```

```python
# 한국어 2단 파이프라인 (V6)
from g2pkk import G2p
print(G2p()("국물 좀 드세요. 신라면 맛있어요."))   # 궁물 좀 드세요. 실라면 마시써요.
```

### 6.4 첫 작업 순서

1. G0 통과 확인
2. C1~C5 (M1) — 코드 변경은 여기서 시작한다
3. G1을 **일본어 화자 검수까지** 통과시킨 뒤 M2로

---

## 7. 리스크 레지스터

| ID | 리스크 | 영향 | 완화 |
|---|---|---|---|
| R1 | 영어 남성 → 일본어 고 F0 여성 동시 이동이 3.96M/9.36M 용량을 초과 | 치명 | F3 2단계 분리. Micro 우선. Nano는 Micro 검증 후 |
| R2 | 신규 심볼 발생 시 체이닝 붕괴 | 높음 | C5(탐지) + C6(완화)를 M3까지 완료 |
| R3 | pyopenjtalk 고유명사 오독 | 중간 | 사용자 사전 슬롯. G1이 오독률을 **측정**하도록 설계 |
| R4 | `mel_fmax`가 여성 고음을 자르는 값 | 중간 | M0에서 실측 기록. 필요 시 학습 전 결정 |
| R5 | 청취 판정자가 1인(단일 리스너, 작은 N) | 중간 | supertonic이 동일 한계를 안고 갔다. 라운드 내 대비만 비교하고 MOS로 부르지 않는다 |
| R6 | anime 계열 클리핑이 고 F0에서 균열로 증폭 | 중간 | M2 클리핑 정책. 교훈 70·71 |
| R7 | 한국어 단일 화자 데이터 미확보 | 중간 | M6 선행 조건. §8 Q3 |
| R8 | F1 설계가 한국어에서 안 맞음 | 중간 | **G5가 바로 그 검증이다.** 실패하면 M1 회귀 |

---

## 8. 사용자 결정 필요

| # | 질문 | 관련 |
|---|---|---|
| Q1 | 일본어 stage-2 목표 화자를 어느 코퍼스로 할 것인가 (고 F0 여성 단일 화자) | M2 |
| Q2 | 최종 산출물의 **배포 계획**이 있는가? 있다면 つくよみちゃん "다른 캐릭터" 조항, No.7 비상업 조건, anime CC0 주장 무효가 stage-1 구성을 제약한다 | M2, M6 |
| Q3 | 한국어 단일 화자 코퍼스를 무엇으로 할 것인가 (신규 녹음 / 라이선스 확보) | M6 |
| Q4 | D1 — 일본어 피치 악센트를 `↑`/`↓`(178 유지) vs `ꜜ`(C6 선행) 중 무엇으로 | M1 |

---

## 변경 이력

| 날짜 | 내용 |
|---|---|
| 2026-08-30 | 최초 작성. V1~V10 실측 기준선 확립. |
| 2026-08-30 | M1 코드 완료. C2 삭제(레지스트리가 custom으로 해석), C5b 추가, D1 결정(`↑`/`↓`), D2 등재, 장음 정책 명시. G1은 화자 검수 대기. |
