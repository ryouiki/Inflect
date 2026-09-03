# 다국어 확장 구현 로드맵

**대상**: `finetune/` adaptation toolkit
**범위**: 일본어(1차) → 한국어(2차)를 **일회성 커스텀 훅이 아니라 툴킷의 기능**으로 지원
**작성일**: 2026-08-30 (최종 갱신 2026-09-04)
**상태**: M1·M5·C6 코드 완료. **M0/G0 통과(2026-09-04, RTX 5090 머신).** 타깃 화자·배포 방침 결정됨(§8 Q1–Q3).
**JA 경로는 하이브리드로 확정** — Micro→arona 직행을 먼저 측정하고, 실패 판정에서만 JSUT 단일화자 stage-1로,
그마저 실패할 때만 C7 다화자로 내려간다(§3.0). G1(c)는 사용자 결정으로 **자동 스크린으로 대체**한다.

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
| V4 | **eSpeak `ko`는 사용 불가.** 음운 규칙이 빠질 뿐 아니라(`옵니다`→`ˈopnidˌɐ`, `신라면`→`sˈinɾɐmjˌʌn`) **후두 대립을 붕괴시킨다** — 최소대립쌍 13개 중 6개(살/쌀·자다/짜다·불/뿔·방/빵·정/쩡·사/싸)가 같은 심볼열이 된다 | espeak-ng 1.52 실행 | 한국어도 커스텀 프론트엔드 **필수**. 2026-08-30 정정 |
| V5 | **`pyopenjtalk-plus` 0.4.1이 prebuilt wheel로 설치된다.** 본가 `pyopenjtalk`는 py3.10 wheel 없음(소스 빌드 필요) | `pip install --only-binary=:all:` | 일본어 의존성이 컴파일러 없이 해결됨 |
| V6 | ~~`g2pkk` + espeak `ko` 조합~~ → **폐기(2026-08-30).** 음운 규칙은 전달되지만 espeak가 후두 대립을 지운다(V4). **대체: `g2pkk` → 발음 한글 → 자모 직접 매핑.** 최소대립쌍 13/13 보존, base 밖 문자 0개, espeak 의존성 제거 | 실행 | 한국어 2단 구조(음운→자모) 확정 |
| V7 | **pyopenjtalk가 supertonic이 어휘사전으로 고쳐야 했던 항목을 그냥 맞게 읽는다.** `抗うつ剤`→コーウツザイ, `対策`→タイサク, `痛み止め薬`→イタミドメヤク, `2026年8月30日`→ニセンニジューロクネンハチガツサンジューニチ | 실행 | supertonic의 `jf-surgical-v1~v6` 아크는 이식 대상이 아님(§5.1) |
| V8 | **2단계 체이닝이 현재 코드로 동작한다.** `export`가 `config.json`+`model.pth`+`runtime/`을 쓰고 `resolve_base_model()`이 로컬 디렉터리를 받는다 | `exporting.export_checkpoint()`, `modeling.resolve_base_model()` | 언어 베이스 → 화자 적응 가능 |
| V9 | ~~심볼 수가 정확히 178이 아니면 체이닝이 깨진다~~ → **해소(2026-08-30, C6).** `load_runtime_components()`가 이제 `>= 178` + 릴리스 접두 일치를 본다 | `modeling.validate_release_compatible_symbols()` | 단일 실패점 제거. JA/KO는 애초에 178이라 체이닝은 이전에도 동작했다 |
| V10 | **다화자 준비가 하드 블록이다.** speaker 값이 2개 이상이면 `prepare`/`audit`이 즉시 실패 | `prepare_dataset()` · `audit_dataset()`의 speaker 가드 | 언어 베이스 단계에 우회 필요 |

### 확인하지 못한 것 — **전부 해소(2026-09-04, G0)**

작성 당시의 세 공백은 모두 메워졌다. 기록은 `inflect-work/env/G0.md`가 정본이다.

- ~~`M:` 미마운트~~ → 마운트 확인, JA/KO/JSUT를 로컬 디스크로 복제하고 검증했다.
  JA는 `metadata/checksums.sha256` **2019/2019 일치**, KO는 `wave` 헤더 전수
  **(1ch, 24-bit, 48 kHz) × 3401**·전사와 1:1, JSUT는 `repeat500` 제외 **7,196** 발화.
  NAS는 9p로 ≈53 files/s이므로 학습·prepare는 복제본에서만 한다.
- ~~`mel_fmin`/`mel_fmax` 미확인~~ → **`mel_fmin 0.0` / `mel_fmax 12000.0`**
  (`sampling_rate 24000`, `filter_length 1024`, `hop_length 256`, `win_length 1024`,
  `n_mel_channels 80`, `segment_size 16384`, `n_speakers 0`). 24 kHz의 Nyquist이므로 mel 손실이
  고음역을 자르지 않는다 — **§7 R4 해소**, 학습 전 결정 사항 없음.
- ~~CUDA 없음~~ → RTX 5090, capability **(12, 0) = sm_120**, torch **2.8.0+cu128**,
  `torch.cuda.is_available() == True`, `pytest` **111 passed / 1 skipped**,
  `INFLECT_TEST_BASE_MODEL=micro pytest -k inventory` **21 passed**(실물 체이닝 실증).
  HF `owensong/Inflect-Micro-v2`는 공개이며 토큰 없이 받아진다.

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
   회귀를 잡는다. **(M1에서 완료)**
2. `modeling.load_runtime_components()`의 `!= 178` 검사를 **`>= 178` + base prefix 일치**로
   완화한다. **(C6에서 완료)** 준비 데이터셋과 런타임 인벤토리가 이제 같은 규칙
   (`validate_release_compatible_symbols()`)을 쓴다.

이 두 개는 서로를 보완한다. (1)은 "심볼을 늘리지 마라", (2)는 "늘려야만 하는 언어가
나왔을 때 막다른 길이 아니게 하라"이다.

**C6가 무엇을 풀었고 무엇을 안 풀었는지**(2026-08-30): JA·KO 둘 다 신규 심볼이 0개라
**체이닝은 C6 이전에도 동작했다**(178 == 178). C6가 없앤 것은 잠재 실패 지점이고,
같이 메운 것이 더 크다 — `warm_start_from_release()`와 `load_runtime_components()`에
테스트가 하나도 없었고, CONTRACT.md가 최소 게이트로 요구하는 "embedding migration by
symbol identity"가 미검증이었다. 일본어 악센트는 `↑`/`↓` 그대로 두었다(D1 불변).

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

### 3.0 JA 경로 결정 — 하이브리드 (2026-09-04)

M3의 F3(2단계)는 "다화자 코퍼스로 언어 베이스"를 전제하지만 그 전제는 C7 미완으로 오늘 실행할 수
없고(§4 C7, `docs/TRAINING.md`), R1은 **측정된 사실이 아니라 사전 우려**다. 한편 arona JA는 1–12 s 창에서
1,282클립/99.7분으로, supertonic이 "~116발화면 암기한다"고 경고한 규모의 약 10배다. 그래서 직행 1런이
R1을 가장 싸게 측정한다. 되돌아갈 때의 stage-1도 **C7이 필요 없는 단일 화자 JSUT**(7,196발화, 로컬
복제 완료)를 먼저 쓴다.

전환 규칙은 **착수 전에 고정**한다.

| 판정점 | 조건 | 행동 |
|---|---|---|
| **T1** JA-A step 3000 (decoder 해제 직전) | step-3000 체크포인트로 OOD 10행을 렌더해 들었을 때 "일본어로 들리지 않음"(정체성·품질 불문 — G3 문구 재사용) | JA-A 중단 → JSUT stage-1(JA-B1) 즉시 착수 |
| **T2** JA-A G4 라운드 | (i) 자동 스크린 통과 후보가 0개, 또는 (ii) 블라인드 페이지의 **절대 바닥(floor, paired와 분리 기록)**에서 "언어 불명/붕괴"가 결정 블록 항목의 과반 | JA-B1 → JA-B2(arona stage-2) 착수. JA-A 선택본은 JA-B2의 paired baseline으로 보존 |
| **T3** JA-B2 G4 라운드 | JA-B2도 T2 조건 실패 **그리고** 자유기술 결함이 "음운/타이밍" 축에 몰림 | **C7 구현** → JVS 여성 화자(+JSUT) 다화자 stage-1. 결함이 "음역/균열" 축이면 C7이 아니라 데이터(고음역 행)를 다시 본다 |
| **T-KO** | KO는 어떤 시나리오에서도 직행(4.825 h 단일화자). KO G4 실패 시에만 KSS stage-1 검토 | 그때 `Arona_KSS` 952/998행이 KSS 텍스트와 동일하므로 stage-2 **검증셋에서 KSS 중복 텍스트를 제거**해야 비교가 성립한다 |
| **T-Nano** | Micro JA G4 통과 전 Nano 착수 금지 | §5 LANGUAGES.md "Use Micro for the first adaptation attempt" |

**직행 경로의 G4 baseline 정의**: 로드맵 G4는 "baseline 대비 우세"인데 직행에는 stage-1 export가 없다.
직행 라운드의 baseline은 **자동 스크린을 통과한 가장 이른 후보 체크포인트**로 한다(학습 진행 효과를 측정).

### M0 — 환경 확정 (CUDA 머신) — **통과 (2026-09-04)**

`inflect-work/env/G0.md`가 기록 정본이다. 요약: sm_120 / torch 2.8.0+cu128 / cuda True /
`pytest` 111 passed·1 skipped / 실물 인벤토리 체이닝 21 passed / Micro 공개 다운로드 성공 /
`mel_fmax 12000` → R4 해소 / JA·KO·JSUT 로컬 복제 및 검증 완료.

G0에서 **공개 툴킷 결함 2건**이 드러나 함께 고쳤다(둘 다 이 문서의 인계 절차를 따라가다 나왔다).

1. `finetune/`에서 `python -m venv .venv`(§6.1의 지시)를 하면 `tests/test_public_safety.py`가
   트리 전체를 `rglob`하며 `.venv/`의 비 UTF-8 파일을 읽어 `UnicodeDecodeError`로 실패했다.
   → 발행 표면이 아닌 트리(dot 디렉터리·`build`·`dist`·`*.egg-info`)를 스캔에서 제외.
2. `pyproject.toml`이 `unidecode`/`num2words`를 선언하지 않았다. 그런데 웜스타트는 릴리스의
   `runtime/text/cleaners.py`를 임포트하고 그 파일이 `unidecode`를 임포트한다 → 깨끗한 설치에서
   `train --base micro`가 사용자 코드와 무관하게 `ModuleNotFoundError`로 죽는다.
   → 릴리스 `requirements.txt`와 같은 하한으로 두 패키지를 선언.

부수로 `[ko]` 주석을 사실에 맞게 고쳤다: g2pkk는 python-mecab-ko를 **선언하지 않고 첫 임포트에서
런타임 pip install**을 한다(관측). 임포트가 네트워크와 쓰기 권한을 요구하는 것은 재현성 위험이라
분석기를 엑스트라에 명시했다.

- [x] CUDA torch 동작, `nvidia-smi`, VRAM 확인
- [x] `M:` 마운트 및 일본어 데이터셋 실물 확인 (+ 로컬 복제·체크섬 검증)
- [x] `finetune/` editable 설치 + `pytest` 전체 통과
- [x] 릴리스 `config.json`의 `mel_fmin`/`mel_fmax`/`filter_length`/`hop_length` 기록

**게이트 G0**: `inflect-adapt --help`, 기존 테스트 스위트 green, Micro 릴리스 다운로드 성공.
→ **통과.** 세 항목 모두 확인. 단, 두 번째 항목은 위 결함 2건을 고친 뒤에야 green이 됐다.

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

- ~~`modeling.py` 178 제약 완화(F2-2)~~ **완료(C6)** — 남은 것은 `--corpus-role`뿐
- `--corpus-role` 구현 (C7, 미착수. 지금은 다화자 데이터셋 준비가 거부된다)
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

### M5 — 한국어 프론트엔드 (구조 검증) — **코드 완료 (2026-08-30)**

- `ko_g2pkk.py`: g2pkk → 발음 한글 → 자모 직접 매핑 (espeak 제외, V4/V6 정정)
- **M1에서 만든 레지스트리에 코드 변경 없이 얹히는지**가 진짜 게이트다.
  얹히지 않으면 F1 설계가 틀린 것이므로 M1로 되돌아간다.

**게이트 G5 — 통과.** 파이프라인 파일(`frontend.py`·`prepare.py`·`audit.py`·`cli.py`·
`exporting.py`) **변경 0줄**. 변경은 `frontends/__init__.py`의 REGISTRY 항목 1개
(설계된 확장점) + `pyproject.toml`의 extra + 검수 스크립트의 한국어 읽기 분기뿐이다.
신규 심볼 0개, 최소대립쌍 13/13 보존.

**남은 것**: 한국어 화자 검수(G1(c)에 해당). 커버리지 스위트 246문장 덤프는 실패 0이며,
실제 전사 표본 검수가 남았다.

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
| C4 | `ja_openjtalk.py` | `frontends/ja_openjtalk.py` | M1 | ✅ 완료. pyopenjtalk-plus. 2026-09-04에 `fy`(フュ) 누락을 JSUT 덤프가 잡아 추가 — 음소 인벤토리 전수 테스트로 잠금 |
| C5 | `--require-no-new-symbols` | `audit.py`, `cli.py` | M1 | ✅ 완료 |
| C5b | export의 동봉 훅 자동 해석 | `cli.py` | M1 | ✅ 완료. 없으면 JA 경로가 end-to-end로 닫히지 않는다 |
| C6 | 178 → `>=178 + prefix` 완화 | `modeling.validate_release_compatible_symbols()` | M3 | ✅ 완료. 마이그레이션 테스트 공백도 같이 메움 |
| C7 | `--corpus-role` (다화자 허용) | `prepare_dataset()` · `audit_dataset()` | M3 | **조건부 보류.** §3.0 T3에서만 착수 — stage-1은 단일화자 JSUT로 성립한다. 기본 동작 불변 |
| C8 | F0 진단 추가 | `evaluation._signal_metrics()` | M4 | ✅ 완료(2026-09-04). median·IQR·유성 프레임 비율. fmax 1000 |
| C9 | ASR/CER 플러그인 (JA/KO) | `examples/transcript_evaluator_asr.py` | M4 | ✅ 완료(2026-09-04). `INFLECT_ASR_MODEL_DIR` + `local_files_only`, 자동 다운로드 금지 유지 |
| C10 | 블라인드 A/B 페이지 생성기 | `examples/build_blind_ab_page.py` · `examples/tally_verdict.py` | M4 | ✅ 완료(2026-09-04). 행별 무작위 라벨·봉인 mapping·실물 앵커 강제·catch 행 |
| C11 | `ko_g2pkk.py` | `frontends/ko_g2pkk.py` | M5 | ✅ 완료. 파이프라인 파일 변경 0 — **G5 통과** |
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

**프론트엔드 파이프라인** (구현됨)
```
원문 → 정규화 → g2pkk (어절 단위) → 발음 한글 → 음절 분해 → 자모 IPA 매핑 → 심볼열
```

**espeak는 체인에서 제외한다.** 후두 대립을 붕괴시키기 때문이다(V4 정정). 한글이
자질문자라, 음운이 이미 적용된 발음 한글을 음소로 바꾸는 것은 기계적 음절 분해다.
의존성은 `g2pkk` 하나(+ wheel로 따라오는 `python-mecab-ko`)로 줄었다.

경음은 `ʼ`(U+02BC), 격음은 `ʰ`(U+02B0) — 둘 다 base 안이라 3중 대립이 심볼을 늘리지 않는다.

**어절 단위로 돌린다.** 문장 전체를 넘기면 g2pkk가 경계를 넘어 연음을 과적용해
`오늘 날씨`→`오늘 랄씨`, `희망을 얘기`→`히망으 럐기`가 된다. 대가는 경계를 넘는 비음화를
놓치는 것(`몇 년`→`멷 년`)인데, 틀린 단어가 아니라 또박또박한 발음이라 안전하다.

**영문·낱자모는 거부한다.** g2pkk가 `IT`를 `읻`으로, `AI`를 `아이`로 읽으면서 라틴
문자를 남기지 않아 출력 검사로는 못 잡는다. 정규화된 **입력**에서 검사한다.
읽기는 `INFLECT_KO_LEXICON`으로 준다.

**알려진 g2pkk 한계**(코드로 고치지 않고 문서화 — 고치려면 수사 읽기를 재구현해야 하고
그건 "추측하지 않는다" 선을 넘는다): `세기`·`층`·`장` 앞 두 자리 이상 수는 자릿수로
읽힌다(`21세기`→`이일세기`). 단위 없는 맨 숫자도 자릿수로 읽힌다. 경음화가 가끔
과소적용된다(`여덟 시`→`여덜 시`).

**일본어보다 쉬운 점**: 어휘 성조가 없어 D1에 해당하는 결정이 없다.

> **검토 필요 (K1-D1)**: ㅐ/ㅔ를 `ɛ`/`e`로 **구분 유지**한다. 현대 서울말에서는 병합됐지만
> 병합은 되돌릴 수 없고 구분 유지는 되돌릴 수 있다. ㅚ/ㅞ는 둘 다 `we`. 청취 판단 항목.

**데이터**: 미확보. HF 캐시의 `Bingsu/KSS_Dataset`은 메타데이터 스텁(12K)이고 오디오는
없다. `fsicoli/common_voice_17_0`(2.3G 캐시)은 다화자라 stage-1 후보다.
**한국어 단일 화자 코퍼스 확보는 M6의 선행 조건이며 사용자 결정 사항이다**(§8 Q3).

---

## 6. CUDA 머신 인계

### 6.1 환경 구축 — 2026-09-04 실행판

이 머신의 저장소는 `/home/ysoya/projects/Inflect`다(작성 당시의 `~/github/...`가 아니다).
Blackwell(sm_120)에서는 **torch를 cu128 인덱스에서 먼저** 깔아야 한다. `pyproject`의 `torch>=2.2`는
sm_120 하한을 강제하지 않으므로, 순서를 바꾸면 기본 PyPI 빌드가 들어와 커널이 없다.

```bash
cd /home/ysoya/projects/Inflect/finetune
python3 -m venv .venv && source .venv/bin/activate && unset LD_LIBRARY_PATH
python -m pip install -U pip
python -m pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[onnx,dev,ja,ko]"
pytest                                    # 111 passed, 1 skipped
INFLECT_TEST_BASE_MODEL=micro pytest -k inventory -p no:warnings    # 21 passed
```

**주의 (실제로 겪은 것)**
- `~/.bashrc`가 `LD_LIBRARY_PATH=/usr/local/cuda/lib64`를 설정한다. 시스템 CUDA lib가 wheel 동봉
  cuDNN/cuBLAS를 가릴 수 있으므로 **torch 셸에서는 `unset LD_LIBRARY_PATH`**.
- 이 머신에는 Python 3.12.3만 있고 `python3-venv`·`ensurepip`가 정상이라 작성 당시의 ensurepip
  우회는 필요 없었다. 3.12는 오히려 유리하다 — `onnxruntime` 1.29는 cp310 wheel이 없다.
- 본가 `pyopenjtalk` 대신 **`pyopenjtalk-plus`**(V5) 는 그대로 유효하다.
- `espeak-ng` CLI는 이 머신에 없지만 `espeakng-loader` wheel이 라이브러리를 제공하므로 설치하지
  않았다. JA/KO 프론트엔드는 espeak을 쓰지 않는다.
- g2pkk는 python-mecab-ko를 **첫 임포트에서 런타임 설치**한다. `[ko]` 엑스트라가 이제 분석기를
  명시하므로 그 부작용에 의존하지 않는다.

### 6.2 데이터셋 마운트

```bash
wsl.exe -d Ubuntu-24.04 -u root -- bash -lc "mkdir -p /mnt/m && mount -t drvfs M: /mnt/m"
```

경로 목록은 `/home/ysoya/projects/supertonic-ja-ft/configs/paths.example.yaml`,
환경 절차는 같은 저장소 `docs/environment.md`. (작성 당시 표기한 `~/github/...`는 이 머신에 없다.)

`/mnt/m`은 9p로 **≈53 files/s · ≈25 MB/s**다. 많은 작은 파일에서 병목이 되므로 학습·prepare는
로컬 복제본에서만 한다. 2026-09-04 복제·검증한 것:

| 데이터셋 | 로컬 경로 | 검증 |
|---|---|---|
| arona/plana JA (processed) | `inflect-work/data/ja` | `sha256sum -c` 2019/2019 |
| arona KR | `inflect-work/data/ko/FINETUNE_Arona_KR` | 3,401파일 · 48 kHz/1ch/24-bit 전수 · 전사 1:1 · 4.825 h |
| JSUT (`repeat500` 제외) | `inflect-work/data/jsut` | 8 하위셋 7,196발화 |

### 6.3 기준선 재현

M0에서 아래를 실행해 이 문서의 표가 그 머신에서도 참인지 확인한다. V4·V6는
2026-08-30에 정정됐다(espeak `ko` 제외) — 아래 한국어 스니펫은 정정 후 기준이다.

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
# 한국어 (V6 정정판: espeak 없이 자모 직접 매핑)
from g2pkk import G2p
print(G2p()("국물 좀 드세요."))   # 궁물 좀 드세요.
```

프론트엔드 자체 확인은 테스트가 대신한다 — `pytest -k "ko_frontend or ja_frontend"`.

### 6.4 인계 상태와 첫 작업 순서

브랜치 `feat/multilingual-frontend-registry`. 프론트엔드 스택과 마이그레이션 경로는
끝났고, 남은 GPU-불필요 작업(C7·C8–C10)은 착수하지 않았다 — 학습 대기 시간에 넣을 수
있도록 남겨둔 것이다.

```bash
git fetch origin && git checkout feat/multilingual-frontend-registry
cd finetune
python -m pip install -e ".[onnx,dev,ja,ko]"
pytest                       # 111 passed, 1 skipped (opt-in 실물 테스트)
```

| 완료 | 내용 |
|---|---|
| M1 | 프론트엔드 레지스트리 + `ja-openjtalk` (신규 심볼 0, 피치 악센트 `↑`/`↓`) |
| M5 | `ko-g2pkk` (신규 심볼 0, 후두 대립 13/13 보존). **G5 통과** — 파이프라인 파일 0줄 |
| C6 | 확장 인벤토리 베이스 허용 + 미검증이던 임베딩 마이그레이션 테스트 |

| 상태 (2026-09-04) | 내용 |
|---|---|
| ✅ **M0/G0** | 통과. §3 M0과 `inflect-work/env/G0.md` |
| ✅ **체이닝 실증** | `INFLECT_TEST_BASE_MODEL=micro pytest -k inventory` → 21 passed (실물 Micro) |
| 🔁 **G1(c)** | 사용자 결정으로 **사람 카나 검수를 자동 스크린으로 대체**한다(아래) |
| ⏳ M2 | 매니페스트·prepare·audit 진행 중 |
| ⏳ C10 → C8 → C9 | GPU 불필요. JA-A/KO-A 학습 시간에 끼워 넣는다 |
| ⏸ C7 `--corpus-role` | **조건부 보류.** §3.0 T3에서만 착수한다 — 그때까지 stage-1은 단일화자 JSUT로 충분하다 |

**G1(c)의 사용자 결정 (2026-09-04)**: 사람 카나 검수 대신 **자동 스크린**을 게이트로 쓴다.
`reading_source == xlsx_reading_exact`인 1,140행에 대해 프론트엔드 덤프의 `reading` 열과 데이터셋
`reading_text`를 카타카나 통일·구두점 제거 후 정규화 편집거리로 비교하고, 분포와 상위 행을 기록한다.
거리 ≥ 0.5 행은 전사 불일치로 보고 매니페스트에서 제외한다. **이것은 로드맵 G1(c)의 예외이며,
"판정자는 청취"라는 원칙을 이 항목에서만 완화한 것이다** — 고유명사 오독(R3)의 실측 상한이 사라지므로
G4 라운드에서 발음 결함이 나오면 여기를 먼저 의심한다.

**첫 작업 순서** (1·2는 완료)

1. ~~G0~~ ✅ 2. ~~체이닝 실증~~ ✅
3. **G1 덤프** — 전 행(JA 1,366 / KO 3,401)에 `examples/frontend_review_dump.py`를 돌려 FAILED를 0으로
   만든다. `prepare`는 한 행만 실패해도 전체를 롤백하므로 이 덤프가 선행 게이트다. 덤프가 잡지 못하는
   **조용한 통과**(`☆ ～ 〜 ~`처럼 JA 구두점 맵에 없어 그대로 Open JTalk에 넘어가는 문자)는 매니페스트
   빌더에서 처리한다.
4. **M2** — 매니페스트(`group_id` = source file), 클리핑 정책, stage-1/stage-2 분리.
5. C8–C10은 GPU를 쓰지 않으므로 학습 대기 시간에 끼워 넣을 수 있다.

**인계받는 사람이 먼저 알아야 할 것**

- `export`에는 **`--package-template micro`가 필수**다. 학습 체크포인트는 base model을
  저장하지 않으므로(run identity 해시에 들어가기 때문) 익스포터가 추론할 수 없고,
  `--verify` 기본값이 true라 없으면 실패한다.
- **다화자 데이터셋은 아직 준비할 수 없다** (C7 미완). stage-1도 단일 화자여야 한다.
- **`--min/--max-duration-seconds`는 필터가 아니라 단언이다.** 범위 밖 행이 하나라도 매니페스트에
  있으면 `inspect_wav`가 던지고 `prepare`가 스테이징을 지우며 전체 실패한다. 길이·텍스트 필터는
  **매니페스트를 만들 때** 끝내고, 이 플래그는 같은 값을 단언하는 용도로만 쓴다.
- **`evaluate`는 행에 `audio`가 있으면 디스크 오디오를 읽고 모델을 로드하지 않는다.**
  `validation.jsonl`을 그대로 넘기면 실물 앵커 지표가 나오고(그것이 "같은 채널" 앵커의 정확한
  용도다), 렌더를 원하면 `audio`·`phonemes`를 제거한 사본을 넘겨야 한다. 후보 체크포인트를
  `--checkpoint`로 바꿔가며 `validation.jsonl`을 넘기면 전 후보가 같은 실물 수치를 낸다.
- **`max_steps`는 run identity에 들어간다**(`_public_options`가 제외하는 것은
  `base_model/prepared_dir/output_dir/preset/resume`뿐). 스텝 수를 바꾼 `--resume`은 거부되므로
  학습 연장은 export → `--base` 체이닝으로 새 run을 만드는 것뿐이다.
- `metrics.jsonl`에는 **타임스탬프도 검증 loss도 없다.** 벽시계는 로그로 재고, 체크포인트 선택은
  `evaluate` + 청취로 한다.
- **알려진 G2P 한계는 `docs/LANGUAGES.md`에 언어별로 정리돼 있다** — 예를 들어
  한국어는 `세기`·`층`·`장` 앞 두 자리 수를 자릿수로 읽는다. 새 오독을 만나면
  프론트엔드를 고치기 전에 먼저 거기를 본다.
- 자동 지표는 전부 스크린이고 판정자는 청취다(§5.2). 이 저장소의 결함 대부분은
  테스트가 아니라 **검수 덤프를 눈으로 훑다가** 나왔다.

---

## 7. 리스크 레지스터

| ID | 리스크 | 영향 | 완화 |
|---|---|---|---|
| R1 | 영어 남성 → 일본어 고 F0 여성 동시 이동이 3.96M/9.36M 용량을 초과 | 치명 | **가설이므로 직행 1런으로 측정한다**(§3.0). 되돌아갈 때 stage-1은 단일화자 JSUT → 그마저 실패면 C7 다화자. Micro 우선, Nano는 Micro 검증 후 |
| ~~R2~~ | ~~신규 심볼 발생 시 체이닝 붕괴~~ | — | **해소.** C5(탐지, M1) + C6(완화) 완료 |
| R3 | pyopenjtalk 고유명사 오독 | 중간→**높음** | 사전 슬롯은 구현됨(`INFLECT_JA_LEXICON`). 그러나 G1(c) 사람 검수가 자동 스크린으로 대체돼(§6.4) **오독률 실측치가 없다**. G4에서 발음 결함이 나오면 여기를 먼저 본다 |
| ~~R4~~ | ~~`mel_fmax`가 여성 고음을 자르는 값~~ | — | **해소(2026-09-04).** `mel_fmax = 12000` = 24 kHz Nyquist. 학습 전 결정 없음 |
| R5 | 청취 판정자가 1인(단일 리스너, 작은 N) | 중간 | supertonic이 동일 한계를 안고 갔다. 라운드 내 대비만 비교하고 MOS로 부르지 않는다 |
| ~~R6~~ | ~~anime 계열 클리핑이 고 F0에서 균열로 증폭~~ | — | **해당 없음.** anime 코퍼스를 쓰지 않는다. arona JA는 loudnorm −23·`clip_rows 0`, arona KR은 진클리핑 0 |
| ~~R7~~ | ~~한국어 단일 화자 데이터 미확보~~ | — | **해소.** §8 Q3 = `FINETUNE_Arona_KR` 3,401클립/4.825 h(검증 완료) |
| ~~R8~~ | ~~F1 설계가 한국어에서 안 맞음~~ | — | **해소.** G5 통과 — `ko_g2pkk.py` 추가 외 파이프라인 파일 0줄 |
| **R9** | 짧은 클립·비어휘 발성이 duration predictor를 훼손 | 중간 | JA <1 s 39클립·KO <1 s 5클립은 길이 하한으로, 구두점만 남는 행은 텍스트 하한으로 매니페스트에서 제외 |
| **R10** | JA 전사 품질 — `C_AronaChan` 170클립은 스크립트 없는 ASR 전사, 소스 표본 severe 3.3% | 중간 | 사용자 결정으로 **포함 + `text_source=asr_only` 태그**. 자동 스크린 상위 tail을 제외하고, JA-A 통과 후 제외 ablation 1회 |
| **R11** | 세션/소스 누출 — 한 소스에서 최대 26세그먼트, 배포 `splits/*`는 텍스트 20개·소스 14개가 교차 | 높음 | `group_id = source_relative_path` + 툴킷 group-aware split. 배포 split 파일 미사용 |
| **R12** | KO arona는 **JP arona와 다른 성우** | 치명(오용 시) | 트랙·데이터셋·`speaker` 값(`arona` vs `arona_kr`) 완전 분리. 체이닝·혼합 금지 |
| **R13** | KO `Chatbot_` 1,299클립(38%)이 다른 파이프라인 — `Arona_` 접두 없음, >16 kHz 에너지 8배, 같은 NAS에 2023-04 arona RVC 체크포인트 존재 | 높음 | prepare 전 **블라인드 스팟체크**(Chatbot 20 + 타 서브코퍼스 20, 봉인 mapping). 불합격 시 제외본으로 run 1 |
| **R14** | 단일 세션 코퍼스 암기 | 중간 | JA-A 8,000 step(≈53 epoch), KO-A 10,000 step. 프리셋 20k(≈133 epoch)는 쓰지 않는다. OOD 텍스트를 모든 청취 게이트에 넣는다 |
| **R15** | 학습 연장이 불가 — `max_steps`가 run identity에 포함 | 낮음 | 스텝 예산을 사전 선언하고, 연장은 export → `--base` 체이닝 새 run으로 |

---

## 8. 사용자 결정 필요

| # | 질문 | 관련 |
|---|---|---|
| ~~Q1~~ | ~~일본어 stage-2 목표 화자~~ | **결정(2026-09-03): `Ja_Voice_AroPla_processed/wavs/arona`** — 1,366클립/110.1분, 48 kHz mono 16-bit, loudnorm −23/TP −1.5, `clip_rows 0`. Plana는 혼입 금지 |
| ~~Q2~~ | ~~배포 계획~~ | **결정(2026-09-03): 배포 없음, 사설 연구.** 따라서 No.7 비상업 조항·つくよみちゃん "다른 캐릭터" 조항·anime CC0 의심은 **학습 입력 사용을 막지 않는다**(권리 기록 decision 13 "매니페스트·캐시·학습 입력 사용 승인, 외부 배포 금지"). 대신 산출물 외부 공개 금지가 유지되고, 매니페스트는 `license`/`distribution_gate`를 보존해 배포 가능 부분집합을 사후 재구성할 수 있게 한다 |
| ~~Q3~~ | ~~한국어 단일 화자 코퍼스~~ | **결정(2026-09-03): `ko_AronaPlana_voice/FINETUNE_Arona_KR`** ("arona kr") — 3,401클립/4.825 h, 48 kHz mono 24-bit, 전사와 1:1. **JP arona와 다른 성우이므로 별개 화자 트랙**(R12). 권리 기록이 어디에도 없어 한 줄 문구가 필요하다(비차단) |
| ~~Q4~~ | ~~D1 — 일본어 피치 악센트 표기~~ | **결정됨: `↑`/`↓`** (§5.1 D1). C6 완료로 `ꜜ`도 가능해졌지만 재개방하지 않음 |
| Q5 | D2 — 일본어 악센트구 경계를 공백 유지 vs base의 `—`로 분리 (§5.1 D2) | 청취 후 |
| Q6 | K1-D1 — 한국어 ㅐ/ㅔ 구분 유지 vs 병합 (§5.3) | 청취 후 |

---

## 변경 이력

| 날짜 | 내용 |
|---|---|
| 2026-08-30 | 최초 작성. V1~V10 실측 기준선 확립. |
| 2026-08-30 | M1 코드 완료. C2 삭제(레지스트리가 custom으로 해석), C5b 추가, D1 결정(`↑`/`↓`), D2 등재, 장음 정책 명시. G1은 화자 검수 대기. |
| 2026-08-30 | M5 코드 완료, **G5 통과**(파이프라인 파일 0줄). V4·V6 정정 — espeak `ko`가 후두 대립을 붕괴시켜 체인에서 제외하고 자모 직접 매핑으로 대체. K1-D1 등재. |
| 2026-08-30 | C6 완료. V9 해소. 인벤토리 검증을 준비 데이터셋·런타임이 공유하고, 미검증이던 임베딩 마이그레이션 경로에 테스트를 넣었다. 버려지는 base 심볼을 `compatibility-report.json`에 보고. |
| 2026-08-30 | 인계 리뷰. **문서의 export 예시 5개가 그대로는 실패하던 것을 수정** (`--package-template` 필수). §6.4를 인계 상태 기준으로 재작성. R2·R8 해소, Q4 결정 반영, Q5·Q6 등재. |
| 2026-09-04 | **M0/G0 통과** (RTX 5090 / sm_120 / torch 2.8.0+cu128 / pytest 111·1 / 실물 체이닝 21 passed). `mel_fmax 12000` 실측 → R4 해소. JA·KO·JSUT 로컬 복제·검증. Q1·Q2·Q3 결정 반영, R6·R7 해소, R9–R15 등재. **JA 경로를 하이브리드로 확정**하고 T1–T3 전환 규칙·직행 G4 baseline 정의를 §3.0에 고정. G1(c)를 자동 스크린으로 대체(예외 명시, R3 상향). C7을 조건부 보류로 이동. §6.1·§6.2·§6.4를 이 머신 기준으로 재작성. **G0을 따라가다 드러난 공개 툴킷 결함 2건 수정** — 문서가 지시한 `finetune/.venv`가 `test_public_safety`를 깨뜨린 것, `pyproject`가 릴리스 런타임의 `unidecode`/`num2words`를 선언하지 않아 깨끗한 설치에서 `train --base micro`가 죽던 것. |
