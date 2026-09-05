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

### M2 — 데이터 준비 — **통과 (2026-09-04)**

준비한 데이터셋 3개. 모두 `added_symbol_count == 0` · `audit.valid true`(strict +
no-new-symbols) · 검증셋 음소 ⊆ 학습셋 음소 · 언어별 필수 심볼이 검증셋에 존재.

| 데이터셋 | 역할 | 규모 | 비고 |
|---|---|---|---|
| `ja-arona-v2` | JA stage-2 (목표 음성) | 1,245행 / 96.96분 / 소스파일 1,188 | `↑` 4,346 · `↓` 4,465 (검증셋 289/278) |
| `ja-jsut-v1` | JA stage-1 (언어 베이스) | 7,101행 / 9.09 h | 단일 여성 화자라 C7 불필요 |
| `ko-arona-v1b` | KO 단독 트랙 | 3,348행 / 4.637 h | 균일 −3 dB 후 재준비(아래) |

- `group_id`는 전부 **소스 녹음 파일**이다. arona JA는 한 소스에서 최대 26개 VAD 세그먼트가
  나오므로 발화 단위 split은 같은 세션을 양쪽에 걸친다. 배포된 `metadata/splits/*`는
  텍스트 20개·소스 14개가 교차하므로 쓰지 않았다.
- **`--min/--max-duration-seconds`는 필터가 아니라 단언이다.** 길이·텍스트 필터는 매니페스트
  빌더가 전부 적용하고, 플래그에는 같은 값을 넘겨 단언으로만 쓴다.
- **`A_CO026` 서브코퍼스 전체 제외**(사용자 결정). 자동 스크린에서 이 조수사 세기 드릴만
  녹음 스크립트 읽기와 p50 0.091 · p90 0.250이었고(다른 12개는 p50 ≤0.045), `九時`를 성우는
  キュウジ로 읽었는데 프론트엔드는 クジ로 읽는 식으로 **텍스트–오디오 정렬이 어긋난다**.
  같은 실패 계열이라 JSUT의 `countersuffix26`도 제외했다.
- `C_AronaChan` 170행은 **포함 + `text_source` 보존**(사용자 결정). ASR 전용 전사라
  R10으로 남고, JA G4 통과 후 제외 ablation으로 영향을 재본다.
- **클리핑 정책은 예상과 다른 곳에서 발동했다.** anime 코퍼스는 애초에 쓰지 않으므로
  §5.2의 교훈 70·71은 해당이 없었고, 문제는 **우리 준비 과정**이었다 — arona KR 원본은
  −0.1 dBFS로 리미팅돼 소스 클리핑이 사실상 없는데, 48→24 kHz 리샘플이 최대 +2.1% 링잉을
  만들고 `prepare`가 그 위를 잘라 **3,348행 중 331행(9.89%)의 상단이 사라졌다**.
  사전 선언대로 로컬 사본 전체에 균일 −3 dB(`sox gain -3`)를 적용해 재준비했다.
  서브코퍼스별 게인과 행 필터는 쓰지 않았다 — 레벨 관계는 행의 성질이 아니라 코퍼스의
  성질이다(교훈 71의 요지). 기록은 `inflect-work/prepared/KO-G2-RECORD.md`.
- 이 과정에서 **공개 툴킷 결함 2건**을 더 고쳤다: `inspect_wav`가 sox·ffmpeg가 24-bit에
  쓰는 WAVE_FORMAT_EXTENSIBLE을 거부했고, `prepare`가 **자기가 만든 클리핑을 보고하지
  않았다**. 이제 `dataset.json.diagnostics.output_clipped_files`가 소스 측 수치와 나란히 나온다.

**게이트 G2**: `phoneme_coverage.json`의 `added_symbol_count == 0`,
group/텍스트 누출 0, 검증셋 음소가 학습셋에 전부 존재. → **세 데이터셋 모두 통과.**

> **운영 교훈 — 프론트엔드를 고치면 그 전에 준비한 데이터셋은 export가 거부한다.**
> `dataset.json`이 훅 소스 해시를 기록하고 `export`가 다시 해시하기 때문이다(설계된 동작,
> `docs/TROUBLESHOOTING.md`에 있다). `fy` 추가로 `ja-arona-v1`/`v2`가 그 상태가 됐고
> `v2`는 재준비했다. `v1`은 T1 기록용으로만 남기고 그 이후 체크포인트는 export할 수 없다.
> **프론트엔드 수정은 prepare 앞에 온다.**

### M3 — F3 다단계 학습 배선 + 일본어 stage 1

- ~~`modeling.py` 178 제약 완화(F2-2)~~ **완료(C6)**
- ~~`--corpus-role`(C7)~~ → **조건부 보류.** stage-1을 단일화자 JSUT로 잡으면 필요가 없다.
  §3.0 T3에서만 착수한다.
- ja-base 학습 (Micro) — **진행 중**: `runs/ja-jsut-stage1-20260904`, 프리셋 20,000 step,
  batch 8 / accum 1, 실측 ≈3.0 step/s. 스테이지 기본값 유지(posterior_warmup 500,
  decoder_unfreeze 3000).
- export → 그 디렉터리를 `--base`로 재로드하는 **체이닝 스모크**
  → 실행 스크립트 `inflect-work/scripts/run_g3.sh`에 게이트 조건까지 고정해 뒀다.

> **JSUT 언어 베이스의 한계를 미리 기록한다.** JSUT는 낭독 평서문이라 학습셋 전체에
> `!`가 1회, `?`가 4회뿐이다(`G2-extra.txt`). 즉 **의문·감탄 프로소디와 고음역은 언어
> 베이스가 가르쳐주지 않고 stage-2의 arona 데이터가 전부 감당해야 한다.** G4에서 고음역
> 결함이 나오면 이 사실을 먼저 본다 — 모델 용량 문제로 오인하기 쉬운 자리다.

**게이트 G3**: stage-1 export를 stage-2의 `--base`로 로드해 1 step 학습이 돈다.
held-out 합성이 일본어로 들린다(정체성·품질 불문).

### M4 — 평가 확장 + 일본어 stage 2

- ~~F4(CER 플러그인, F0 진단, 블라인드 페이지)~~ **완료(C8·C9·C10, 2026-09-04)**.
  청취 파이프라인은 실물 렌더로 예행 검증했다(`listening/rehearsal`, page_key
  `rehearsal-not-a-verdict`) — 행별 라벨 재섞임·실물 앵커 강제·catch 행 동작 확인.
- 고 F0 여성 화자 적응 — stage-2 (`scripts/run_stage2.sh`)
- 체크포인트 선택 규칙을 **데이터 보기 전에** 선언 → `scripts/run_g4.sh` 머리에 고정했다:
  후보 {4000,5000,6000,7000,final} · 스크린은 탈락 전용 · **baseline = 스크린을 통과한
  가장 이른 후보** · 선택은 `latest_within_noise`.

> **직행 arm은 존재하지 않는다.** 로드맵 G4는 "baseline 대비 우세"를 요구하고 원래 계획은
> JA-A 선택본을 paired baseline으로 쓰려 했는데, JA-A는 T1에서 멈췄고 그 체크포인트들은
> `fy` 수정으로 데이터셋 해시가 어긋나 export도 안 된다. 그래서 baseline을 위와 같이
> 재정의했다 — **결과를 보기 전에** 정한 것이고, 이 라운드가 측정하는 것은 "2단계가 직행보다
> 낫다"가 아니라 "학습 진행이 무엇을 벌었나"임을 분명히 해 둔다.
> 실물 앵커는 `prepared/<ds>/val-real.jsonl`로 **렌더와 같은 행 id·같은 evaluate 경로**를
> 지나게 만들었다(교훈 79의 렌더 관례 규칙).

**게이트 G4**: 사전 선언한 규칙으로 고른 체크포인트가 블라인드 청취에서 baseline 대비
우세. 고음역 행이 페이지에 포함되어 있을 것. 여기서 처음으로 "일본어 파인튜닝 성공"을
말할 수 있다.


**G4 결과 (2026-09-04 → 05)**: JA(JSUT stage-1 → arona stage-2 8,000 step)·KO(직행 10,000 step) 두 라운드
모두 스크린은 통과했으나 **사용자 청취 판정 "쓸 수 없음"** — 전 렌더에 심한 링잉(정적 격자 톤). 순서
JA > KO > JSUT. 원인은 §7 R16, 전체 진단은 `inflect-work/evals/diag/RINGING-DIAGNOSIS.md`. **G4 미통과.**
"일본어 파인튜닝 성공"은 아직 말할 수 없다. 이 결함은 언어·화자가 아니라 **적응 레시피(학습 코어)**의
문제이므로, 다음 단계는 M-level이 아니라 툴킷 학습 절차의 수정이다.

### 3.1 링잉 대응 1차 라운드 (2026-09-05) — **처방 기각, 계측 존속**

사용자 승인으로 학습 코어를 열고(개선안 b, C13–C16) KO 직행에서 2 arm을 10,000 step 돌렸다.
`ko-arona-v1b`, batch 8, 대조군 `ko-arona-micro-direct-20260904`와 동일 조건.

| arm | 추가 플래그 |
|---|---|
| K-A | `--adversarial-gating --adversarial-ramp-steps 1000 --decoder-lr-warmup-steps 300 --generator-ema-decay 0.999` |
| K-B | K-A + `--decoder-polish-mode recon --stft-loss-weight 1.0 --decoder-proximal-weight 0.1` |

종점 평가(같은 라운드 매니페스트 40행, 같은 evaluate 경로, p50):

| | 실물 | 대조군 | **K-A** | **K-B** |
|---|---:|---:|---:|---:|
| `grid_tone_excess_db` | −0.13 | 8.15 | **8.27** | **11.53** |
| `steady_tone_artifact_score` | 0.00 | 29.86 | 27.09 | 7.37 |
| `fold_periodic_excess_db` | −0.16 | 4.17 | 3.85 | 19.49 |
| `f0_median_hz` | 363.9 | 362.7 | 376.9 | **93.76** |
| `clips_grid_tone_flagged` /40 | 0 | 40 | 40 | 40 |
| `clips_f0_locked_to_frame_grid` /40 | 0 | 0 | 0 | **31** |

**K-A는 대조군과 구별되지 않는다.** 음역·IQR은 건강했으므로 언어·화자 적응 자체는 대조군만큼
됐고, 콤만 그대로다.

**K-B는 훨씬 나쁘다.** 추적 음고 중앙값이 화자의 364 Hz가 아니라 콤 주파수 93.76 Hz로 붕괴했다.
그런데 recon 폴리시 동안 자기 손실은 계속 좋아졌다(mel 0.936 → 0.726). **재구성 손실은
MR-STFT를 포함해 이 아티팩트에 민감하지 않다** — 더 열심히 최적화하는 것은 탈출구가 아니다.

**진단의 인과 사슬이 기각됐다.** 추론경로 `z_dc_rms`(릴리스 0.737):

| step | 대조군 | K-A |
|---:|---:|---:|
| 1000 | 1.459 | 1.453 |
| 3000 | 1.421 | 1.750 |
| 10000 | **1.190** | **1.528** |

대조군은 학습이 진행되며 드리프트가 줄어든다 — 적대 항이 z를 되당기고 있었다. 게이팅은 그 힘을
없애 드리프트를 **늘렸다**. 그리고 결정적으로 **드리프트 1.190과 1.528이 같은 콤(8.15 대 8.27 dB)을
낸다** — 종점에서 드리프트는 콤을 지배하는 변수가 아니다. 학습경로(포스테리어 z)와 추론경로 둘 다
울리므로 두 경로의 불일치도 아니다.

**남은 것은 계측이다.** 실물 40행 0/40, 두 렌더 40/40, 사이드카가 콤이 켜지는 구간(step 500 → 1000)을
짚었다.

**단, 이 라운드의 결론은 2026-09-05 감사로 한 단계 내려갔다(§3.2).** K-A·K-B를 포함한 실패 런
전부가 **정답과 생성음에 다른 스펙트럼 바닥값을 쓰는 mel 손실**로 학습됐다. 따라서 위 수치는
"이 처방이 무효"가 아니라 "깨진 목적함수 아래에서 이 처방을 재본 결과"다. 스왑 테스트로
디코더를 배제했다는 서술도 과하다 — 반대 방향(릴리스 z → 적응 디코더)은 시행되지 않았다.

다음 후보(수정): (i) z의 프레임 간 구조·채널 공분산과 디코더의 상호작용, (ii) 디코더의
anti-imaging 부재 자체(업샘플러 커널·필터, 단 kernel `[16,16,4,4]`이 stride `[8,8,2,2]`의 정수배라
단순 uneven-overlap 설명은 부정확하다), (iii) **0–12 kHz 안** 좁은 톤에 대한 손실의 해상도·민감도와
중간 업샘플 단계의 이미징. ~~`mel_fmax 12000`이 12 kHz 위를 제외하는가~~ 는 **삭제** — 24 kHz의
Nyquist가 12 kHz이므로 제외되는 것이 없다.
상세 측정은 `inflect-work/runs/REMEDY-B-VERIFICATION.md`.

### 3.2 외부 감사와 확정 결함 5건 (2026-09-05) — **수정 완료, 대조 실험 사용성 탈락**

두 번째 외부 감사가 저장소 함수를 실제로 실행해 결함을 재현했고, 나도 전부 재확인했다.
**전부 초기 커밋 `1da708c`부터 있던 것이고 이번 브랜치가 만든 것이 아니다.**

| ID | 결함 | 확인 방법 | 상태 |
|---|---|---|---|
| D1 | mel 손실이 정답 `sqrt(\|X\|²+1e-6)`(바닥 1e-3)과 생성 `clamp_min(1e-5)`를 비교 — 같은 파형이 `ln(100)=4.605`, 실물 음성끼리 0.07–0.52. 조용한 셀의 기울기가 잡음을 **보상**(RMS 1e-6에서 −840750) | 저장소 함수 CPU 실행 | ✅ C17 |
| D2 | 단계 경계 resume 시 새 그룹 lr이 0으로 남음. decoder는 종단 단계라 영구 | 실제 `train_adaptation` 재현 | ✅ C18 |
| D3 | `evaluate`가 원음 매니페스트에서 모델을 열지 않는데 리포트에 `model_dir`이 남음. 혼합 매니페스트는 `audio`를 조용히 무시 | 코드 분기 + 리포트 29건 검사 | ✅ C19 |
| D4 | 배포 splitter가 `1.5초`를 `1.` + `5초`로 분할(220 ms 무음·시드 재추첨 삽입) | shipped source 13입력 실행 | ✅ C20 |
| D5 | `_validate`의 `manual_seed`가 학습 RNG를 오염 → `validation_interval`이 학습 결과를 바꿈 | 코드 확인 | ✅ C18 |

**기존 판정에 미치는 영향**: D2는 어떤 런도 밟지 않았고(resume한 두 런 모두 단계 중간),
D3은 eval 매니페스트 19개 전부 text-only라 G4·K-A/K-B 판정이 유효하다. **D1만이 실패 런 전부에
영향을 준다.**

콤 주입 실험(실물 6클립): 주입 레벨 1e-4에서 legacy mel이 0.2190 → **0.2124로 내려가고**,
통일 mel과 MR-STFT는 전 범위 단조 증가한다. 즉 "재구성 손실은 이 아티팩트에 민감하지 않다"는
이전 서술은 틀렸다. 다만 **격자 점수가 같다고 콤 진폭이 같은 것은 아니며**, K-B 악화에 대한
D1의 기여도는 대조 학습 전까지 미확정이다.

**대조 실험 결과 (2026-09-05, 완료)** — 상세: `inflect-work/runs/MEL-AB-2026-09-05.md`

같은 커밋 `0727f7d`·같은 시드에서 `--mel-loss-legacy-floor` 하나만 다른 두 arm을 1,500 step 돌렸다.
개입 범위는 손실 값이 아니라 옵션·identity·입력 일치로 검증했다(`training-options.json`이 한 필드만
다르고, step 500까지 검증 렌더가 비트 동일 — warm-up 동안 추론이 `enc_q`를 쓰지 않기 때문이다).

**사전 선언 규칙의 판정은 "악화"다**(격자 톤 초과 문장별 차이 중앙값 step 1000 +3.73, final +1.63).
규칙을 사후에 바꾸지 않는다. 그러나 종점에서 세 스크린 중 둘은 개선을 가리켰고(steady-tone −4.62,
fold −1.78), 계기를 검사한 결과 **격자 톤 초과가 이 비교에 유효하지 않았다.**

`grid_tone_excess_db`는 on-grid/off-grid **비율**이고 분모는 그 렌더 자신의 광대역 바닥이다.
각 렌더의 자기 레벨 기준으로 40행을 다시 재면(실물 척도 병기):

| 관측치 p50 | 실물 | legacy | matched | 차이 | matched가 낮은 문장 |
|---|---:|---:|---:|---:|---:|
| 콤, 신호 대비 dB | −52.3 | −35.9 | **−39.0** | **−3.16** | 37/40 |
| 광대역 바닥, 신호 대비 dB | −52.0 | −42.2 | **−46.7** | **−4.64** | 40/40 |
| **콤, 절대 평균 PSD** | — | — | — | **+0.99** | **9/40** |
| 광대역 바닥, 절대 평균 PSD | — | — | — | −0.62 | 27/40 |
| 레벨 dBFS | −20.0 | −36.4 | −32.5 | **+3.97** | 1/40 |
| on/off 비율(스크린) | −0.1 | 6.1 | 7.7 | +1.63 | 3/40 |

**"신호 대비"와 "절대"는 같은 데이터의 다른 질문이고, 둘 다 옳다.** 신호 대비 지표는 순수
게인에 불변이므로 "콤이 얼마나 있는가"에 답하지 않는다. 절대 평균 PSD는 `레벨 + rms_dbfs`로
파생되며, 그 값에서 **matched의 콤은 줄지 않았다**(중앙값 +0.99 dB, 40문장 중 9개에서만 감소).
줄어든 것처럼 보인 이유는 전체 전력이 3.97 dB 커졌기 때문이고, 그중 일부는 matched 렌더가 더
짧고(40/40) 무음 프레임 비율이 더 낮다는(38/40) 사실이 설명한다.

따라서 **4단 규칙을 사후 관측치에 적용해 "부분 개선"이라고 쓴 앞선 문장은 철회한다.** 판정
등급은 **"개선 불확실"**이다. 바닥이 상대 기준으로 4.6 dB 내려간 것은 예측대로지만, 절대 기준
차이는 −0.62 dB에 그친다. 이 값들은 선택된 격자 bin의 평균 PSD이지 분리된 링잉 성분이 아니며,
길이·무음 비율·스펙트럼 구성 차이는 어떤 정규화로도 제거되지 않는다.

**정정된 문구**: 앞선 초안의 "mel 결함은 지배 원인이 아니다"는 이 데이터가 지지하는 진술이
아니다. 지지되는 것은 **"이번 초기 실험에서 mel 수정만으로는 링잉이 해결되지 않았다"**이며,
콤이 실물까지 16.4 dB 격차 중 3.2 dB만 좁혔다는 사실이 그 근거다. 1,500 step · 단일 seed ·
폴리시 미진입 · 청취 없음으로 지배 원인의 유무는 판별할 수 없다.

**사전 선언과 사후 탐색의 구분**: 사전 선언은 4단 규칙과 그것이 걸린 `grid_tone_excess_db`,
그리고 그 판정("악화")뿐이다. 사후 탐색은 **둘**이고 서로 다른 답을 냈다 — 상대 레벨
(`grid_tone_level_db`·`off_grid_level_db`, 2026-09-05, "개선")과 절대 평균 PSD 재측정
(2026-09-06, "감소 아님"). 어느 쪽도 사전 판정을 덮지 않는다. 계기 세 개가 세 답을 낸 것이
이 라운드에서 얻은 가장 이전 가능한 교훈이다. 문장별 원시값 120행(3체크포인트 × 40문장,
`legacy.*`·`matched.*`·`diff.*`, 파생 절대 PSD·길이·무음 비율 포함)을
`inflect-work/evals/M-mel-ab-per-sentence.csv`에 보존했고
`scripts/verdict_mel_ab.py --csv`로 재생성된다. 재현 지점은 태그 `mel-ab-2026-09-05`(임시 옵션이
살아 있는 마지막 커밋)이다.

**청취 판정 (2026-09-06)** — `inflect-work/listening/mel-ab-20260906/`, 8행, 실물 앵커·탈락 control 포함,
봉인 mapping. **두 arm 모두 품질 1("사람 목소리로 들리지 않는다") 8/8, 언어 인식 "아니오" 8/8.**
실물은 품질 5가 8/8이고 강제 선택 `most_natural`은 실물 7회 + 미응답 1회, catch 행 소음 바닥 0.
강제 선택 `most_blurred`는 M-legacy 7 대 M-matched 1. 자유기술 8행 전부: 1.5k arm은 **"발음 템포에
음성 대신 ringing 기계음만"**, 10k control은 "발음도 들리지만 ringing이 심하다".

**결론: 두 arm 모두 사용 불가.** 이것은 승격 보류가 아니라 **사용성 기준 탈락**이다. 절대 척도
(품질·언어)에서는 두 arm이 구분되지 않았고, 강제 선택에서만 7:1로 legacy가 더 흐렸다 — "귀로
구분 불가"는 과한 표현이므로 철회한다. 그 차이의 크기는 확정하지 않는다. 이 페이지는 클립별로
−24 dBFS를 목표하고 피크 가드를 걸었기 때문에 33트랙 중 9개가 목표에 못 미쳤고 한 행에서 2.33 dB
차가 났다(빌더는 이후 페이지 공통 목표로 수정, C22).

**mel 수정(D1)은 유지한다.** 결함 수정 자체는 타당하고, 이 실험이 보인 것은 "이번 초기 실험에서
mel 수정만으로는 해결되지 않았다"이다. 또한 **1,500 step은 초기 아티팩트 진단 지점이지 최종
음질 비교 지점이 아니다.** control이 발음을 들려준 것은 legacy가 우수하다는 증거가 아니고,
matched의 후반 효과도 배제되지 않았다. mel 수정을 근거로 장기 학습을 늘릴 근거는 없다.

### 3.3 잠재 표현 × 디코더 완전 교차 (2026-09-06) — **잠재가 지배, 디코더는 미해명**

상세: `inflect-work/evals/diag/cross-latent-decoder/FINDING.md`

**1,500 step arm으로는 이 실험을 할 수 없다.** 두 arm 모두 디코더가 전 구간 lr 0이고 231개
텐서가 base Micro와 비트 동일하므로, 둘끼리 교환하면 같은 텐서를 자신과 비교한다. 디코더는
디코더가 7,000 step 학습된 10,000 step 런에서 가져왔다. KO 심볼이 릴리스 178과 완전히 동일해
같은 12문장의 같은 토큰열을 두 사전 경로에 통과시킬 수 있고, 모든 셀의 내용이 일치한다.

on/off 비율 dB (실물 −0.1 근처, 탈락 렌더 +6 이상):

| | D-release | D-control | D-KB |
|---|---:|---:|---:|
| **L-release** | **−0.20** | **1.53** | **−0.24** |
| L-adapted | 6.33 | 8.01 | 9.89 |
| L-posterior | 6.25 | 7.48 | 9.29 |

**릴리스 잠재는 7,000 step 학습된 적응 디코더 두 개를 포함한 세 디코더 전부에서 깨끗하고,
적응 잠재는 릴리스 디코더를 포함한 세 디코더 전부에서 울린다. 12문장 전부, 세 디코더 전부.**

- **적응 디코더 가중치는 링잉의 필수조건이 아니다.** 릴리스 디코더만으로도 적응 잠재는
  울린다. 다만 이것을 "디코더를 사실상 면제한다"로 쓴 앞선 문장은 철회한다 — **같은 디코더
  구조가 특정 잠재 분포에 취약할 가능성은 이 격자로 배제되지 않으며**, 릴리스 디코더에는
  anti-imaging 필터가 없다. §3.1의 유보는 "한 방향은 시행됐다"까지만 해소됐다.
- **"디코더 학습이 악화시켰다"는 지표를 명시해야 성립한다.** 적응 잠재에서 **비율 지표**는
  D-release 6.33 → D-control 8.01 → D-KB 9.89로 악화하지만, **상대 레벨 지표**는 −44.48 →
  −46.47 → −47.66으로 **반대 방향**이다. 어느 쪽이 실제로 더 나쁜 소리인지는 청취하지 않았다.
- **사후 탐색은 수렴하지 않았다.** 채널별 DC 제거는 비율 지표에서 크게 개선(6.25 → 3.35)하지만
  콤 레벨 지표에서 D-control을 6.4 dB 악화시킨다. RMS 정합은 레벨 지표에서 일관되게 개선하고
  비율에서는 거의 무효. **두 지표가 동의하는 것은 "시간 축 평활은 도움이 안 된다" 하나뿐이다.**
  사전 선언한 개입 실험이 필요하다.

계기 수정은 C21로 등재했다. **콤 발생 시점도 좁혔다**: 50 step 간격 사이드카에서 추론 경로가
step 500 −0.24 → step 550 **+4.76**(M-legacy) / **+6.63**(M-matched), 즉 linguistic 단계 시작 후
50 step 안이다. (앞선 판에 "K-A 사이드카"로 적은 것은 오기다. K-A/K-B는 500 step 간격이라 550
지점이 없다.)

한계: 단일 seed · 1,500 step · 폴리시 미진입 · 청취 없음. **재구성 경로에는 양성 대조가 없다** —
릴리스 체크포인트에 `enc_q`가 없어(`FRESH_PREFIXES`) 두 런 모두 posterior를 새로 초기화했으므로,
"이 디코더로 실물을 콤 없이 재구성하는 posterior가 존재한다"는 증거는 아직 없다.

### 3.4 경로 진단 준비 (2026-09-06) — **진행 중**

청취 판정이 문제를 다시 규정했다. 1,500 step 두 arm은 링잉이 얹힌 음성이 아니라 **음성이 없는
링잉**이고, 10k control만 "어색하지만 말"이었다. 즉 지금 실패하고 있는 것은 잔여 잡음 제거가
아니라 **말소리 형성**이다. 다음 학습을 정하기 전에, 기존 체크포인트만으로 같은 문장에서 두
경로를 갈라 본다.

1. **재구성**: 실물 → `enc_q` → `dec` (학습이 디코더에 주는 sampled `z`)
2. **추론**: 음소 → `enc_p`/`dp`/`flow` → `dec`

두 축을 분리해 읽는다. **콤은 측정**(스크린 세 값 + 절대 PSD), **말소리 성립 여부는 청취**로
판정하며, 품질 척도의 1("사람 목소리로 들리지 않는다")과 2("말이지만 어색하다") 사이가 그
경계다. 자연성 점수 차이는 보조 정보이고 두 단계 차이를 필수조건으로 삼지 않는다.

**측정 완료 (2026-09-06)** — 상세 `inflect-work/evals/diag/path/FINDING.md`,
원시값 `path.json`·`path.csv`(760행), 청취 페이지 `listening/path-20260906/`.

재현이 먼저 통과했다. `prior_latent`+`dec`는 배포 경로 `model.infer`와 **비트 동일**(0.00e+00)이고,
control@10k 재구성은 2×2의 L-posterior × D-control 셀을 **0.0000 dB**로 재현한다(12/12).

격자 톤 초과, 40문장 중앙값(실물 −0.13):

| 체크포인트 | recon-sampled | recon-mean | infer |
|---|---:|---:|---:|
| matched@500 | **+4.54** | +9.45 | **−0.20** |
| legacy@500 | **+3.78** | +5.73 | **−0.20** |
| matched@1000 | +11.15 | +11.64 | +10.36 |
| matched@1500 | +8.55 | +8.82 | +7.68 |
| legacy@1500 | +8.96 | +9.38 | +6.03 |
| control@10000 | +8.11 | +8.38 | +8.12 |

- **P1 성립**: 추론@500은 −0.20 dB, +1 dB 초과 1/40. 예측대로 릴리스 사전 수준이다.
- **P2: 재구성@500은 이미 울린다.** 같은 체크포인트·같은 문장·릴리스 그대로의 디코더인데
  추론은 깨끗하고(0/40이 +4 초과) 재구성은 울린다(matched 33/40, mean 40/40).
  **기록은 "warm-up 종료 시점에 콤이 재구성 경로에 이미 존재한다"까지다.** 릴리스에 `enc_q`가
  없어 양성 대조가 없으므로 500 step 미수렴과 설계 결함을 이 실험은 구분하지 못한다.
- **P6**: `recon-mean`이 `recon-sampled`보다 나쁘고 그 격차는 step 500에서만 크다(matched +4.68 dB,
  38/40). 표기는 "step 500 출력은 샘플링에 민감하다"까지이며 **분산·KL 처방으로 쓰지 않는다.**
- **사후 관측**: f0가 93.75 Hz에 잠기는 것은 1,500 step arm에만 나타난다(recon 34/40, infer 39/40).
  청취가 "말로 들린다"고 한 10k control은 실물과 같은 360 Hz대이고 잠김 0/40인데 콤 초과는
  +8.1로 비슷하다. **초과 크기는 말소리 성립과 무관해 보이고 f0 잠김이 그것과 정합한다**는
  가설이 서지만, 계기를 청취 대체물로 쓰지 않는다.

**청취 대기**: 0038 한 문장 8트랙, −24.00 dBFS 공통 목표(편차 0.0000 dB), catch 바이트 동일,
안내문에 말소리/언어/이해 세 질문 구분. 판정 경계는 품질 1 대 2다. **후속 학습 선택에 쓰는
결론은 한 문장으로 확정하지 않고**, 분기가 정해지면 다른 두 문장에서 핵심 후보만 재확인한다.
어떤 지표 하나로도 원인을 배제하거나 처방을 확정하지 않는다.
**새 장기 학습은 그 뒤 별도로 승인받는다.**

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
| C13 | 적대항 게이팅 + 램프 + 디코더 lr 워밍업 | `training._adversarial_weight()` · `_decoder_lr_scale()` · `_scaled_decoder_lr()` | M7 | ✅ 완료(2026-09-05). 기본 off. 게이트 구간에도 D는 계속 학습 |
| C14 | recon-only 폴리시 + MR-STFT + proximal | `training._enabled_groups()` · `_multi_resolution_stft_loss()` · `_proximal_loss()` | M7 | ✅ 완료(2026-09-05). recon은 디코더만 학습, D 정지. STFT는 해상도 평균(PWG 관례) |
| C15 | 업샘플러 동결 · posterior 사이드카 · 생성기 EMA | `training._apply_stage()` · `checkpoint.save_posterior_sidecar()` · `exporting.export_checkpoint()` | M7 | ✅ 완료(2026-09-05). 업샘플러 동결은 그룹 분리가 아니라 기울기 마스크 — 옵티마이저 state 형태 불변 |
| C16 | 프레임 격자 스크린 + fp32 mel | `grid_screens.py`(신규) · `evaluation._grid_screens()` · `training._mel_from_spec()` | M7 | ✅ 완료(2026-09-05). 실물 40 대 렌더 40에서 grid·steady-tone이 0/40 대 40/40으로 완전 분리. fp32 mel은 AMP 런의 수치를 바꾸는 의도된 변경 |
| C17 | mel 크기 스펙트럼 공식 통일(D1) | `training_data.magnitude_spectrogram()`(신규) · `training._mel_from_waveform()` | M7 | ✅ 완료(2026-09-05). 같은 파형의 mel L1이 정확히 0. 임시 `--mel-loss-legacy-floor`는 대조 실험 후 제거 예정. **모든 런의 `loss_mel`·`loss_g` 절대값이 바뀌므로 이전 값과 비교 불가** |
| C18 | 경계 resume lr(D2) + 검증 RNG 격리(D5) | `training.train_adaptation()` resume 분기 · `training._validate()` | M7 | ✅ 완료(2026-09-05). 경계 재개가 무중단 런과 1e-18 이내로 일치. 초기 커밋부터 있던 결함 |
| C19 | evaluate 출처 명시(D3) | `evaluation.evaluate_checkpoint()` | M7 | ✅ 완료(2026-09-05). `source.mode`·합성 수·무시된 audio 수·혼합 검사. `ok`의 의미는 의도적으로 불변 |
| C20 | 배포 splitter 숫자 가드(D4) | `exporting._SENTENCE_BOUNDARY` · 신규 `tests/test_export_runtime_split.py` | M6 | ✅ 완료(2026-09-05). **기존 export 15개는 각자 사본을 갖고 있어 재-export 필요.** 공개 HF 패키지는 영향 없음 |
| C21 | 콤 절대 레벨 관측치 추가 | `grid_screens.grid_comb_metrics()` (`grid_tone_level_db`·`off_grid_level_db`) | M7 | ✅ 완료(2026-09-05). **격자 톤 초과는 비율이라 바닥이 다른 두 렌더의 순위를 뒤집는다** — mel A/B에서 40행 중 37행을 거꾸로 매겼다. 검출은 비율, 렌더 비교는 레벨 |
| C22 | 청취 페이지 공통 목표 RMS | `examples/build_blind_ab_page.py` (`page_target_rms_dbfs`·`crest_factor_db`·`LEVEL_FLOOR_DBFS`·`--catch-system`) | M7 | ✅ 완료(2026-09-06). 클립별 레벨링이라 피크 가드가 걸린 클립만 조용해졌다 — 지난 페이지 33트랙 중 9개가 목표 미달, 한 행 2.33 dB 차. 이제 모든 트랙이 도달 가능한 최대값 하나를 페이지 전체에 적용하고(지난 페이지 재생성 시 −27.28 dBFS, 편차 0.0001 dB), 바닥 −30 dBFS를 넘기면 중단한다. **`limited_by`는 시스템 이름을 담으므로 페이지에 박히는 `axes`가 아니라 봉인된 mapping 최상위에 기록한다** |
| C23 | 문장별 CSV에 절대 PSD 파생 열 | `inflect-work/scripts/verdict_mel_ab.py --csv` | M7 | ✅ 완료(2026-09-06). 스크린 세 값이 모두 순수 게인에 불변이라 절대량을 말하지 못한다. `레벨 + rms_dbfs`(원래 출력)와 `레벨 + 고정 기준`(공통 재생 레벨)을 나눠 기록하고 길이·무음 비율을 함께 남긴다. **라이브러리 키는 추가하지 않았다 — 항등식으로 파생된다** |

학습 코어(`training.py`)와 임베딩 마이그레이션(`checkpoint.py`)은 원래 **변경 대상이
아니었다.** 2026-09-05 사용자가 링잉 대응(개선안 b)을 승인하면서 이 제약을 해제했고,
C13–C16이 그 결과다. 새 옵션은 전부 기본 off이고 기본 경로의 손실·스케줄은 20 step
비교에서 마지막 자리까지 동일함을 확인했다. **분할 로직은 여전히 변경 대상이 아니고
실제로 손대지 않았다.**

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
| **R16** | **디코더 업샘플 격자 톤(링잉).** 신선한 `enc_q`/mean-only `flow`가 동결 디코더를 역산하며 z가 릴리스 분포를 벗어나고(채널평균 RMS 0.74 → 1.4–1.5), anti-imaging 없는 HiFi-GAN 디코더가 그 z를 93.75/750/6000 Hz 격자 톤으로 냄; 신선한 D의 해제 충격이 정적 톤을 응집; 체이닝은 이를 두 번 겪음 | **치명 — 2026-09-04 G4 두 라운드 모두 사용자 판정 "쓸 수 없음"** | 진단 `inflect-work/evals/diag/RINGING-DIAGNOSIS.md`. 2026-09-05 개선안 (b) 구현(C13–C16) 후 KO 2 arm 10,000 step 실행. **처방 두 개 모두 기각, 원인 가설도 기각.** 게이팅은 종점 콤을 바꾸지 못했고(8.15 → 8.27 dB) 드리프트는 오히려 늘렸다(1.190 → 1.528). recon 폴리시는 크게 악화시켰다(11.53 dB, 추적 음고 중앙값이 93.76 Hz로 붕괴, 40 중 31행 격자 잠김). **드리프트는 콤을 지배하지 않는다** — 1.190과 1.528이 같은 콤을 낸다. 남은 산출물은 계측이다(실물 0/40 · 렌더 40/40). 상세 §3.1 |
| **R17** | **배포된 Micro 디코더는 alias-free가 꺼져 있다.** `config.json`에 `decoder_alias_free` 없음(→ False), `model.pth`에 Snake/필터 텐서 없음 — 표준 HiFi-GAN V1. README·ARCHITECTURE의 "alias-reduced decoder"와 불일치 | 중간 | 저장소 소유자 보고. 켜려면 재학습 필요(장기) |

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
| 2026-09-04 | M2 통과. `ja-arona-v2`(A_CO026 제외)·`ja-jsut-v1`·`ko-arona-v1b` 준비. **KO 클리핑 정책 발동** — 소스가 아니라 우리 리샘플이 331행(9.89%)의 상단을 잘라내고 있었고, 선언대로 균일 −3 dB 후 재준비. **툴킷 결함 2건 추가 수정**(WAVEX 거부, 출력단 클리핑 미보고). C8·C9·C10 완료, 청취 파이프라인 실물 예행 검증. G3/stage-2/G4 실행 스크립트와 선언 규칙 고정. `fy` 수정이 사전 준비 데이터셋의 export를 무효화하는 것을 확인 — **프론트엔드 수정은 prepare 앞에 온다**. |
| 2026-09-05 | **G4 두 라운드 미통과(링잉).** 진단 완료: 디코더 업샘플 격자 톤(93.75/750/6000 Hz 배수), 1차 원인 z 드리프트 + 신선한 D 충격 + 체이닝, anti-imaging 없는 디코더가 enabler. R16·R17 등재. E1(디코더 조기 해제) 기각. 처방은 학습 코어 변경이라 사용자 결정 대기. |
| 2026-09-05 | **외부 감사 확정 결함 5건 수정 — C17–C20.** mel 양쪽 바닥값 통일(같은 파형 L1이 `ln(100)` → 0), 경계 resume lr, 검증 RNG 격리, evaluate 출처 명시, 배포 splitter 숫자 가드. **전부 초기 커밋부터 있던 결함.** D1은 실패 런 전부에 영향을 주므로 K-A/K-B 결론을 "깨진 목적함수 아래의 측정"으로 강등했고, `docs/TRAINING.md`·`docs/TROUBLESHOOTING.md`의 인과 서술 5건(flow 경로 오류, 재구성 손실 민감도, 드리프트 추적, 경로 불일치, F0로 적응 성공 판단)을 정정했다. pytest 330 → 364. 1,500 step mel 대조 실험 진행 중. |
| 2026-09-05 | **KO 2 arm 판정 — 처방 기각, 계측 존속.** K-A(게이팅)는 대조군과 구별되지 않았고(grid 8.27 대 8.15 dB), K-B(recon 폴리시)는 더 나빠졌다(11.53 dB, f0 중앙값 93.76 Hz). 잠재 드리프트가 콤을 지배하지 않음을 확인 — 진단의 인과 사슬 기각. `docs/TRAINING.md`·`docs/TROUBLESHOOTING.md`의 권장 문구를 측정 결과로 교체했다. 다음 후보는 z 스케일이 아니라 z의 프레임 간 구조, 그리고 디코더의 anti-imaging 부재 자체. |
| 2026-09-05 | **개선안 (b) 구현 — C13–C16 완료.** 사용자 승인으로 학습 코어 제약 해제. 손잡이 9개 전부 기본 off, 기본 경로 20 step 비교에서 레거시 열 불일치 0. pytest 155 → 330, ruff 74 → 68(신규 지적 0). 스크린 임계는 계획값에서 실측으로 이동(G 2.0 → 4.0 dB, fold 3.0 → 6.0 dB, steady-tone 기본 off → on) — 근거는 `inflect-work/runs/REMEDY-B-VERIFICATION.md`. **아직 링잉이 사라진다는 증거는 없다**: 600 step 스모크에서 게이팅을 켜고도 콤이 step 200에 나타났고(사이드카 G −0.24 → +7.18 dB), 다만 잠재 드리프트는 대조군보다 낮았다(z_dc_rms 1.035 대 1.19–1.42). 판정은 KO 3 arm 10,000 step이 한다. |
| 2026-09-06 | **mel A/B 청취 판정 = 사용성 탈락, 계기 3건이 3답.** 두 arm 모두 품질 1 8/8, 언어 "아니오" 8/8, 자유기술 "발음 템포에 ringing 기계음만". 절대 평균 PSD로 재측정한 결과 **"부분 개선 −3.16 dB"는 철회** — 절대 on-grid는 +0.99 dB(9/40)이고 전체 전력이 +3.97 dB 커진 것이며, matched 렌더가 더 짧고(40/40) 무음이 적다(38/40). 등급을 "개선 불확실"로 재분류. 과장 3건 정정("귀로 구분 불가", "디코더 사실상 면제", 지표 미명시한 "디코더 학습이 악화")과 오기 1건(K-A 사이드카 → M-legacy). **mel 수정(D1)은 유지.** C22·C23 완료. §3.4 경로 진단(재구성 대 추론) 착수 — 새 장기 학습은 진단·청취 후 별도 승인. |
