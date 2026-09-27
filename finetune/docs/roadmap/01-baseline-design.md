# 기준선과 설계 (§1 · §2)

> [로드맵 목차](README.md)의 한 부분이다. 절 번호(§)는 옛 `MULTILINGUAL_ROADMAP.md`의 번호를 그대로 쓴다.

## 1. 검증된 기준선 (2026-08-30 실측)

아래는 추정이 아니라 이 저장소에서 실행해 확인한 결과다. 재현 명령은 [HANDOFF 부록 §6.3](../HANDOFF.md)에 있다.

> 코드 인용은 **심볼 이름이 정본**이다. 줄 번호는 편집으로 즉시 어긋나므로 쓰지 않는다.

| # | 사실 | 근거 | 함의 |
|---|---|---|---|
| V1 | **일본어 음소 전체가 릴리스 178심볼 인벤토리 안에 있다.** OpenJTalk 음소 집합(모음5·무성화5·`N`·`cl`·`pau` + 자음 30여)을 IPA로 매핑했을 때 신규 심볼 **0개** | `symbols.BASE_SYMBOLS` 대조 | 임베딩 마이그레이션이 전 행 복사. 랜덤 초기화 행 없음 |
| V2 | **한국어도 신규 심볼 0개.** espeak-ng `ko` 출력 중 base 밖 문자는 `-`(U+002D) 하나뿐이며, 이는 경음 표지이므로 `ʼ`(U+02BC, base에 존재)로 매핑하면 해소 | 동일 | JA/KO 모두 178 계약 유지 가능 |
| V3 | **eSpeak `ja`는 사용 불가.** 한자마다 영어 단어 "chinese letter"(`tʃˈaɪniːzlˈe̞tə`)를 리터럴로 방출. 카나 입력에서도 base 밖 문자 `ä`·`̞`(U+031E)·`ᵝ`(U+1D5D) 발생 | espeak-ng 1.52 실행 | 일본어는 커스텀 프론트엔드 **필수** |
| V4 | **eSpeak `ko`는 사용 불가.** 음운 규칙이 빠질 뿐 아니라(`옵니다`→`ˈopnidˌɐ`, `신라면`→`sˈinɾɐmjˌʌn`) **후두 대립을 붕괴시킨다** — 최소대립쌍 13개 중 6개(살/쌀·자다/짜다·불/뿔·방/빵·정/쩡·사/싸)가 같은 심볼열이 된다 | espeak-ng 1.52 실행 | 한국어도 커스텀 프론트엔드 **필수**. 2026-08-30 정정 |
| V5 | **`pyopenjtalk-plus` 0.4.1이 prebuilt wheel로 설치된다.** 본가 `pyopenjtalk`는 py3.10 wheel 없음(소스 빌드 필요) | `pip install --only-binary=:all:` | 일본어 의존성이 컴파일러 없이 해결됨 |
| V6 | ~~`g2pkk` + espeak `ko` 조합~~ → **폐기(2026-08-30).** 음운 규칙은 전달되지만 espeak가 후두 대립을 지운다(V4). **대체: `g2pkk` → 발음 한글 → 자모 직접 매핑.** 최소대립쌍 13/13 보존, base 밖 문자 0개, espeak 의존성 제거 | 실행 | 한국어 2단 구조(음운→자모) 확정 |
| V7 | **pyopenjtalk가 <prior-ja-project>이 어휘사전으로 고쳐야 했던 항목을 그냥 맞게 읽는다.** `抗うつ剤`→コーウツザイ, `対策`→タイサク, `痛み止め薬`→イタミドメヤク, `2026年8月30日`→ニセンニジューロクネンハチガツサンジューニチ | 실행 | <prior-ja-project>의 어휘사전 수술 연작는 이식 대상이 아님(§5.1) |
| V8 | **2단계 체이닝이 현재 코드로 동작한다.** `export`가 `config.json`+`model.pth`+`runtime/`을 쓰고 `resolve_base_model()`이 로컬 디렉터리를 받는다 | `exporting.export_checkpoint()`, `modeling.resolve_base_model()` | 언어 베이스 → 화자 적응 가능 |
| V9 | ~~심볼 수가 정확히 178이 아니면 체이닝이 깨진다~~ → **해소(2026-08-30, C6).** `load_runtime_components()`가 이제 `>= 178` + 릴리스 접두 일치를 본다 | `modeling.validate_release_compatible_symbols()` | 단일 실패점 제거. JA/KO는 애초에 178이라 체이닝은 이전에도 동작했다 |
| V10 | **다화자 준비가 하드 블록이다.** speaker 값이 2개 이상이면 `prepare`/`audit`이 즉시 실패 | `prepare_dataset()` · `audit_dataset()`의 speaker 가드 | 언어 베이스 단계에 우회 필요 |

### 확인하지 못한 것 — **전부 해소(2026-09-04, G0)**

작성 당시의 세 공백은 모두 메워졌다. 기록은 `<work-dir>/env/G0.md`가 정본이다.

- ~~`<nas-drive>:` 미마운트~~ → 마운트 확인, JA/KO/JSUT를 로컬 디스크로 복제하고 검증했다.
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
  ko_g2pkk.py          # g2pkk + espeak(ko) IPA 단계                          [M5] (2026-08-30 보완: espeak 제외, g2pkk → 발음 한글 → 자모 직접 매핑으로 구현. V6, §5.3)
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

> 의존성 선언이 필요한 이유: [CUSTOM_G2P.md](../CUSTOM_G2P.md)가 "외부 아티팩트를 요구하면
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
음성 정체성도 판정하지 못한다. <prior-ja-project> 프로젝트의 가장 값비싼 교훈이 여기 적용된다 —
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

