# 인계 — 새 환경에서 셋업하고 이어가기

**대상 브랜치**: `main` (2026-09-27에 `feat/training-core-remedy-b`를 머지하고 그 브랜치를 지웠다)
**갱신일**: 2026-09-27

이 문서는 **"어디서 무엇을 깔고, 무엇을 먼저 읽고, 무엇이 저장소 밖에 있는가"** 만 다룬다.
진행 상태·판정·결정의 정본은 [로드맵](roadmap/README.md)이고,
여기 있는 상태 요약은 그 문서의 색인일 뿐이다. 둘이 다르면 로드맵이 맞다.

---

## 1. 지금 어디인가 (2026-10-02 기준)

| 단계 | 상태 | 로드맵 |
|---|---|---|
| M0 / G0 환경 확정 | ✅ 통과 (2026-09-04, RTX 5090) | [02-milestones.md](roadmap/02-milestones.md) M0 |
| M1 JA 프론트엔드 | 코드 완료. G1(c)는 자동 스크린으로 대체(사용자 결정) | M1, 이 문서 부록 §6.4 |
| M2 데이터 준비 | ✅ 통과 (2026-09-04) — `ja-spkA-v2` · `ja-jsut-v1` · `ko-spkA-v1b`. 다음 학습부터 `ja-spkA-v3`(전사 30행 교정, §3.38–§3.39. 2026-09-29에 읽기 교정 1건을 lexicon으로 더해 다시 준비했고 분할은 같다). 2026-09-27부터 `prepare`가 변환 클리핑을 거부한다(C28, 게인 0이면 기존 자료판과 같다). `ko-spkA-v1b`의 입력(−3 dB 사본)은 게인 0으로 통과한다. KO를 무가공 원본에서 다시 준비하면 `--input-gain-db -1.6`이 필요하고 검증셋이 바뀐다 — KO 재개 때 정한다 | M2 |
| M3 stage-1 (JSUT) | 학습 완료, 그 export와 체크포인트에서 stage-2 진행(G3 판정 줄은 로드맵에 따로 없음) | M3 |
| M4 / G4 청취 | ❌ **미통과** — 링잉(R16)이 남아 있다 | M4, [10-risks.md](roadmap/10-risks.md) R16 |
| 링잉 대응 | 라운드 1차 ~ J6 · B1 · B2까지 진행. **채택된 처방 없음.** J3가 링잉 감소 후보(연구 기준)이고, 발음 · 음높이 품질 회복은 미완 | [README 현재 상태](roadmap/README.md), §3.1–§3.41 |
| M5 KO 프론트엔드 | 코드 완료, G5 통과. 실제 전사 표본 검수 남음 | M5 |
| M6 KO 적응 + 배포 패키징 | 미착수 (C12 미완) | M6 |

**마지막 라운드**: J6(2026-09-30 ~ 10-02) — J5a에서 생성기 clip(한도 500) 또는 판별기 clip(생략)만 바꾼 두 후보. 두 후보 모두 "그 밖" · 미채택이고, 2026-10-02에 확정했다(§3.41). 요약은 "종합 품질 · 음높이 회복이 정체됐다"이다. 음높이는 세 행 모두 J5a와 같은 2였다. clipping 탐색은 마무리했다. 음높이를 물은 세 페이지에서 J2 계열 여섯 런은 모두 음높이 2, J1은 모두 0이다.
J1 · J5a · J5b(와 J3)는 서로 다른 비교 기준으로 보존한다([README 기준 런](roadmap/README.md)). 보조 비교 B2는 미결이다(§3.37).

**다음 하나**(사용자 결정, 2026-10-02): **J7** — J5a에서 판별기만 fresh로 바꾼 한 런(`init_from_discriminator: fresh`, C32). 학습 · 관문 통과, 청취 대기(2026-10-02). Q7 준비(설계 v0.2, 평가 확장 문장)는 병행하되 J7의 선행조건이 아니다. J7에서 음높이 · 발음 회복의 유망한 신호가 없으면 Q7 파일럿 Q7B-1이 다음 우선순위다(실행은 별도 승인).
새 세션은 사용자 결정 없이 새 학습·청취 라운드를 시작하지 않는다.

**lexicon 전달**: 일본어 lexicon은 `INFLECT_JA_LEXICON`으로만 읽힌다. 설정하지 않으면 경고 없이 빈 lexicon이 된다. `dataset.json`은 lexicon 내용과 그 해시를 기록한다. export 패키지는 lexicon 파일을 담지 않으므로, export와 배포 런타임에서도 같은 파일을 넘겨야 한다(해시가 다르면 거부된다). 2026-09-29에 다시 준비한 `ja-spkA-v3`로 학습한 모델은 새 lexicon이, J1–J4는 이전 lexicon이 필요하다(파일은 `<work-dir>`에 있다).

**남은 사용자 결정**: J7 청취, 평가 확장 10문장 검토(새로 작성한 초안, 봉인 전), Q7B-1 실행 승인과 GPU 예산, 교사 합성 자료 사용, Q7-A 범위([Q7 설계 v0.2](roadmap/11-q7-ja-pretraining-design.md) §6). 2026-09-29에 정한 것: A1의 lexicon 1건은 반영했다. §8 Q5는 현재 표기를 유지하고, Q6은 KO 재개 때까지 보류한다. B2 재청취는 하지 않는다. 벤치마크 0004는 이미 교정됐고, 남은 것은 prepared 검증 목록의 옛 전사(분할 재현용, 학습에 쓰이지 않음)다.

---

## 2. 저장소 안과 밖

이 프로젝트의 기록은 **두 곳**에 나뉘어 있다. 새 환경에서 가장 먼저 헷갈리는 지점이다.

| 위치 | 내용 | 어디에 있나 |
|---|---|---|
| `Inflect/finetune/` (git) | 툴킷 코드·테스트·문서·예제 스크립트 | GitHub `ryouiki/Inflect` 브랜치 `main` |
| `<work-dir>/` (공개 저장소 아님) | 준비 데이터셋·런·체크포인트·청취 페이지·판정 파일·런 기록(`runs/*.md`)·진단(`evals/diag/`)·실행 스크립트(`scripts/`)·`env/G0.md` | **CUDA 머신(WSL2 Ubuntu-24.04) 로컬 디스크에만** 있다 |
| NAS 드라이브 | 원본 음성 코퍼스(화자 A JA/KR, JSUT) | 윈도우 NAS 드라이브 → WSL `<nas-mount>` |

- 로드맵이 `runs/…`, `listening/…`, `scripts/render_decoder_pair.py` 등을 인용하면 **전부 `<work-dir>/` 기준 경로**다.
  이 저장소의 `Inflect/scripts/`와는 무관하다.
- `<work-dir>/`와 원본 음성은 **외부 공개 금지**(§8 Q2: 배포 없음, 사설 연구). 공개 저장소에 커밋하지 않는다.
- 공개 문서에는 비공개 자료를 가명으로만 쓴다(화자 A · B, `ja-spkA-*`, 하위 자료 S1… · K1…). 실제 대응은 공개 저장소 밖에 둔다.
- 따라서 **GPU 작업(학습·렌더·청취 페이지 생성)은 CUDA 머신에서만** 이어갈 수 있다.
  다른 환경에서 할 수 있는 것은 코드·테스트·문서 작업이다(§4 표).

---

## 3. 환경별 셋업

공통 전제: Python ≥ 3.10(3.12 권장 — `onnxruntime` 1.29는 cp310 wheel이 없다), git.

### 3.1 CUDA 머신 (WSL2 Ubuntu-24.04, RTX 5090 / sm_120) — 학습·렌더 가능

이 문서 부록 §6.1이 실행판이다. 요점만:

```bash
cd <repo>   # 이 머신의 기존 클론 위치
git fetch origin && git checkout main && git pull
cd finetune
python3 -m venv .venv && source .venv/bin/activate && unset LD_LIBRARY_PATH
python -m pip install -U pip
python -m pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128   # 반드시 먼저
python -m pip install -e ".[onnx,dev,ja,ko]"
pytest
INFLECT_TEST_BASE_MODEL=micro pytest -k inventory -p no:warnings
```

- **torch를 cu128 인덱스에서 먼저** 깐다. 순서가 바뀌면 sm_120 커널 없는 PyPI 빌드가 들어온다.
- `~/.bashrc`의 `LD_LIBRARY_PATH=/usr/local/cuda/lib64`가 wheel 동봉 cuDNN을 가리므로 torch 셸에서는 `unset`.
- 데이터셋 마운트(부록 §6.2): `wsl.exe -d Ubuntu-24.04 -u root -- bash -lc "mkdir -p <nas-mount> && mount -t drvfs <nas-drive>: <nas-mount>"`.
  9p라 느리므로(≈53 files/s) 학습·prepare는 `<work-dir>/data/` 로컬 복제본에서만 한다.
- GPU 작업은 한 번에 하나만 돌린다.

### 3.2 Linux (GPU 없음) — 코드·테스트·문서

```bash
git clone https://github.com/ryouiki/Inflect.git && cd Inflect
cd finetune
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU 빌드로 용량 절약
python -m pip install -e ".[dev,ja,ko]"      # ONNX 작업이면 ,onnx 추가
pytest
```

실물 체이닝 테스트(`INFLECT_TEST_BASE_MODEL=micro`)는 HF에서 Micro를 받아 CPU로도 돌지만 느리다. 필요할 때만.

### 3.3 Windows 11 (네이티브) — 코드·테스트·문서

GPU 작업은 WSL2의 CUDA 머신 셸(§3.1)에서 한다. 네이티브는 편집·테스트용이다.

```powershell
git clone https://github.com/ryouiki/Inflect.git; cd Inflect
cd finetune
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev,ja,ko]"
pytest
```

- 실행 정책 오류가 나면 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
- `[ko]`의 `python-mecab-ko`가 wheel을 못 찾으면 KO 없이 `".[dev,ja]"`로 설치하고 KO 테스트는 Linux/WSL에서 돌린다.
  (네이티브 Windows에서의 `[ja,ko]` wheel 설치는 이 문서 작성 시점에 실측하지 않았다 — 처음 해보는 사람이 결과를 여기에 적는다.)
- `.venv`를 `finetune/` 안에 두어도 `test_public_safety.py`가 dot 디렉터리를 건너뛰므로 괜찮다(G0 결함 1 수정).

### 3.4 iOS 태블릿 · 웹 세션 (Claude Code on the web 등)

로컬 실행 환경이 없으므로 **문서 읽기·리뷰·문서 수정** 위주로 쓴다. 클라우드 세션은 §3.2 절차로 CPU 테스트까지 돌릴 수 있다.
`<work-dir>/`에는 접근할 수 없으므로 런 기록이 필요한 질문은 CUDA 머신 세션으로 넘긴다.

---

## 4. 어떤 환경에서 무엇을 할 수 있나

| 작업 | CUDA 머신 | Linux/Win11 (CPU) | 태블릿·웹 |
|---|---|---|---|
| 문서 읽기·수정, 로드맵 갱신 | ✅ | ✅ | ✅ |
| 코드 수정 + `pytest` | ✅ | ✅ | 웹 세션만 |
| 프론트엔드 덤프(`examples/frontend_review_dump.py`) | ✅ | ✅ | 웹 세션만 |
| `prepare` / `audit` (원본 음성 필요) | ✅ | ❌ | ❌ |
| 학습·렌더·`evaluate`·청취 페이지 생성 | ✅ | ❌ | ❌ |
| 청취·판정 (사용자) | 페이지를 연 브라우저 어디서든 | | |

---

## 5. 이어가기 전에 읽을 것 (순서)

1. 이 문서 §1–§2.
2. [로드맵 목차](roadmap/README.md)의 "현재 상태" → [08-joint-adaptation.md](roadmap/08-joint-adaptation.md)(§3.33–§3.39) → [10-risks.md](roadmap/10-risks.md) R16 → §8.
3. 이 문서 부록 §6.4 **"인계받는 사람이 먼저 알아야 할 것"** — `export --package-template micro` 필수, `--min/--max-duration-seconds`는
   필터가 아니라 단언, `max_steps`가 run identity에 들어가 `--resume` 연장은 거부된다(전체 상태를 이어 연장하려면 `--branch-from`, 새 레시피는 `--init-from`, TRAINING.md), 등.
4. [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — 링잉 계측의 한계(짧은 창의 수치, 비율 대 절대 레벨)가 정리돼 있다. 측정할 수 없으면 아무것도 보고하지 않는 규칙은 TRAINING.md에 있다.
5. [TRAINING.md](TRAINING.md) — 개선안 b의 옵션(전부 기본 off), `--init-from`, `--discriminator-update-order`의 의미.

**작업 규칙** (로드맵이 지켜온 것):
- 게이트와 판정 규칙은 **데이터를 보기 전에** 고정하고 커밋한다.
- 자동 지표는 스크린이고 판정자는 청취다. 청취 페이지는 블라인드·실물 앵커·catch 복제를 쓴다.
- 새 학습·새 청취 라운드는 **사용자 승인 후에만** 연다.
- 프론트엔드를 고치면 그 전 데이터셋은 export가 거부한다 — 프론트엔드 수정은 prepare 앞에 온다.

---

## 6. 세션을 마칠 때

다음 세션(다른 기기·다른 콘솔)이 이어받을 수 있도록:

1. 로드맵의 해당 파일(`roadmap/`)과 [CHANGELOG.md](roadmap/CHANGELOG.md)에 한 줄을 남긴다. 목차 README의 "현재 상태"도 바뀌었으면 고친다.
2. 이 문서 §1 표와 "다음 하나"가 바뀌었으면 고친다(갱신일 포함).
3. `<work-dir>/`에만 있는 새 기록은 로드맵에 **경로를 인용**해 둔다 — 다른 환경에서는 그 파일을 볼 수 없으므로,
   결론 문장은 로드맵 본문에 있어야 한다.
4. 커밋·푸시한다(`main`, `ryouiki/Inflect`에만). 푸시하지 않은 작업은 다른 기기에서 보이지 않는다. 공개 문서에는 가명만 쓴다.

---

## 부록 — CUDA 머신 인계 원문 (옛 로드맵 §6, 2026-09-27에 이 문서로 옮김)

아래는 옛 로드맵 §6을 그대로 옮긴 것이다(익명화만 적용). 로드맵 파일과 옛 기록의 "§6.x"는 이 부록의 같은 번호를 가리킨다.

### 6.1 환경 구축 — 2026-09-04 실행판

이 머신의 저장소는 `<repo>`다(작성 당시의 `<other-repo>/...`가 아니다).
Blackwell(sm_120)에서는 **torch를 cu128 인덱스에서 먼저** 깔아야 한다. `pyproject`의 `torch>=2.2`는
sm_120 하한을 강제하지 않으므로, 순서를 바꾸면 기본 PyPI 빌드가 들어와 커널이 없다.

```bash
cd <repo>/finetune
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
wsl.exe -d Ubuntu-24.04 -u root -- bash -lc "mkdir -p <nas-mount> && mount -t drvfs <nas-drive>: <nas-mount>"
```

경로 목록은 `<prior-ja-project>/configs/paths.example.yaml`,
환경 절차는 같은 저장소 `docs/environment.md`. (작성 당시 표기한 `<other-repo>/...`는 이 머신에 없다.)

`<nas-mount>`은 9p로 **≈53 files/s · ≈25 MB/s**다. 많은 작은 파일에서 병목이 되므로 학습·prepare는
로컬 복제본에서만 한다. 2026-09-04 복제·검증한 것:

| 데이터셋 | 로컬 경로 | 검증 |
|---|---|---|
| spkA/spkB JA (processed) | `<work-dir>/data/ja` | `sha256sum -c` 2019/2019 |
| 화자 A KR | `<work-dir>/data/ko/spkA_KR` | 3,401파일 · 48 kHz/1ch/24-bit 전수 · 전사 1:1 · 4.825 h |
| JSUT (`repeat500` 제외) | `<work-dir>/data/jsut` | 8 하위셋 7,196발화 |

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

*(2026-09-27 보완: 아래는 2026-09-04 당시의 기록이다. 지금은 본문 §3.1의 `main` 절차를 따른다.)*

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
| ✅ **M0/G0** | 통과. §3 M0과 `<work-dir>/env/G0.md` |
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
  학습 연장은 export → `--base` 체이닝으로 새 run을 만드는 것뿐이다. *(2026-09-27 보완: 이제 `--branch-from`으로 전체 상태를 이어받아 연장할 수 있다. 셔플 순서만 분기점에서 다시 시작된다.)*
- `metrics.jsonl`에는 **타임스탬프도 검증 loss도 없다.** 벽시계는 로그로 재고, 체크포인트 선택은
  `evaluate` + 청취로 한다.
- **알려진 G2P 한계는 `docs/LANGUAGES.md`에 언어별로 정리돼 있다** — 예를 들어
  한국어는 `세기`·`층`·`장` 앞 두 자리 수를 자릿수로 읽는다. 새 오독을 만나면
  프론트엔드를 고치기 전에 먼저 거기를 본다.
- 자동 지표는 전부 스크린이고 판정자는 청취다(§5.2). 이 저장소의 결함 대부분은
  테스트가 아니라 **검수 덤프를 눈으로 훑다가** 나왔다.

---

