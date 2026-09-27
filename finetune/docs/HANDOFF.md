# 인계 — 새 환경에서 셋업하고 이어가기

**대상 브랜치**: `feat/training-core-remedy-b` (main 대비 79커밋, 마지막 커밋 `91ff9f5` · 2026-09-22)
**갱신일**: 2026-09-27

이 문서는 **"어디서 무엇을 깔고, 무엇을 먼저 읽고, 무엇이 저장소 밖에 있는가"** 만 다룬다.
진행 상태·판정·결정의 정본은 [MULTILINGUAL_ROADMAP.md](MULTILINGUAL_ROADMAP.md)이고,
여기 있는 상태 요약은 그 문서의 색인일 뿐이다. 둘이 다르면 로드맵이 맞다.

---

## 1. 지금 어디인가 (2026-09-27 기준)

| 단계 | 상태 | 로드맵 |
|---|---|---|
| M0 / G0 환경 확정 | ✅ 통과 (2026-09-04, RTX 5090) | §3 M0 |
| M1 JA 프론트엔드 | 코드 완료. G1(c)는 자동 스크린으로 대체(사용자 결정) | §3 M1, §6.4 |
| M2 데이터 준비 | ✅ 통과 (2026-09-04) — `ja-arona-v2` · `ja-jsut-v1` · `ko-arona-v1b` | §3 M2 |
| M3 stage-1 (JSUT) | 학습 완료, 그 export를 `--base`로 stage-2 진행(G3 통과 줄은 로드맵에 따로 없음) | §3 M3 |
| M4 / G4 청취 | ❌ **미통과** — 전 렌더에 링잉(디코더 업샘플 격자 톤, R16) | §3 M4, §7 R16 |
| 링잉 대응 (학습 코어 개선안 b, C13–C23) | 코드 완료 · 라운드 1차 ~ Q까지 진행. **채택된 처방 없음** | §3.1 – §3.21 |
| M5 KO 프론트엔드 | 코드 완료, G5 통과. 실제 전사 표본 검수 남음 | §3 M5 |
| M6 KO 적응 + 배포 패키징 | 미착수 (C12 미완) | §3 M6 |

**마지막 라운드**: 청취 Q(2026-09-21, 외부 리뷰 각주 2026-09-22) — 같은 J1 잠재를 J1·I1 디코더로 낸 쌍이
네 축·직접 순위 모두 동률 → **이번 디코더 교체 후보를 닫았다.** 종료 범위는 "선정한 두 10k 체크포인트의
교체 비교"로 좁혀져 있다(각주 R2).

**다음 하나**: **권고 없음 — 선택은 사용자에게 있다**(§3.21 각주 R3). 구간 진단은 "L·M이 답하지 못한
어떤 질문에 답하고, 결과에 따라 어떤 개입을 고르거나 버리는가"를 적은 **별도 계획**이 있을 때만 연다.
새 세션은 사용자 결정 없이 새 학습·청취 라운드를 시작하지 않는다.

**남은 사용자 결정**: §8 Q5(JA 악센트구 경계 표기), Q6(KO ㅐ/ㅔ 병합) — 둘 다 청취 후.

---

## 2. 저장소 안과 밖

이 브랜치의 기록은 **두 곳**에 나뉘어 있다. 새 환경에서 가장 먼저 헷갈리는 지점이다.

| 위치 | 내용 | 어디에 있나 |
|---|---|---|
| `Inflect/finetune/` (git) | 툴킷 코드·테스트·문서·예제 스크립트 | GitHub `ryouiki/Inflect` 브랜치 `feat/training-core-remedy-b` |
| `inflect-work/` (git 아님) | 준비 데이터셋·런·체크포인트·청취 페이지·판정 파일·런 기록(`runs/*.md`)·진단(`evals/diag/`)·실행 스크립트(`scripts/`)·`env/G0.md` | **CUDA 머신(WSL2 Ubuntu-24.04) 로컬 디스크에만** 있다 |
| NAS `M:` | 원본 음성 코퍼스(arona JA/KR, JSUT) | 윈도우 드라이브 `M:` → WSL `/mnt/m` |

- 로드맵이 `runs/…`, `listening/…`, `scripts/render_decoder_pair.py` 등을 인용하면 **전부 `inflect-work/` 기준 경로**다.
  이 저장소의 `Inflect/scripts/`와는 무관하다.
- `inflect-work/`와 원본 음성은 **외부 공개 금지**(§8 Q2: 배포 없음, 사설 연구). 커밋하지 않는다.
- 따라서 **GPU 작업(학습·렌더·청취 페이지 생성)은 CUDA 머신에서만** 이어갈 수 있다.
  다른 환경에서 할 수 있는 것은 코드·테스트·문서 작업이다(§4 표).

---

## 3. 환경별 셋업

공통 전제: Python ≥ 3.10(3.12 권장 — `onnxruntime` 1.29는 cp310 wheel이 없다), git.

### 3.1 CUDA 머신 (WSL2 Ubuntu-24.04, RTX 5090 / sm_120) — 학습·렌더 가능

로드맵 §6.1이 실행판이다. 요점만:

```bash
cd /home/ysoya/projects/Inflect
git fetch origin && git checkout feat/training-core-remedy-b && git pull
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
- 데이터셋 마운트(§6.2): `wsl.exe -d Ubuntu-24.04 -u root -- bash -lc "mkdir -p /mnt/m && mount -t drvfs M: /mnt/m"`.
  9p라 느리므로(≈53 files/s) 학습·prepare는 `inflect-work/data/` 로컬 복제본에서만 한다.
- GPU 작업은 한 번에 하나만 돌린다.

### 3.2 Linux (GPU 없음) — 코드·테스트·문서

```bash
git clone https://github.com/ryouiki/Inflect.git && cd Inflect
git checkout feat/training-core-remedy-b
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
git checkout feat/training-core-remedy-b
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
`inflect-work/`에는 접근할 수 없으므로 런 기록이 필요한 질문은 CUDA 머신 세션으로 넘긴다.

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
2. [MULTILINGUAL_ROADMAP.md](MULTILINGUAL_ROADMAP.md) 머리말 → §3.21(마지막 라운드와 각주) → §7 R16 → §8.
3. §6.4 **"인계받는 사람이 먼저 알아야 할 것"** — `export --package-template micro` 필수, `--min/--max-duration-seconds`는
   필터가 아니라 단언, `max_steps`가 run identity에 들어가 연장은 export → `--base` 체이닝뿐, 등.
4. [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — 링잉 계측의 한계(짧은 창의 `None`, 비율 대 절대 레벨)가 정리돼 있다.
5. [TRAINING.md](TRAINING.md) — 개선안 b의 옵션(전부 기본 off)과 의미.

**작업 규칙** (로드맵이 지켜온 것):
- 게이트와 판정 규칙은 **데이터를 보기 전에** 고정하고 커밋한다.
- 자동 지표는 스크린이고 판정자는 청취다. 청취 페이지는 블라인드·실물 앵커·catch 복제를 쓴다.
- 새 학습·새 청취 라운드는 **사용자 승인 후에만** 연다.
- 프론트엔드를 고치면 그 전 데이터셋은 export가 거부한다 — 프론트엔드 수정은 prepare 앞에 온다.

---

## 6. 세션을 마칠 때

다음 세션(다른 기기·다른 콘솔)이 이어받을 수 있도록:

1. 로드맵의 해당 절과 **변경 이력** 표에 한 줄을 남긴다.
2. 이 문서 §1 표와 "다음 하나"가 바뀌었으면 고친다(갱신일 포함).
3. `inflect-work/`에만 있는 새 기록은 로드맵에 **경로를 인용**해 둔다 — 다른 환경에서는 그 파일을 볼 수 없으므로,
   결론 문장은 로드맵 본문에 있어야 한다.
4. 커밋·푸시한다. 푸시하지 않은 작업은 다른 기기에서 보이지 않는다.
