# 코드 변경 목록과 부록 (§4 · §5)

> [로드맵 목차](README.md)의 한 부분이다. 절 번호(§)는 옛 `MULTILINGUAL_ROADMAP.md`의 번호를 그대로 쓴다.

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
| C23 | 문장별 CSV에 절대 PSD 파생 열 | `<work-dir>/scripts/verdict_mel_ab.py --csv` | M7 | ✅ 완료(2026-09-06). 스크린 세 값이 모두 순수 게인에 불변이라 절대량을 말하지 못한다. `레벨 + rms_dbfs`(원래 출력)와 `레벨 + 고정 기준`(공통 재생 레벨)을 나눠 기록하고 길이·무음 비율을 함께 남긴다. **라이브러리 키는 추가하지 않았다 — 항등식으로 파생된다** |

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

### 5.2 <prior-ja-project>에서 반영하는 것

같은 사용자의 일본어 파인튜닝 선행 프로젝트(`<prior-ja-project>`, 실험 124건 ·
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

- ONNX forensic → PyTorch parity → wav-to-latent → adapter 스택 전체. <prior-ja-project> 전제의
  대부분은 "학습용 체크포인트 미공개"에서 파생됐다. Inflect는 학습 가능한 체크포인트와
  warm-start 경로를 공개한다 — **우회로가 통째로 불필요하다.**
- <prior-ja-project>의 어휘사전 수술 연작. orthography 프론트엔드(모델이
  텍스트를 직접 소비) 제약에서 나온 싸움이며, V7이 그 실패 클래스의 소멸을 보인다.
  다만 **고유명사 오독이라는 새 실패 클래스**가 생기므로 사용자 사전 슬롯은 유지한다(M1).
- `midhigh` 비음 학습 표적 (교훈 81 — 소진됨). 새 학습 목적함수로 제안하지 않는다.
- <prior-ja-project>의 base · 출하 후보 자산. Supertonic 아키텍처 전용.

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

