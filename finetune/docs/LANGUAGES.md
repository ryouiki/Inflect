# Languages and frontend configuration

## One configured language per checkpoint

An adaptation run produces a checkpoint tied to the language and symbol
inventory recorded in its prepared dataset. It does not make the original
checkpoint multilingual, and it does not expose a runtime language selector.

For example, separate English and Spanish runs produce separate English and
Spanish checkpoints. Combining their files does not produce a bilingual model.

## Bundled language frontends

Some languages are not served by eSpeak. For those, the toolkit ships a named
frontend that is selected directly:

```bash
python -m inflect_finetune prepare \
  --manifest data/metadata.jsonl \
  --audio-root data/audio \
  --language ja \
  --frontend ja-openjtalk \
  --output prepared/ja
```

| Name | Language | Extra | Notes |
| --- | --- | --- | --- |
| `ja-openjtalk` | `ja` | `ja` | Open JTalk G2P with pitch-accent marks |
| `ko-g2pkk` | `ko` | `ko` | g2pkk phonology mapped directly from Hangul |

A bundled frontend is a name for a custom frontend that ships with the toolkit.
It is recorded in `dataset.json` as a custom frontend plus a `registry` block,
so source hashing, declared-symbol enforcement, determinism checks, and
deployment packaging behave exactly as they do for your own hook. `export`
recovers the bundled hook file by itself; `--frontend-hook` is not needed.

Install its extra before preparing:

```bash
python -m pip install ".[ja]"   # or ".[ko]"
```

The underlying dictionary is a third-party artifact under its own license. A
package exported with a bundled frontend is not self-contained.

### Japanese

`ja-openjtalk` maps every Open JTalk phone into the released symbol inventory,
so a Japanese dataset adds no embedding rows. Confirm this after preparing:

```bash
python -m inflect_finetune audit --dataset prepared/ja --require-no-new-symbols
```

Pitch accent is written with `↑` (rise at the start of an accent phrase) and
`↓` (fall after the accent nucleus). Both characters are in the released
inventory, so the symbol count does not change, but neither carries a useful
pretrained meaning — the corpus has to teach them.

Long vowels stay as repeated vowels rather than a length mark, so each mora
keeps its own symbol and its own predicted duration. Devoiced vowels fold to
their plain counterpart; whether that distinction deserves its own symbol is a
listening question, not something to assume.

Open JTalk collapses `、`, `。`, `！`, `？`, and `・` into one undifferentiated
pause, so the text is split on punctuation and the writer's marks are restored
around each phonemized chunk. Brackets and quotation marks are removed during
normalization because they have no reading.

Numbers are normalized before Open JTalk sees them: digit-grouping commas are
removed so `3,000` reads as one number, and a decimal point does not end a
sentence. Verify these against your own transcripts — number formatting is
exactly where a frontend fails quietly.

Dump readings for a fluent speaker before training:

```bash
python examples/frontend_review_dump.py \
  --sentences examples/japanese_review_suite.txt \
  --frontend ja-openjtalk --language ja \
  --output review/ja.tsv
```

The bundled suite covers frontend behavior, not your corpus. Run it on a random
sample of your own transcripts as well, and record the misreading rate rather
than assuming it is zero.

Proper nouns are where Open JTalk misreads. Supply a reading lexicon as a JSON
object of surface/reading pairs and point `INFLECT_JA_LEXICON` at it:

```json
{"鷹神": "たかかみ"}
```

Its contents are part of the hashed frontend metadata, so changing the lexicon
correctly invalidates an export prepared with the previous one.

### Korean

`ko-g2pkk` runs g2pkk to obtain the pronunciation as Hangul, then maps that
surface form to phonemes directly. Hangul is featural, so the mapping is a
mechanical syllable decomposition once the phonology has been applied.

eSpeak is deliberately not in this chain. Its Korean voice merges the
three-way laryngeal contrast — 살/쌀, 자다/짜다, 불/뿔, 방/빵, 정/쩡, and 사/싸
each come back as a single phoneme string. A merged pair is a phonemic
collision no amount of data can undo.

Tense consonants are written with `ʼ` and aspirated ones with `ʰ`, both from
the released inventory, so the contrast costs no new symbols. Korean has no
lexical pitch accent, so nothing corresponds to the Japanese accent marks;
spaces are eojeol boundaries.

Phonology is applied one eojeol at a time. Given a whole sentence, g2pkk
applies liaison across word boundaries and produces different words: 오늘 날씨
becomes 오늘 랄씨 and 희망을 얘기 becomes 히망으 럐기. The cost is that genuine
cross-boundary nasalization is missed — 몇 년 stays 멷 년 rather than 면 년 —
which reads as careful speech rather than as the wrong word.

**Latin letters and bare jamo are refused.** g2pkk reads some acronyms
incorrectly while consuming them entirely: `IT` becomes the syllable 읻 and
`AI` becomes 아이, leaving nothing behind to detect. Checking the output would
miss exactly the cases that matter, so the check runs on the normalized input.
Supply readings through a lexicon and point `INFLECT_KO_LEXICON` at it:

```json
{"AI": "에이아이", "IT": "아이티", "TV": "티비"}
```

`examples/korean_reading_lexicon.json` is a starting point. Nothing is applied
by default — the frontend does not guess a reading.

Numbers are left for g2pkk, which reads digit-grouping commas correctly
(3,000 → 삼천). Only the decimal point is rewritten, as 점, because g2pkk
leaves it unread.

Known g2pkk limits, to check against your own transcripts:

- 세기, 층, and 장 take a digit-by-digit reading for multi-digit numbers
  (21세기 → 이일세기, not 이십일세기). Write the number in Hangul or add a
  lexicon entry.
- A bare number with no counter is read digit by digit.
- Tensification is occasionally under-applied (여덟 시 → 여덜 시).

Dump readings for a fluent speaker before training:

```bash
python examples/frontend_review_dump.py \
  --sentences examples/korean_review_suite.txt \
  --frontend ko-g2pkk --language ko \
  --output review/ko.tsv
```

## eSpeak frontend

**Do not use eSpeak for Japanese or Korean.** It has voices for both, and
neither is usable. For Japanese it cannot read kanji at all — it emits the
literal English words "chinese letter" for each one. For Korean it merges the
tense/plain consonant contrast, so 살 and 쌀 become the same phoneme string.
Use `ja-openjtalk` and `ko-g2pkk`.

Confirm that eSpeak produces a sensible result for your language before relying
on it. Availability of a voice is not evidence of usable output, and the way it
fails is not always visible without checking minimal pairs.

The built-in frontend uses `phonemizer` with eSpeak NG:

```bash
python -m inflect_finetune prepare \
  --manifest data/metadata.jsonl \
  --audio-root data/audio \
  --language es \
  --frontend espeak \
  --output prepared/es
```

Use an eSpeak language or voice code appropriate to the transcripts and
recordings. Language codes, stress behavior, punctuation preservation, and
normalization must be recorded in `dataset.json`.

Before training:

1. Confirm eSpeak supports the requested code in the installed version.
2. Inspect normalized text and phonemes for a representative sample.
3. Ask a fluent speaker to review names, loanwords, abbreviations, and numbers.
4. Run `audit` and resolve unknown symbols.
5. Verify validation phonemes are covered by training.

## Symbol inventory migration

The base checkpoint and prepared language may use different symbols. Migration
must:

- copy shape-compatible generator weights;
- match text embedding rows by symbol string, never numeric position;
- deterministically initialize newly added symbol rows;
- record copied, new, and unused symbols;
- reject ambiguous or duplicate symbol definitions;
- preserve the prepared symbol order in the exported package.

New symbol initialization makes training possible; it does not provide a
pronunciation. The corpus must teach the acoustic realization and timing.

## Normalization is language-specific

Numbers, currencies, dates, abbreviations, casing, punctuation, and symbols
cannot be normalized reliably with one universal rule set. Users must verify
that normalized text matches what the speaker says.

Do not reuse English-specific normalization for a different language without
review. If the built-in frontend cannot represent the intended reading, use a
custom frontend rather than editing prepared phonemes by hand.

## Quality expectations

Availability of an eSpeak voice is not evidence that Inflect will train well on
that language. Languages with substantially different phonology, writing
systems, timing, or prosody may require more data, frontend work, model
capacity, and optimization changes.

Nano has less capacity and less tolerance for poor coverage than Micro. Use
Micro for the first adaptation attempt unless footprint is the primary
constraint and the Micro workflow has already been validated.

Every release should state:

- language and regional variety;
- frontend name and version;
- symbol inventory;
- speaker and corpus provenance;
- training and validation quantities;
- known pronunciation limitations;
- whether fluent speakers evaluated held-out output.
