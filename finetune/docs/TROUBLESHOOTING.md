# Troubleshooting

Start with the first failing stage. Do not work around preparation or audit
errors by editing prepared files.

## Command or module not found

Install from the `finetune` directory in an activated environment:

```bash
python -m pip install -e .
python -m inflect_finetune --help
```

If the editable install is not wanted, build and install the wheel instead.

## Manifest rejected

Check that:

- the file is UTF-8 JSONL or CSV;
- every non-empty JSONL line is one JSON object;
- `audio` and `text` are strings;
- audio paths are relative to `--audio-root`;
- paths do not contain traversal outside the root;
- referenced files exist;
- JSON quoting and escaping are valid.

## Audio cannot be decoded

Preserve the original file, decode it with a trusted audio tool, and convert a
copy to a standard uncompressed WAV. Confirm sample rate, channels, duration,
and finite samples. Do not rename a compressed file to `.wav`.

## eSpeak or phonemizer failure

Confirm the installed eSpeak NG library is available, the language code exists,
and `phonemizer` can process a short sentence independently. Record package and
eSpeak versions when reporting the problem.

If only specific text fails, reduce it to the smallest reproducible input and
inspect unsupported symbols, language mixing, and normalization.

## Bundled frontend failure

A bundled frontend needs its extra installed, for example `pip install ".[ja]"`
for `ja-openjtalk`. It also requires the language it declares, so
`--frontend ja-openjtalk` must be paired with `--language ja`.

If preparation fails on specific rows, reduce the transcript to the smallest
failing text. A row that is only punctuation, brackets, or a bare long-vowel
mark has no reading and is rejected rather than silently dropped; fix the
transcript instead of the frontend.

Export recovers the bundled hook from the prepared dataset. If it still asks
for `--frontend-hook`, the dataset was prepared with `--frontend custom` and
your own file, or `--prepared-dataset` was not passed.

A hook source hash mismatch at export means the toolkit version changed after
preparation. Re-prepare with the current version rather than overriding it.

## Unknown symbols

Do not delete unknown symbols from prepared text. Determine whether:

- the frontend emitted an undeclared symbol;
- Unicode normalization produced a different code point;
- the base inventory lacks a valid target-language phoneme;
- punctuation or a word-boundary marker was omitted from the inventory.

Correct the frontend or symbol declaration, rerun `prepare`, and audit again.
New valid symbols should be initialized and learned; they should not be mapped
silently to unrelated base symbols.

## Out of GPU memory

Use a smaller validated memory preset, reduce the configured batch workload,
shorten training segments only if the trainer supports that change, or increase
gradient accumulation. Restart after clearing the failed process.

Do not compare runs as equivalent if batch semantics, segment length, precision,
or optimizer behavior changed. The mel projection and the waveform transform now
run in full precision regardless of the surrounding cast, which fixed a
half-precision window that made the transform return complex32 and quantized the
mel target as well as the prediction. Mixed-precision loss values are therefore
not comparable across that change; the largest observed difference in a log-mel
value is 0.025, and runs with `--no-amp` are unaffected.

## Resume rejected

Compare the recorded toolkit version, base checkpoint, dataset hash, symbols
hash, frontend metadata, model configuration, and optimizer schema. Resume
rejection usually means the run inputs changed.

Note that this release adds option fields for its own reasons, so every
checkpoint written before it is unresumable even at their defaults.

`identity fields differ: ['options']` after an upgrade usually means the run
inputs did not change and the toolkit did. The public options are part of the
run identity, so a release that adds an option field makes every checkpoint
written before it unresumable, including at that field's default. This is the
guard doing its job. Finish the work in a new run, or export the interrupted
one and chain from it.

Start a new run rather than forcing incompatible state to load.

## Training loss improves but audio worsens

Stop and inspect held-out audio. Common failures include decoder buzz, metallic
resonance, excessive sibilance, clipped endings, duration collapse, speaker
drift, and overfitting. One of those has a mechanical explanation and automatic
detection; see the section on ringing below.

Select checkpoints using matched held-out listening and diagnostics, not loss
alone. More steps can make adaptation worse.

## A steady tone or hum behind the voice

The giveaway is that it does not stop when the speaking does. A comb of tones at
multiples of the sample rate over the frame hop sits under the whole render,
including the silence between words, and on headphones it localizes as a single
pitch behind the voice. For Micro at 24 kHz with hop 256 that is 93.75 Hz and
its multiples.

Confirm it from an evaluation report rather than by ear. `grid_tone_excess_db`
is zero by construction for real speech and measured 8.15 dB at the median on a
rejected run against -0.13 dB on the speaker's own recordings.
`steady_tone_artifact_score` separated the same two sets completely, 29.9
against 0.00. `clips_f0_locked_to_frame_grid` counts clips where the pitch
tracker reported the comb as the voice; one failing checkpoint scored 134 of
160 there. The training run also writes the two cheap screens into each
`validation/step-*.json`, so the arrival can be dated to within a validation
interval instead of discovered at the end.

The cause found by investigation was drift in the latents, not in the decoder
weights. The released checkpoint carries no posterior encoder, so a fresh one
starts every run, and while the decoder is frozen the adversarial gradients
still reach the posterior encoder through it with nothing anchoring where they
go. They do not reach the flow, which the KL term updates instead. Feeding the drifted latents to the released decoder
rang harder than feeding them to the adapted one. That was read at the time as
ruling the decoder out, which it does not: the reverse case, released latents
through the adapted decoder, was never run, so how the two interact is still
open. The upsampler has no anti-imaging filter, which is why this particular
grid is where the energy lands.

There is no known fix, and the search for one is mid-flight. The two training
controls that exist for this were tried on one corpus at 10,000 steps each and
did not work: gating the adversarial term left the artifact where it was, and a
reconstruction-only decoder polish made it substantially worse, collapsing the
median tracked pitch onto the comb frequency. But every one of those runs
trained against a mel loss that floored its two sides differently, so what
those numbers measure is those controls under a broken objective.

That defect is fixed, and the controlled comparison has been run. Against
signal power it looked like progress: the broadband floor fell 4.64 dB and the
comb 3.16 dB, on 40 and 37 of 40 sentences. Measured absolutely the same
sentences say something else. On-grid power rose 0.99 dB, with the fixed arm
lower on only 9 of 40, while total power rose 3.97 dB on 39 of 40 and the
renders came out shorter with less silence in them. The comb did not shrink;
the render got louder. Both arms then failed listening outright at 1,500 steps,
scored "not a human voice" on every row, so the round is a usability rejection
rather than a deferral at that checkpoint. It is not a prediction for longer
training: the Japanese base was equally not speech at step 1,000 on either
path and reached 4 of 5 by step 20,000. The mel fix stays in: it is a defect fix on its own
terms, and this experiment did not resolve the artifact by itself.

That episode is the reason this page keeps three readings of the comb. The
ratio, the level and the absolute power ranked the two arms three different
ways, and the listening test agreed with none of them about which was better.
Screens eliminate; the listener decides.

Crossing latents against decoders has since narrowed where to look. The same
twelve sentences were rendered through three decoders with the latents held
fixed. Latents from the released prior path came out clean through all three,
including two decoders that had trained for 7000 steps; latents from an
adapted run rang through all three, including the released decoder. That held
on 12 of 12 sentences in every decoder. So adapted decoder weights are not
necessary for the comb, and the latents are where it enters. That is not the
same as clearing the decoder: this architecture may still be what turns those
particular latents into a comb, and the released decoder has no anti-imaging
filter. Which property of the latents matters is not settled either: removing
their per-channel offset helps on one screen and hurts on the other, matching
their scale helps modestly, and smoothing them in time helps on neither.

Naming the screen matters when reporting that grid. Decoder training looks
progressively worse on the ratio, 6.33 then 8.01 then 9.89 dB across the
released, control and reconstruction-polish decoders, and progressively better
on the relative level over the same three, -44.48 then -46.47 then -47.66 dB.
Which of those is the worse sound has not been listened to.

The listening round that followed reframed the problem. At 1,500 steps neither
arm produced speech at all, only ringing at the tempo of the sentence, while a
10,000-step control was scored as awkward but recognisable speech. What is
failing is the formation of speech, not a residual noise on top of it.

Splitting the two paths on existing checkpoints then separated the two
problems outright. A 10,000-step run reconstructs and infers as recognisable
speech while carrying 8.1 dB of comb excess; the 1,500-step arms carry the
same 7.7 to 8.6 dB and are not speech at all. **How much comb a render has
does not predict whether speech formed in it.** Treat them as two problems.

The same split found a working path that later broke. At step 500, with the
prior, flow and decoder all still frozen, inference produced clean speech,
scored "speech but awkward" with no defect at all, in a voice the listener
described as an English-speaking man reading Korean text. A thousand steps of
linguistic adaptation later the same path is not speech and is the most
smeared track on the page. Both 1,500-step paths also hand their pitch to the
comb, tracked at 93.75 Hz on 34 and 39 of 40 sentences, while every render
scored as speech tracks near the speaker's own 360 Hz. Pitch lock is not
necessary for a render to fail, but nothing that locked has passed.

Two more sentences, the highest-pitched one and the one with the most silence,
gave the same picture on every track, so it is not a property of one sentence.
What was confirmed is speech with no defect on all three, scored 2, 3 and 2
for naturalness. Whether it is Korean is a separate axis and it did not pass:
the language answers were "hard to say", "hard to say" and "no", and the free
text on one sentence says outright that it did not sound Korean. So the
released prior already renders these Korean phonemes as defect-free human
speech in a foreign voice, and the first thousand steps of adaptation take
that away. Speech preservation and Korean intelligibility are reported apart.

Walking every checkpoint the longer runs saved, with pitch lock as the
instrument, puts a shape on that. Every run locks within the first 500 steps
of linguistic adaptation, while its decoder is frozen. One gated run was
unlocked at step 1000 and locked by 2000, but the other run under the same
gating locked on all 40 sentences at 1000, so "gating delays the lock" is a
single observation that did not reproduce. The
lock releases only where a run both trained its decoder and had the
adversarial term on: two runs did, and both released at the first checkpoint
after unfreezing. A run that trained its decoder on reconstruction alone
stayed locked through seven thousand decoder steps, and a run that kept its
decoder frozen with the adversarial term on stayed locked to the end. The comb
excess stayed between 6 and 12 dB through all of it, released or not. Pitch
lock is a screen, and the checkpoints where it releases have not yet been
listened to; treat "released" as "released", not as "speech".

The Japanese speaker's runs close the obvious explanation. A run that unfroze
its decoder at step 500, the moment linguistic adaptation begins, still locked
on 40 of 40 sentences by step 1000, so a decoder that can respond does not
prevent the collapse, and a run with the adversarial term off throughout
locks just as early, so that term is not necessary for it either. Neither
result excludes a contribution or an interaction. Every configuration tried collapses in the first 500
steps of linguistic adaptation. What every run shared and none varied is a
posterior encoder initialised fresh and warmed for 500 steps, a linguistic
learning rate of half the base rate, and a KL weight of one. Those are the
axes a follow-up has to move, one at a time.

### Why the Japanese language base did not ring

The first adaptation this toolkit ran, a released model onto nine hours of a
Japanese corpus for twenty thousand steps, does not show this failure at all.
Measured with the same instrument on forty sentences, both its paths sit
between 1 and 4 dB of grid excess at every checkpoint from step 1000 onward,
and its pitch never locks to the comb: at worst five sentences of forty, at
step 20000 two. The Korean runs at step 1000 sit at 7 to 11 dB with all forty
locked. Whatever goes wrong is not something this architecture does whenever
it meets a new language or a female speaker.

The voice run that followed it warm-started from that base but did not inherit
its posterior encoder: the export carries the text path and the decoder, and a
hundred tensors of posterior are initialised fresh. That run locks partially,
thirteen to sixteen sentences at step 1000, and works its way back out by step
7000 while its pitch climbs toward the target speaker. Three regimes, then,
and they differ in more than one thing at once: data volume, pitch range and
starting point all move together, so none of this attributes a cause.

Two short runs test the one axis that can be isolated. From the same export,
byte-identical, one run starts a fresh posterior and one inherits the trained
one; nothing else differs. On the instrument the inheriting run is
consistently lower, and both are clean while the linguistic path is still
frozen.

Listened to, the base at 20,000 steps is speech on both paths, scored 4 of 5
with a defect "noticeable if you listen for it", which is the level the earlier
informal impression had described. At step 1000 it is not speech at all, on
either path, even though the instrument had it unlocked at 205 Hz with a comb
excess of 3 dB. That is the second time an unlocked render has failed as
speech, so an unlocked pitch says the comb has not captured it and nothing
more; the difference between this base and the failing runs at step 1000 is
between an unlocked non-speech and a locked one. The replay under current code
was not speech at 1000 either, so that comparison could not be judged; the
listener called it the more smeared of the two on every sentence. And the
inherited posterior did not turn the voice runs into speech at 500, 1000 or
2000 on two sentences of three, for all that the instrument preferred it;
on the highest-pitched sentence alone it reached "speech but awkward" where
the fresh posterior did not. Inheriting the posterior on its own was not sufficient as
a remedy in this condition; whether the fresh initialisation contributed to the
failure stays open, since nothing yet separates a posterior that was fine when
handed over and degraded during warm-up from one that never handled this
speaker's audio. How much longer the original run needed to form speech, and
what the voice run lost and when, are the next questions; which other state to
inherit stays a later candidate.

### When speech forms, and what adaptation removes

Two questions that mattered have answers now, both from checkpoints that
already existed.

The Japanese base had formed speech by step 2,000, the earliest point
listened to: two of three sentences on the inference path and all three on
reconstruction. Quality then keeps climbing for the rest of the run, from a
median naturalness of 2 at 2,000 to 5 at 20,000 as the defect grade falls
from 2 to 1. Speech forming and quality finishing are separate milestones, so
a run that has one has not necessarily reached the other. Note also that
speech was there before the decoder unfroze at 3,000; naturalness does jump
at the first checkpoint after unfreezing, but one run on one grid cannot
attribute that.

The same number of updates, 2,000, produced no speech in the Korean and
voice runs, so the two conditions behave differently at equal step counts.
That is as far as it goes: 2,000 steps is roughly the third epoch of the
Japanese corpus and the fourteenth of the voice corpus, the starting weights
differ, and so do data volume and target pitch range, so this does not rule
out training length and does not say which of those matters. Speech forming
is also not the same as being heard as Japanese: the language question was
answered "yes" on all three sentences first at step 4,000, with 3,000
answered "hard to say" on all three and nothing observed in between.

On the voice side the loss is now visible directly. The frozen base prior,
reading the voice speaker's Japanese text before any adaptation, is speech on
all three sentences at naturalness 3 to 4 with a barely noticeable defect.
That output is the base speaker's voice reading the target speaker's text;
the target voice had not been acquired. After two thousand steps of
adaptation the same sentences score 1. Within the observed window,
adaptation degraded inference output that was already there; whether that is
permanent, or recoverable with more training, has not been tested. The
inherited posterior's reconstruction failure, separately, was present before
warm-up began: reconstructing the voice speaker's own recordings through it
fails on two of three sentences at step zero. Those scores sit on the floor,
so whether warm-up made it worse still cannot be read, and reconstruction
exercises posterior and decoder together, so it does not single out the
posterior. The instrument had preferred step zero by 2.7 dB, which again did
not survive listening.

The degradation is not confined to the new speaker's text. Rendering the base
corpus's own sentences through the adapted checkpoint drops naturalness from 5
to 2 on both sentences listened to, with the language answer going from yes to
no, so whatever adaptation did to the inference path it did to text the model
already read well.

Shrinking the corpus to the voice corpus's size did not reproduce it. A subset
of the base corpus matched to the voice corpus on rows, hours and number of
passes, trained from the same export with the same inherited posterior and the
same schedule, came back level with the full-corpus control on both sentences
listened to: same naturalness grade, same defect grade, still heard as the
right language, and the duplicated track agreed exactly on both. The
full-corpus control held up on those sentences too, so the round had a working
positive reference. That is where the reading stops — the size axis stays open,
because the difference the round measured came out undecided under its own
two-grade bar rather than at zero effect.

Read it narrowly for a second reason. One flag moved three things at once: how
much data, how many times each row is seen, and which rows were drawn. What it
did not move is whether the model had seen the rows before — both arms train on
material the base already fit, so that term is held fixed here, and it is the
switch to the new speaker's corpus that changes it. Separately, because the
subset is drawn from the corpus the base already trained on, none of this
speaks to repeating a small amount of *new* material, which is the voice
condition; that is a limit on generalising, not a confound between the arms.
What the round does do is move the search: the remaining differences on that
side are the speaker's own voice characteristics, the pitch range, the
alignment, and whether the inherited posterior matches the data it now meets.

Restoring one module at a time narrows it a little and not enough. Putting the
flow back to its pre-adaptation weights, with everything else left adapted,
raises naturalness by one grade on both sentences and turns one sentence's
language answer from no back to yes; restoring the text encoder and duration
predictor moves one sentence by one grade and the other not at all. The
pre-declared bar for a recovery was two grades on both sentences, so module
attribution stays undecided — which is not the same as no effect, and the free
text agreed that the restored variants sound better than the adapted one. All
of them keep the worse defect grade, so none returns to the original quality.
Restoring one group leaves a mixture of weights that never trained together,
so a variant that does not recover does not clear its group. Note also that
the posterior and the discriminator kept training through this window, so
naming a module says nothing yet about what moved it.

The flow restore was then put to the listener once more, on two sentences of
the new speaker's text it had never been heard on, with the adapted track
duplicated on both rows. This time the duplicate agreed exactly, all three
axes, on both sentences, so the outcome does not hang on that reading: the
restore gained one grade on one sentence and two on the other, and the
pre-declared bar of two grades on both was not met. It is the fourth sentence
in a row where restoring the flow scores higher than leaving it adapted, and
the second page where that does not add up to a recovery. Both restores still
carry the worse defect grade. Gains from separate pages are not summed, so
module attribution stays undecided, and re-listening on this axis stops here;
what remains untested is which member of the corpus-change bundle matters.

The restart itself was then taken out of the question. A second run started
from the same export with the same inherited posterior and the same schedule,
adapting back onto the corpus the export came from, so the only flag that
differed was the dataset. Its options file matches the first run on every key,
and both checkpoints at step 500 render all forty sentences bit-identically,
which they must while warm-up freezes the inference path. Listened to on two
sentences never used before, that run scores within one grade of the
pre-adaptation output on both, keeps the same defect grade, and is still heard
as Japanese; the run that had changed corpus scored two or more grades lower
on both sentences in the same round, and its language answers went to no. So
under these conditions, with the target data unchanged, no large degradation
was observed. That is not the same as restarting being harmless: this run
revisits data it had already converged on, and its inherited posterior was
fitted to exactly that data, so the comparison changes several things at once
under one flag. Which member of that bundle matters is untested, and the
mechanism stays open.

The same restores were then listened to on the new speaker's own text, with
the pre-adaptation and adapted renders in the same round. Naturalness there
falls from 4 and 5 before adaptation to 1 on both sentences after it, and both
language answers go from yes to no, so the degradation shows on the target text
as well as on the base corpus. The flow restore gains two grades on both
sentences, which would meet the pre-declared bar, but this page's catch pair
disagreed: the same bytes scored 1 and 2, one grade apart and across the
speech boundary. Scoring that sentence with the duplicate instead drops the
gain to one grade and changes the classification, so the comparison is
undecided and module attribution still is. The signal is the largest seen so
far and it points the same way as the base-corpus page, where the same restore
gained one grade on each sentence; the two pages are not added together,
because the bar is two grades on both sentences of a single page. Restoring
the text encoder and duration predictor gains one grade per sentence there,
also undecided. Both restores keep the worse defect grade on this text too,
and the listener noted that even the best restore had shifted the delivery and
rung more than the pre-adaptation render.

The first of those has been moved. A run with the posterior warmed for 1,500
steps instead of 500, nothing else changed, was listened to against the
original at the same offset into adaptation: every adapted track on every
sentence scored "not a human voice", including the reconstruction at the
adaptation boundary. So that schedule did not prevent the absence of speech within the observed
early window, and the readiness manipulation itself did not take, which leaves
the readiness hypothesis untested rather than refuted. What the run did show is
that a fresh posterior locks its pitch onto the comb by step 1000 with the
linguistic path and decoder both frozen, so the collapse does not need
linguistic adaptation at all; posterior-only training reaches it. The
500-step posterior that had not locked may simply have stopped earlier.

The same lever was then pulled on the inherited posterior, in the other
language, with adaptation length held equal. That is a different cell and not
a repeat: the earlier warm-up run started its posterior from scratch on the
Korean corpus with a different voice actress and was heard 500 updates into
adaptation, while this one inherits the posterior from the base export, adapts
on the Japanese voice corpus, and runs three thousand steps so that both arms
take exactly fifteen hundred adaptation updates. Neither comparison cleared
its pre-declared two-grade bar. The reconstruction at each arm's own
adaptation boundary scored 2 in both arms on both sentences, and the inference
at equal adaptation length scored 1 in both arms on both sentences; both gains
are zero. The duplicated reconstruction track agreed exactly on all three axes
in both rows, so neither reading hangs on the duplicate, and the baseline
scores sat low enough, 2 and 1, that a two-grade gain was arithmetically
available. Read that as the manipulation not clearing its own bar rather than
as the schedule leaving the output alone: with both gains at zero, the
hypothesis that adaptation should start from a better-prepared posterior is
untested here, not refuted. Lengthening warm-up is also not a single variable
— the discriminator keeps training through it, so its update count and its
learning rate move as well, along with the data stream position and the
optimizer state — which is why this reads as a statement about that schedule
and not about posterior readiness. A longer warm-up still, a lower posterior
learning rate, and gating the adversarial term during warm-up all remain
untried. The instrument had put the two boundary reconstructions within 0.14 dB
of each other, the two arms' mel windows 0.011 apart and their KL windows 0.31
apart (mel 0.9881 / KL 3.2167 in the 500-step warm-up arm, 0.9774 / 3.5254 in
the 1,500-step arm); the listening put both arms at the same score on both
sentences, which is a description of where those readings landed, not the
reason for the verdict.

Taking the critic out of the warm-up did not move either question. A run
identical to the one above except that the generator's adversarial and
feature-matching terms are held at zero for the fifteen hundred warm-up steps,
and restored to full weight for the fifteen hundred adaptation steps, was
listened to against it on two fresh sentences. Neither row met the
pre-declared bar of two grades on both sentences: the boundary reconstruction
scored 2 against 2 on one sentence and 1 against 1 on the other, and the
inference at equal adaptation length scored 1 against 1 on both. The
intervention itself is visible in the log rather than inferred -- every gated
row carries a zero weight with both generator-side terms null and a finite
discriminator loss, and the first adapted row is back at full weight -- so
what did not clear the bar is the schedule's effect on what the listener
heard, not the manipulation. Read the gains per sentence rather than as one
number: scoring the duplicated baseline instead of the original turns one
sentence's gain from zero to one, which is still short of the bar and does not
change the row. That one is not a partial improvement in the candidate: the
candidate track scored 2 under either scoring and did not move at all, and the
gain appeared only because the baseline's own score for the same bytes fell
from 2 to 1. That duplicate is worth noting on its own: the same bytes
scored 2 and 1 on that sentence, one grade apart and across the speech
boundary, so whether that track counts as speech at all is undecided here.
Seven axis cells were left blank, all on the second sentence, which is why
nothing is said about its language axis. The listener's free text on both
sentences was that little is voiced and the rest rings, and added that the
phonemes that do come through sound closer to the base corpus's speaker than
to the target -- an observation recorded as such, since voice identity is not
an axis this page scored.

Training past the decoder unfreeze was then tried on the inherited recipe,
which no run had done. Ten thousand steps, seven thousand of them with the
decoder training, and the same two sentences heard at the last frozen
checkpoint, at six thousand, and at the cap. The output at the cap is speech
on both sentences and is heard as Japanese on both: naturalness 3 and 2 with
the language answer yes, against naturalness 1 with "hard to say" and "no" at
the last frozen checkpoint. The duplicated track agreed exactly on all three
axes in both rows, so none of that hangs on a single reading.

It did not clear the bars. Recovery was pre-declared as naturalness 3 or
better with the language answer yes on both sentences, and one sentence came
in at 2. Usability adds a defect grade of 1 or better on top of that, and the
defect grade is 2 on both sentences -- the ringing is still there, and the
listener said so in the free text on both. The gain over the last frozen
checkpoint is two grades on one sentence and one on the other, which is short
of a bar that asks for two on both. So: speech and the right language on both
sentences, and neither pre-declared threshold met. Write those as the separate
findings they are, and do not promote the first into "usable".

Two things about that run limit what it can carry. The host killed it twice
while it was writing a checkpoint, and the loader's position is not
checkpointed, so the data order restarted at step 6500 and again at 8000; this
is not an uninterrupted ten thousand steps, and the discontinuities sit between
the very checkpoints the contrasts compare. And the instrument moved in the
same window -- the pitch lock released completely after the unfreeze while the
comb stayed at six to nine decibels -- but the lock is measured over forty
sentences and the naturalness over the two that were heard, so they are not
the same set. On one of those two the lock released further between six
thousand and the cap while the naturalness did not move at all.

One thing to know before reading any two runs against each other: repeating a
run with the same nominal settings does not reproduce the same trained weights.
Two runs here were configured identically over their first five hundred steps —
same seed, same corpus, and options that differ only in values which take effect
later — and at step 500 every one of the posterior encoder's hundred tensors
differs between them, by up to 9.4e-03, as does every one of the discriminator's
hundred and eleven, by up to 5.0e-02, while the four hundred and ten tensors
held frozen match bit for bit. What produces that variation has not been
isolated: nothing here separates GPU kernel nondeterminism from mixed precision
from anything else, and no experiment to do so is planned. Rendering is stable
by a separate check, the one every round already runs — the same checkpoint
re-rendered at a fixed seed comes back byte-identical. So a contrast between two
runs carries this variation alongside whatever was manipulated, its size is
unmeasured, and a single repeat pair would be one observation of it rather than
a bound on it. The verdicts already recorded are arithmetic on listened scores
and do not move. What does not follow is reading a gain of zero as evidence that
the variation is small, and a verdict that treats a gain of one grade or less as
its result is weakened by this rather than supported.

Four things were tried and measured and did not work, under that caveat.
Gating the generator's adversarial term while the decoder is frozen leaves the
artifact unchanged and raises the latent drift, because that term had been
pulling the latents back. A reconstruction-only decoder polish raises the
artifact while its own reported losses fall. Freezing the decoder for the whole
run does not prevent the comb either; an ablation that never unfroze it showed
the comb by step 1000 with 94 per cent of frames locked to the grid. Unfreezing
earlier was worse early and no better at the end.
Restoring the released decoder at export time makes it louder, not quieter.

The cause is therefore still open. The latent drift the investigation first
blamed does not track the artifact on its own: two runs ending at drift 1.190
and 1.528 produced 8.15 and 8.27 dB of comb, which rules out a single scalar
mean as the controlling variable and rules out nothing else about the latents.
Both the training path and the inference path ring, which shows a mismatch
between them is not the whole story rather than showing there is none. Treat
the screens as the reliable part and the explanations as provisional.

## Output is intelligible but pronunciation is wrong

Verify the source transcript, normalized text, phonemes, symbol coverage, and
training examples for the affected sound. If the frontend is wrong, fix it and
re-prepare the corpus. Training longer will not reliably correct a systematically
wrong phoneme sequence.

## Adapted voice sounds unlike the speaker

Check speaker consistency, recording conditions, data quantity and coverage,
and whether the run changed language and voice simultaneously. Speaker
similarity is not guaranteed by the fixed-voice architecture.

## ONNX model fails to parse or load

Confirm the file is fully downloaded and its checksum matches. Test with the
documented ONNX Runtime and ONNX versions. Re-export from the same inference
checkpoint and validate every graph before upload.

An ONNX file existing on disk is not evidence that export succeeded.

## Reporting an issue

Include:

- exact command;
- toolkit commit or version;
- operating system, Python, PyTorch, CUDA, eSpeak, and ONNX Runtime versions;
- base model and checksum;
- redacted manifest schema and failing row;
- prepared dataset and symbols hashes;
- complete error text;
- minimal reproducible input.

Do not attach private speech data without the speaker's permission.
