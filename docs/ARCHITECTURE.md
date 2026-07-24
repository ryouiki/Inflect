# Inflect v2 Architecture

Inflect v2 is a family of compact, fixed-voice English text-to-speech models.
Both releases use the same end-to-end VITS-family deployment design and return
a 24 kHz waveform without an external vocoder.

## Inference path

```mermaid
flowchart LR
  A["English text"] --> B["Normalization"]
  B --> C["Phonemization"]
  C --> D["Transformer text encoder"]
  D --> E["Duration + latent speech model"]
  E --> F["Residual coupling flow"]
  F --> G["Alias-reduced neural decoder"]
  G --> H["24 kHz mono waveform"]
```

The public runtime contains:

1. **Text normalization.** Numbers, common abbreviations, punctuation, and
   whitespace are normalized before phonemization.
2. **English phoneme frontend.** The normalized sequence is converted to the
   token inventory expected by the checkpoint.
3. **Text and duration model.** A transformer encoder builds linguistic
   representations and a stochastic duration predictor estimates alignment.
4. **Latent speech generator.** Monotonic alignment and a latent-variable path
   map encoded text to acoustic latents.
5. **Residual coupling flows.** Invertible transformations shape the latent
   distribution for generation.
6. **Integrated waveform decoder.** An alias-reduced neural decoder emits the
   final 24 kHz waveform.

## Micro and Nano

| | Inflect-Micro-v2 | Inflect-Nano-v2 |
| --- | ---: | ---: |
| Complete inference parameters | 9,356,513 | 3,966,721 |
| FP32 weights | 37.53 MB | 15.97 MB |
| Output | 24 kHz mono | 24 kHz mono |
| Public API | Shared | Shared |

Micro uses wider hidden and decoder capacity. Nano narrows the same deployment
pattern rather than removing the waveform generator or requiring another
model. The published counts cover every learned module loaded for
text-to-waveform inference. Training-only discriminators are not counted.

## Long text

The runtime detects punctuation-aware boundaries, synthesizes bounded chunks,
applies short edge fades, and joins them with controlled pauses. This improves
memory predictability and avoids truncating the model to a small character
limit.

It is important to state what this is not:

- it is not autoregressive waveform streaming;
- it does not provide a measured time-to-first-audio guarantee;
- chunks do not share one global acoustic plan, so a boundary can occasionally
  expose a small change in rhythm or energy.

## Controls

The validated runtime exposes:

- `speed`: speaking-rate multiplier;
- `variation`: stochastic delivery strength;
- `seed`: deterministic reproduction of a stochastic generation;
- `device`: CPU or CUDA execution.

Pitch adjustment in the public playground is a restrained output control, not
a learned speaker or emotion control.

## Open-weight boundary

The model repositories publish:

- deployable weights;
- text frontend and inference runtime;
- configuration and dependency metadata;
- held-out samples and evaluation artifacts;
- integrity hashes and third-party notices.

The private corpus-generation pipeline and full optimization infrastructure
are not part of the open-weight release. The public architecture description
therefore focuses on the model that users can download and execute.
