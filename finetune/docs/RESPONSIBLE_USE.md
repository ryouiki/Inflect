# Consent, licensing, and responsible use

## Speaker consent

Use speech only when the speaker has knowingly authorized its use for model
training and the intended distribution. Consent to publish recordings does not
automatically include consent to train a synthetic voice.

Document:

- who owns or controls the recordings;
- what training and distribution uses were authorized;
- whether commercial use is permitted;
- whether the resulting model may be redistributed;
- how withdrawal requests are handled where applicable.

Do not train on private calls, leaked recordings, scraped personal media, or a
person's voice without permission.

## Voice identity

An adapted checkpoint is a fixed synthetic voice learned from its corpus. Do
not present it as the real speaker, use it for impersonation, or use it to
mislead listeners about who said something.

Where a generated voice resembles an identifiable person, disclose that the
audio is synthetic and follow applicable laws, platform rules, and contractual
restrictions.

## Data licensing

The Inflect code or base checkpoint license does not grant rights to a user's
adaptation dataset. Users are responsible for transcript, audio, dictionary,
frontend, and model-output rights.

An exported checkpoint cannot be distributed more broadly than its data and
dependency licenses permit. Preserve required attribution and notices.

## Language and cultural review

For new languages, involve fluent speakers in pronunciation and acceptability
review. Automated ASR and phonemization scores do not detect all offensive,
misleading, or culturally inappropriate output.

Document regional variety and known limitations rather than describing a model
as supporting an entire language based on one speaker or corpus.

## Security and privacy

Treat manifests and custom frontend hooks as untrusted inputs. Do not publish
source paths, user names, credentials, or private metadata in reports or
checkpoints.

Review exported files for:

- absolute paths;
- source transcripts that were not intended for release;
- speaker identifiers and personal information;
- embedded credentials or remote URLs;
- training logs containing private data.

## Release disclosure

An adapted model card should state:

- that it is community-adapted and not an official Inflect checkpoint;
- base model and toolkit version;
- fixed voice and configured language;
- data source, consent, and license;
- evaluation procedure and fluent-speaker involvement;
- known failure modes;
- intended and prohibited uses;
- contact or reporting path for harmful outputs.

Technical capability does not remove the obligation to obtain consent or use
the model responsibly.
