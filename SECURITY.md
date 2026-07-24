# Security Policy

Inflect is a local speech-generation project. Security issues may involve code
execution, model artifacts, dependency handling, or unsafe integration.

## Reporting

If you find a security issue, do not open a public issue with exploit details.

Send a private report to the repository owner through a GitHub security
advisory. If that is unavailable, email
[owen.aw.song@gmail.com](mailto:owen.aw.song@gmail.com). Do not include exploit
details in a public issue.

Include:

- affected files or workflow
- reproduction steps
- impact
- suggested fix if known

## Voice and Data Safety

Do not submit:

- private voice recordings
- credentials
- Hugging Face tokens
- API keys
- copyrighted datasets without clear rights
- model checkpoints containing private training data

## Supported versions

Security fixes target the latest Inflect v2 model packages and the current
`main` branch. Legacy v1 releases may not receive the same fixes.

## Responsible Use

Inflect is intended for local creative tooling, accessibility research,
education, and product prototyping.

Do not use generated speech for impersonation, fraud, harassment, deceptive
attribution, or other unlawful activity.
