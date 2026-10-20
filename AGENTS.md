# Public repository safety

This repository is public. Never commit secrets, credentials, private source
documents, raw private traces, sensitive logs, or local environment files.

Before each push, review the exact staged file list and diff and run an
appropriate secret check. Stage explicit task files, preserving unrelated
workspace changes. Automated scanning does not replace manual review.

Do not print credential values. If a secret is found, stop publication and
arrange removal and rotation without repeating the value.
