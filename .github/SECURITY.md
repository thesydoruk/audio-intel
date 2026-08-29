# Security policy

## Supported versions

Security fixes land on the latest `main` of this repository.

## Reporting a vulnerability

Please **do not** open a public GitHub issue for a security problem.

Report privately through
[GitHub Security Advisories](https://github.com/thesydoruk/audio-intel/security/advisories/new).

Include:

- a description of the issue and its impact
- steps to reproduce, or a proof of concept
- affected version or commit, if known

You should get an acknowledgement within a few days. A fix or a reasoned
response will follow as soon as the issue can be assessed.

This service accepts uploaded media and optional Hugging Face tokens. Treat
`.env`, `HF_TOKEN`, and model caches as secrets; they must never appear in
issues, pull requests, or logs you paste publicly.
