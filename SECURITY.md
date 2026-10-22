# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x  | ✅ |

## Reporting a Vulnerability

**Please do not open a public issue for security problems.**

Report privately via GitHub's Security Advisory form:

> https://github.com/ssinder/ampml/security/advisories/new

Include:

- Type of issue (e.g. path traversal, code execution via untrusted input, dependency vulnerability)
- Full path(s) of the affected file(s)
- Steps to reproduce
- Impact assessment
- Suggested mitigation (optional)

You can expect an initial response within 7 days. If the report is accepted, a
fix will be released as a patch version, and you will be credited in the
advisory unless you prefer otherwise.

## Scope

This project is a **research and educational machine learning pipeline**. It
executes local data-preprocessing scripts and trains models on local files. It is
not a network service and does not accept untrusted input at runtime.

The following are in scope:

- Vulnerabilities that allow arbitrary code execution when processing
  attacker-supplied files (e.g. feature CSV or FASTA inputs)
- Path traversal in scripts that accept user-supplied paths
- Insecure deserialization (`pickle`) of model files received from untrusted sources
- Dependency vulnerabilities with a demonstrated exploit in this codebase

The following are **out of scope**:

- The accuracy or scientific validity of the models (please open a regular issue)
- Findings from automated scanners without a demonstrated impact
- Vulnerabilities in upstream dependencies that are not reachable from this code

## Note on model files

This repository ships pre-trained models as `.pkl` files. **Pickle files can
execute arbitrary code when loaded.** Only load model files from this repository
or from a source you trust. If you need to verify integrity, use the SHA-256
checksums recorded in `results/training_metadata.json`.

If you replace the model files in `results/models/`, regenerate the checksums
and update that file — the verification audit (`src/99_verify_test_metrics.py`)
validates them.
