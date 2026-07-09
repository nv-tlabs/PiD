# PiD Contribution Rules

Thanks for your interest in contributing to PiD! This document covers the
local setup, code style, and PR / sign-off flow.

## Development setup

Follow the install steps in [`README.md`](README.md) to get a working conda
environment, then install the dev extras and the pre-commit hook:

```bash
pip install -e ".[dev]"
pre-commit install
```

`pre-commit install` registers the hooks in `.pre-commit-config.yaml` so they
run on every `git commit`. The configured hooks are:

- `trailing-whitespace`, `end-of-file-fixer`, `check-symlinks`
- `ruff-format` (formatter)
- `ruff-check --fix` (lint, auto-fix where possible)

To run the hooks across the whole repo without committing:

```bash
pre-commit run --all-files
```

Each time you make a commit, you may see
```bash
$ git commit
Trim Trailing Whitespace.................................................Failed
- hook id: trailing-whitespace
- exit code: 1
- files were modified by this hook

Fixing README.md

Fix End of Files.........................................................Passed
Check for broken symlinks............................(no files to check)Skipped
ruff fix.................................................................Passed
ruff format..............................................................Passed
```

Then `git add .` again and `git commit` again.

## Code style

- Python target is **3.12**; formatter and linter are both **ruff**
  (configured in `pyproject.toml`). Ruff replaces black + isort + flake8
  here — please don't introduce other formatters.
- Line length is **120**.
- Imports are sorted by ruff's isort rules; `pid` is the first-party package.
- Follow existing conventions in the relevant file / submodule / module when
  adding or extending code. If a directory has an established pattern,
  match it rather than introducing a new one.

## Submitting a pull request

1. Fork the repo and create a topic branch off `main`:

   ```bash
   git checkout -b my-feature main
   ```

2. Make your change, and make sure `pre-commit run --all-files` is clean.
3. Sign off every commit — see [Signing your work](#signing-your-work).
4. Open a PR against `main`. In the description, summarize the change and
   link any related issue.
5. Keep the PR focused — split unrelated changes into separate PRs. If your
   branch falls behind, rebase onto `main` rather than merging it back in.

## Licensing

This project is licensed under the [Apache License 2.0](LICENSE). By
contributing, you agree that your contribution will be released under the
same license. Make sure you have the right to submit your work and that it
introduces no license or patent conflict.

## Signing your work

We require that all contributors sign off on their commits. The sign-off
certifies that the contribution is your original work, or that you have the
right to submit it under a compatible license. Commits without a
`Signed-off-by` trailer will not be accepted.

To sign off, use the `-s` (or `--signoff`) flag when committing:

```bash
git commit -s -m "Add cool feature."
```

This appends the following line to your commit message:

```
Signed-off-by: Your Name <your@email.com>
```

Full text of the DCO:

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
1 Letterman Drive
Suite D4700
San Francisco, CA, 94129

Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

(b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

(c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

(d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
```

Thanks in advance for your patience as we review your contributions — we
appreciate them!
