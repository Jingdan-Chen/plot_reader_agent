# Installation and maintenance contract

Read README.md before changing the setup or runtime. This repository is a local
plot-reading application with remote visual inference and local measurements.

## Install for the user's chosen agent

- Codex setup: run `python3 scripts/setup.py --agent codex --smoke-test`, then
  `uv run plot-reader web --agent codex`. Plot inference is exactly
  `gpt-6-luna`, high reasoning effort by default, priority/Fast service tier.
- Claude setup: run `python3 scripts/setup.py --agent claude --smoke-test`, then
  `uv run plot-reader web --agent claude`. Plot inference is exactly
  `claude-haiku-5-5`, high effort by default.
- Each analysis uses the user's selected effort for every model call, including
  repairs, tool decisions, reviews and clarification resumes. OpenAI options are
  light/medium/high/xhigh (light maps to low); Claude options are
  low/medium/high/extra (extra maps to xhigh). Persist the choice with the run.
- Check Tesseract, Poppler, Python >=3.11, uv, the selected CLI and account access.
  The setup wrapper reports missing system packages; use the README commands.
- A smoke test makes one real inference call. Run it when the user asks you to
  finish installation and verify the application, as these setup prompts do.
- If authentication needs the user's interaction, give the exact sign-in command.
  Never ask for account credentials or tokens in chat or tracked files.
- Do not substitute a different model or silently switch inference/OCR providers.
- Keep the server process running and give the actual printed localhost URL.

## Preserve the calling design

The assistant interprets images through a strict JSON schema. The host owns
source files, validates tool arguments, executes local numerical tools, and
retains uncertainty. General-purpose inference-worker tools remain disabled.
The bounded scatter-template protocol has four decisions and two scans maximum.
Do not convert template heatmap scores into probabilities or ground-truth points.
Do not display raw model reasoning in the latest-action box.

PDF input uses the packaged PaddleOCR skill or an explicitly selected installed
copy. Count pages locally, ask for a range above 10 pages, isolate it before OCR,
then let the user choose an exact extracted figure. Keep image upload immediate.

The complete-curve checkpoint is immutable. Verify its manifest; do not edit its
files or bypass hashes. Implement changes in the application layer or introduce
an explicitly reviewed new checkpoint.

## Verify and publish

Use `uv run pytest -q` and `node --test tests/test_browser_watch.cjs tests/test_timing_view.cjs` for this
release. Run meaningful additional checks when changing provider commands,
coordinate mapping, checkpoint handling or job recovery. Distinguish mocked
provider tests from a live model smoke test when reporting results.

Publish only the source-only export produced by `scripts/build_release.py`.
Local uploads, jobs, environment files, account config and historical development
campaigns are not release inputs. The existing user's data and Git history must
not be reset to prepare a fresh public repository.
