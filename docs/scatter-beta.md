# Beta scatter workflow

The beta branch adds an alternative application-layer path for all-points scatter
requests. The frozen numerical engines and complete-curve checkpoint are unchanged.
Select **Beta · concurrent source review** to try it. **Previous workflow** remains
the default because repeat trials still exposed legend and duplicate false positives.
The selected provider, model and per-run effort remain unchanged for all its calls.
Point queries and normalization retain the previous selected-measurement path.

## Calling design

1. One schema-constrained setup call identifies labeled axis ticks, independent
   checks, plot/legend rectangles and distinct marker series. It does not draft
   the entire point inventory. Local OCR of isolated axis-label strips challenges
   inferred check values when it finds a readable, monotonic sequence of labels.
   One bounded setup repair is allowed. OCR remains fallible; missing labels do
   not manufacture calibration evidence.
2. The existing optional template dispatcher keeps its four-decision/two-scan
   budget. Source-template responses remain proposals, not probabilities.
3. Local Python supplements templates with plotted appearance exemplars and
   marker-plus-line fitting. The latter tests bounded centers, glyph size/stroke
   and line nuisance ink. Exemplars are unverified appearance hypotheses;
   source identity review remains required. No curve interpolation or assumed
   sampling grid creates points.
4. Near-duplicate and weak corner seeds are grouped into primary/reserve batches.
   Every retained candidate belongs to one batch. Reserve seeds are also reviewed,
   so grouping does not silently remove missing markers. Each batch shows native
   source patches, local context and the source legend.
5. Up to four independent runner instances review batches concurrently. Each
   inherits the frozen provider/effort profile and has its own process, response,
   work directory and bounded repair. Parent cancellation terminates all children.
6. Model selections must pass local source-support checks. Line-only centers and
   unsupported corner/frame assignments are retained in `local-acceptance.json`
   as rejected claims, rather than converted into measurements. Fixed source
   template centers cannot be moved by a model or by this validation.
7. A source-wide overlay audit records missing markers, ghosts and uncertainty.
   Failed review batches cannot be declared complete. The local numerical engine
   measures the resulting inventory. Large unnormalized inventories are measured
   in bounded calls of at most 100 selections, without truncation.

Artifacts include `scatter-setup.json`, `scatter-axis-evidence.json`, `marker-grounding.json`, individual batch
results, `marker-inventory.json`, `local-acceptance.json`, `scatter-coverage.json`
and the existing timing/table/overlay files. Concurrent call durations can sum to
more than wall time; the timing summary accounts for overlap.

## Optional JEV image trial

JEV is **off by default**. Configure `TYPESAFE_API_KEY` in the server environment,
then explicitly choose:

- **Evaluate alongside the agent**: shadow classification, not shown to the agent.
- **Give classifications to the agent**: advisory classifications supplied with
  source patches. The selected agent still reviews every candidate.

Candidate image sheets are sent to TypeSafe's experimental image endpoint using
pinned `jev-1.14.0`. JEV does not supply coordinates, accept measurements, or delete
candidates. Its probabilities are not calibrated scientific confidence. Invalid,
missing or failed answers fall back to ordinary source review and remain recorded.
API keys are never included in artifacts, CLI worker environments or release inputs.
The legacy `jev.py` text-routing adapter is not used by this path.

References: [TypeSafe image input](https://docs.typesafe.ai/images),
[Choice responses](https://docs.typesafe.ai/primitives/choice),
[model versions](https://docs.typesafe.ai/models).

## Validation limits

Synthetic tests cover concurrent ownership/cancellation, frozen effort inheritance,
exact candidate accounting, source coordinates, line-only rejection, marker/line
support, fixed templates, duplicate ambiguity, large inventories and JEV failure
handling. Live runs are separate from mocked provider tests.

Development trials use private local artifacts under `artifacts/beta-evaluation/`;
these are excluded from release exports. Clear-marker and negative-region checks
on the reported screenshot are a partial development diagnostic, not a complete
independent truth set. Speed gains do not establish full recall or scientific
measurement accuracy. Crossings, clipping, inseparable markers and source/legend
appearance mismatch remain explicit limitations.

## Initial screenshot trials (2026-11-01)

These are individual development runs on the user's six-series conversion plot,
with `gpt-6-luna`, medium effort, and priority service in every selected-agent call.
They are not a statistical latency benchmark. Historical 600–675 second runs used
**high** effort, so they are not the default-setting comparator.

| Workflow | Wall time | Clear locations / 27 | Correct series + location / 27 |
| --- | ---: | ---: | ---: |
| Previous workflow, fresh medium run | 221.3 s | 20 | 0 (all output series were unresolved) |
| Beta, JEV off, before the local axis-label guard | 112.1 s | 25 | 23 |
| Beta with advisory JEV, earlier gate revision | 127.4 s | 14 | 12 |
| Beta repeat, local axis-label guard enabled | 98.9 s | 24 | 22 |

The location tolerance was four native pixels. The 27 markers were selected by
visual inspection as a partial diagnostic; they do not cover every overlap,
endpoint or glyph. The checks evolved during development and are not a held-out
accuracy benchmark. The no-JEV beta produced 84 rows, including duplicate/ambiguous
claims; its coverage audit did not assert completeness. Zero predictions fell in
the selected negative regions, which does not establish global precision.

The JEV trial made 25 batches; six received HTTP 503 responses. Its fallible hints
did not improve this full-image run, despite rejecting several line-only seeds in
a smaller pilot. JEV remains off by default. The selected agent still reviewed all
candidate batches when TypeSafe was unavailable.

Visual review also caught an unprinted 300-minute tick used as an independent
check. The beta now challenges that value using isolated local OCR: the readable
labels in the source are 0, 200, 400 and 600. This is an application-layer guard;
it neither changes the frozen axis mapping nor treats OCR as ground truth.

The repeat with the guard chose the printed 400-minute check correctly. However,
its model setup returned no legend exclusion, then the review accepted 45 points
in the legend rectangle. It returned 135 rows and failed the completeness audit.
This demonstrates why the 27-marker diagnostic cannot establish full-image
precision and why beta is opt-in. The next priority is an independently checked
legend mask, followed by source-based duplicate resolution; neither a faster
model nor parallel review alone resolves these errors.

Verification for this iteration: the full Python suite passed (205 tests), then
all six workflow API cases passed after the final input-validation change. All
12 browser/timing tests and checkpoint verification passed. Live selected-agent trials and the live
JEV trial are reported above, separately from mocked provider tests.
