# Scatter extraction workflows

Main uses **Concurrent source review** by default for all-points scatter requests.
This promotes the non-JEV beta pipeline. Choose **Classical (legacy)** in
Extraction settings to retain the previous sequential workflow. Both use exactly
`gpt-6-luna` (priority/Fast) or `claude-haiku-5-5`, with **high** effort by default.
A user's explicit effort applies to every call, including concurrent children,
repairs, reviews and clarification resumes.

The saved workflow identifier remains `beta` for concurrent review and `legacy`
for classical review, preserving existing requests. A missing workflow field in
an older saved run continues to use classical review. A missing effort uses the
new high default when that run is explicitly rerun; recorded profiles and explicit
medium/low choices are preserved. Point queries and normalization retain the
previous selected-measurement path. The immutable curve checkpoint is unchanged.

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

## Branch boundary

JEV image classification, its controls, prompts, network client, optional inference
metadata and direct runtime dependency are excluded from main. Main rejects new,
restarted or resumed requests that require JEV instead of silently changing them.
The pre-existing optional text adapter and immutable checkpoint copies are not
part of scatter extraction and have not been changed.

The [beta branch](https://github.com/Jingdan-Chen/plot_reader_agent/tree/beta)
preserves the experimental integration and its original defaults for comparison.
No API key or private job history is included in either published branch.

## Validation and limitations

Tests cover independent concurrent workers, cancellation, exact candidate
accounting, per-run effort inheritance, native source pixels, line-only rejection,
fixed templates, duplicate ambiguity, large inventories, printed-tick checks,
workflow selection and rejection of JEV requests on main. CLI command tests use
mocked responses; live image/JSON smoke tests are reported separately.

Earlier individual LUNA medium-effort trials of the concurrent pipeline took
99–112 seconds versus 221 seconds for a fresh classical run. These are development
observations, not performance guarantees for the new high default. A partial
27-marker check found 24–25 locations within four pixels, versus 20 for classical.
That subset does not establish global recall or precision: duplicate claims
remained, and one run with an incorrect layout accepted 45 legend-region points.

Coverage audits and local rejected-claim records remain visible. Review source
alignment, legend exclusions, duplicates and overlapping/boundary markers before
using the exported table. Changing the default effort does not establish that
those recognition limitations have been resolved. Private trial images, jobs and
historical development campaigns remain outside source-only releases.
