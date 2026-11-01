(function (root) {
  'use strict';
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const seconds = value => typeof value === 'number' && Number.isFinite(value) ? `${value.toFixed(3)} s` : '—';
  const names = {starting:'Starting analysis',prepare:'Prepare figure',inspect:'Inspect and plan features',
    measure:'Measure selected features',coordinate_review:'Review source coordinates',
    complete_plan:'Plan complete curves',complete_validate:'Validate curve plan',complete_extract:'Extract complete curves',
    review:'Review results and explain',clarification:'Process clarification',user_wait:'Wait for your clarification',
    correct_points:'Recalculate corrected points'};
  const table = (headers, rows) => `<div class="table-wrap"><table><thead><tr>${headers.map(h=>`<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map(cell=>`<td>${esc(cell)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  const callTable = calls => table(['Call','Workflow stage','Wall time','Status'], calls.map((call,index)=>[
    `${index+1}. ${call.name}`,names[call.stage]||call.stage,seconds(call.elapsed_seconds),call.status]));

  function render(timing, downloadUrl) {
    if (!timing || !timing.summary) return '<section class="timing-report"><h3>Time cost analysis</h3><p>Timing was not recorded for this saved run. Start a new extraction to measure its time cost.</p></section>';
    const s = timing.summary;
    const summary = table(['Time cost','Wall time'],[
      ['Total elapsed',seconds(s.total_seconds)],
      ['Waiting for model responses',seconds(s.model_wait_seconds)],
      ['Local Python work',seconds(s.local_python_seconds)],
      ['Waiting for your clarification',seconds(s.user_wait_seconds)],
      ['Other host work and scheduling',seconds(s.host_overhead_seconds)]
    ]);
    const stages = table(['Workflow stage','Wall time','Model wait','Local Python'],(timing.stages||[]).map(stage=>[
      names[stage.name]||stage.name,seconds(stage.elapsed_seconds),seconds(stage.model_wait_seconds),seconds(stage.local_python_seconds)]));
    const model = timing.model_calls || [], local = timing.local_python_calls || [];
    const download = downloadUrl ? `<a href="${esc(downloadUrl)}">↓ Timing JSON</a>` : '';
    return `<section class="timing-report" aria-label="Time cost analysis"><div class="eyebrow">HOST-MEASURED WALL TIME</div><h3>Time cost analysis</h3>${summary}<p>Summary rows below total add up to total elapsed. Stage and call rows include the same work; do not add them again.</p><h4>Workflow stages</h4>${stages}<details><summary>Model responses · ${model.length} calls</summary>${model.length ? callTable(model) : '<p>No model calls.</p>'}</details><details open><summary>Local Python · ${local.length} calls</summary>${local.length ? callTable(local) : '<p>No local Python calls recorded.</p>'}</details><p>Total includes clarification waits. Upload, PDF figure discovery, browser rendering and downloads are outside this analysis. Model waits include CLI startup, image transfer, provider queueing and inference. Local calls include child tools such as Tesseract. Concurrent calls may overlap.</p>${download}</section>`;
  }
  if (typeof module === 'object' && module.exports) module.exports = {render};
  root.PlotReaderTiming = {render};
})(globalThis);
