(() => {
  const token = new URLSearchParams(location.search).get('token') || '';
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const measurementDefaults = JSON.parse($('#measurement-defaults').textContent);
  const state = {step: 1, upload: null, page: null, crop: null, cropBox: null, imageDims: null,
    targetConfirmed: false, category: 'spectroscopy', mode: 'selected', jobId: null, eventSource: null,
    drawing: false, result: null, timingCleanup: null};
  const api = async (path, options = {}) => {
    const headers = new Headers(options.headers || {});
    headers.set('X-Plot-Reader-Token', token);
    if (options.body && !(options.body instanceof FormData) && !headers.has('Content-Type')) headers.set('Content-Type', 'application/json');
    const response = await fetch(path, {...options, headers});
    const data = response.headers.get('content-type')?.includes('json') ? await response.json() : await response.text();
    if (!response.ok) throw new Error(data?.error || String(data));
    return data;
  };
  const showError = message => { const box = $('#form-error'); box.textContent = message; box.classList.toggle('hidden', !message); };
  const toast = message => { const box = $('#toast'); box.textContent = message; box.classList.remove('hidden'); setTimeout(() => box.classList.add('hidden'), 2600); };
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const setStep = step => {
    state.step = step;
    $$('[data-panel]').forEach(node => node.classList.toggle('hidden', Number(node.dataset.panel) !== step));
    $$('.step').forEach(node => { const n = Number(node.dataset.step); node.classList.toggle('active', n === step); node.classList.toggle('done', n < step); });
    window.scrollTo({top: 0, behavior: 'smooth'});
  };
  const showPreview = async (url, label, status = 'READY') => {
    const rail = $('#rail-preview'); rail.classList.remove('empty');
    rail.innerHTML = `<img alt="${esc(label)}" src="${esc(url)}">`;
    $('#rail-title').textContent = label;
    $('#rail-status').textContent = status;
  };
  const apiPost = (url, data) => api(url, {method: 'POST', body: JSON.stringify(data)});
  const authed = url => `${url}${url.includes('?') ? '&' : '?'}token=${encodeURIComponent(token)}`;
  const updateContinue = () => { $('#to-channel').disabled = !(state.upload && state.targetConfirmed); };
  const showLatestAction = (message, at) => {
    if (!message) return;
    $('#latest-action').textContent = String(message).slice(0, 220);
    $('#latest-action-time').textContent = at ? new Date(at).toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'}) : '';
  };
  let sourceEpoch = 0, figureTimer = null;
  const stopFigureWatch = () => { clearTimeout(figureTimer); figureTimer = null; };
  const clearSource = () => {
    sourceEpoch++; stopFigureWatch();
    showLatestAction('Waiting for an extraction request.');
    state.upload = null; state.page = null; state.crop = null; state.figureRun = null; state.targetConfirmed = false;
    for (const id of ['source-card', 'pdf-range', 'pdf-intake-status', 'pdf-figures', 'ocr-details']) $(`#${id}`).classList.add('hidden');
    $('#pdf-figures').replaceChildren(); $('#file-input').value = '';
    $('#rail-title').textContent = 'Waiting for a figure'; $('#rail-status').textContent = 'EMPTY';
    $('#rail-preview').className = 'rail-preview empty';
    $('#rail-preview').innerHTML = '<div class="empty-art"><span>⌗</span><i>∿</i></div><span>Your plot will appear here</span>';
    updateContinue();
  };
  const showRange = source => {
    $('#pdf-range').classList.remove('hidden');
    $('#range-description').textContent = `This PDF has ${source.page_count} pages. Choose the pages to search for pictures.`;
    for (const id of ['page-start','page-end']) { $(`#${id}`).max = source.page_count; $(`#${id}`).disabled = false; }
    $('#page-start').value = 1; $('#page-end').value = Math.min(10, source.page_count);
    $('#find-figures').disabled = false;
  };
  const selectFigure = async (figure, intake, source, epoch) => {
    try {
      const target = await apiPost('/api/target', {upload_id: source.id, figure_id: figure.id, figure_run_id: intake.run_id});
      if (epoch !== sourceEpoch || state.upload?.id !== source.id) return;
      state.page = target.page; state.crop = target.crop; state.targetConfirmed = true;
      state.upload.explanation = target.explanation; state.figureRun = intake.run_id;
      const url = authed(`/api/source/${source.id}/figures/${intake.run_id}/${figure.id}`);
      await showPreview(url, `${figure.title}${figure.page ? ` · page ${figure.page}` : ''}`, 'SELECTED');
      $('#rail-note').innerHTML = '<span class="rail-note-icon">i</span><span>The selected picture will be used for extraction.</span>';
      $$('.figure-card').forEach(card => card.classList.toggle('selected', card.dataset.figure === figure.id));
      updateContinue(); setStep(2);
    } catch (err) { if (epoch === sourceEpoch) showError(err.message); }
  };
  const renderIntake = (intake, source, epoch) => {
    const status = $('#pdf-intake-status'); status.classList.remove('hidden');
    if (intake.status === 'ready') {
      stopFigureWatch();
      const figures = intake.figures || [];
      $('#rail-status').textContent = figures.length ? 'CHOOSE FIGURE' : 'NO PICTURES';
      status.textContent = figures.length ? `Choose a picture from pages ${intake.page_start}–${intake.page_end}.` : 'No pictures were found in these pages. Choose another page range to try again.';
      const gallery = $('#pdf-figures'); gallery.replaceChildren(); gallery.classList.remove('hidden');
      for (const figure of figures) {
        const button = document.createElement('button'); button.type = 'button'; button.className = 'figure-card'; button.dataset.figure = figure.id;
        button.innerHTML = `<img loading="lazy" src="${authed(`/api/source/${source.id}/figures/${intake.run_id}/${figure.id}`)}" alt="${esc(figure.title)} preview"><span><strong>${esc(figure.title)}</strong><small>${figure.page ? `Page ${figure.page}` : 'Page not identified'} · ${figure.image_size.join(' × ')} px</small></span><p>${esc(figure.caption || 'Select this picture')}</p>`;
        button.addEventListener('click', () => selectFigure(figure, intake, source, epoch)); gallery.append(button);
      }
      const report = intake.report || {};
      $('#ocr-details').classList.remove('hidden');
      $('#ocr-report').textContent = [`${report.model || 'PaddleOCR'} · ${report.backend || ''}`, `${report.page_count} pages · ${report.markdown_count} Markdown files · ${report.picture_count} pictures`, ...Object.entries(report.timings || {}).map(([k,v]) => `${k}: ${v}`), `Run: ${report.run_path || ''}`, `BibTeX: ${report.bibtex_path || ''}`, ...(report.warnings || [])].join('\n');
      showRange(source); $('#page-start').value = intake.page_start; $('#page-end').value = intake.page_end;
      $('#range-description').textContent = 'Search a different page range if the picture you need is missing.';
      $('#find-figures').textContent = 'Find figures';
    } else if (intake.status === 'error') {
      stopFigureWatch();
      $('#rail-status').textContent = 'OCR ERROR';
      status.textContent = `Figure discovery failed (${intake.stage}): ${intake.error}. Your source and prior runs are preserved.`;
      showRange(source); $('#page-start').value = intake.page_start || 1; $('#page-end').value = intake.page_end || Math.min(10, source.page_count);
      $('#find-figures').textContent = 'Retry figure discovery';
    } else {
      status.textContent = intake.message || 'Finding pictures with PaddleOCR…';
      $('#rail-status').textContent = 'FINDING FIGURES';
    }
  };
  const watchFigures = async (source, epoch) => {
    try {
      const intake = await api(`/api/source/${source.id}/figures`);
      if (epoch !== sourceEpoch || state.upload?.id !== source.id) return;
      renderIntake(intake, source, epoch);
      if (intake.status === 'processing') figureTimer = setTimeout(() => watchFigures(source, epoch), 1500);
    } catch (err) {
      if (epoch !== sourceEpoch) return;
      $('#pdf-intake-status').textContent = `Waiting to reconnect: ${err.message}`;
      figureTimer = setTimeout(() => watchFigures(source, epoch), 3000);
    }
  };
  const startFigureDiscovery = async (range = {}) => {
    const source = state.upload, epoch = sourceEpoch;
    if (!source || source.kind !== 'pdf') return;
    stopFigureWatch(); showError(''); state.targetConfirmed = false; updateContinue();
    $('#pdf-figures').classList.add('hidden'); $('#ocr-details').classList.add('hidden');
    $('#pdf-intake-status').classList.remove('hidden'); $('#pdf-intake-status').textContent = 'Preparing figure discovery…';
    $('#find-figures').disabled = true;
    for (const id of ['page-start','page-end']) $(`#${id}`).disabled = true;
    try {
      const intake = await apiPost(`/api/source/${source.id}/figures`, range);
      if (epoch !== sourceEpoch) return;
      renderIntake(intake, source, epoch);
      if (intake.status === 'processing') figureTimer = setTimeout(() => watchFigures(source, epoch), 1000);
    } catch (err) {
      if (epoch !== sourceEpoch) return;
      showError(err.message); showRange(source); $('#pdf-intake-status').textContent = 'Figure discovery has not started.';
    }
  };
  const setSource = async source => {
    state.upload = source; state.targetConfirmed = false; state.crop = null;
    $('#source-card').classList.remove('hidden'); $('#source-kind').textContent = source.kind.toUpperCase();
    $('#source-name').textContent = source.original_name;
    $('#source-meta').textContent = `${(source.bytes / 1024 / 1024).toFixed(1)} MB · ${source.kind === 'pdf' ? `${source.page_count} pages` : `${source.image_size.join(' × ')} px`}`;
    showError(''); updateContinue();
    if (source.kind === 'image') {
      state.crop = source.target.crop; state.targetConfirmed = true; state.page = null;
      await showPreview(authed(`/api/source/${source.id}/preview`), source.original_name, 'READY');
      $('#rail-note').innerHTML = '<span class="rail-note-icon">i</span><span>Your uploaded image is the extraction target.</span>';
      updateContinue(); setStep(2);
    } else {
      $('#rail-title').textContent = source.original_name; $('#rail-status').textContent = 'PDF';
      $('#rail-note').innerHTML = '<span class="rail-note-icon">i</span><span>Selected PDF pages are sent to PaddleOCR (AI Studio) to identify pictures.</span>';
      if (source.requires_page_range) showRange(source); else await startFigureDiscovery();
    }
  };
  const sourceUpload = async file => {
    if (!file) return;
    clearSource(); setStep(1); showError(''); const epoch = sourceEpoch;
    const body = new FormData(); body.append('file', file);
    $('#source-name').textContent = 'Uploading…'; $('#source-card').classList.remove('hidden');
    try {
      const source = await api('/api/upload', {method: 'POST', body});
      if (epoch === sourceEpoch) await setSource(source);
    } catch (err) { if (epoch === sourceEpoch) { $('#source-card').classList.add('hidden'); showError(err.message); } }
  };
  $('#file-input').addEventListener('change', event => sourceUpload(event.target.files[0]));
  $('#drop-zone').addEventListener('dragover', event => { event.preventDefault(); $('#drop-zone').classList.add('dragging'); });
  $('#drop-zone').addEventListener('dragleave', () => $('#drop-zone').classList.remove('dragging'));
  $('#drop-zone').addEventListener('drop', event => { event.preventDefault(); $('#drop-zone').classList.remove('dragging'); sourceUpload(event.dataTransfer.files[0]); });
  $('#remove-source').addEventListener('click', clearSource);
  $('#find-figures').addEventListener('click', () => {
    const start = Number($('#page-start').value), end = Number($('#page-end').value);
    if (!Number.isInteger(start) || !Number.isInteger(end) || start < 1 || end < start || end > state.upload.page_count) return showError(`Choose a valid page range from 1 to ${state.upload.page_count}.`);
    startFigureDiscovery({page_start: start, page_end: end});
  });
  $('#to-channel').addEventListener('click', () => { if (state.targetConfirmed) setStep(2); });
  $$('.choice-card').forEach(card => card.addEventListener('click', () => {
    state.category = card.dataset.category; $$('.choice-card').forEach(x => x.classList.toggle('selected', x === card));
    $('#spectroscopy-options').classList.toggle('hidden', state.category !== 'spectroscopy');
    $('#plot-description-wrap').classList.toggle('hidden', state.category !== 'plot');
    $('#other-description-wrap').classList.toggle('hidden', state.category !== 'other');
    updateModeFields();
  }));
  $$('.back-button').forEach(button => button.addEventListener('click', () => setStep(Number(button.dataset.back))));
  $$('.next-button').forEach(button => button.addEventListener('click', () => setStep(Number(button.dataset.next))));
  $$('input[name="mode"]').forEach(input => input.addEventListener('change', () => {
    state.mode = input.value; $$('.mode-option').forEach(x => x.classList.toggle('selected', $('input', x).checked)); updateModeFields();
  }));
  const defaultTargets = () => {
    if (state.mode === 'selected' && $('#x-positions').value.trim()) return measurementDefaults.x_positions;
    const choice = measurementDefaults[state.mode][state.category];
    if (typeof choice === 'string') return choice;
    const kind = state.category === 'spectroscopy' ? $('#spectrum-type').value : $('#plot-kind').value;
    return choice[kind] || choice.default;
  };
  const updateTargetHint = () => {
    const fallback = defaultTargets();
    $('#feature-targets').placeholder = fallback;
    $('#feature-default-hint').textContent = `Optional. Leave blank to use: ${fallback}.`;
  };
  $('#plot-kind').addEventListener('change', updateTargetHint);
  $('#spectrum-type').addEventListener('change', () => updateModeFields());
  $('#x-positions').addEventListener('input', updateTargetHint);
  const updateModeFields = () => {
    const selected = state.mode === 'selected';
    const isNMR = state.category === 'spectroscopy' && $('#spectrum-type').value === 'NMR';
    $('#nmr-options').classList.toggle('hidden', !isNMR);
    $('#nmr-nucleus').disabled = !isNMR;
    $('#x-positions').closest('.field-block').classList.toggle('hidden', !selected);
    $('#normalize').closest('.check-row').classList.toggle('hidden', !selected || state.category !== 'spectroscopy');
    $('#normalization-wrap').classList.toggle('hidden', !$('#normalize').checked || !selected);
    $('#feature-targets').closest('.field-block').querySelector('label').textContent = selected ? 'Which features or series should be measured?' : 'Which panels and series should be traced?';
    updateTargetHint();
  };
  $('#normalize').addEventListener('change', updateModeFields);
  const summary = () => {
    const spectrum = $('#spectrum-type').value;
    const isNMR = state.category === 'spectroscopy' && spectrum === 'NMR';
    const channel = state.category === 'spectroscopy' ? `${spectrum}${isNMR && $('#nmr-nucleus').value ? ` · ${$('#nmr-nucleus').value}` : ''}` : state.category === 'plot' ? $('#plot-kind').value : $('#channel-description').value.trim();
    return {channel, mode: state.mode, targets: $('#feature-targets').value.trim() || defaultTargets(), xPositions: $('#x-positions').value.trim(), normalization: $('#normalize').checked, reference: $('#normalization-reference').value.trim()};
  };
  $('#to-review').addEventListener('click', () => {
    showError(''); const info = summary();
    if (state.category === 'other' && !info.channel) return showError('Describe the figure type and what you want measured.');
    if (info.normalization && !info.reference) return showError('Describe which feature should be the normalization reference.');
    const positions = info.xPositions.split(/[,;\n]+/).map(x=>x.trim()).filter(Boolean);
    if (positions.some(x=>!Number.isFinite(Number(x)))) return showError('Enter x positions as finite numbers separated by commas.');
    $('#review-summary').textContent = `${info.channel} · ${info.mode === 'complete' ? 'complete curves, experimental' : 'selected features'}${info.normalization ? ` · normalized to ${info.reference}` : ''}`;
    $('#request-review').innerHTML = `<div class="review-line"><span>SOURCE</span><strong>${esc(state.upload.original_name)}${state.page ? ` · page ${state.page}` : ''}</strong></div><div class="review-line"><span>FIGURE</span><strong>${esc(state.upload.explanation || 'Uploaded image')}</strong></div><div class="review-line"><span>CHANNEL</span><strong>${esc(info.channel)}</strong></div><div class="review-line"><span>EXTRACTION</span><strong>${info.mode === 'complete' ? '<span class="mode-tag amber">EXPERIMENTAL</span> Complete curves' : 'Selected peaks or points'}</strong></div><div class="review-line"><span>MEASURE</span><strong>${esc(info.targets || 'Requested x positions')}${info.xPositions ? ` · x = ${esc(info.xPositions)}` : ''}</strong></div><div class="review-line"><span>REFERENCE</span><strong>${info.normalization ? esc(info.reference) : 'No normalization'}</strong></div>`;
    $('#launch-actions').classList.remove('hidden'); $('#result-view').classList.add('hidden'); $('#job-progress').classList.add('hidden'); $('#clarification').classList.add('hidden'); $('#begin-extraction').disabled = false; setStep(4);
  });
  $('#begin-extraction').addEventListener('click', async () => {
    if (state.result && $('#begin-extraction').textContent.startsWith('New extraction')) {
      state.eventSource?.close(); state.eventSource = null;
      state.upload = null; state.page = null; state.crop = null; state.cropBox = null;
      state.imageDims = null; state.targetConfirmed = false; state.result = null; state.jobId = null;
      state.category = 'spectroscopy'; state.mode = 'selected';
      $('#remove-source').click();
      $('#feature-targets').value = '';
      $('#x-positions').value = ''; $('#channel-description').value = '';
      $('#plot-kind').value = 'Scatter plot'; $('#spectrum-type').value = 'NMR';
      $('#nmr-nucleus').value = ''; $('#normalize').checked = false;
      $('#normalization-reference').value = '';
      $('input[name="mode"][value="selected"]').checked = true;
      $$('.mode-option').forEach(x => x.classList.toggle('selected', $('input', x).checked));
      $$('.choice-card').forEach(x => x.classList.toggle('selected', x.dataset.category === 'spectroscopy'));
      $('#spectroscopy-options').classList.remove('hidden');
      $('#plot-description-wrap').classList.add('hidden'); $('#other-description-wrap').classList.add('hidden');
      $('#result-view').classList.add('hidden'); $('#job-progress').classList.add('hidden');
      $('#clarification').classList.add('hidden'); $('#launch-actions').classList.add('hidden');
      $('#begin-extraction').textContent = 'Begin extraction ↗'; showError(''); updateModeFields(); setStep(1);
      return;
    }
    const info = summary();
    const nucleus = state.category === 'spectroscopy' && $('#spectrum-type').value === 'NMR' && $('#nmr-nucleus').value ? `, ${$('#nmr-nucleus').value} spectrum` : '';
    const description = state.category === 'spectroscopy' ? `${$('#spectrum-type').value}${nucleus}` : state.category === 'plot' ? $('#plot-kind').value : $('#channel-description').value.trim();
    const body = {upload_id: state.upload.id, category: state.category, mode: state.mode,
      description, targets: info.targets, x_positions: info.xPositions, normalization: info.normalization,
      normalization_reference: info.reference};
    try {
      const job = await apiPost('/api/jobs', body); state.jobId = job.id; $('#begin-extraction').disabled = true; $('#launch-actions').classList.add('hidden');
      $('#job-progress').classList.remove('hidden'); $('#progress-title').textContent = 'Preparing your extraction'; $('#progress-status').textContent = 'QUEUED'; $('#progress-bar').style.width = '12%';
      await followJob(job.id);
    } catch (err) { showError(err.message); $('#begin-extraction').disabled = false; }
  });
  const stageText = {queued:'In the extraction queue',prepare:'Preparing confirmed figure',inspect:'Inspecting ticks and feature candidates',measure:'Measuring selected features locally',coordinate_review:'Checking selected pixels against the source',complete_plan:'Planning panels and series',complete_validate:'Validating the curve plan locally',complete_extract:'Extracting complete curves locally',review:'Reviewing output and writing explanation',clarification:'Waiting for your image review'};
  const followJob = async jobId => {
    state.jobWatch?.stop(); state.eventSource?.close(); state.jobId=jobId;
    showError('');
    state.jobWatch=PlotReaderJobWatch({
      getStatus:()=>api(`/api/jobs/${jobId}`),
      openStream:()=>new EventSource(`/api/jobs/${jobId}/events?token=${encodeURIComponent(token)}`),
      onStatus:async job=>{
        if(state.jobId!==jobId)return;
        if(job.latest_action)showLatestAction(job.latest_action.message,job.latest_action.at);
        if(['complete','failed','cancelled','interrupted'].includes(job.status)) return renderTerminal(job);
        if(job.status==='needs_input') {renderClarification(job.plan?.clarifications||[]);return;}
        $('#clarification').classList.add('hidden');$('#launch-actions').classList.add('hidden');
        $('#job-progress').classList.remove('hidden'); updateProgress(job.stage,job.status);
      },
      onProgress:event=>{
        if(state.jobId!==jobId)return;
        if(['stage','luna_started','luna_completed','engine_started','review_note','clarification'].includes(event.type)&&event.message)showLatestAction(event.message,event.at);
        if(event.type==='stage')updateProgress(event.stage,'RUNNING');
        else if(event.type==='review_note')$('#progress-detail').textContent=event.notes?.join(' · ')||event.message;
      },
      onError:()=>{$('#progress-detail').textContent='Connection interrupted. Checking for saved progress…';}
    });
  };
  const updateProgress = (stage, status) => {
    $('#progress-title').textContent = stageText[stage] || 'Processing figure'; $('#progress-status').textContent = String(status || 'RUNNING').toUpperCase();
    const pct = ({queued:8,prepare:20,inspect:32,complete_plan:36,complete_validate:48,complete_extract:61,measure:61,coordinate_review:70,review:88,clarification:45})[stage] || 54;
    $('#progress-bar').style.width = `${pct}%`; $('#progress-detail').textContent = 'The assistant interprets the image; the local engine calculates coordinates and uncertainty.';
  };
  const renderClarification = questions => {
    $('#job-progress').classList.add('hidden'); $('#clarification').classList.remove('hidden'); $('#launch-actions').classList.add('hidden');
    const q = questions[0];
    if (!q) { $('#clarification').innerHTML = '<h3>Review needed</h3><p>There is not enough image evidence to continue automatically. Start a new run with a more specific target description.</p>'; return; }
    const region = q.region ? `<div class="focus-preview"><img src="${authed(`/api/jobs/${state.jobId}/region/${encodeURIComponent(q.id)}`)}" alt="Highlighted ambiguous plot feature"></div>` : '';
    $('#clarification').innerHTML = `<h3>Help LUNA confirm the figure</h3><p>${esc(q.question)}</p>${region}<div class="clarification-options">${(q.choices||[]).map((choice,i)=>`<label class="clarification-choice"><input type="radio" name="answer-choice" value="${esc(choice)}" ${i===0?'checked':''}><span>${esc(choice)}</span></label>`).join('')}</div><label class="answer-label" for="answer-freeform">Or describe what you see</label><input id="answer-freeform" type="text" placeholder="Optional clarification"><div class="clarification-actions"><button id="skip-clarification" class="button secondary" type="button">Skip this item</button><button id="send-clarification" class="button primary" type="button">Continue review →</button></div>`;
    const submitAnswer=async skip=>{
      const jobId=state.jobId;
      const answer=$('#answer-freeform').value.trim()||$('input[name="answer-choice"]:checked')?.value||'';
      $('#send-clarification').disabled=true;$('#skip-clarification').disabled=true;showError('');
      try {
        await apiPost(`/api/jobs/${jobId}/answer`,{question_id:q.id,answer,skip});
        $('#clarification').classList.add('hidden');await followJob(jobId);
      } catch(err) {
        // A repeated click or stale tab may refer to an answered question.
        // Recover to durable status without submitting another inference.
        try {const fresh=await api(`/api/jobs/${jobId}`);
          if(fresh.status!=='needs_input'){await followJob(jobId);return;}
        } catch {}
        showError(err.message);$('#send-clarification').disabled=false;$('#skip-clarification').disabled=false;
      }
    };
    $('#send-clarification').addEventListener('click',()=>submitAnswer(false));
    $('#skip-clarification').addEventListener('click',()=>submitAnswer(true));
  };

  const renderTerminal = async job => {
    showError('');
    $('#job-progress').classList.add('hidden'); $('#clarification').classList.add('hidden'); $('#launch-actions').classList.remove('hidden');
    if (job.status === 'interrupted') {showError('Server restarted. Saved artifacts are retained; use Run history to explicitly restart this extraction.');return;}
    if (job.status === 'failed') { showError(job.error || 'The extraction could not be completed.'); $('#begin-extraction').textContent = 'Start a new run'; $('#begin-extraction').disabled = false; return; }
    if (job.status === 'cancelled') { $('#progress-title').textContent = 'Extraction cancelled'; $('#begin-extraction').textContent = 'Run again'; $('#begin-extraction').disabled = false; return; }
    const result = await api(`/api/jobs/${job.id}/result`); state.result = result; $('#result-view').classList.remove('hidden'); $('#launch-actions').classList.remove('hidden'); $('#begin-extraction').textContent = 'New extraction ↗'; $('#begin-extraction').disabled = false;
    const reviewCount = (result.rows||[]).filter(r=>r.status!=='resolved').length;
    const exp = result.experimental ? '<div class="result-banner experimental">⚠ Complete curve extraction is experimental. Review the overlay and unresolved measurements carefully.</div>' : `<div class="result-banner ${reviewCount?'experimental':''}">${reviewCount ? `⚠ Extraction completed · ${reviewCount} rows need review` : '✓ Measurements completed'} · Check source alignment before using the values.</div>`;
    const rows = result.rows || []; const cols = ['panel','series','feature','status','x','y','height_relative','x_uncertainty','y_uncertainty','x_unit','y_unit','reason','pixel','method','review_id','review_note'];
    const head = ['Panel','Series','Feature','Status','x','y','Relative height','x uncertainty','y uncertainty','x unit','y unit','Note','Source pixel','Method','Review ID','Review note'];
    const table = `<div class="table-wrap"><table><thead><tr>${head.map(c=>`<th>${esc(c)}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${cols.map(k=>`<td class="${['unresolved','needs_review'].includes(row.status)?'unresolved':''}">${k==='status'&&['unresolved','needs_review'].includes(row[k])?'Needs review':k==='height_relative'&&row[k]!=null?fmt(row[k]):(row[k] == null || row[k] === '' ? '—' : esc(k==='x_unit'||k==='y_unit'?row[k]:typeof row[k]==='number'?fmt(row[k]):row[k]))}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
    const imageUrl = authed(`/api/jobs/${job.id}/artifact/overlay.png`);
    const issues = (result.review_issues||[]).length ? `<div class="result-issues">Review notes: ${result.review_issues.map(esc).join(' · ')}</div>` : '';
    $('#result-view').innerHTML = `${exp}<div id="point-review"></div><div class="eyebrow result-eyebrow">MEASUREMENTS · ${rows.length} ROWS</div>${table}<button id="copy-table" class="copy-button" type="button">▣ Copy table</button><div class="result-comment">${esc(result.comment)}</div>${issues}<div class="result-image"><img src="${imageUrl}" alt="Annotated plot measurement overlay"></div><div class="download-row"><a href="${authed(`/api/jobs/${job.id}/artifact/table.csv`)}">↓ CSV</a><a href="${authed(`/api/jobs/${job.id}/artifact/result.json`)}">↓ JSON</a><a href="${imageUrl}" download="plot-reader-overlay.png">↓ Overlay</a></div><details class="provenance"><summary>Measurement details and source pixels</summary><pre>${esc(JSON.stringify(result.measurement,null,2))}</pre></details>${PlotReaderTiming.render(result.timing, authed(`/api/jobs/${job.id}/artifact/timing.json`))}`;
    state.timingCleanup?.();
    state.timingCleanup = PlotReaderTiming.mount($('#result-view .timing-report'), result.timing);
    if (!result.experimental) renderPointReview(job, result);
    $('#copy-table').addEventListener('click', async () => {
      const tsv = [cols.join('\t'), ...rows.map(row => cols.map(k => row[k] == null ? '' : String(k==='pixel'&&Array.isArray(row[k])?JSON.stringify(row[k]):row[k]).replace(/[\t\n]/g,' ')).join('\t'))].join('\n');
      try { await navigator.clipboard.writeText(tsv); toast('Table copied as TSV'); } catch { toast('Clipboard unavailable in this browser'); }
    });
    const overlayImg = $('#result-view .result-image img'); overlayImg.addEventListener('load', () => showPreview(imageUrl, 'Annotated extraction', result.experimental ? 'EXPERIMENTAL' : 'COMPLETE'));
    $('#progress-bar').style.width = '100%';
  };
  const renderPointReview = (job, result) => {
    const root = $('#point-review'); const rows=[...(result.rows||[])]; const originalCount=rows.length; const pending=new Map();
    root.innerHTML=`<div class="point-review-head"><strong>Check points against the source</strong><button type="button" id="rerun-grounded">Re-extract with local marker review</button></div>
      <p class="quiet">Choose a feature, then click its center on the original image. Save corrections to recalculate the table and downloads in a new history entry.</p>
      <div class="point-review-controls"><label>Feature <select id="review-feature">${rows.map((r,i)=>`<option value="${i}">${esc(r.feature)} · ${esc(r.series)}${r.status!=='resolved'?' · needs review':''}</option>`).join('')}</select></label>
      <label>Zoom <select id="review-zoom"><option value="1">Fit</option><option value="2">2×</option><option value="3">3×</option></select></label>
      <label><input type="checkbox" id="review-show-points" checked> Show points</label>
      <button type="button" id="review-unresolved">Mark selected unresolved</button><button type="button" id="review-remove">Remove spurious point</button></div>
      <div class="point-review-controls"><label>New feature <input id="new-point-label" placeholder="Feature label"></label><label>Series <select id="new-point-series">${[...new Set(rows.map(r=>r.series))].map(s=>`<option>${esc(s)}</option>`).join('')}</select></label><button type="button" id="review-add">Add a missing marker</button></div>
      <div class="point-review-scroll"><canvas id="review-canvas" aria-label="Original source with editable measurement centers"></canvas></div>
      <div class="point-review-actions"><span id="review-pending" role="status">No pending corrections</span><button type="button" id="discard-point-edits" disabled>Discard edits</button><button type="button" id="save-point-edits" disabled>Save corrected table</button></div>`;
    const canvas=$('#review-canvas'); const ctx=canvas.getContext('2d'); const source=new Image();
    const current=()=>Number($('#review-feature').value);
    const pixelAt=i=>pending.has(i)?(pending.get(i)==='omit'?null:pending.get(i)):rows[i].pixel;
    const draw=()=>{
      if (!source.complete||!source.naturalWidth) return;
      ctx.clearRect(0,0,canvas.width,canvas.height); ctx.drawImage(source,0,0);
      if ($('#review-show-points').checked) rows.forEach((row,i)=>{
        const p=pixelAt(i);if(!p)return;
        ctx.strokeStyle=i===current()?'#007bba':pending.has(i)?'#147a3d':row.status==='resolved'?'#d13a30':'#b46b00';
        ctx.lineWidth=i===current()?2:1;
        ctx.beginPath();ctx.arc(p[0],p[1],i===current()?7:4,0,Math.PI*2);ctx.stroke();
      });
      const n=pending.size;$('#review-pending').textContent=n?`${n} pending correction${n===1?'':'s'} · save to update values`:'No pending corrections';
      $('#save-point-edits').disabled=!n;$('#discard-point-edits').disabled=!n;
    };
    source.onload=()=>{canvas.width=source.naturalWidth;canvas.height=source.naturalHeight;draw();};
    source.src=authed(`/api/jobs/${job.id}/artifact/source-image.png`);
    canvas.addEventListener('click',event=>{
      const bounds=canvas.getBoundingClientRect();
      const x=(event.clientX-bounds.left)*canvas.width/bounds.width, y=(event.clientY-bounds.top)*canvas.height/bounds.height;
      if (x>=0&&x<canvas.width&&y>=0&&y<canvas.height&&rows.length) {pending.set(current(),[Number(x.toFixed(2)),Number(y.toFixed(2))]);draw();}
    });
    $('#review-feature').onchange=draw;$('#review-show-points').onchange=draw;
    $('#review-zoom').onchange=()=>{canvas.style.width=`${Number($('#review-zoom').value)*100}%`;};
    $('#review-unresolved').onclick=()=>{if(rows.length){pending.set(current(),null);draw();}};
    $('#review-remove').onclick=()=>{if(rows.length){pending.set(current(),'omit');draw();}};
    $('#review-add').onclick=()=>{
      const label=$('#new-point-label').value.trim();if(!label){toast('Give the new feature a label first');return;}
      const index=rows.length;rows.push({feature:label,series:$('#new-point-series').value,pixel:null,status:'unresolved'});
      const option=document.createElement('option');option.value=index;option.textContent=`${label} · new marker`;
      $('#review-feature').append(option);$('#review-feature').value=index;$('#new-point-label').value='';draw();toast('Click the new marker center on the source');
    };
    $('#discard-point-edits').onclick=()=>{pending.clear();rows.splice(originalCount);while($('#review-feature').options.length>originalCount)$('#review-feature').remove(originalCount);if(current()>=originalCount)$('#review-feature').value=0;draw();};
    $('#save-point-edits').onclick=async()=>{
      $('#save-point-edits').disabled=true;
      try{const next=await apiPost(`/api/jobs/${job.id}/corrections`,{edits:[...pending].filter(([index])=>index<originalCount).map(([index,pixel])=>({index,pixel:pixel==='omit'?null:pixel,omit:pixel==='omit'})),additions:[...pending].filter(([index,pixel])=>index>=originalCount&&Array.isArray(pixel)).map(([index,pixel])=>({label:rows[index].feature,series:rows[index].series,pixel}))});
        state.jobId=next.id; await renderTerminal(await api(`/api/jobs/${next.id}`));toast('Corrected table saved; original run retained');
      }catch(err){showError(err.message);$('#save-point-edits').disabled=false;}
    };
    $('#rerun-grounded').onclick=async()=>{
      $('#rerun-grounded').disabled=true;
      try{const next=await apiPost(`/api/jobs/${job.id}/restart`,{});state.jobId=next.id;$('#result-view').classList.add('hidden');followJob(next.id);}
      catch(err){showError(err.message);$('#rerun-grounded').disabled=false;}
    };
  };
  const fmt = value => Number(value).toPrecision(6).replace(/(?:\.0+|(?:(\.\d*?)0+))$/, '$1');
  $('#cancel-job').addEventListener('click', async () => { if (!state.jobId) return; try { await apiPost(`/api/jobs/${state.jobId}/cancel`, {}); $('#progress-title').textContent = 'Cancelling extraction'; } catch (err) { showError(err.message); } });
  $('#history-toggle').addEventListener('click', async () => { $('#history-drawer').classList.remove('hidden'); await renderHistory(); });
  $('#close-history').addEventListener('click', () => $('#history-drawer').classList.add('hidden'));
  const renderHistory = async () => {
    try {
      const jobs = await api('/api/jobs'); const list = $('#history-list');
      list.innerHTML = jobs.length ? jobs.map(job=>`<div class="history-item" data-id="${esc(job.id)}"><span class="status-pill">${esc(job.status.toUpperCase())}</span><strong>${esc(job.display_name|| (job.engine==='experimental-round9_full'?'Complete curves · Experimental':'Selected measurements'))}</strong><small>${esc(job.created_at||'')}${job.stage?` · ${esc(job.stage)}`:''}</small></div>`).join('') : '<p class="quiet">No saved runs yet.</p>';
      $$('.history-item',list).forEach(item=>item.addEventListener('click',async()=>{
        const id=item.dataset.id; state.jobWatch?.stop(); $('#history-drawer').classList.add('hidden'); state.jobId=id; setStep(4);
        try { const job=await api(`/api/jobs/${id}`); if(job.status==='needs_input') renderClarification(job.plan?.clarifications||[]); else if(job.status==='complete') await renderTerminal(job); else if(['interrupted','failed','cancelled'].includes(job.status)) {$('#job-progress').classList.remove('hidden');$('#progress-title').textContent='Run stopped';$('#progress-detail').textContent='Saved files are retained. Start a new explicit run to retry inference.';$('#launch-actions').classList.add('hidden');const restart=document.createElement('button');restart.className='button primary';restart.textContent='Start a new run →';restart.onclick=async()=>{try{const next=await apiPost(`/api/jobs/${id}/restart`,{});state.jobId=next.id;followJob(next.id);}catch(err){showError(err.message);}};$('#job-progress').append(restart);} else if(['queued','running','cancel_requested'].includes(job.status)) { followJob(id); } else {$('#progress-detail').textContent=job.error||job.status;$('#job-progress').classList.remove('hidden');} }
        catch(err){showError(err.message);}
      }));
    } catch (err) { showError(err.message); }
  };
  $$('.step').forEach(button => button.addEventListener('click', () => { const target=Number(button.dataset.step); if(target<=state.step) setStep(target); }));
  updateModeFields(); renderHistory().catch(()=>{});
})();
