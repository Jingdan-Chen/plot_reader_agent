(function (root) {
  'use strict';
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const finite = value => typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : 0;
  const seconds = value => typeof value === 'number' && Number.isFinite(value) ? `${value.toFixed(3)} s` : '—';
  const percent = value => `${(100 * value).toFixed(2)}%`;
  const names = {starting:'Starting analysis',prepare:'Prepare figure',inspect:'Inspect and plan features',
    measure:'Measure selected features',coordinate_review:'Review source coordinates',
    complete_plan:'Plan complete curves',complete_validate:'Validate curve plan',complete_extract:'Extract complete curves',
    review:'Review results and explain',clarification:'Process clarification',user_wait:'Wait for your clarification',
    correct_points:'Recalculate corrected points'};
  const modes = {summary:'Overall time',stages:'Workflow stages',model:'Model calls',python:'Python calls'};
  const stageName = name => names[name] || name || 'Unspecified';
  const color = index => `hsl(${(89 + index * 137.508) % 360} 62% 65%)`;
  const table = (headers, rows) => `<div class="table-wrap"><table><thead><tr>${headers.map(h=>`<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map(cell=>`<td>${esc(cell)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  const callTable = calls => table(['Call','Workflow stage','Wall time','Status'], calls.map((call,index)=>[
    `${index+1}. ${call.name}`,stageName(call.stage),seconds(call.elapsed_seconds),call.status]));

  function breakdown(timing, mode = 'summary') {
    const s = timing.summary || {}, model = timing.model_calls || [], local = timing.local_python_calls || [];
    let items, note, centerLabel;
    if (mode === 'stages') {
      centerLabel = 'STAGE WALL TIME';
      note = 'Stages include the model waits and Python work within them. Repeated visits to a stage are combined.';
      items = (timing.stages || []).map(stage=>({label:stageName(stage.name),value:finite(stage.elapsed_seconds),
        details:[['Model wait',seconds(stage.model_wait_seconds)],['Local Python',seconds(stage.local_python_seconds)],
          ['Other / clarification',seconds(Math.max(0,finite(stage.elapsed_seconds)-finite(stage.model_wait_seconds)-finite(stage.local_python_seconds)))],
          ['Model calls',model.filter(c=>c.stage===stage.name).length],['Python calls',local.filter(c=>c.stage===stage.name).length]]}));
    } else if (mode === 'model' || mode === 'python') {
      const calls = mode === 'model' ? model : local;
      centerLabel = 'SUM OF CALLS';
      note = `Each slice is one ${mode === 'model' ? 'model response' : 'local Python'} call. Percentages use the sum of call wall times; overlapping calls can make this larger than elapsed time.`;
      items = calls.map((call,index)=>({label:`${index+1}. ${call.name}`,value:finite(call.elapsed_seconds),
        details:[['Workflow stage',stageName(call.stage)],['Status',call.status || 'Unknown'],
          ...(call.provider ? [['Provider',call.provider]] : []),...(call.model ? [['Model',call.model]] : []),
          ['Started after',seconds(call.start)],['Finished after',seconds(call.end)]],
        description:mode === 'model' ? 'CLI round trip, including startup, transfer, provider queueing, inference and response validation.' : 'Wall time in this Python worker or subprocess, including any child tools such as Tesseract.'}));
    } else {
      centerLabel = 'TOTAL ELAPSED';
      note = 'These portions partition total elapsed time. Hover a slice or legend entry for details; tap or use keyboard focus as well.';
      items = [
        {label:'Waiting for model responses',value:finite(s.model_wait_seconds),color:'#b8ee59',
          details:[['Model calls',model.length],['Failed / cancelled calls',model.filter(c=>c.status!=='completed').length]],
          description:'Time waiting for validated model responses, including CLI startup, image transfer, provider queueing and inference.'},
        {label:'Local Python work',value:finite(s.local_python_seconds),color:'#67d6df',details:[['Timed Python calls',local.length]],
          description:'Combined wall time of local tool calls and curve-engine subprocesses. Overlapping work is counted once.'},
        {label:'Waiting for your clarification',value:finite(s.user_wait_seconds),color:'#efbd71',details:[],
          description:'Time paused for your clarification, including the interval before the resume task starts.'},
        {label:'Other host work and scheduling',value:finite(s.host_overhead_seconds),color:'#9eacc7',details:[],
          description:'Remaining host work, including thread scheduling, event handling, small operations and saving intermediate artifacts.'}
      ];
    }
    const total = items.reduce((sum,item)=>sum+item.value,0);
    return {mode,items:items.map((item,index)=>({...item,color:item.color||color(index),share:total ? item.value/total : 0})),
      total,centerLabel,displayTotal:mode==='summary' ? finite(s.total_seconds) : total,note};
  }

  function slicePath(start, sweep) {
    const point = (radius,angle) => {
      const radians = (angle-90)*Math.PI/180;
      return `${(120+radius*Math.cos(radians)).toFixed(5)},${(120+radius*Math.sin(radians)).toFixed(5)}`;
    };
    if (sweep >= 359.999999) return 'M120,19 A101,101 0 1,1 120,221 A101,101 0 1,1 120,19 Z M120,56 A64,64 0 1,0 120,184 A64,64 0 1,0 120,56 Z';
    const large = sweep > 180 ? 1 : 0;
    return `M${point(101,start)} A101,101 0 ${large},1 ${point(101,start+sweep)} L${point(64,start+sweep)} A64,64 0 ${large},0 ${point(64,start)} Z`;
  }

  function chartMarkup(chart) {
    let angle = 0;
    const slices = chart.items.map((item,index)=>{
      if (!item.value || !chart.total) return '';
      const sweep = 360*item.share, start = angle;
      angle += sweep;
      return `<path class="timing-slice" d="${slicePath(start,sweep)}" fill="${item.color}" fill-rule="evenodd" role="button" tabindex="0" aria-label="${esc(`${item.label}: ${seconds(item.value)}, ${percent(item.share)} of this chart`)}" aria-describedby="timing-tooltip" data-timing-index="${index}"></path>`;
    }).join('');
    const legend = chart.items.map((item,index)=>`<button type="button" class="timing-legend-item" data-timing-index="${index}" aria-describedby="timing-tooltip"><i style="background:${item.color}" aria-hidden="true"></i><span>${esc(item.label)}<small>${seconds(item.value)} <b>· ${percent(item.share)}</b></small></span></button>`).join('');
    return `<div class="timing-chart-layout"><svg class="timing-donut" viewBox="0 0 240 240" role="group" aria-label="${esc(modes[chart.mode])} interactive pie chart"><circle cx="120" cy="120" r="82.5" fill="none" stroke="#29323b" stroke-width="37"></circle>${slices}<text class="timing-center-label" x="120" y="111" text-anchor="middle">${chart.centerLabel}</text><text class="timing-center-value" x="120" y="136" text-anchor="middle">${seconds(chart.displayTotal)}</text></svg><div class="timing-legend" aria-label="${esc(modes[chart.mode])} legend">${legend || '<p>No calls recorded in this view.</p>'}</div></div><p class="timing-chart-note">${esc(chart.note)}</p>`;
  }

  function tooltipMarkup(item) {
    return `<strong>${esc(item.label)}</strong><div class="timing-tooltip-value">${seconds(item.value)}<span>${percent(item.share)} of this chart</span></div>${item.details.length ? `<dl>${item.details.map(([label,value])=>`<div><dt>${esc(label)}</dt><dd>${esc(value)}</dd></div>`).join('')}</dl>` : ''}${item.description ? `<p>${esc(item.description)}</p>` : ''}`;
  }

  function render(timing, downloadUrl) {
    if (!timing || !timing.summary) return '<section class="timing-report"><h3>Time cost analysis</h3><p>Timing was not recorded for this saved run. Start a new extraction to measure its time cost.</p></section>';
    const s = timing.summary, model = timing.model_calls || [], local = timing.local_python_calls || [];
    const summary = table(['Time cost','Wall time'],[
      ['Total elapsed',seconds(s.total_seconds)],['Waiting for model responses',seconds(s.model_wait_seconds)],
      ['Local Python work',seconds(s.local_python_seconds)],['Waiting for your clarification',seconds(s.user_wait_seconds)],
      ['Other host work and scheduling',seconds(s.host_overhead_seconds)]
    ]);
    const stages = table(['Workflow stage','Wall time','Model wait','Local Python'],(timing.stages||[]).map(stage=>[
      stageName(stage.name),seconds(stage.elapsed_seconds),seconds(stage.model_wait_seconds),seconds(stage.local_python_seconds)]));
    const tabs = Object.entries(modes).map(([mode,label],index)=>`<button type="button" role="tab" id="timing-tab-${mode}" aria-controls="timing-pane" aria-selected="${index===0}" tabindex="${index===0 ? 0 : -1}" data-timing-mode="${mode}">${label}</button>`).join('');
    const download = downloadUrl ? `<a href="${esc(downloadUrl)}">↓ Timing JSON</a>` : '';
    return `<section class="timing-report" aria-label="Time cost analysis"><div class="eyebrow">HOST-MEASURED WALL TIME</div><h3>Time cost analysis</h3><div class="timing-tabs" role="tablist" aria-label="Timing breakdown">${tabs}</div><div id="timing-pane" role="tabpanel" aria-labelledby="timing-tab-summary" data-timing-chart>${chartMarkup(breakdown(timing))}</div><div id="timing-tooltip" class="timing-tooltip" role="tooltip" hidden></div><details class="timing-exact"><summary>Exact timing values</summary>${summary}<h4>Workflow stages</h4>${stages}<h4>Model responses · ${model.length} calls</h4>${model.length ? callTable(model) : '<p>No model calls.</p>'}<h4>Local Python · ${local.length} calls</h4>${local.length ? callTable(local) : '<p>No local Python calls recorded.</p>'}</details><p>Total includes clarification waits. Upload, PDF figure discovery, browser rendering and downloads are outside this analysis. Stage and call views describe the same work as the overall breakdown.</p>${download}</section>`;
  }

  function mount(container, timing) {
    if (!container || !timing || !timing.summary) return;
    let chart = breakdown(timing), active = null, pinned = null;
    const tooltip = container.querySelector('.timing-tooltip');
    const pane = container.querySelector('[data-timing-chart]');
    const tabButtons = Array.from(container.querySelectorAll('[data-timing-mode]'));
    const targets = () => Array.from(pane.querySelectorAll('[data-timing-index]'));
    const targetFor = target => target && target.closest ? target.closest('[data-timing-index]') : null;
    const hide = () => {
      tooltip.hidden = true; active = null;
      targets().forEach(target=>target.classList.remove('is-active'));
    };
    const position = (event,target) => {
      const rect = tooltip.getBoundingClientRect(), anchor = target.getBoundingClientRect();
      const width = container.ownerDocument.documentElement.clientWidth;
      const height = container.ownerDocument.documentElement.clientHeight;
      const x = event && Number.isFinite(event.clientX) ? event.clientX : anchor.left+anchor.width/2;
      const y = event && Number.isFinite(event.clientY) ? event.clientY : anchor.top+anchor.height/2;
      let left = x+16, top = y+16;
      if (left+rect.width>width-12) left = x-rect.width-16;
      if (top+rect.height>height-12) top = y-rect.height-16;
      tooltip.style.left = `${Math.max(12,Math.min(left,width-rect.width-12))}px`;
      tooltip.style.top = `${Math.max(12,Math.min(top,height-rect.height-12))}px`;
    };
    const show = (target,event) => {
      const index = Number(target.dataset.timingIndex), item = chart.items[index];
      if (!item) return;
      active = index;
      targets().forEach(element=>element.classList.toggle('is-active',Number(element.dataset.timingIndex)===index));
      tooltip.innerHTML = tooltipMarkup(item); tooltip.hidden = false;
      position(event,target);
    };
    const switchMode = mode => {
      if (!(mode in modes)) return;
      pinned = null; hide(); chart = breakdown(timing,mode);
      tabButtons.forEach(button=>{
        const selected = button.dataset.timingMode===mode;
        button.setAttribute('aria-selected',String(selected)); button.tabIndex = selected ? 0 : -1;
      });
      pane.setAttribute('aria-labelledby',`timing-tab-${mode}`);
      pane.innerHTML = chartMarkup(chart);
    };
    container.addEventListener('pointerover',event=>{
      const target = targetFor(event.target);
      if (target && pinned===null) show(target,event);
    });
    container.addEventListener('pointermove',event=>{
      const target = targetFor(event.target);
      if (target && active!==null && pinned===null) position(event,target);
    });
    container.addEventListener('pointerout',event=>{
      const target = targetFor(event.target), next = targetFor(event.relatedTarget);
      if (target && target!==next && pinned===null) hide();
    });
    container.addEventListener('focusin',event=>{
      const target = targetFor(event.target);
      if (target) show(target);
    });
    container.addEventListener('focusout',event=>{
      if (targetFor(event.target) && pinned===null) hide();
    });
    container.addEventListener('click',event=>{
      const mode = event.target.closest('[data-timing-mode]');
      if (mode) { switchMode(mode.dataset.timingMode); return; }
      const target = targetFor(event.target);
      if (!target) { pinned=null; hide(); return; }
      const index = Number(target.dataset.timingIndex);
      if (pinned===index) { pinned=null; hide(); }
      else { pinned=index; show(target,event.detail ? event : null); }
    });
    container.addEventListener('keydown',event=>{
      if (event.key==='Escape') { pinned=null; hide(); return; }
      const tab = event.target.closest('[data-timing-mode]');
      if (tab && ['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) {
        event.preventDefault();
        const index = tabButtons.indexOf(tab), count = tabButtons.length;
        const next = event.key==='Home' ? 0 : event.key==='End' ? count-1 : (index+(event.key==='ArrowRight' ? 1 : -1)+count)%count;
        switchMode(tabButtons[next].dataset.timingMode); tabButtons[next].focus();
      } else if (event.target.matches('.timing-slice') && ['Enter',' '].includes(event.key)) {
        event.preventDefault();
        const target = event.target, index = Number(target.dataset.timingIndex);
        if (pinned===index) { pinned=null; hide(); }
        else { pinned=index; show(target); }
      }
    });
    // A floating detail box follows the hovered slice and closes when scrolled.
    // Abort listeners when results are replaced, so old reports retain no handlers.
    const view = container.ownerDocument.defaultView;
    const events = new view.AbortController();
    const onViewportChange = () => {
      if (!container.isConnected) events.abort();
      pinned=null; hide();
    };
    view.addEventListener('scroll',onViewportChange,{capture:true,passive:true,signal:events.signal});
    view.addEventListener('resize',onViewportChange,{passive:true,signal:events.signal});
    return () => { events.abort(); pinned=null; hide(); };
  }
  const api = {render,mount,breakdown,slicePath,tooltipMarkup};
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.PlotReaderTiming = api;
})(globalThis);
