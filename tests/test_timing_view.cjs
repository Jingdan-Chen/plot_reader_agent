const test = require('node:test');
const assert = require('node:assert/strict');
const {render,breakdown,slicePath,tooltipMarkup} = require('../plot_reader/browser_ui/timing-view.js');

test('interactive pie includes all breakdown views and exact call values', () => {
  const html = render({summary:{total_seconds:9,model_wait_seconds:6,local_python_seconds:2,user_wait_seconds:0,host_overhead_seconds:1},
    stages:[{name:'inspect',elapsed_seconds:8,model_wait_seconds:6,local_python_seconds:2}],
    model_calls:[{name:'selected-plan',stage:'inspect',elapsed_seconds:6,status:'completed'}],
    local_python_calls:[{name:'inspect_plot',stage:'inspect',elapsed_seconds:1.5,status:'completed'},
      {name:'scan_scatter_templates',stage:'coordinate_review',elapsed_seconds:.5,status:'failed'}]},'/timing.json');
  for (const value of ['Time cost analysis','9.000 s','6.000 s','inspect_plot','1.500 s','scan_scatter_templates','0.500 s','failed','selected-plan','Inspect and plan features','href="/timing.json"']) assert.ok(html.includes(value), value);
  for (const value of ['interactive pie chart','data-timing-mode="summary"','data-timing-mode="stages"',
    'data-timing-mode="model"','data-timing-mode="python"','role="tooltip"','Exact timing values']) assert.ok(html.includes(value),value);
  assert.equal((html.match(/class="timing-slice"/g)||[]).length,3);
  assert.ok(!html.includes('<details open>'));
});

test('older saved results do not receive invented timing estimates', () => {
  const html = render(null,'/not-available');
  assert.match(html,/Timing was not recorded/);
  assert.ok(!html.includes('href='));
});

test('timing names and download URL are escaped', () => {
  const html = render({summary:{},stages:[],model_calls:[],local_python_calls:[
    {name:'<img src=x onerror=alert(1)>',stage:'<script>',elapsed_seconds:1,status:'failed'}]},'" onclick="x');
  assert.ok(!html.includes('<img'));
  assert.ok(!html.includes('<script>'));
  assert.ok(html.includes('&quot; onclick=&quot;x'));
});

test('call pies use summed call times while overall pie uses deduplicated wall time', () => {
  const timing={summary:{total_seconds:4,local_python_seconds:4},local_python_calls:[
    {name:'outer',elapsed_seconds:4},{name:'inner',elapsed_seconds:2}]};
  const overall=breakdown(timing),calls=breakdown(timing,'python');
  assert.equal(overall.total,4);
  assert.equal(overall.displayTotal,4);
  assert.equal(calls.total,6);
  assert.equal(calls.centerLabel,'SUM OF CALLS');
  assert.equal(calls.items[0].share,4/6);
  assert.equal(calls.items[1].share,2/6);
  assert.match(calls.note,/overlapping calls/);
});

test('stage hover details retain model and Python portions and call counts', () => {
  const timing={summary:{total_seconds:10},stages:[{name:'inspect',elapsed_seconds:10,
    model_wait_seconds:6,local_python_seconds:3}],model_calls:[{stage:'inspect'}],local_python_calls:[{stage:'inspect'},{stage:'inspect'}]};
  const item=breakdown(timing,'stages').items[0];
  const html=tooltipMarkup(item);
  for(const value of ['Inspect and plan features','10.000 s','100.00%','Model wait','6.000 s','Local Python','3.000 s','1.000 s','Model calls','Python calls']) assert.ok(html.includes(value),value);
  assert.deepEqual(item.details.slice(-2),[['Model calls',1],['Python calls',2]]);
});

test('call hover details include status, provider, model and timing offsets safely', () => {
  const timing={summary:{total_seconds:4},model_calls:[{name:'<script>',stage:'review',elapsed_seconds:2,
    status:'failed',provider:'claude',model:'haiku',start:1,end:3}]};
  const html=tooltipMarkup(breakdown(timing,'model').items[0]);
  for(const value of ['&lt;script&gt;','failed','claude','haiku','Review results and explain','1.000 s','3.000 s']) assert.ok(html.includes(value),value);
  assert.ok(!html.includes('<script>'));
});

test('one full slice, zero values and tiny calls retain correct proportions', () => {
  const single=slicePath(0,360);
  assert.equal((single.match(/A101,101/g)||[]).length,2);
  assert.equal((single.match(/A64,64/g)||[]).length,2);
  assert.ok(!slicePath(0,.0001).includes('NaN'));
  const tiny=breakdown({summary:{},local_python_calls:[{name:'large',elapsed_seconds:1000},
    {name:'tiny',elapsed_seconds:.00001},{name:'zero',elapsed_seconds:0}]},'python');
  assert.equal(tiny.items.length,3);
  assert.ok(tiny.items[1].share>0 && tiny.items[1].share<.000001);
  const empty=render({summary:{total_seconds:0}});
  assert.ok(!empty.includes('class="timing-slice"'));
  assert.ok(!empty.includes('NaN'));
});
