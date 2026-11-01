const test = require('node:test');
const assert = require('node:assert/strict');
const {render} = require('../plot_reader/browser_ui/timing-view.js');

test('timing table renders summary, per-stage times and every Python call', () => {
  const html = render({summary:{total_seconds:9,model_wait_seconds:6,local_python_seconds:2,user_wait_seconds:0,host_overhead_seconds:1},
    stages:[{name:'inspect',elapsed_seconds:8,model_wait_seconds:6,local_python_seconds:2}],
    model_calls:[{name:'selected-plan',stage:'inspect',elapsed_seconds:6,status:'completed'}],
    local_python_calls:[{name:'inspect_plot',stage:'inspect',elapsed_seconds:1.5,status:'completed'},
      {name:'scan_scatter_templates',stage:'coordinate_review',elapsed_seconds:.5,status:'failed'}]},'/timing.json');
  for (const value of ['Time cost analysis','9.000 s','6.000 s','inspect_plot','1.500 s','scan_scatter_templates','0.500 s','failed','selected-plan','Inspect and plan features','href="/timing.json"']) assert.ok(html.includes(value), value);
  assert.ok(html.includes('<details open>'));
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
