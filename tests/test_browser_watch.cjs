const test=require('node:test');
const assert=require('node:assert/strict');
const watchJob=require('../plot_reader/browser_ui/job-watch.js');
const flush=()=>new Promise(resolve=>setImmediate(resolve));
function harness(initial={status:'running',stage:'review'}) {
  const h={job:initial,statuses:[],timers:new Map(),streams:[],next:0,errors:[]};
  h.watch=watchJob({getStatus:async()=>h.job,onStatus:async job=>h.statuses.push(job),
    openStream:()=>{const s={closed:false,close(){this.closed=true;}};h.streams.push(s);return s;},
    schedule:fn=>{const id=++h.next;h.timers.set(id,fn);return id;},unschedule:id=>h.timers.delete(id),
    onError:err=>h.errors.push(err)});
  h.tick=async()=>{const [id,fn]=h.timers.entries().next().value;h.timers.delete(id);await fn();await flush();};
  return h;
}
test('replayed answered question cannot strand a resumed job',async()=>{
  const h=harness();await flush();
  h.streams[0].onmessage({data:JSON.stringify({type:'clarification',questions:[{id:'already-answered'}]})});await flush();
  assert(h.statuses.every(job=>job.status==='running'));
  assert.equal(h.streams[0].closed,false);
  h.job={status:'complete'};await h.tick();
  assert.equal(h.statuses.at(-1).status,'complete');assert.equal(h.timers.size,0);assert(h.streams[0].closed);
});
test('disconnect and missed completion recover through status polling',async()=>{
  const h=harness();await flush();h.streams[0].onerror();h.job={status:'complete'};await h.tick();
  assert.equal(h.statuses.at(-1).status,'complete');assert.equal(h.streams.length,1);
});
test('snapshot uses current status, including a new clarification',async()=>{
  const h=harness();await flush();h.job={status:'needs_input',plan:{clarifications:[{id:'new-question'}]}};
  h.streams[0].onmessage({data:JSON.stringify({type:'snapshot',status:'running'})});await flush();
  assert.equal(h.statuses.at(-1).plan.clarifications[0].id,'new-question');assert.equal(h.timers.size,0);
});
test('already completed jobs do not open an event stream',async()=>{
  const h=harness({status:'complete'});await flush();assert.equal(h.streams.length,0);assert.equal(h.timers.size,0);
});
test('stopping an old watcher ignores its in-flight response',async()=>{
  let respond;let rendered=false;
  const watcher=watchJob({getStatus:()=>new Promise(resolve=>respond=resolve),
    openStream:()=>{throw Error('must not open');},onStatus:()=>rendered=true});
  watcher.stop();respond({status:'complete'});await flush();assert.equal(rendered,false);
});
