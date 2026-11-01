/* Status is authoritative. Streamed events may include old log entries. */
(function (root) {
  function watchJob({getStatus, openStream, onStatus, onProgress=()=>{}, onError=()=>{},
                     schedule=setTimeout, unschedule=clearTimeout, interval=2000}) {
    let stopped=false, busy=false, refreshAgain=false, timer=null, stream=null;
    const settled=new Set(['complete','failed','cancelled','interrupted','needs_input']);
    function stop() {
      stopped=true;
      if(timer!==null) unschedule(timer);
      timer=null;stream?.close();stream=null;
    }
    async function refresh() {
      if(stopped) return;
      if(busy){refreshAgain=true;return;}
      busy=true;
      if(timer!==null) unschedule(timer);
      timer=null;
      try {
        const job=await getStatus();
        if(stopped)return;
        await onStatus(job);
        if(stopped)return;
        if(settled.has(job.status)){stop();return;}
        if(!stream) {
          stream=openStream();
          stream.onmessage=message=>{
            if(stopped)return;
            let event;try{event=JSON.parse(message.data);}catch{return;}
            // Never display a question or completed result directly from a
            // replayed event. Fetch the job's current durable state instead.
            if(['snapshot','clarification','complete','failed','cancelled'].includes(event.type)) refresh();
            else onProgress(event);
          };
          stream.onerror=()=>{
            if(stopped)return;
            stream?.close();stream=null;
            // Polling continues even when SSE is disconnected or suspended.
          };
        }
      } catch(error) {if(!stopped)onError(error);}
      finally {
        busy=false;
        if(!stopped){const delay=refreshAgain?0:interval;refreshAgain=false;timer=schedule(refresh,delay);}
      }
    }
    refresh();
    return {stop,refresh};
  }
  if(typeof module==='object'&&module.exports) module.exports=watchJob;
  else root.PlotReaderJobWatch=watchJob;
})(globalThis);
