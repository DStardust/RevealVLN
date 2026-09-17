async function runClientTests(createPoller) {
  const passed = [];
  const check = (ok, reason) => { if (!ok) throw new Error(reason); };
  const data = n => ({completed: n, lanes: [{gpu:3},{gpu:4},{gpu:5}], monitor:{has_snapshot:true,stale:false}});
  const response = (d, status=200, type="application/json") => ({ok:status===200,status:status,headers:{get:()=>type},json:async()=>d});
  function rig(fetch, initial) {
    let next=1; const tasks=new Map(), renders=[], connections=[], controllers=[];
    class Controller { constructor(){this.signal={aborted:false};controllers.push(this);} abort(){this.signal.aborted=true;} }
    const p=createPoller({url:"api/status",fetch:fetch,initial:initial,
      setTimer:(fn,ms)=>{const id=next++;tasks.set(id,{fn,ms});return id;},
      clearTimer:id=>tasks.delete(id),AbortController:Controller,
      render:d=>renders.push(d),connection:v=>connections.push(v)});
    return {p,tasks,renders,connections,controllers};
  }
  {
    const r=rig(async()=>response(data(10))); await r.p.refresh();
    check(r.renders[0].completed===10 && r.connections[0].connected,"success render");
    check([...r.tasks.values()].some(t=>t.ms===10000),"success cadence");r.p.stop();passed.push("success_and_cadence");
  }
  {
    let calls=0; const r=rig(async()=>{calls++;if(calls<3)throw new TypeError("Failed to fetch");return response(data(20));},data(9));
    await r.p.refresh();check(r.p.state().last.completed===9 && r.renders.length===0,"failed fetch retains snapshot");
    check([...r.tasks.values()].some(t=>t.ms===2000),"first backoff");
    await r.p.refresh();check([...r.tasks.values()].some(t=>t.ms===4000),"second backoff");
    await r.p.refresh();check(r.p.state().last.completed===20 && r.p.state().failures===0,"recovery clears failures");
    r.p.stop();passed.push("network_failure_retention_backoff_recovery");
  }
  {
    let resolve, calls=0; const waiting=new Promise(r=>{resolve=r;});
    const r=rig(()=>{calls++;return waiting;});const first=r.p.refresh();await r.p.refresh();
    check(calls===1 && r.p.state().running,"no overlapping requests");
    resolve(response(data(30)));await first;r.p.stop();passed.push("single_inflight");
  }
  {
    const r=rig(()=>new Promise(()=>{}),data(9));const waiting=r.p.refresh();
    [...r.tasks.values()].find(t=>t.ms===8000).fn();await waiting;
    check(r.controllers[0].signal.aborted && !r.p.state().running,"timeout abort");
    check(!r.connections[0].connected && r.connections[0].error==="请求超时","timeout displayed");
    check(r.p.state().last.completed===9,"timeout retains last snapshot");r.p.stop();passed.push("timeout_abort_and_retry");
  }
  for (const [name,res] of [["http_error",response(data(1),503)],["proxy_html",response(data(1),200,"text/html")],["schema_error",response({})]]) {
    const r=rig(async()=>res,data(9));await r.p.refresh();
    check(r.renders.length===0 && !r.connections[0].connected && r.p.state().last.completed===9,name);
    r.p.stop();passed.push(name);
  }
  {
    let resolve;const r=rig(()=>new Promise(done=>{resolve=done;}));const pending=r.p.refresh();r.p.stop();resolve(response(data(1)));await pending;
    check(r.renders.length===0 && r.tasks.size===0,"no updates or timers after stop");passed.push("stop_cleanup");
  }
  return {passed:true,tests:passed.length,cases:passed,scope:"V8 mocked network/timers; not an end-user browser or SSH test"};
}
