(function (root) {
  "use strict";
  function createPoller(options) {
    let running = false, timer = null, failures = 0, last = options.initial || null;
    let stopped = false, activeController = null;
    const schedule = options.setTimer || setTimeout;
    const cancel = options.clearTimer || clearTimeout;
    const Controller = options.AbortController || AbortController;
    function later(ms) {
      if (timer !== null) cancel(timer);
      timer = schedule(function () { timer = null; refresh(); }, ms);
    }
    async function refresh() {
      if (running || stopped) return;
      if (timer !== null) { cancel(timer); timer = null; }
      running = true;
      let timeout = null, timedOut = false;
      const controller = new Controller();
      activeController = controller;
      try {
        const deadline = new Promise(function (_, reject) {
          timeout = schedule(function () {
            timedOut = true;
            controller.abort();
            reject(new Error("请求超过 8 秒，正在重连"));
          }, options.timeoutMs || 8000);
        });
        const load = async function () {
          const response = await options.fetch(options.url, {cache: "no-store", credentials: "same-origin", signal: controller.signal});
          if (!response.ok) throw new Error("HTTP " + response.status + "；请检查转发或代理登录");
          const type = response.headers.get("content-type") || "";
          if (type.indexOf("application/json") < 0) throw new Error("接口返回的不是 JSON，可能进入代理登录页");
          const data = await response.json();
          if (!data || !data.monitor || !Array.isArray(data.lanes) || data.lanes.length !== 3) throw new Error("状态响应格式不完整");
          return data;
        };
        const data = await Promise.race([load(), deadline]);
        if (stopped) return;
        last = data; failures = 0;
        options.render(data);
        options.connection({connected: true, last: last, error: null, failures: 0});
      } catch (error) {
        if (!stopped) {
          failures += 1;
          const message = timedOut ? "请求超时" : (error && error.message ? error.message : String(error));
          options.connection({connected: false, last: last, error: message, failures: failures});
        }
      } finally {
        if (timeout !== null) cancel(timeout);
        activeController = null; running = false;
        if (!stopped) later(failures ? Math.min(30000, 2000 * Math.pow(2, failures - 1)) : 10000);
      }
    }
    return {refresh: refresh, stop: function () {
      stopped = true; if (timer !== null) cancel(timer);
      if (activeController) activeController.abort();
    }, state: function () { return {running: running, failures: failures, last: last}; }};
  }
  root.Q35NMonitor = {createPoller: createPoller};
  if (!root.document) return;
  const doc = root.document;
  const el = id => doc.getElementById(id);
  const text = (id, value) => { el(id).textContent = value; };
  const stamp = unix => unix == null ? "尚无" : new Date(unix * 1000).toLocaleString();
  const label = state => ({FAILED: "已停止（失败）", PRODUCING: "生产中", CLOSED: "已结束", FIRST_STRICT_GATE: "首批审核"}[state] || state);
  const initial = JSON.parse(el("bootstrap").textContent);
  let last = initial.monitor.has_snapshot ? initial : null;
  let lastReceivedMillis = Date.now();
  let connected = false, lastError = "正在连接", failures = 0;
  function render(data) {
    last = data;
    lastReceivedMillis = Date.now();
    text("count", `已处理 ${data.completed} / ${data.total_routes} 条；严格合格 ${data.strict_routes} 条 / ${data.strict_decisions} 动作`);
    el("bar").value = data.completed;
    text("note", data.note);
    text("training", `训练：${data.training_status} · 最终检查点 ${data.checkpoint_updates} 步 · 本页面不会启动训练。`);
    const b = data.benchmark || {};
    text("benchmark", `完整基准：${b.status || "UNKNOWN"} · ${b.completed == null ? "—" : b.completed} / ${b.planned || b.total || "—"} 条 · 评测检查点 ${b.checkpoint_updates || "—"} 步。`);
    const rows = el("rows"); rows.replaceChildren();
    for (const lane of data.lanes) {
      const tr = doc.createElement("tr");
      for (const value of [lane.gpu, label(lane.stage), `${lane.completed} / ${lane.target}`, lane.strict_routes, lane.strict_decisions, (lane.result || {}).error || "—"]) {
        const td = doc.createElement("td"); td.textContent = value; tr.appendChild(td);
      }
      rows.appendChild(tr);
    }
    const old = data.old_data_audit || {};
    text("old-summary", `${old.status || "未知"} · 合格 ${old.strict_routes == null ? "—" : old.strict_routes} 条 / ${old.instruction_conditioned_decisions == null ? "—" : old.instruction_conditioned_decisions} 动作；不计入上方新波。`);
    text("old", JSON.stringify(old, null, 2));
    text("final", Object.keys(data.final_merge || {}).length ? JSON.stringify(data.final_merge) : "尚未完成最终合并；滚动合格数不是全波闭合结果。");
  }
  function banner() {
    const meta = last && last.monitor;
    const age = meta && meta.age_seconds != null ? Math.max(0, meta.age_seconds + (Date.now() - lastReceivedMillis) / 1000) : null;
    const stale = !connected || !meta || meta.stale || age > 30;
    el("connection").className = "banner " + (stale ? "offline" : "live");
    if (!connected) text("connection", `网页连接中断 / 尚未确认：${lastError}。正在自动重连（连续失败 ${failures} 次）。下方保留上次快照，不是实时进度。请确认 SSH / VS Code 端口转发仍在线。`);
    else if (meta.stale || age > 30) text("connection", `网页连接正常，但服务端数据暂未更新：${meta.source_error || "采集快照已过期"}。下方是最近成功快照。`);
    else text("connection", "网页连接正常 · 服务端快照新鲜。GPU 的 FAILED 是生产状态，与网页断连不同。");
    text("last-success", `最近成功采集：${stamp(meta && meta.snapshot_unix)}${age == null ? "" : "（" + Math.floor(age) + " 秒前）"}`);
  }
  const url = new URL("api/status", doc.baseURI).href;
  text("api-url", "本页状态接口：" + url);
  if (last) render(last);
  const poller = createPoller({url: url, fetch: root.fetch.bind(root), initial: last, render: render,
    connection: value => { connected = value.connected; lastError = value.error; failures = value.failures; banner(); }});
  el("retry").addEventListener("click", poller.refresh);
  root.addEventListener("online", poller.refresh);
  doc.addEventListener("visibilitychange", () => { if (!doc.hidden) poller.refresh(); });
  root.addEventListener("pagehide", event => { if (!event.persisted) poller.stop(); });
  root.addEventListener("pageshow", event => { if (event.persisted) poller.refresh(); });
  root.setInterval(banner, 1000);
  banner(); poller.refresh();
})(globalThis);
