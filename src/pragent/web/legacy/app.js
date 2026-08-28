"use strict";

function $(sel) { return document.querySelector(sel); }
function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); }
function addText(el, text) { el.appendChild(document.createTextNode(text)); }

// ---- Tab 切换 ----
document.querySelectorAll(".tab").forEach(function (btn) {
  btn.addEventListener("click", function () {
    document.querySelectorAll(".tab").forEach(function (b) { b.classList.remove("active"); });
    document.querySelectorAll(".tabpanel").forEach(function (p) { p.classList.remove("active"); });
    btn.classList.add("active");
    $("#tab-" + btn.dataset.tab).classList.add("active");
  });
});

async function api(url, opts, retried) {
  const request = Object.assign({}, opts || {});
  const headers = new Headers(request.headers || {});
  const apiKey = sessionStorage.getItem("pragent-api-key");
  if (apiKey) headers.set("X-PRA-Key", apiKey);
  request.headers = headers;
  const resp = await fetch(url, request);
  if (resp.status === 401 && !retried) {
    const entered = window.prompt("此 PRAgent 服务需要 API key：");
    if (entered) {
      sessionStorage.setItem("pragent-api-key", entered);
      return api(url, opts, true);
    }
  }
  if (!resp.ok) {
    let msg = "请求失败（" + resp.status + "）";
    try { msg = (await resp.json()).detail || msg; } catch (e) { /* 保持默认 */ }
    throw new Error(msg);
  }
  return resp.json();
}

document.querySelectorAll(".research-ui-link").forEach(function (researchLink) {
  researchLink.addEventListener("click", async function (event) {
    event.preventDefault();
    try {
      await api("/api/ui-auth", { method: "POST" });
      window.location.assign(researchLink.href);
    } catch (err) {
      window.alert("无法打开研究工作区：" + err.message);
    }
  });
});

async function apiStream(url, opts, onEvent, retried) {
  const request = Object.assign({}, opts || {});
  const headers = new Headers(request.headers || {});
  const apiKey = sessionStorage.getItem("pragent-api-key");
  if (apiKey) headers.set("X-PRA-Key", apiKey);
  request.headers = headers;
  const resp = await fetch(url, request);
  if (resp.status === 401 && !retried) {
    const entered = window.prompt("此 PRAgent 服务需要 API key：");
    if (entered) {
      sessionStorage.setItem("pragent-api-key", entered);
      return apiStream(url, opts, onEvent, true);
    }
  }
  if (!resp.ok) {
    let msg = "请求失败（" + resp.status + "）";
    try { msg = (await resp.json()).detail || msg; } catch (e) { /* 保持默认 */ }
    throw new Error(msg);
  }
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  function parseFrames(final) {
    let parsed = 0;
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const frame = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      for (const line of frame.split("\n")) {
        const trimmed = line.trim();
        if (trimmed.startsWith("data:")) {
          onEvent(JSON.parse(trimmed.slice(5).trim()));
          parsed++;
        }
      }
    }
    if (final && buf.trim()) {
      // 流结束时若有未以空行结尾的尾帧，也尝试解析（防御性处理）
      const trimmed = buf.trim();
      if (trimmed.startsWith("data:")) {
        onEvent(JSON.parse(trimmed.slice(5).trim()));
        parsed++;
      }
      buf = "";
    }
    return parsed;
  }
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    parseFrames(false);
  }
  buf += decoder.decode();
  parseFrames(true);
}

function renderHits(el, hits) {
  for (const h of hits) {
    const div = document.createElement("div");
    const label = "[" + h.title + (h.year ? "（" + h.year + "）" : "") + " 第" + h.page + "页] ";
    addText(div, label + h.text);
    el.appendChild(div);
  }
}

// ---- 检索 ----
$("#search-btn").addEventListener("click", runSearch);
$("#search-q").addEventListener("keydown", function (e) { if (e.key === "Enter") runSearch(); });

function renderWebPapers(el, papers) {
  for (const p of papers) {
    const div = document.createElement("div");
    div.className = "hit";
    const head = document.createElement("div");
    head.className = "hit-head";
    const t = document.createElement("span");
    t.className = "hit-title";
    addText(t, p.title + (p.year ? "（" + p.year + "）" : ""));
    const links = document.createElement("span");
    const abs = document.createElement("a");
    abs.href = p.url;
    abs.target = "_blank";
    abs.rel = "noopener";
    addText(abs, "摘要页");
    links.appendChild(abs);
    if (p.pdf_url) {
      addText(links, " | ");
      const pdf = document.createElement("a");
      pdf.href = p.pdf_url;
      pdf.target = "_blank";
      pdf.rel = "noopener";
      addText(pdf, "PDF");
      links.appendChild(pdf);
    }
    head.append(t, links);
    const meta = document.createElement("div");
    meta.className = "hit-meta";
    addText(meta, (p.authors || []).join("、") + " — arXiv 联网");
    const text = document.createElement("div");
    text.className = "hit-text";
    addText(text, p.abstract || "");
    div.append(head, meta, text);
    el.appendChild(div);
  }
}

async function runSearch() {
  const q = $("#search-q").value.trim();
  const out = $("#search-results");
  clear(out);
  if (!q) return;
  if ($("#web-check").checked) {
    try {
      const data = await api("/api/websearch?q=" + encodeURIComponent(q) + "&top=5");
      if (!data.papers.length) { addText(out, "未找到相关论文（arXiv 以英文为主，建议用英文查询）。"); return; }
      renderWebPapers(out, data.papers);
      addText(out, "");
    } catch (err) {
      addText(out, "联网检索失败：" + err.message);
    }
    return;
  }
  try {
    const data = await api("/api/search?q=" + encodeURIComponent(q) + "&top=10");
    if (!data.hits.length) { addText(out, "未找到相关内容。"); return; }
    for (const h of data.hits) {
      const div = document.createElement("div");
      div.className = "hit";
      const head = document.createElement("div");
      head.className = "hit-head";
      const t = document.createElement("span");
      t.className = "hit-title";
      addText(t, h.title + (h.year ? "（" + h.year + "）" : "") + " 第" + h.page + "页");
      const score = document.createElement("span");
      score.className = "hit-score";
      addText(score, "得分 " + h.score.toFixed(3));
      head.append(t, score);
      const meta = document.createElement("div");
      meta.className = "hit-meta";
      addText(meta, h.path);
      const text = document.createElement("div");
      text.className = "hit-text";
      addText(text, h.text);
      div.append(head, meta, text);
      out.appendChild(div);
    }
  } catch (err) {
    addText(out, "检索失败：" + err.message);
  }
}

// ---- 问答 ----
$("#ask-btn").addEventListener("click", runAsk);
$("#ask-q").addEventListener("keydown", function (e) { if (e.key === "Enter") runAsk(); });

function addErrorNote(el, text) {
  const div = document.createElement("div");
  div.className = "ask-error";
  addText(div, text);
  el.appendChild(div);
}

function showThinking(el) {
  const wrap = document.createElement("div");
  wrap.className = "thinking";
  const spinner = document.createElement("span");
  spinner.className = "spinner";
  const textWrap = document.createElement("span");
  textWrap.className = "thinking-text";
  const label = document.createElement("span");
  addText(label, "正在思考");
  textWrap.appendChild(label);
  for (let i = 0; i < 3; i++) {
    const dot = document.createElement("span");
    dot.className = "think-dot";
    dot.style.animationDelay = (i * 0.2) + "s";
    textWrap.appendChild(dot);
  }
  wrap.append(spinner, textWrap);
  el.appendChild(wrap);
}

function renderAskSources(el, sources) {
  if (!sources || !sources.length) return;
  const head = document.createElement("div");
  head.className = "src-head";
  addText(head, "来源：");
  el.appendChild(head);
  for (const s of sources) {
    const div = document.createElement("div");
    if (s.web) {
      addText(div, "[" + s.n + "] " + s.title + (s.year ? "（" + s.year + "）" : "") + " [arXiv 联网] ");
      const a = document.createElement("a");
      a.href = s.path;
      a.target = "_blank";
      a.rel = "noopener";
      addText(a, s.path);
      div.appendChild(a);
    } else {
      const loc = (s.page !== null && s.page !== undefined) ? " 第" + s.page + "页 — " : " — ";
      addText(div, "[" + s.n + "] " + s.title + (s.year ? "（" + s.year + "）" : "") + loc + s.path + (s.catalog ? "（库藏）" : ""));
    }
    el.appendChild(div);
  }
}

function renderRetrievalOnly(hintEl, sourcesEl, data) {
  hintEl.classList.remove("hidden");
  addText(hintEl, "未配置 PRA_LLM_API_KEY，仅显示检索结果；配置后获得生成式回答。");
  renderHits(sourcesEl, data.hits || []);
  if (data.web_papers && data.web_papers.length) {
    const head = document.createElement("div");
    head.className = "src-head";
    addText(head, "—— 联网检索（arXiv）——");
    sourcesEl.appendChild(head);
    renderWebPapers(sourcesEl, data.web_papers);
  }
}

async function askClassic(q, web) {
  const data = await api("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question: q, web: web }),
  });
  return data;
}

async function runAsk() {
  const q = $("#ask-q").value.trim();
  const answerEl = $("#ask-answer");
  const sourcesEl = $("#ask-sources");
  const hintEl = $("#ask-hint");
  const statusEl = $("#ask-status");
  clear(answerEl);
  clear(sourcesEl);
  hintEl.classList.add("hidden");
  if (!q) return;
  const web = $("#ask-web").checked;
  function setStatus(text, isError) {
    statusEl.classList.remove("hidden");
    statusEl.textContent = text;
    statusEl.style.color = isError ? "#a8071a" : "#888";
  }
  // 等待阶段动态反馈：旋转 spinner + 闪烁省略号 + 实时等待秒数
  showThinking(answerEl);
  setStatus("已连接服务，正在检索与生成…");
  let phase = "thinking";
  const startedAt = Date.now();
  const ticker = setInterval(function () {
    if (phase !== "thinking") return;
    const secs = Math.round((Date.now() - startedAt) / 1000);
    setStatus("已连接服务，正在检索与生成…（已等待 " + secs + " 秒）");
  }, 1000);
  function setPhase(next) {
    phase = next;
    if (next !== "thinking") clearInterval(ticker);
  }
  console.info("[pra] ask/stream start:", q.slice(0, 40));

  let gotAnyEvent = false;
  let context = null;
  let acc = "";
  let streamErr = null;
  let frameCount = 0;
  try {
    await apiStream(
      "/api/ask/stream",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, web: web }),
      },
      function (event) {
        gotAnyEvent = true;
        frameCount++;
        if (event.type === "context") {
          context = event;
          setPhase("streaming");
          if (event.retrieval_only) {
            console.info("[pra] retrieval_only");
            clear(answerEl);
            setStatus("未配置 LLM，仅展示检索结果");
          } else {
            setStatus("检索完成（" + (event.sources || []).length + " 条来源），正在生成回答…");
          }
        } else if (event.type === "delta") {
          acc += event.text;
          answerEl.textContent = acc;
          setStatus("正在生成回答…（已收到 " + frameCount + " 帧）");
        } else if (event.type === "complete") {
          console.info("[pra] complete, verification=", event.verification);
          setPhase("done");
          setStatus("完成（共 " + frameCount + " 帧，用时 " + Math.round((Date.now() - startedAt) / 1000) + " 秒）");
          if (event.verification && !event.verification.ok) {
            addErrorNote(sourcesEl, "引用验证未通过（" + event.verification.code + "）：" + event.verification.message);
          }
        } else if (event.type === "error") {
          streamErr = event.message;
          setPhase("error");
        }
      }
    );
    console.info("[pra] stream finished, frames=", frameCount);
  } catch (err) {
    console.warn("[pra] stream failed:", err);
    streamErr = err.message;
  }

  if (!gotAnyEvent) {
    // 流式端点不可用（旧版服务/网络错误）：降级为一次性问答
    setPhase("fallback");
    clear(answerEl);
    setStatus("流式端点无响应，切换普通模式重试…");
    try {
      const data = await askClassic(q, web);
      if (data.retrieval_only) {
        setStatus("完成（普通模式，仅检索结果）");
        renderRetrievalOnly(hintEl, sourcesEl, data);
        return;
      }
      if (data.answer) addText(answerEl, data.answer);
      renderAskSources(sourcesEl, data.sources);
      setStatus("完成（普通模式）");
      return;
    } catch (err2) {
      addText(answerEl, "问答失败：" + err2.message);
      setStatus("问答失败：" + err2.message, true);
      try {
        const search = await api("/api/search?q=" + encodeURIComponent(q) + "&top=8");
        if (search.hits.length) {
          const head = document.createElement("div");
          head.className = "src-head";
          addText(head, "—— 检索结果 ——");
          sourcesEl.appendChild(head);
          renderHits(sourcesEl, search.hits);
        }
      } catch (e) { /* 检索也失败则忽略 */ }
      return;
    }
  }

  if (streamErr) {
    clear(answerEl);
    addText(answerEl, "生成中断：" + streamErr);
    addErrorNote(sourcesEl, "（流式中断：" + streamErr + "）");
    setStatus("流式中断：" + streamErr, true);
  }
  if (!context) return;
  if (context.retrieval_only) {
    renderRetrievalOnly(hintEl, sourcesEl, context);
    return;
  }
  renderAskSources(sourcesEl, context.sources);
}

// ---- 论文库 ----
$("#lib-btn").addEventListener("click", loadLibrary);
$("#lib-q").addEventListener("keydown", function (e) { if (e.key === "Enter") loadLibrary(); });
$("#reindex-btn").addEventListener("click", reindex);

async function loadLibrary() {
  const q = $("#lib-q").value.trim();
  const tbody = $("#lib-table tbody");
  const info = $("#lib-info");
  clear(tbody);
  clear(info);
  try {
    const data = await api("/api/papers?q=" + encodeURIComponent(q) + "&limit=100&offset=0");
    addText(info, "共 " + data.total + " 篇");
    for (const p of data.items) {
      const tr = document.createElement("tr");
      const cells = [
        p.title + (p.has_text ? "" : "（扫描版无文本）"),
        (p.authors || []).join("、"),
        p.year || "-",
        String(p.page_count),
        String(p.chunk_count),
        p.path,
      ];
      for (const c of cells) {
        const td = document.createElement("td");
        addText(td, c);
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    }
  } catch (err) {
    addText(info, "加载失败：" + err.message);
  }
}

async function reindex() {
  const info = $("#lib-info");
  clear(info);
  addText(info, "正在重新索引…");
  try {
    const r = await api("/api/reindex", { method: "POST" });
    clear(info);
    addText(
      info,
      "重新索引完成：新增 " + r.added + "，更新 " + r.updated + "，未变化 " + r.unchanged +
      "，失败 " + r.failed + "，删除 " + r.removed + "，无文本 " + r.skipped_no_text
    );
    loadLibrary();
  } catch (err) {
    clear(info);
    addText(info, "重新索引失败：" + err.message);
  }
}

// ---- Agent ----
$("#agent-btn").addEventListener("click", runAgent);
$("#agent-q").addEventListener("keydown", function (e) { if (e.key === "Enter") runAgent(); });
$("#agent-new").addEventListener("click", newAgentSession);
$("#agent-project").addEventListener("change", function () {
  localStorage.setItem("pra-agent-project", $("#agent-project").value || "");
});

function hide(el) { el.classList.add("hidden"); }
function unhide(el) { el.classList.remove("hidden"); }

function agentSessionId() {
  let id = sessionStorage.getItem("pra-agent-session");
  if (!id) {
    id = (window.crypto && crypto.randomUUID)
      ? crypto.randomUUID()
      : "s-" + Date.now() + "-" + Math.random().toString(16).slice(2);
    sessionStorage.setItem("pra-agent-session", id);
  }
  return id;
}

async function newAgentSession() {
  if (agentBusy) return;
  const currentId = sessionStorage.getItem("pra-agent-session");
  if (currentId) {
    try {
      await api("/api/agent/sessions/" + encodeURIComponent(currentId), { method: "DELETE" });
    } catch (err) {
      agentStatus("无法清空当前会话：" + err.message, true);
      return;
    }
  }
  sessionStorage.removeItem("pra-agent-session");
  clear($("#agent-flow"));
  hide($("#agent-pending"));
  clear($("#agent-pending"));
  $("#agent-q").value = "";
  $("#agent-project").disabled = false;
  agentStatus("已开始新会话");
}

async function loadAgentProjects() {
  const select = $("#agent-project");
  const preferred = localStorage.getItem("pra-agent-project") || "";
  try {
    const data = await api("/api/v1/projects?limit=200");
    for (const project of data.items || []) {
      const option = document.createElement("option");
      option.value = project.id;
      addText(option, project.title || project.id);
      select.appendChild(option);
    }
    if ([...select.options].some(function (option) { return option.value === preferred; })) {
      select.value = preferred;
    }
  } catch (err) {
    agentStatus("项目列表加载失败：" + err.message, true);
  }
}

function restoreAgentHistory(events) {
  clear($("#agent-flow"));
  for (const event of events || []) {
    if (event.type === "message" && event.role === "user") {
      agentAddUser(event.content || "");
    } else if (event.type === "message" && event.role === "assistant") {
      const block = agentStartAssistant();
      block.raw = event.content || "";
      agentRenderText(block);
    } else if (event.type === "tool") {
      agentAddTool(event.name, event.args, event.result, event.code);
    }
  }
}

async function restoreAgentSession() {
  const sessionId = sessionStorage.getItem("pra-agent-session");
  if (!sessionId) return;
  try {
    const state = await api("/api/agent/sessions/" + encodeURIComponent(sessionId));
    restoreAgentHistory(state.history);
    currentRunId = state.run_id || null;
    if (state.project_id) {
      $("#agent-project").value = state.project_id;
      localStorage.setItem("pra-agent-project", state.project_id);
    }
    $("#agent-project").disabled = Boolean((state.history || []).length || state.pending);
    if (state.pending) {
      renderAgentPending(state.pending);
      agentStatus("已恢复等待确认的操作");
    } else if ((state.history || []).length) {
      agentStatus("已恢复会话历史");
    }
  } catch (err) {
    if (!String(err.message || "").includes("404")) {
      agentStatus("会话恢复失败：" + err.message, true);
    }
  }
}

function agentScroll() {
  const flow = $("#agent-flow");
  flow.scrollTop = flow.scrollHeight;
}

function agentAddUser(q) {
  const div = document.createElement("div");
  div.className = "msg msg-user";
  addText(div, q);
  $("#agent-flow").appendChild(div);
  agentScroll();
}

function agentStartAssistant() {
  const node = document.createElement("div");
  node.className = "msg msg-assistant";
  const textEl = document.createElement("div");
  textEl.className = "agent-text";
  node.appendChild(textEl);
  $("#agent-flow").appendChild(node);
  agentScroll();
  return { raw: "", textEl: textEl, node: node };
}

function agentRenderText(block) {
  const parts = block.raw.split(/(\[E:[^\]]+\])/g);
  block.textEl.replaceChildren();
  for (const part of parts) {
    if (/^\[E:[^\]]+\]$/.test(part)) {
      const mark = document.createElement("mark");
      mark.className = "evidence";
      addText(mark, part);
      block.textEl.appendChild(mark);
    } else {
      addText(block.textEl, part);
    }
  }
  agentScroll();
}

function agentAddTool(name, args, result, code) {
  const card = document.createElement("div");
  card.className = "tool-card";
  const head = document.createElement("div");
  const nameSpan = document.createElement("span");
  nameSpan.className = "tool-name";
  addText(nameSpan, name);
  head.appendChild(nameSpan);
  let argsText = "";
  try { argsText = JSON.stringify(args || {}); } catch (e) { /* 忽略 */ }
  if (argsText && argsText !== "{}") {
    addText(head, "(" + argsText.slice(0, 200) + ")");
  }
  if (code && code !== "ok" && code !== "confirmed") {
    addText(head, " [" + code + "]");
  }
  card.appendChild(head);
  if (result) {
    const r = document.createElement("div");
    r.className = "tool-result";
    addText(r, String(result).slice(0, 500));
    card.appendChild(r);
  }
  $("#agent-flow").appendChild(card);
  agentScroll();
}

function agentAddNote(text, cls) {
  const div = document.createElement("div");
  div.className = "note " + cls;
  addText(div, text);
  $("#agent-flow").appendChild(div);
  agentScroll();
}

function renderAgentPending(event) {
  const box = $("#agent-pending");
  clear(box);
  const card = document.createElement("div");
  card.className = "pending-card";
  const h = document.createElement("h3");
  addText(h, "待确认操作：" + event.name);
  const summary = document.createElement("div");
  summary.className = "pending-summary";
  addText(summary, event.summary || "");
  const digest = document.createElement("div");
  digest.className = "pending-digest";
  addText(digest, "绑定摘要 SHA-256：" + (event.digest || ""));
  const actions = document.createElement("div");
  actions.className = "pending-actions";
  const ok = document.createElement("button");
  ok.className = "pending-ok";
  addText(ok, "确认执行");
  ok.addEventListener("click", function () { agentConfirm(true); });
  const no = document.createElement("button");
  no.className = "pending-no";
  addText(no, "取消");
  no.addEventListener("click", function () { agentConfirm(false); });
  actions.append(ok, no);
  card.append(h, summary, digest, actions);
  box.appendChild(card);
  unhide(box);
}

let agentBusy = false;
function agentSetBusy(busy) {
  agentBusy = busy;
  $("#agent-btn").disabled = busy;
  $("#agent-q").disabled = busy;
}

function agentStatus(text, isError) {
  const el = $("#agent-status");
  unhide(el);
  el.textContent = text;
  el.style.color = isError ? "#a8071a" : "#888";
}

async function consumeAgentStream(url, body) {
  const assistantBlock = agentStartAssistant();
  showThinking(assistantBlock.textEl);
  let status = null;
  try {
    await apiStream(
      url,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      },
      function (event) {
        if (event.type === "session") return;
        if (event.type === "assistant_delta") {
          assistantBlock.raw += event.text;
          agentRenderText(assistantBlock);
          agentStatus("正在生成…");
        } else if (event.type === "tool") {
          agentAddTool(event.name, event.args, event.result, event.code);
        } else if (event.type === "verification") {
          agentAddNote(event.message, "note-verify");
        } else if (event.type === "error") {
          agentAddNote(event.message, "note-error");
        } else if (event.type === "pending") {
          renderAgentPending(event);
        } else if (event.type === "run") {
          currentRunId = event.run_id;
        } else if (event.type === "complete") {
          status = event.status;
        }
      }
    );
  } catch (err) {
    agentAddNote("请求失败：" + err.message, "note-error");
  }
  if (!assistantBlock.raw) {
    assistantBlock.node.remove();
  }
  return status;
}

let currentRunId = null;

async function runAgent() {
  const q = $("#agent-q").value.trim();
  if (!q || agentBusy) return;
  if (!$("#agent-pending").classList.contains("hidden")) {
    agentStatus("上方有待确认操作，请先确认或取消。", true);
    return;
  }
  agentSetBusy(true);
  $("#agent-project").disabled = true;
  $("#agent-q").value = "";
  hide($("#agent-pending"));
  clear($("#agent-pending"));
  agentAddUser(q);
  agentStatus("正在检索与思考…");
  const status = await consumeAgentStream("/api/agent/chat", {
    session_id: agentSessionId(),
    project_id: $("#agent-project").value || null,
    question: q,
  });
  if (status === "awaiting_confirmation") {
    agentStatus("等待确认（见上方卡片）");
  } else {
    agentStatus("完成（" + (status || "done") + "）", status === "failed" || status === "blocked");
  }
  agentSetBusy(false);
  refreshAgentRuns();
}

async function agentConfirm(confirm) {
  if (agentBusy) return;
  agentSetBusy(true);
  hide($("#agent-pending"));
  clear($("#agent-pending"));
  agentAddNote(confirm ? "已确认执行，继续运行…" : "已取消该操作。", "note-dim");
  agentStatus(confirm ? "已确认，正在执行…" : "已取消");
  const status = await consumeAgentStream("/api/agent/confirm", {
    session_id: agentSessionId(),
    confirm: confirm,
  });
  if (status === "awaiting_confirmation") {
    agentStatus("等待确认（见上方卡片）");
  } else {
    agentStatus("完成（" + (status || "done") + "）", status === "failed" || status === "blocked");
  }
  agentSetBusy(false);
  refreshAgentRuns();
}

// ---- Run 审计侧栏 ----
async function refreshAgentRuns() {
  try {
    const params = new URLSearchParams({ limit: "50" });
    const sessionId = sessionStorage.getItem("pra-agent-session");
    if (sessionId) params.set("session_id", sessionId);
    const data = await api("/api/agent/runs?" + params.toString());
    const box = $("#agent-runs");
    clear(box);
    if (!data.items.length) { addText(box, "暂无 Agent run。"); return; }
    for (const r of data.items) {
      const row = document.createElement("div");
      row.className = "run-row run-" + r.status;
      row.title = (r.objective || "") + "（" + r.run_id + "）";
      addText(row, r.status + " · " + (r.objective || "").slice(0, 24));
      row.addEventListener("click", function () { loadRunEvents(r.run_id, row); });
      box.appendChild(row);
    }
  } catch (err) { /* 侧栏失败不打断主流程 */ }
}

async function loadRunEvents(runId, rowEl) {
  try {
    const data = await api("/api/agent/runs/" + encodeURIComponent(runId) + "/events");
    const box = $("#agent-run-events");
    clear(box);
    if (rowEl) {
      document.querySelectorAll(".run-row").forEach(function (r) { r.classList.remove("selected"); });
      rowEl.classList.add("selected");
    }
    for (const e of data.items) {
      const div = document.createElement("div");
      div.className = "event-row";
      const type = document.createElement("span");
      type.className = "event-type";
      addText(type, "#" + e.seq + " " + e.event_type);
      div.appendChild(type);
      let payloadText = "";
      try { payloadText = JSON.stringify(e.payload || {}); } catch (err) { /* 忽略 */ }
      if (payloadText && payloadText !== "{}") {
        addText(div, " " + payloadText.slice(0, 120));
      }
      box.appendChild(div);
    }
  } catch (err) {
    const box = $("#agent-run-events");
    clear(box);
    addText(box, "加载事件失败：" + err.message);
  }
}

async function initializeAgent() {
  await loadAgentProjects();
  await restoreAgentSession();
  await refreshAgentRuns();
}

initializeAgent();

loadLibrary();
