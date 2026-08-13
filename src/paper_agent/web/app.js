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
  const apiKey = sessionStorage.getItem("paper-agent-api-key");
  if (apiKey) headers.set("X-Paper-Agent-Key", apiKey);
  request.headers = headers;
  const resp = await fetch(url, request);
  if (resp.status === 401 && !retried) {
    const entered = window.prompt("此 Pagent 服务需要 API key：");
    if (entered) {
      sessionStorage.setItem("paper-agent-api-key", entered);
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

async function apiStream(url, opts, onEvent, retried) {
  const request = Object.assign({}, opts || {});
  const headers = new Headers(request.headers || {});
  const apiKey = sessionStorage.getItem("paper-agent-api-key");
  if (apiKey) headers.set("X-Paper-Agent-Key", apiKey);
  request.headers = headers;
  const resp = await fetch(url, request);
  if (resp.status === 401 && !retried) {
    const entered = window.prompt("此 Pagent 服务需要 API key：");
    if (entered) {
      sessionStorage.setItem("paper-agent-api-key", entered);
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
      addText(div, "[" + s.n + "] " + s.title + (s.year ? "（" + s.year + "）" : "") + " 第" + s.page + "页 — " + s.path);
    }
    el.appendChild(div);
  }
}

function renderRetrievalOnly(hintEl, sourcesEl, data) {
  hintEl.classList.remove("hidden");
  addText(hintEl, "未配置 PAPER_LLM_API_KEY，仅显示检索结果；配置后获得生成式回答。");
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
  console.info("[pagent] ask/stream start:", q.slice(0, 40));

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
            console.info("[pagent] retrieval_only");
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
          console.info("[pagent] complete, verification=", event.verification);
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
    console.info("[pagent] stream finished, frames=", frameCount);
  } catch (err) {
    console.warn("[pagent] stream failed:", err);
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

loadLibrary();
