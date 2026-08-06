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

async function api(url, opts) {
  const resp = await fetch(url, opts);
  if (!resp.ok) {
    let msg = "请求失败（" + resp.status + "）";
    try { msg = (await resp.json()).detail || msg; } catch (e) { /* 保持默认 */ }
    throw new Error(msg);
  }
  return resp.json();
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

async function runAsk() {
  const q = $("#ask-q").value.trim();
  const answerEl = $("#ask-answer");
  const sourcesEl = $("#ask-sources");
  const hintEl = $("#ask-hint");
  clear(answerEl);
  clear(sourcesEl);
  hintEl.classList.add("hidden");
  if (!q) return;
  let data;
  try {
    data = await api("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q, web: $("#ask-web").checked }),
    });
  } catch (err) {
    // LLM/联网调用失败：显示错误并降级展示本地检索结果
    addText(answerEl, "问答失败：" + err.message);
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
  if (data.retrieval_only) {
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
    return;
  }
  if (data.answer) addText(answerEl, data.answer);
  if (data.sources && data.sources.length) {
    const head = document.createElement("div");
    head.className = "src-head";
    addText(head, "来源：");
    sourcesEl.appendChild(head);
    for (const s of data.sources) {
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
      sourcesEl.appendChild(div);
    }
  }
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
