"use strict";

const STORAGE_KEY = "pragent-prototype-v1-1-state";

const STAGES = [
  { id: "start", label: "目标", title: "开始一项新研究", status: "等待你的目标" },
  { id: "plan", label: "计划", title: "确认研究计划", status: "正在界定研究范围" },
  { id: "approval", label: "确认", title: "批准即将执行的操作", status: "等待关键操作确认" },
  { id: "papers", label: "论文", title: "筛选进入精读的论文", status: "正在解释候选论文" },
  { id: "progress", label: "执行", title: "执行研究计划", status: "正在分阶段处理" },
  { id: "deepread", label: "精读", title: "逐篇检查研究证据", status: "已完成结构化精读" },
  { id: "compare", label: "对比", title: "形成跨论文比较与洞察", status: "正在综合共识与差异" },
  { id: "review", label: "综述", title: "编辑证据化综述草稿", status: "研究成果可以交付" }
];

const PAPERS = [
  { id: "p1", score: 96, year: 2025, venue: "KDD", title: "Large Language Models for Multi-Task Recommendation: A Semantic Routing Framework", reason: "直接研究 LLM 语义表示如何参与多任务路由，方法和你的问题高度一致。", tag: "核心方法" },
  { id: "p2", score: 93, year: 2024, venue: "RecSys", title: "Semantic Task Alignment for Multi-Behavior Recommendation", reason: "提供任务对齐表征，可作为传统 MMoE/PLE 与 LLM 增强方案之间的桥梁。", tag: "任务对齐" },
  { id: "p3", score: 91, year: 2024, venue: "WWW", title: "Foundation Models Meet Recommender Systems: A Survey and Taxonomy", reason: "综述覆盖全面，适合建立术语、分类法和相关工作入口。", tag: "领域综述" },
  { id: "p4", score: 88, year: 2023, venue: "NeurIPS", title: "Text-Enhanced Multi-Task Learning for Sparse Recommendation", reason: "给出文本语义缓解稀疏性的实证结果，但使用的不是完整大模型。", tag: "关键基线" },
  { id: "p5", score: 84, year: 2025, venue: "arXiv", title: "Instruction-Tuned User Modeling for Multi-Domain Recommendation", reason: "跨域设置相关，但论文尚未正式发表，证据权重需要降低。", tag: "前沿工作" },
  { id: "p6", score: 78, year: 2022, venue: "SIGIR", title: "Progressive Layered Extraction for Multi-Task Recommender Systems", reason: "不使用 LLM，但属于必要的经典架构基线，适合方法对照。", tag: "经典基线" }
];

const DEFAULT_MARKDOWN = `# 大模型增强多任务推荐：方法、证据与研究空白

## 摘要

近年的研究开始使用语言模型补充多任务推荐中的语义信息，但现有证据不支持“大模型在所有任务上普遍更优”的判断。其收益主要出现在行为稀疏、任务语义可描述或需要跨域知识迁移的场景。[1, E-021]

## 1. 从共享参数到语义任务关系

PLE 等经典方法通过分层专家缓解任务之间的参数冲突。新方法将任务描述、物品文本或领域知识编码为语义表示，用于控制专家路由或增加对齐约束。[1, E-014]

## 2. 当前证据的边界

公开实验尚未充分控制参数量、预训练数据和推理成本。更稳妥的结论是：语义增强具有条件性价值，而非替代所有轻量多任务基线。

## 3. 值得验证的研究空白

在统一数据划分和计算预算下，对比冻结 LLM、轻量文本编码器和无语义 PLE，并报告逐任务收益、负迁移率以及成本。
`;

const DEFAULT_STATE = {
  projectName: "大模型增强多任务推荐",
  workspacePath: "D:\\Research\\llm-mtl-review",
  activeChatId: "chat-main",
  chatOrder: ["chat-main"],
  chatMeta: {
    "chat-main": { title: "主题调研到综述", status: "正在规划" }
  },
  chatSnapshots: {},
  memoryPublished: false,
  memoryUpdatedAt: "",
  memoryFile: "research-summary.md",
  memoryContent: "",
  stage: 0,
  maxVisited: 0,
  goal: "",
  plan: {
    question: "大模型如何增强多任务推荐系统，并在哪些条件下带来稳定收益？",
    years: "2023–2026",
    paperCount: "6",
    focus: "语义表征、任务关系建模、可复现性、与 PLE/MMoE 的公平对比",
    output: "证据化中文综述"
  },
  selectedPapers: ["p1", "p2", "p3", "p4", "p6"],
  approvals: ["search", "download", "model"],
  progressStep: 0,
  paused: false,
  customDimensions: [],
  stageRevisions: {},
  staleFromStage: null,
  reviewEdited: false,
  markdownDraft: DEFAULT_MARKDOWN,
  feedback: {},
  extraMessages: []
};

let state = loadState();
let progressTimer = null;
let toastTimer = null;

const stageNav = document.querySelector("#stage-nav");
const artifactContent = document.querySelector("#artifact-content");
const conversation = document.querySelector("#conversation");
const composer = document.querySelector("#composer");
const goalInput = document.querySelector("#goal-input");
const sendButton = document.querySelector("#send-button");
const modalBackdrop = document.querySelector("#modal-backdrop");
const modal = document.querySelector("#modal");
const toast = document.querySelector("#toast");

function cloneDefaultState() {
  return JSON.parse(JSON.stringify(DEFAULT_STATE));
}

function loadState() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
    if (!saved || typeof saved.stage !== "number" || saved.stage > STAGES.length - 1) {
      return cloneDefaultState();
    }
    return {
      ...cloneDefaultState(),
      ...saved,
      plan: { ...DEFAULT_STATE.plan, ...(saved.plan || {}) },
      chatMeta: { ...DEFAULT_STATE.chatMeta, ...(saved.chatMeta || {}) },
      chatOrder: saved.chatOrder || DEFAULT_STATE.chatOrder,
      chatSnapshots: saved.chatSnapshots || {},
      customDimensions: saved.customDimensions || [],
      stageRevisions: saved.stageRevisions || {},
      feedback: saved.feedback || {},
      extraMessages: saved.extraMessages || []
    };
  } catch (_error) {
    return cloneDefaultState();
  }
}

function saveState() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  const saveLabel = document.querySelector("#save-state");
  if (saveLabel) {
    saveLabel.textContent = "已保存到当前浏览器";
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

const CHAT_FIELDS = [
  "stage", "maxVisited", "goal", "plan", "selectedPapers", "approvals",
  "progressStep", "paused", "customDimensions", "stageRevisions", "staleFromStage", "reviewEdited", "markdownDraft", "extraMessages"
];

function saveActiveChatSnapshot() {
  const snapshot = {};
  CHAT_FIELDS.forEach((field) => {
    snapshot[field] = JSON.parse(JSON.stringify(state[field]));
  });
  state.chatSnapshots[state.activeChatId] = snapshot;
  if (state.chatMeta[state.activeChatId]) {
    state.chatMeta[state.activeChatId].status = currentChatStatus();
  }
}

function restoreChatSnapshot(chatId) {
  saveActiveChatSnapshot();
  const snapshot = state.chatSnapshots[chatId];
  if (!snapshot) return;
  CHAT_FIELDS.forEach((field) => {
    state[field] = JSON.parse(JSON.stringify(snapshot[field]));
  });
  state.activeChatId = chatId;
  render();
}

function createChat(title) {
  saveActiveChatSnapshot();
  const chatId = `chat-${Date.now()}`;
  const defaults = cloneDefaultState();
  CHAT_FIELDS.forEach((field) => {
    state[field] = JSON.parse(JSON.stringify(defaults[field]));
  });
  state.activeChatId = chatId;
  state.chatOrder.unshift(chatId);
  state.chatMeta[chatId] = { title: title || `新研究对话 ${state.chatOrder.length}`, status: "等待目标" };
  if (state.memoryPublished) {
    state.extraMessages.push({
      stage: 0,
      role: "agent",
      text: `已加载项目公共记忆 ${state.memoryFile}。你可以在它的基础上提出新问题，不需要重复前一个对话。`,
      meta: "研究 Agent · 项目记忆已加载"
    });
  }
  render();
  closeModal();
  showToast("已在当前项目中新建对话");
}

function createProject(name, workspacePath) {
  state = cloneDefaultState();
  state.projectName = name;
  state.workspacePath = workspacePath;
  state.chatMeta["chat-main"].title = "第一个研究对话";
  localStorage.removeItem(STORAGE_KEY);
  render();
  closeModal();
  showToast("项目和第一个对话已创建");
}

function showToast(message) {
  window.clearTimeout(toastTimer);
  toast.textContent = message;
  toast.classList.add("show");
  toastTimer = window.setTimeout(() => toast.classList.remove("show"), 2400);
}

function render() {
  const stage = STAGES[state.stage];
  document.querySelector("#page-title").textContent = stage.title;
  document.querySelector("#agent-status").textContent = stage.status;
  document.querySelector("#sidebar-project-name").textContent = state.projectName;
  document.querySelector("#header-project-name").textContent = state.projectName;
  document.querySelector("#header-chat-title").textContent = state.chatMeta[state.activeChatId]?.title || "新对话";
  document.querySelector("#workspace-path").textContent = state.workspacePath;
  document.querySelector("#workspace-path").title = state.workspacePath;
  document.querySelector("#memory-count").textContent = state.memoryPublished ? "1" : "0";
  document.querySelector("#context-scope").textContent = `可询问并修改“目标”至“${stage.label}”阶段`;
  renderChatList();
  renderStageNav();
  renderConversation();
  renderArtifact();
  configureComposer();
  saveState();

  if (stage.id === "progress" && state.progressStep < 4 && !state.paused) {
    startProgressTimer();
  } else {
    stopProgressTimer();
  }
}

function renderChatList() {
  const list = document.querySelector("#chat-list");
  list.innerHTML = state.chatOrder.map((chatId) => {
    const meta = state.chatMeta[chatId] || { title: "未命名对话", status: "已保存" };
    const isActive = chatId === state.activeChatId;
    return `
      <button class="history-item ${isActive ? "active" : ""}" type="button" data-chat="${chatId}">
        <span class="history-icon ${isActive ? "" : "muted"}">话</span>
        <span><strong>${escapeHtml(meta.title)}</strong><small>${escapeHtml(isActive ? currentChatStatus() : meta.status)}</small></span>
      </button>`;
  }).join("");
}

function currentChatStatus() {
  if (state.memoryPublished && state.stage === 7) return "已写入项目公共记忆";
  return `${STAGES[state.stage].label}阶段 · 原型`;
}

function renderStageNav() {
  stageNav.innerHTML = STAGES.map((stage, index) => {
    const active = index === state.stage;
    const visited = index <= state.maxVisited;
    const className = active ? "active visited" : visited ? "visited" : "locked";
    const number = index < state.maxVisited ? "✓" : index + 1;
    return `
      <button class="stage-button ${className}" type="button" data-stage="${index}" ${visited ? "" : "disabled"}>
        <span class="stage-number">${number}</span><span class="stage-label">${stage.label}</span>
      </button>`;
  }).join("");
}

function message(role, body, meta) {
  return `
    <article class="message ${role}">
      <div class="message-meta">${meta || (role === "user" ? "你" : "研究 Agent")}</div>
      <div class="message-body">${body}</div>
    </article>`;
}

function stageMessages() {
  const goal = escapeHtml(state.goal || "帮我调研大模型如何增强多任务推荐，重点关注 2023 年后的方法和可复现方案。");
  const selectedCount = state.selectedPapers.length;
  const messages = [
    [
      message("agent", `<p>告诉我这个对话要研究的问题或最终想交付什么。我会先读取项目公共记忆，再把目标整理成计划。</p>
        <div class="inline-card"><strong>当前项目：${escapeHtml(state.projectName)}</strong><br>所有中间文件与输出都将归档到 ${escapeHtml(state.workspacePath)}。</div>`, "研究 Agent · 现在")
    ],
    [
      message("user", `<p>${goal}</p>`),
      message("agent", `<p>我先把它收敛成一个可执行的研究问题，并补上时间范围、筛选标准和交付形式。</p><p>右侧计划可以直接编辑。确认前不会发生联网或模型调用。</p>`, "研究 Agent · 已分析目标")
    ],
    [
      message("user", `<p>研究计划没问题，继续。</p>`),
      message("agent", `<p>开始前，我把会联网、写入本地和消耗模型额度的操作集中列出。</p><p>你可以取消其中任何一项；未批准的能力不会执行。</p>`, "研究 Agent · 等待批准")
    ],
    [
      message("agent", `<p>我聚合检索了 3 个学术来源，去重后得到 24 篇候选论文，并按与你的研究问题的相关性排序。</p><p>我推荐精读其中 5 篇。你可以查看理由并修改选择。</p>`, "研究 Agent · 检索完成")
    ],
    [
      message("user", `<p>使用选中的 ${selectedCount} 篇论文继续。</p>`),
      message("agent", `<p>任务已拆成四个阶段。我会先完成全文准备，再逐篇精读，最后综合比较。</p><p>你可以离开页面，任务状态会被保存；也可以在阶段边界暂停。</p>`, "研究 Agent · 正在执行")
    ],
    [
      message("agent", `<p>${selectedCount} 篇论文已经完成结构化精读。我为每个关键判断保留了可回到原文的证据片段。</p><p>建议先抽查最关键的一篇，再进入跨论文比较。</p>`, "研究 Agent · 精读完成")
    ],
    [
      message("agent", `<p>比较结果显示：LLM 的主要收益来自语义补充和任务关系建模，但在高密度行为数据上未必稳定优于轻量基线。</p><p>右侧已区分共识、冲突和仍缺少证据的研究空白。</p>`, "研究 Agent · 综合完成")
    ],
    [
      message("agent", `<p>综述草稿已生成。每个关键结论都能展开查看对应论文与原文证据。</p><p>你可以继续对话修改章节，也可以导出 Markdown 或 DOCX。</p>`, "研究 Agent · 可以交付")
    ]
  ];

  return messages[state.stage] || [];
}

function renderConversation() {
  const extras = state.extraMessages
    .filter((entry) => entry.stage === state.stage)
    .map((entry) => message(entry.role, `<p>${escapeHtml(entry.text)}</p>`, entry.meta));
  conversation.innerHTML = [...stageMessages(), ...extras].join("") + renderQuickPrompts();
  conversation.scrollTop = conversation.scrollHeight;
}

function renderQuickPrompts() {
  const prompts = [
    ["基于项目公共记忆继续找研究空白", "先解释这个项目已有的研究结论"],
    ["把检索范围改为 2024 年以后", "为什么建议精读 6 篇？"],
    ["模型会收到哪些项目内容？", "取消下载，只保留在线检索"],
    ["为什么没有选择第 5 篇？", "增加一篇经典基线"],
    ["暂停后会保留哪些中间文件？", "查看已经完成的精读"],
    ["更详细解释 LLM-MTR 的语义路由器", "修改实验结论，强调它只在稀疏任务提升"],
    ["新增“推理成本”比较维度", "详细比较 Text-MTL 和 PLE"],
    ["把研究空白改写成可执行实验", "修改第二节，减少概括性表述"]
  ][state.stage];
  return `<div class="conversation-prompts"><span>可以继续问</span>${prompts.map((prompt) => `<button type="button" data-prompt="${escapeHtml(prompt)}">${escapeHtml(prompt)}</button>`).join("")}</div>`;
}

function configureComposer() {
  if (state.stage === 0) {
    goalInput.disabled = false;
    goalInput.placeholder = "例如：帮我调研大模型如何增强多任务推荐，重点关注 2023 年后的方法和可复现方案。";
    goalInput.value = state.goal;
    sendButton.textContent = "开始规划";
  } else {
    goalInput.disabled = false;
    goalInput.value = "";
    goalInput.placeholder = "继续补充要求，Agent 会在当前阶段调整方案……";
    sendButton.textContent = "发送";
  }
}

function renderArtifact() {
  const renderers = [renderStart, renderPlan, renderApproval, renderPapers, renderProgress, renderDeepRead, renderCompare, renderReview];
  const stageId = STAGES[state.stage].id;
  const revision = state.stageRevisions[stageId];
  const revisionBanner = revision ? `<div class="revision-banner"><strong>已根据对话修改 · v2</strong><span>${escapeHtml(revision)}</span></div>` : "";
  const staleBanner = state.staleFromStage !== null && state.stage > state.staleFromStage
    ? `<div class="revision-banner stale"><strong>上游已修改</strong><span>“${STAGES[state.staleFromStage].label}”阶段发生变化，当前及后续产物需要重新生成。</span><button class="secondary-button" type="button" data-action="regenerate-downstream">重新生成</button></div>`
    : "";
  artifactContent.innerHTML = staleBanner + revisionBanner + renderers[state.stage]();
}

function heading(eyebrow, title, description, badge = "") {
  return `
    <div class="artifact-heading">
      <div><span class="eyebrow">${eyebrow}</span><h2>${title}</h2><p>${description}</p></div>
      ${badge ? `<span class="heading-badge">${badge}</span>` : ""}
    </div>`;
}

function renderStart() {
  return `
    <div class="artifact-empty">
      <div class="empty-inner">
        <span class="eyebrow">${escapeHtml(state.projectName)} · 新对话</span>
        <h2>这个对话要继续研究什么？</h2>
        <p>所有对话共享项目工作区和公共记忆，但各自保留独立的消息、计划与研究产物。</p>
        ${state.memoryPublished ? `<div class="memory-banner"><span class="memory-icon">MD</span><span><strong>已加载项目公共记忆</strong><small>${escapeHtml(state.memoryFile)} · 来自“主题调研到综述”</small></span><button class="text-button" type="button" data-action="show-memory">查看</button></div>` : `<div class="memory-banner muted"><span class="memory-icon">夹</span><span><strong>项目工作区已连接</strong><small>${escapeHtml(state.workspacePath)}</small></span><button class="text-button" type="button" data-action="show-project">设置</button></div>`}
        <div class="goal-examples">
          <button class="example-card" type="button" data-example="帮我调研大模型如何增强多任务推荐，重点关注 2023 年后的方法、稳定收益和可复现方案。">
            <strong>主题调研到综述</strong><span>围绕一个研究问题，完成检索、筛选、精读、比较和写作。</span>
          </button>
          <button class="example-card" type="button" data-example="精读这篇论文，解释它的核心方法、实验是否可靠，以及和 PLE 的主要区别。">
            <strong>单篇论文精读</strong><span>导入论文后，获得结构化解读和可回溯原文的证据。</span>
          </button>
          <button class="example-card" type="button" data-example="帮我找近三年关于 Evidence-grounded Research Agent 的代表论文，先给候选清单和筛选理由。">
            <strong>发现与筛选论文</strong><span>聚合多个来源、去重排序，再由你决定进入后续流程的论文。</span>
          </button>
          <button class="example-card" type="button" data-example="比较这几篇论文的方法假设、数据集、指标和结论冲突，并指出还没有被解决的问题。">
            <strong>比较已有论文</strong><span>从已有论文集合出发，形成证据化比较矩阵和研究空白。</span>
          </button>
        </div>
      </div>
    </div>`;
}

function renderPlan() {
  return `
    ${heading("步骤 1 · 研究计划", "先把问题定义清楚", "这些字段会约束后续检索、筛选和综述，不满意可以直接修改。", "尚未执行")}
    <section class="card">
      <div class="form-grid">
        <label class="field-label wide">核心研究问题
          <textarea rows="3" data-plan="question">${escapeHtml(state.plan.question)}</textarea>
          <span class="field-help">最终综述必须直接回答这个问题。</span>
        </label>
        <label class="field-label">时间范围
          <input data-plan="years" value="${escapeHtml(state.plan.years)}">
        </label>
        <label class="field-label">目标精读数量
          <select data-plan="paperCount">
            ${["4", "6", "8", "12"].map((value) => `<option ${state.plan.paperCount === value ? "selected" : ""}>${value}</option>`).join("")}
          </select>
        </label>
        <label class="field-label wide">重点关注
          <textarea rows="2" data-plan="focus">${escapeHtml(state.plan.focus)}</textarea>
        </label>
        <label class="field-label wide">交付形式
          <select data-plan="output">
            <option selected>证据化中文综述</option>
            <option>相关工作章节草稿</option>
            <option>研究方向与 gap 报告</option>
          </select>
        </label>
      </div>
    </section>
    <section class="card">
      <div class="section-row"><h3>Agent 拟定的执行步骤</h3><span class="badge teal">可调整</span></div>
      <ol>
        <li>聚合 arXiv、Semantic Scholar、Crossref，按标题与 DOI 去重。</li>
        <li>根据主题相关性、时间、发表渠道和全文可得性推荐候选论文。</li>
        <li>你确认后，对入选论文生成结构化精读卡和逐字证据。</li>
        <li>形成比较矩阵、共识/冲突/空白，再生成带引用的综述。</li>
      </ol>
    </section>
    <div class="action-bar"><span>确认计划只会进入操作授权页，不会立即联网。</span><div><button class="secondary-button" type="button" data-action="save-plan">保存修改</button><button class="primary-button" type="button" data-action="confirm-plan">确认研究计划</button></div></div>`;
}

function renderApproval() {
  const approvals = [
    { id: "search", title: "联网检索学术来源", detail: "发送检索词和筛选条件；不会发送本地论文正文。", badge: "联网" },
    { id: "download", title: "下载并索引入选论文", detail: "仅在论文筛选确认后执行；文件保存在本机。", badge: "本地写入" },
    { id: "model", title: "批量调用模型进行精读与综合", detail: "预计 6 篇论文、8–14 次模型调用；提交选中的文本片段。", badge: "模型额度" }
  ];
  return `
    ${heading("步骤 2 · 操作授权", "一次看清即将发生什么", "按影响集中确认，后续不会为同一批次的每篇论文反复打断你。", "3 项待确认")}
    <section class="card">
      <div class="approval-list">
        ${approvals.map((item) => `
          <label class="approval-item">
            <input type="checkbox" data-approval="${item.id}" ${state.approvals.includes(item.id) ? "checked" : ""}>
            <span><strong>${item.title}</strong><small>${item.detail}</small></span>
            <span class="badge ${item.id === "search" ? "blue" : item.id === "model" ? "amber" : "gray"}">${item.badge}</span>
          </label>`).join("")}
      </div>
    </section>
    <div class="cost-note"><strong>原型说明：</strong>本页只是验证授权逻辑。点击批准不会联网、下载文件或调用模型。</div>
    <div class="action-bar"><span id="approval-summary">已选择 ${state.approvals.length}/3 项</span><div><button class="secondary-button" type="button" data-stage="1">返回修改计划</button><button class="primary-button" type="button" data-action="approve-actions">批准并开始检索</button></div></div>`;
}

function renderPapers() {
  return `
    ${heading("步骤 3 · 候选论文", "Agent 推荐，你做最终决定", "24 篇候选已去重。这里展示 6 篇代表结果及其进入精读的理由。", "已选 ${state.selectedPapers.length} 篇")}
    <div class="toolbar">
      <div class="toolbar-group"><input class="search-mini" id="paper-filter" placeholder="筛选标题或标签"><button class="subtle-button" type="button" data-action="select-recommended">恢复 Agent 推荐</button></div>
      <div class="toolbar-group"><span class="badge gray">3 个来源</span><span class="badge teal">24 篇去重结果</span></div>
    </div>
    <div class="paper-list" id="paper-list">
      ${PAPERS.map((paper) => `
        <label class="paper-row" data-search="${escapeHtml(`${paper.title} ${paper.tag}`.toLowerCase())}">
          <input class="paper-check" type="checkbox" data-paper="${paper.id}" ${state.selectedPapers.includes(paper.id) ? "checked" : ""}>
          <span><h3>${paper.title}</h3><span class="paper-meta">${paper.year} · ${paper.venue} · <span class="badge gray">${paper.tag}</span></span><p class="paper-reason"><strong>推荐理由：</strong>${paper.reason}</p></span>
          <span class="score"><strong>${paper.score}</strong><small>相关性</small></span>
        </label>`).join("")}
    </div>
    <div class="action-bar"><span id="paper-selection-summary">已选择 ${state.selectedPapers.length} 篇；建议 4–8 篇。</span><div><button class="secondary-button" type="button" data-stage="2">返回授权</button><button class="primary-button" type="button" data-action="continue-papers" ${state.selectedPapers.length < 3 ? "disabled" : ""}>使用已选论文继续</button></div></div>`;
}

function renderProgress() {
  const labels = [
    ["准备全文", "下载、解析并建立可检索文本", "6/6"],
    ["结构化精读", "抽取问题、方法、实验、限制和证据", "5/5"],
    ["跨论文比较", "对齐方法假设、数据和结论", "9 个维度"],
    ["生成综述", "组织章节并校验引用覆盖", "4 个章节"]
  ];
  const complete = state.progressStep >= 4;
  return `
    ${heading("步骤 4 · 执行过程", complete ? "研究处理已经完成" : state.paused ? "任务已在安全点暂停" : "Agent 正在按计划执行", "离开页面不会丢失任务；失败时会保留已完成阶段，并给出可重试项。", complete ? "可以检查结果" : state.paused ? "已暂停" : "原型模拟中")}
    <div class="progress-layout">
      <section class="card">
        ${labels.map((item, index) => {
          const done = index < state.progressStep;
          const running = index === state.progressStep && !complete;
          return `<div class="progress-stage ${done ? "done" : running ? "running" : ""}"><span class="progress-icon">${done ? "✓" : index + 1}</span><span><strong>${item[0]}</strong><small>${item[1]}</small></span><span>${done ? item[2] : running ? state.paused ? "已暂停" : "进行中" : "等待"}</span></div>`;
        }).join("")}
      </section>
      <div>
        <section class="card"><div class="section-row"><h3>本次运行</h3><span class="badge ${complete ? "teal" : "blue"}">${complete ? "完成" : "运行中"}</span></div><div class="metric-list"><div class="metric"><span>入选论文</span><strong>${state.selectedPapers.length} 篇</strong></div><div class="metric"><span>证据片段</span><strong>${complete ? "47" : 12 + state.progressStep * 9} 条</strong></div><div class="metric"><span>模型调用</span><strong>${complete ? "11" : 2 + state.progressStep * 2} 次</strong></div><div class="metric"><span>失败项目</span><strong>0</strong></div></div></section>
        <section class="card"><h3>Agent 正在做什么</h3><p>${complete ? "已检查结构化输出、证据来源范围和引用覆盖，可以逐篇抽查。" : state.paused ? "所有已完成结果均已保留。恢复后从当前阶段继续。" : "仅使用你确认的论文，逐项保存可恢复的中间结果。"}</p></section>
      </div>
    </div>
    <div class="action-bar"><span>原型会自动模拟进度，也可以直接完成。</span><div>${complete ? "" : `<button class="secondary-button" type="button" data-action="toggle-pause">${state.paused ? "恢复任务" : "暂停任务"}</button><button class="secondary-button" type="button" data-action="quick-complete">快速完成演示</button>`}<button class="primary-button" type="button" data-action="open-deepread" ${complete ? "" : "disabled"}>检查精读结果</button></div></div>`;
}

function renderDeepRead() {
  return `
    ${heading("步骤 5 · 单篇精读", "先验证论文，再相信综合结论", "每个判断都附带证据入口；证据展示原文、页码和抽取来源。", "47 条证据")}
    <div class="deep-layout">
      <nav class="paper-index" aria-label="已精读论文"><button class="active" type="button">01 · LLM-MTR</button><button type="button">02 · TaskAlign</button><button type="button">03 · FM Survey</button><button type="button">04 · Text-MTL</button><button type="button">05 · PLE</button></nav>
      <div>
        <section class="card"><div class="section-row"><div><span class="eyebrow">核心论文 · 2025 KDD</span><h3>Large Language Models for Multi-Task Recommendation</h3></div><span class="badge teal">证据完整</span></div><p class="paper-meta">Zhang et al. · 18 页 · 引用元数据已核对</p></section>
        <section class="deep-card"><h3>研究问题</h3><p>在多个行为任务存在语义差异时，LLM 表征能否改善共享专家的任务路由，并缓解负迁移？</p></section>
        <section class="deep-card"><h3>核心方法</h3><div class="claim"><p>方法先用冻结语言模型编码任务描述和物品文本，再用语义路由器控制共享专家权重；训练阶段不更新语言模型参数。</p><button class="evidence-link" type="button" data-action="open-evidence">查看证据 E-014 · 第 4 页</button></div></section>
        <section class="deep-card"><h3>实验结论</h3><div class="claim"><p>在稀疏行为任务上提升更明显；高频点击任务与强 PLE 基线差距较小，不能概括为所有任务稳定提升。</p><button class="evidence-link" type="button" data-action="open-evidence">查看证据 E-021 · 第 9 页</button></div></section>
        <section class="deep-card"><h3>局限与复现风险</h3><p>未报告路由器对提示措辞的敏感性；两个私有数据集无法复现；公开数据实验缺少相同参数量的文本编码器消融。</p></section>
      </div>
    </div>
    <div class="action-bar"><span>建议至少抽查一条方法证据和一条实验结论。</span><div><button class="secondary-button" type="button" data-stage="4">返回运行</button><button class="primary-button" type="button" data-action="open-compare">进入跨论文比较</button></div></div>`;
}

function renderCompare() {
  const hasCostDimension = state.customDimensions.includes("推理成本");
  return `
    ${heading("步骤 6 · 跨论文比较", "把论文放进同一组问题里比较", "可以在左侧对话中追问任意单元格，或直接要求 Agent 新建比较维度。", `5 篇 · ${9 + state.customDimensions.length} 个维度`)}
    <div class="matrix-wrap"><table class="matrix"><thead><tr><th>论文</th><th>LLM 的角色</th><th>任务关系</th><th>主要收益</th>${hasCostDimension ? '<th class="new-dimension">推理成本 <span class="badge teal">对话新增</span></th>' : ""}<th>关键限制</th></tr></thead><tbody>
      <tr><td><strong>LLM-MTR</strong><br>2025</td><td>冻结语义编码器</td><td>语义路由</td><td>稀疏任务改善明显 <button class="text-button" data-action="open-evidence">[E-021]</button></td>${hasCostDimension ? '<td class="new-dimension">中等：离线编码，在线仅路由</td>' : ""}<td>私有数据、提示敏感性未测</td></tr>
      <tr><td><strong>TaskAlign</strong><br>2024</td><td>任务描述表示</td><td>对齐损失</td><td>减少任务冲突</td>${hasCostDimension ? '<td class="new-dimension">中等：增加对齐训练</td>' : ""}<td>训练成本较高</td></tr>
      <tr><td><strong>FM Survey</strong><br>2024</td><td>分类与综述</td><td>不适用</td><td>统一术语</td>${hasCostDimension ? '<td class="new-dimension">不适用</td>' : ""}<td>缺少统一实证</td></tr>
      <tr><td><strong>Text-MTL</strong><br>2023</td><td>轻量文本编码</td><td>共享底座</td><td>冷启动提升</td>${hasCostDimension ? '<td class="new-dimension">低：轻量编码器</td>' : ""}<td>并非完整 LLM</td></tr>
      <tr><td><strong>PLE</strong><br>2022</td><td>无</td><td>分层专家</td><td>强且低成本的基线</td>${hasCostDimension ? '<td class="new-dimension">低：无语言模型</td>' : ""}<td>缺少开放语义知识</td></tr>
    </tbody></table></div>
    <div class="insight-grid">
      <article class="insight-card"><strong>共识</strong><p>语义信息在数据稀疏或任务标签信息不足时更有价值。</p></article>
      <article class="insight-card"><strong>冲突</strong><p>LLM 是否稳定优于轻量文本编码器，现有实验证据不足。</p></article>
      <article class="insight-card"><strong>研究空白</strong><p>缺少控制参数量和推理成本后的公平比较与跨数据集复现。</p></article>
    </div>
    <section class="card" style="margin-top:14px"><div class="section-row"><h3>证据不足提醒</h3><span class="badge amber">1 项</span></div><p>“LLM 能减少所有多任务场景的负迁移”目前没有被所选论文共同支持，综述中不会作为确定结论。</p></section>
    <div class="action-bar"><span>综合结论会继承当前论文集合和证据版本。</span><div><button class="secondary-button" type="button" data-stage="5">返回精读</button><button class="primary-button" type="button" data-action="open-review">生成综述草稿</button></div></div>`;
}

function renderReview() {
  return `
    ${heading("步骤 7 · 证据化综述", "草稿是可编辑、可复用的项目记忆", "在对话中修改内容；确认后导出 MD，并让项目内后续对话以它为公共上下文。", state.memoryPublished ? "公共记忆已更新" : state.reviewEdited ? "草稿 v2" : "草稿 v1")}
    ${state.memoryPublished ? `<div class="memory-success"><span class="memory-icon">MD</span><span><strong>${escapeHtml(state.memoryFile)} 已成为项目公共记忆</strong><small>项目中的所有新对话都可以读取它；再次导出会创建新版本。</small></span><button class="secondary-button" type="button" data-action="new-chat">基于此记忆新建对话</button></div>` : ""}
    <div class="review-layout">
      <nav class="outline-index" aria-label="综述提纲"><button class="active" type="button">摘要</button><button type="button">1. 问题背景</button><button type="button">2. 方法分类</button><button type="button">3. 实证比较</button><button type="button">4. 空白与建议</button><button type="button">参考文献</button></nav>
      <article class="draft ${state.reviewEdited ? "edited" : ""}">
        <h2>大模型增强多任务推荐：方法、证据与研究空白</h2>
        <p><strong>摘要。</strong>近年的研究开始使用语言模型补充多任务推荐中的语义信息，但现有证据不支持“大模型在所有任务上普遍更优”的判断。其收益主要出现在行为稀疏、任务语义可描述或需要跨域知识迁移的场景 <button class="citation" data-action="open-evidence">[1, E-021]</button>。</p>
        <h3>1. 从共享参数到语义任务关系</h3>
        <p>PLE 等经典方法通过分层专家缓解任务之间的参数冲突，却主要依赖行为信号学习任务关系。新方法将任务描述、物品文本或领域知识编码为语义表示，用于控制专家路由或增加对齐约束 <button class="citation" data-action="open-evidence">[1, E-014]</button>。这种变化并没有取消传统架构，而是在其上增加语义条件。</p>
        <h3>2. 当前证据的边界</h3>
        <p>多篇工作报告了稀疏任务或冷启动场景的改善，但公开实验尚未充分控制参数量、预训练数据和推理成本。部分最强结果来自无法复现的私有数据。因此，更稳妥的结论是：语义增强具有条件性价值，而非替代所有轻量多任务基线。</p>
        <h3>3. 值得验证的研究空白</h3>
        <p>下一步应在统一数据划分和计算预算下，对比冻结 LLM、轻量文本编码器和无语义 PLE，并报告逐任务收益、负迁移率以及成本。该设计能区分收益究竟来自语言知识，还是来自额外参数与训练信号。</p>
      </article>
    </div>
    <div class="action-bar"><span>公共记忆保存在项目工作区，可由多个对话读取和继续修改。</span><div><button class="secondary-button" type="button" data-action="edit-markdown">编辑 Markdown</button><button class="secondary-button" type="button" data-action="export-docx">另存 DOCX</button><button class="primary-button memory-primary" type="button" data-action="publish-memory">${state.memoryPublished ? "导出新版 MD 并更新公共记忆" : "导出 MD 并设为项目公共记忆"}</button></div></div>`;
}

function advanceTo(index) {
  state.stage = index;
  state.maxVisited = Math.max(state.maxVisited, index);
  render();
}

function syncPlanFromDom() {
  document.querySelectorAll("[data-plan]").forEach((element) => {
    state.plan[element.dataset.plan] = element.value.trim();
  });
}

function syncApprovals() {
  state.approvals = [...document.querySelectorAll("[data-approval]:checked")].map((input) => input.dataset.approval);
  const summary = document.querySelector("#approval-summary");
  if (summary) summary.textContent = `已选择 ${state.approvals.length}/3 项`;
  saveState();
}

function syncPapers() {
  state.selectedPapers = [...document.querySelectorAll("[data-paper]:checked")].map((input) => input.dataset.paper);
  const summary = document.querySelector("#paper-selection-summary");
  const continueButton = document.querySelector("[data-action='continue-papers']");
  if (summary) summary.textContent = `已选择 ${state.selectedPapers.length} 篇；建议 4–8 篇。`;
  if (continueButton) continueButton.disabled = state.selectedPapers.length < 3;
  saveState();
}

function startProgressTimer() {
  if (progressTimer) return;
  progressTimer = window.setInterval(() => {
    if (state.stage !== 4 || state.paused || state.progressStep >= 4) {
      stopProgressTimer();
      return;
    }
    state.progressStep += 1;
    render();
  }, 1400);
}

function stopProgressTimer() {
  if (progressTimer) {
    window.clearInterval(progressTimer);
    progressTimer = null;
  }
}

function openModal(content) {
  modal.innerHTML = content;
  modalBackdrop.hidden = false;
  document.body.style.overflow = "hidden";
  const closeButton = modal.querySelector("[data-action='close-modal']");
  if (closeButton) closeButton.focus();
}

function closeModal() {
  modalBackdrop.hidden = true;
  modal.innerHTML = "";
  document.body.style.overflow = "";
}

function openFeedback() {
  const template = document.querySelector("#feedback-template");
  openModal(template.innerHTML);
  const stage = STAGES[state.stage];
  document.querySelector("#feedback-screen").textContent = `P${state.stage + 1} · ${stage.title}`;
  document.querySelector("#feedback-input").value = state.feedback[stage.id] || "";
}

function feedbackText() {
  const entries = STAGES
    .filter((stage) => state.feedback[stage.id])
    .map((stage, index) => `${index + 1}. [P${STAGES.indexOf(stage) + 1} ${stage.title}]\n${state.feedback[stage.id]}`);
  return entries.length ? `PRAgent Web V1 原型反馈\n\n${entries.join("\n\n")}` : "";
}

function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) {
    return navigator.clipboard.writeText(text);
  }
  const helper = document.createElement("textarea");
  helper.value = text;
  helper.style.position = "fixed";
  helper.style.opacity = "0";
  document.body.appendChild(helper);
  helper.select();
  document.execCommand("copy");
  helper.remove();
  return Promise.resolve();
}

function openEvidence() {
  openModal(`
    <div class="modal-header"><div><span class="eyebrow">原文证据 E-021</span><h2 id="modal-title">稀疏任务上的收益更明显</h2></div><button class="close-button" type="button" data-action="close-modal" aria-label="关闭">×</button></div>
    <blockquote class="evidence-quote">“The improvements are most pronounced on sparse conversion behaviors, while gains on the high-frequency click task remain marginal compared with PLE.”</blockquote>
    <div class="source-detail"><div><span>论文</span><strong>Large Language Models for Multi-Task Recommendation</strong></div><div><span>位置</span><strong>第 9 页 · Results</strong></div><div><span>证据状态</span><strong>逐字匹配 · 当前版本</strong></div><div><span>用途</span><strong>实验结论与综述摘要</strong></div></div>
    <p class="modal-copy">正式产品中，这里还会显示 PDF 页面定位和前后文，并允许固定为研究笔记。</p>
    <div class="modal-actions"><button class="secondary-button" type="button" data-action="pin-note">固定为笔记</button><button class="primary-button" type="button" data-action="close-modal">返回结果</button></div>`);
}

function openSettings() {
  openModal(`
    <div class="modal-header"><div><span class="eyebrow">原型范围外</span><h2 id="modal-title">模型与数据设置</h2></div><button class="close-button" type="button" data-action="close-modal">×</button></div>
    <p class="modal-copy">设置不会成为研究流程的主入口。正式产品计划在这里管理模型、默认预算、数据目录和隐私边界。</p>
    <section class="card"><div class="section-row"><h3>默认研究模型</h3><span class="badge teal">deepseek-v4-flash</span></div><p>仅为原型展示，本页面不会读取你的 .env。</p></section>
    <section class="card"><div class="section-row"><h3>论文与产物</h3><span class="badge gray">保存在本机</span></div><p>原型不会访问现有论文库或 SQLite 数据库。</p></section>
    <div class="modal-actions"><button class="primary-button" type="button" data-action="close-modal">知道了</button></div>`);
}

function openRunDetail() {
  openModal(`
    <div class="modal-header"><div><span class="eyebrow">研究任务运行详情</span><h2 id="modal-title">当前任务可以恢复</h2></div><button class="close-button" type="button" data-action="close-modal">×</button></div>
    <div class="source-detail"><div><span>当前阶段</span><strong>${STAGES[state.stage].label}</strong></div><div><span>已选论文</span><strong>${state.selectedPapers.length} 篇</strong></div><div><span>联网授权</span><strong>${state.approvals.includes("search") ? "已批准" : "未批准"}</strong></div><div><span>原型运行</span><strong>demo-web-v1-001</strong></div></div>
    <p class="modal-copy">正式产品中，此处显示运行日志、模型调用、失败原因、取消和恢复入口；普通用户不需要先进入独立“任务中心”。</p>
    <div class="modal-actions"><button class="primary-button" type="button" data-action="close-modal">返回</button></div>`);
}

function openProjectDialog(isNew = false) {
  openModal(`
    <div class="modal-header"><div><span class="eyebrow">${isNew ? "新建项目" : "项目设置"}</span><h2 id="modal-title">${isNew ? "先选择项目工作区" : escapeHtml(state.projectName)}</h2></div><button class="close-button" type="button" data-action="close-modal">×</button></div>
    <p class="modal-copy">工作区用于保存下载的论文、中间文件、导出结果和项目公共记忆。一个项目可以包含多个独立对话。</p>
    <label class="field-label">项目名称<input id="project-name-input" value="${isNew ? "" : escapeHtml(state.projectName)}" placeholder="例如：多任务推荐论文研究"></label>
    <label class="field-label" style="margin-top:12px">项目工作区<div class="path-input-row"><input id="workspace-input" value="${isNew ? "D:\\Research\\new-paper-project" : escapeHtml(state.workspacePath)}" placeholder="选择或输入本地文件夹"><button class="secondary-button" type="button" data-action="choose-workspace">选择文件夹</button></div></label>
    <div class="workspace-preview"><span class="memory-icon">夹</span><span><strong>这个目录归项目所有</strong><small>不会把其他项目的论文、记忆或对话混入当前上下文。</small></span></div>
    <div class="modal-actions"><button class="secondary-button" type="button" data-action="close-modal">取消</button><button class="primary-button" type="button" data-action="${isNew ? "create-project-confirm" : "save-project-settings"}">${isNew ? "创建项目并进入" : "保存项目设置"}</button></div>`);
}

function openNewChatDialog() {
  openModal(`
    <div class="modal-header"><div><span class="eyebrow">${escapeHtml(state.projectName)}</span><h2 id="modal-title">在项目中新建对话</h2></div><button class="close-button" type="button" data-action="close-modal">×</button></div>
    <p class="modal-copy">新对话有独立的消息和研究阶段，但会自动读取项目公共记忆与工作区文件。</p>
    ${state.memoryPublished ? `<div class="memory-banner"><span class="memory-icon">MD</span><span><strong>将加载 ${escapeHtml(state.memoryFile)}</strong><small>由前一个研究对话生成，可继续修改或深入研究。</small></span></div>` : `<div class="cost-note">当前项目还没有公共记忆。新对话仍可以读取工作区中的论文和文件。</div>`}
    <label class="field-label">对话名称<input id="chat-title-input" value="基于综述继续研究" placeholder="例如：验证研究空白"></label>
    <div class="modal-actions"><button class="secondary-button" type="button" data-action="close-modal">取消</button><button class="primary-button" type="button" data-action="create-chat-confirm">创建对话</button></div>`);
}

function openProjectContext() {
  const accessibleStages = STAGES.slice(0, state.maxVisited + 1).map((stage) => stage.label).join("、");
  openModal(`
    <div class="modal-header"><div><span class="eyebrow">对话可用上下文</span><h2 id="modal-title">当前及历史阶段均可追问和修改</h2></div><button class="close-button" type="button" data-action="close-modal">×</button></div>
    <div class="source-detail"><div><span>当前对话阶段</span><strong>${STAGES[state.stage].label}</strong></div><div><span>可访问阶段</span><strong>${accessibleStages}</strong></div><div><span>项目公共记忆</span><strong>${state.memoryPublished ? state.memoryFile : "尚未创建"}</strong></div><div><span>项目工作区</span><strong>${escapeHtml(state.workspacePath)}</strong></div></div>
    <p class="modal-copy">对话修改某个历史阶段后，依赖该阶段的后续产物会标记为需要重新生成；原型用“已根据对话修改 · v2”展示这种关系。</p>
    <div class="modal-actions"><button class="primary-button" type="button" data-action="close-modal">返回对话</button></div>`);
}

function openMemory() {
  if (!state.memoryPublished) {
    openModal(`<div class="modal-header"><div><span class="eyebrow">项目公共记忆</span><h2 id="modal-title">尚未创建公共记忆</h2></div><button class="close-button" type="button" data-action="close-modal">×</button></div><p class="modal-copy">完成综述后，使用突出按钮“导出 MD 并设为项目公共记忆”。之后项目中的所有新对话都会加载它。</p><div class="modal-actions"><button class="primary-button" type="button" data-action="close-modal">知道了</button></div>`);
    return;
  }
  openModal(`
    <div class="modal-header"><div><span class="eyebrow">项目公共记忆</span><h2 id="modal-title">${escapeHtml(state.memoryFile)}</h2></div><button class="close-button" type="button" data-action="close-modal">×</button></div>
    <div class="source-detail"><div><span>保存位置</span><strong>${escapeHtml(state.workspacePath)}\\${escapeHtml(state.memoryFile)}</strong></div><div><span>更新时间</span><strong>${escapeHtml(state.memoryUpdatedAt)}</strong></div><div><span>可见范围</span><strong>当前项目的所有对话</strong></div><div><span>来源</span><strong>主题调研到综述</strong></div></div>
    <pre class="memory-preview">${escapeHtml(state.memoryContent)}</pre>
    <div class="modal-actions"><button class="secondary-button" type="button" data-action="edit-memory">修改公共记忆</button><button class="primary-button" type="button" data-action="new-chat">基于记忆新建对话</button></div>`);
}

function openMarkdownEditor(editingMemory = false) {
  const content = editingMemory ? state.memoryContent : state.markdownDraft;
  openModal(`
    <div class="modal-header"><div><span class="eyebrow">Markdown 编辑器</span><h2 id="modal-title">${editingMemory ? "修改项目公共记忆" : "修改综述草稿"}</h2></div><button class="close-button" type="button" data-action="close-modal">×</button></div>
    <p class="modal-copy">这里的修改会保存为新版本；公共记忆更新后，新建对话读取最新版本。</p>
    <textarea class="markdown-editor" id="markdown-editor" rows="20">${escapeHtml(content)}</textarea>
    <div class="modal-actions"><button class="secondary-button" type="button" data-action="close-modal">取消</button><button class="primary-button" type="button" data-action="${editingMemory ? "save-memory-edit" : "save-markdown"}">保存为新版本</button></div>`);
}

function downloadMarkdown() {
  const blob = new Blob([state.markdownDraft], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = state.memoryFile;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function publishMemory() {
  state.memoryPublished = true;
  state.memoryContent = state.markdownDraft;
  state.memoryUpdatedAt = new Date().toLocaleString("zh-CN", { hour12: false });
  state.chatMeta[state.activeChatId].status = "已写入项目公共记忆";
  downloadMarkdown();
  render();
  showToast("MD 已导出，并设为项目公共记忆");
}

function respondInContext(text) {
  let answer = "我已结合当前阶段和之前的研究记录回答。正式产品会把相关产物与证据一并带入本轮对话。";
  const stageId = STAGES[state.stage].id;
  const isModification = /修改|改成|改为|补充|删除|强调|减少|增加/.test(text);
  const historicalStageKeywords = [
    [0, /目标|研究问题/],
    [1, /计划|时间范围|交付形式/],
    [2, /授权|确认|联网|下载/],
    [3, /论文筛选|候选论文|入选论文/],
    [4, /执行|任务阶段/],
    [5, /精读|实验结论|方法结论/],
    [6, /对比|比较矩阵/],
    [7, /综述|草稿|章节/]
  ];
  const requestedStage = historicalStageKeywords.find(([index, pattern]) => index <= state.stage && pattern.test(text));
  const targetStageIndex = isModification && requestedStage ? requestedStage[0] : state.stage;
  const targetStageId = STAGES[targetStageIndex].id;

  if (state.stage === 5 && /路由|细节|详细/.test(text)) {
    answer = "LLM-MTR 并不在每次推荐时调用大模型。它离线编码任务描述和物品文本，再把语义向量交给门控网络；门控网络为每个任务生成专家权重。论文第 4 页的 E-014 支持这一结构，第 9 页的 E-021 只支持稀疏任务收益，不能外推到所有任务。";
  } else if (state.stage === 6 && /新增|维度|成本/.test(text)) {
    if (!state.customDimensions.includes("推理成本")) state.customDimensions.push("推理成本");
    state.stageRevisions.compare = "通过对话新增“推理成本”维度，并回填 5 篇论文。";
    answer = "已新增“推理成本”维度，并从精读卡与方法描述中回填 5 篇论文。无法得到精确数值的论文只标记相对等级，不编造成本数据。";
  } else if (state.stage === 6 && /Text-MTL|PLE|详细|比较/.test(text)) {
    answer = "Text-MTL 和 PLE 都避免在线调用大模型。Text-MTL 额外使用轻量文本编码器，主要改善冷启动；PLE 只依赖行为信号，计算更低且是更严格的非语义基线。当前证据不足以断言 Text-MTL 在高密度任务上稳定优于 PLE。";
  } else if (isModification) {
    state.stageRevisions[targetStageId] = text;
    if (targetStageIndex < state.stage) state.staleFromStage = targetStageIndex;
    if (targetStageIndex === 7) state.reviewEdited = true;
    answer = `已修改“${STAGES[targetStageIndex].label}”阶段产物并保存为 v2。${targetStageIndex < state.stage ? `当前“${STAGES[state.stage].label}”及后续产物已标记为需要重新生成。` : "依赖它的后续阶段会继承新版本。"}原始版本仍可回看。`;
  } else if (/为什么|之前|检索|选择|第 5 篇/.test(text)) {
    answer = "我查阅了之前的计划与论文筛选记录：第 5 篇属于尚未正式发表的跨域工作，主题相关但证据权重较低，因此没有进入默认精读集合。你仍可以回到“论文”阶段将它加入，并重新生成受影响的后续产物。";
  } else if (state.memoryPublished && /记忆|结论|项目/.test(text)) {
    answer = `当前对话已经加载 ${state.memoryFile}。它保存了前一对话的研究问题、入选论文、关键证据、比较结论和研究空白；我会把它当作项目背景，但不会把其中的推断冒充为新证据。`;
  }

  state.extraMessages.push({ stage: state.stage, role: "user", text, meta: "你 · 追问或修改" });
  state.extraMessages.push({ stage: state.stage, role: "agent", text: answer, meta: `研究 Agent · 已读取目标至${STAGES[state.stage].label}阶段` });
  goalInput.value = "";
  render();
}

document.addEventListener("click", (event) => {
  const chatButton = event.target.closest("[data-chat]");
  if (chatButton) {
    restoreChatSnapshot(chatButton.dataset.chat);
    return;
  }

  const stageButton = event.target.closest("[data-stage]");
  if (stageButton && !stageButton.disabled) {
    const index = Number(stageButton.dataset.stage);
    if (index <= state.maxVisited) advanceTo(index);
    return;
  }

  const example = event.target.closest("[data-example]");
  if (example) {
    state.goal = example.dataset.example;
    goalInput.value = state.goal;
    goalInput.focus();
    showToast("示例已填入左侧输入框");
    saveState();
    return;
  }

  const promptButton = event.target.closest("[data-prompt]");
  if (promptButton) {
    goalInput.value = promptButton.dataset.prompt;
    composer.requestSubmit();
    return;
  }

  const actionElement = event.target.closest("[data-action]");
  if (!actionElement) return;
  const action = actionElement.dataset.action;

  if (action === "reset") {
    if (window.confirm("重置原型进度和本地反馈？此操作不会影响正式数据。")) {
      stopProgressTimer();
      state = cloneDefaultState();
      localStorage.removeItem(STORAGE_KEY);
      closeModal();
      render();
      showToast("原型已重置");
    }
  } else if (action === "new-project") {
    openProjectDialog(true);
  } else if (action === "show-project") {
    openProjectDialog(false);
  } else if (action === "choose-workspace") {
    document.querySelector("#workspace-input").value = "D:\\Research\\paper-agent-web-v1";
    showToast("原型：已选择示例工作区");
  } else if (action === "create-project-confirm") {
    const name = document.querySelector("#project-name-input").value.trim();
    const workspace = document.querySelector("#workspace-input").value.trim();
    if (!name || !workspace) {
      showToast("项目名称和工作区都不能为空");
      return;
    }
    createProject(name, workspace);
  } else if (action === "save-project-settings") {
    const name = document.querySelector("#project-name-input").value.trim();
    const workspace = document.querySelector("#workspace-input").value.trim();
    if (!name || !workspace) {
      showToast("项目名称和工作区都不能为空");
      return;
    }
    state.projectName = name;
    state.workspacePath = workspace;
    render();
    closeModal();
    showToast("项目设置已保存");
  } else if (action === "new-chat") {
    openNewChatDialog();
  } else if (action === "create-chat-confirm") {
    const title = document.querySelector("#chat-title-input").value.trim();
    createChat(title);
  } else if (action === "show-context") {
    openProjectContext();
  } else if (action === "show-memory") {
    openMemory();
  } else if (action === "edit-memory") {
    openMarkdownEditor(true);
  } else if (action === "edit-markdown") {
    openMarkdownEditor(false);
  } else if (action === "save-markdown") {
    state.markdownDraft = document.querySelector("#markdown-editor").value;
    state.reviewEdited = true;
    state.stageRevisions.review = "已在 Markdown 编辑器中修改综述草稿。";
    render();
    closeModal();
    showToast("综述草稿已保存为 v2");
  } else if (action === "save-memory-edit") {
    state.memoryContent = document.querySelector("#markdown-editor").value;
    state.markdownDraft = state.memoryContent;
    state.memoryUpdatedAt = new Date().toLocaleString("zh-CN", { hour12: false });
    state.reviewEdited = true;
    render();
    closeModal();
    showToast("项目公共记忆已更新");
  } else if (action === "publish-memory") {
    publishMemory();
  } else if (action === "regenerate-downstream") {
    state.stageRevisions[STAGES[state.stage].id] = `已基于“${STAGES[state.staleFromStage].label}”阶段 v2 重新生成。`;
    state.staleFromStage = null;
    render();
    showToast("当前及后续产物已基于上游修改重新生成");
  } else if (action === "show-settings") {
    openSettings();
  } else if (action === "show-run-detail") {
    openRunDetail();
  } else if (action === "save-plan") {
    syncPlanFromDom();
    render();
    showToast("计划修改已保存");
  } else if (action === "confirm-plan") {
    syncPlanFromDom();
    advanceTo(2);
  } else if (action === "approve-actions") {
    syncApprovals();
    if (!state.approvals.includes("search")) {
      showToast("需要批准联网检索，才能进入候选论文页");
      return;
    }
    advanceTo(3);
  } else if (action === "select-recommended") {
    state.selectedPapers = ["p1", "p2", "p3", "p4", "p6"];
    render();
    showToast("已恢复 Agent 推荐的 5 篇论文");
  } else if (action === "continue-papers") {
    syncPapers();
    if (state.selectedPapers.length < 3) {
      showToast("至少选择 3 篇才能形成比较与综述");
      return;
    }
    state.progressStep = 0;
    state.paused = false;
    advanceTo(4);
  } else if (action === "toggle-pause") {
    state.paused = !state.paused;
    render();
    showToast(state.paused ? "已在当前阶段安全暂停" : "任务已恢复");
  } else if (action === "quick-complete") {
    state.progressStep = 4;
    state.paused = false;
    render();
    showToast("演示任务已完成");
  } else if (action === "open-deepread") {
    if (state.progressStep >= 4) advanceTo(5);
  } else if (action === "open-compare") {
    advanceTo(6);
  } else if (action === "open-review") {
    advanceTo(7);
  } else if (action === "open-evidence") {
    openEvidence();
  } else if (action === "pin-note") {
    closeModal();
    showToast("原型：证据已固定为研究笔记");
  } else if (action === "open-feedback") {
    openFeedback();
  } else if (action === "save-feedback") {
    state.feedback[STAGES[state.stage].id] = document.querySelector("#feedback-input").value.trim();
    saveState();
    closeModal();
    showToast("本页反馈已保存");
  } else if (action === "copy-feedback") {
    const text = feedbackText();
    if (!text) {
      showToast("还没有保存任何反馈");
      return;
    }
    copyText(text).then(() => showToast("全部反馈已复制"));
  } else if (action === "close-modal") {
    closeModal();
  } else if (action === "export-docx") {
    showToast("原型：DOCX 另存入口有效，未生成真实文件");
  }
});

document.addEventListener("change", (event) => {
  if (event.target.matches("[data-approval]")) syncApprovals();
  if (event.target.matches("[data-paper]")) syncPapers();
});

document.addEventListener("input", (event) => {
  if (event.target.id === "paper-filter") {
    const query = event.target.value.trim().toLowerCase();
    document.querySelectorAll(".paper-row").forEach((row) => {
      row.hidden = query && !row.dataset.search.includes(query);
    });
  }
});

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = goalInput.value.trim();
  if (!text) {
    showToast("先输入一个研究目标或补充要求");
    goalInput.focus();
    return;
  }

  if (state.stage === 0) {
    state.goal = text;
    if (text.includes("多任务") || text.includes("推荐")) {
      state.plan.question = "大模型如何增强多任务推荐系统，并在哪些条件下带来稳定收益？";
    } else {
      state.plan.question = text.replace(/[。！？!?]$/, "") + "？";
    }
    advanceTo(1);
  } else {
    respondInContext(text);
  }
});

goalInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    composer.requestSubmit();
  }
});

modalBackdrop.addEventListener("click", (event) => {
  if (event.target === modalBackdrop) closeModal();
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !modalBackdrop.hidden) closeModal();
});

render();
