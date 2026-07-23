(function () {
  "use strict";

  const NODE_WIDTH = 188;
  const NODE_HEIGHT = 94;
  const MAX_LOG_EVENTS = 1200;
  const DEFAULT_PROGRESS_REMINDER_MINUTES = 5;
  const PROGRESS_REMINDER_STORAGE_KEY = "daaam.progressReminderMinutes";
  const PROGRESS_MILESTONES = [25, 50, 75, 90];
  const DEPTH_REFRESH_INTERVAL_MS = 1600;
  const TERMINAL_STATUSES = new Set([
    "completed",
    "failed",
    "cancelled",
    "canceled",
    "stopped",
    "aborted",
    "error",
    "success",
    "done",
    "resumed",
  ]);

  const CATEGORY_META = {
    input: { label: "数据输入", color: "#6aa7ff" },
    data: { label: "数据输入", color: "#6aa7ff" },
    ingest: { label: "数据接入", color: "#6aa7ff" },
    depth: { label: "深度估计", color: "#48cbe5" },
    model: { label: "模型推理", color: "#48cbe5" },
    perception: { label: "视觉感知", color: "#48cbe5" },
    segmentation: { label: "实例分割", color: "#48cbe5" },
    tracking: { label: "跨帧跟踪", color: "#56d5c3" },
    assignment: { label: "关键帧", color: "#8bd6aa" },
    grounding: { label: "语义理解", color: "#a78bfa" },
    semantic: { label: "语义理解", color: "#a78bfa" },
    embedding: { label: "语义嵌入", color: "#c18df2" },
    mapping: { label: "三维建图", color: "#f4a261" },
    geometry: { label: "三维几何", color: "#f4a261" },
    fusion: { label: "几何融合", color: "#f4a261" },
    hydra: { label: "Hydra 图谱", color: "#f4a261" },
    region: { label: "区域聚类", color: "#eecb68" },
    room: { label: "房间生成", color: "#eecb68" },
    quality: { label: "质量验收", color: "#56d58c" },
    artifact: { label: "产物提交", color: "#56d58c" },
    output: { label: "产物输出", color: "#56d58c" },
    dynamic: { label: "动态隔离", color: "#eecb68" },
    service: { label: "查询服务", color: "#6aa7ff" },
    default: { label: "处理模块", color: "#8293aa" },
  };

  const STATUS_LABELS = {
    idle: "待机",
    pending: "等待",
    queued: "已排队",
    waiting: "等待",
    ready: "就绪",
    running: "运行中",
    active: "运行中",
    processing: "处理中",
    starting: "启动中",
    stopping: "停止中",
    completed: "已完成",
    complete: "已完成",
    success: "已完成",
    passed: "已通过",
    done: "已完成",
    failed: "失败",
    error: "异常",
    warning: "警告",
    cancelled: "已取消",
    canceled: "已取消",
    stopped: "已停止",
    skipped: "已跳过",
    succeeded: "已完成",
    resumed: "已恢复",
    stale: "状态陈旧",
    planned: "已规划",
    blocked: "被阻断",
    unknown: "未知",
  };

  const state = {
    workflows: [],
    activeWorkflow: null,
    activeRun: null,
    activeRunId: null,
    processId: null,
    snapshot: null,
    selectedNodeId: null,
    nodeRuntime: new Map(),
    parameters: {},
    parameterDefinitions: [],
    selectedPresetId: null,
    parametersDirty: false,
    command: "",
    commandWarnings: [],
    events: [],
    eventKeys: new Set(),
    eventCursor: 0,
    logLevel: "all",
    activeCategories: new Set(),
    graphLayout: null,
    view: { x: 0, y: 0, scale: 1, manual: false },
    drag: null,
    previewSequence: 0,
    previewTimer: null,
    pollTimer: null,
    pollGeneration: 0,
    pollFailures: 0,
    pollCount: 0,
    workflowSwitching: false,
    starting: false,
    stopping: false,
    lastSync: null,
    elapsedTimer: null,
    resizeTimer: null,
    progressReminderMinutes: DEFAULT_PROGRESS_REMINDER_MINUTES,
    progressReminder: {
      runKey: "",
      armed: false,
      nextHeartbeatAt: 0,
      milestones: new Set(),
      terminalNotified: false,
      wasLong: false,
      titlePending: false,
    },
    depthPreview: {
      runId: null,
      frames: [],
      index: -1,
      followLatest: true,
      available: false,
      complete: false,
      live: false,
      source: null,
      minimumDepthM: 0.25,
      maximumDepthM: 5.0,
      loading: false,
      error: "",
      dismissedRunId: null,
      requestSequence: 0,
      indexSequence: 0,
      refreshTimer: null,
      objectUrl: null,
      frameStats: null,
    },
  };

  const dom = {};

  function cacheDom() {
    const ids = [
      "active-workflow-name",
      "global-status",
      "global-status-label",
      "header-run-id",
      "header-progress",
      "header-elapsed",
      "preview-button",
      "start-button",
      "stop-button",
      "workflow-select",
      "workflow-description",
      "workflow-node-count",
      "workflow-edge-count",
      "refresh-runs-button",
      "run-list",
      "preset-list",
      "preset-count",
      "progress-reminder-select",
      "progress-reminder-indicator",
      "progress-reminder-copy",
      "connection-card",
      "connection-title",
      "connection-copy",
      "canvas-workflow-name",
      "canvas-caption",
      "category-filters",
      "zoom-out-button",
      "zoom-reset-button",
      "zoom-in-button",
      "fit-view-button",
      "dag-viewport",
      "dag-scene",
      "edge-layer",
      "node-layer",
      "canvas-loading",
      "canvas-empty",
      "depth-preview-window",
      "depth-preview-status",
      "close-depth-preview",
      "depth-preview-image",
      "depth-preview-loading",
      "depth-preview-empty",
      "depth-preview-error",
      "depth-preview-meta",
      "depth-preview-valid",
      "depth-preview-previous",
      "depth-preview-next",
      "depth-preview-counter",
      "depth-preview-follow",
      "depth-legend-minimum",
      "depth-legend-maximum",
      "stage-summary",
      "last-updated-time",
      "left-panel",
      "right-panel",
      "open-left-panel",
      "close-left-panel",
      "open-right-panel",
      "close-right-panel",
      "panel-scrim",
      "detail-tab",
      "parameters-tab",
      "detail-tab-panel",
      "parameters-tab-panel",
      "parameter-dirty-dot",
      "node-detail-empty",
      "node-detail-content",
      "reset-parameters-button",
      "parameter-form",
      "no-parameters",
      "inline-command",
      "inline-command-warnings",
      "copy-command-inline",
      "bottom-dock",
      "event-count-label",
      "autoscroll-checkbox",
      "clear-logs-button",
      "collapse-dock-button",
      "log-filters",
      "log-stream",
      "log-placeholder",
      "live-indicator",
      "quality-summary-label",
      "quality-score",
      "quality-list",
      "toast-region",
      "command-dialog",
      "close-command-dialog",
      "dialog-command",
      "dialog-command-warnings",
      "preview-status",
      "copy-command-dialog",
      "start-from-dialog",
    ];
    ids.forEach((id) => {
      dom[toCamelCase(id)] = document.getElementById(id);
    });
    dom.workspaceGrid = document.querySelector(".workspace-grid");
    dom.gridBackdrop = document.querySelector(".grid-backdrop");
  }

  function toCamelCase(value) {
    return String(value).replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
  }

  function createElement(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function createSvgElement(tag, attributes) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.entries(attributes || {}).forEach(([name, value]) => {
      if (value !== undefined && value !== null) node.setAttribute(name, String(value));
    });
    return node;
  }

  function safeString(value, fallback) {
    if (value === undefined || value === null || value === "") return fallback || "";
    if (typeof value === "string") return value;
    if (typeof value === "number" || typeof value === "boolean") return String(value);
    try {
      return JSON.stringify(value);
    } catch (_error) {
      return fallback || "";
    }
  }

  function finiteNumber(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  function objectEntriesAsArray(value, nameKey) {
    if (Array.isArray(value)) return value;
    if (!value || typeof value !== "object") return [];
    return Object.entries(value).map(([key, item]) => {
      if (item && typeof item === "object" && !Array.isArray(item)) {
        return { [nameKey || "id"]: key, ...item };
      }
      return { [nameKey || "id"]: key, value: item };
    });
  }

  function shortId(value) {
    const text = safeString(value, "—");
    if (text.length <= 15) return text;
    return `${text.slice(0, 7)}…${text.slice(-5)}`;
  }

  function truncate(value, length) {
    const text = safeString(value, "");
    if (text.length <= length) return text;
    return `${text.slice(0, Math.max(0, length - 1))}…`;
  }

  function normalizeStatus(value) {
    let raw = value;
    if (raw && typeof raw === "object") {
      raw = raw.status || raw.state || raw.phase || raw.result || raw.value;
    }
    const status = safeString(raw, "idle").toLowerCase().replace(/[\s-]+/g, "_");
    if (["in_progress", "started", "executing", "busy"].includes(status)) return "running";
    if (["complete", "succeeded", "successful", "pass", "passed", "ok", "done"].includes(status)) {
      return "completed";
    }
    if (["failure", "errored", "fatal"].includes(status)) return "failed";
    if (["canceling", "cancelling", "terminating"].includes(status)) return "stopping";
    if (["not_started", "not_run", "created"].includes(status)) return "pending";
    return status || "idle";
  }

  function statusClass(value) {
    const status = normalizeStatus(value);
    if (["running", "starting", "active", "processing"].includes(status)) return "running";
    if (["completed", "success", "passed", "done", "resumed"].includes(status)) return "completed";
    if (["failed", "error", "aborted", "cancelled", "canceled", "blocked"].includes(status)) return "failed";
    if (["warning", "stopping", "stale"].includes(status)) return status === "stale" ? "warning" : status;
    return "pending";
  }

  function statusLabel(value) {
    const status = normalizeStatus(value);
    return STATUS_LABELS[status] || safeString(value, "未知");
  }

  function isRunningStatus(value) {
    return ["running", "starting", "active", "processing", "stopping"].includes(normalizeStatus(value));
  }

  function isTerminalStatus(value) {
    return TERMINAL_STATUSES.has(normalizeStatus(value));
  }

  function categoryKey(value) {
    const raw = safeString(value, "default").toLowerCase().replace(/[\s-]+/g, "_");
    if (CATEGORY_META[raw]) return raw;
    if (raw.includes("depth") || raw.includes("stereo")) return "depth";
    if (raw.includes("segment") || raw.includes("fastsam") || raw.includes("mask")) return "segmentation";
    if (raw.includes("track") || raw.includes("sort") || raw.includes("reid")) return "tracking";
    if (raw.includes("assign") || raw.includes("keyframe") || raw.includes("clip")) return "assignment";
    if (raw.includes("ground") || raw.includes("dam") || raw.includes("describe")) return "grounding";
    if (raw.includes("embed") || raw.includes("sentence")) return "embedding";
    if (raw.includes("hydra") || raw.includes("khronos")) return "hydra";
    if (raw.includes("map") || raw.includes("fusion") || raw.includes("mesh") || raw.includes("dsg")) return "mapping";
    if (raw.includes("room")) return "room";
    if (raw.includes("region") || raw.includes("cluster")) return "region";
    if (raw.includes("quality") || raw.includes("gate") || raw.includes("valid")) return "quality";
    if (raw.includes("input") || raw.includes("source") || raw.includes("dataset")) return "input";
    if (raw.includes("semantic")) return "semantic";
    if (raw.includes("output") || raw.includes("artifact")) return "output";
    return raw || "default";
  }

  function categoryMeta(value) {
    const key = categoryKey(value);
    return CATEGORY_META[key] || {
      label: safeString(value, CATEGORY_META.default.label),
      color: CATEGORY_META.default.color,
    };
  }

  async function apiRequest(path, options) {
    const requestOptions = { ...(options || {}) };
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), requestOptions.timeout || 18000);
    delete requestOptions.timeout;
    requestOptions.headers = {
      Accept: "application/json",
      ...(requestOptions.body !== undefined ? { "Content-Type": "application/json" } : {}),
      ...(requestOptions.headers || {}),
    };
    if (requestOptions.body !== undefined && typeof requestOptions.body !== "string") {
      requestOptions.body = JSON.stringify(requestOptions.body);
    }
    requestOptions.signal = controller.signal;

    try {
      const response = await fetch(path, requestOptions);
      const raw = await response.text();
      let payload = null;
      if (raw) {
        try {
          payload = JSON.parse(raw);
        } catch (_error) {
          payload = raw;
        }
      }
      if (!response.ok) {
        const detail =
          payload && typeof payload === "object"
            ? payload.detail || payload.message || payload.error
            : payload;
        const error = new Error(safeString(detail, `请求失败（HTTP ${response.status}）`));
        error.status = response.status;
        error.payload = payload;
        throw error;
      }
      return payload || {};
    } catch (error) {
      if (error && error.name === "AbortError") throw new Error("请求超时，请检查服务状态");
      throw error;
    } finally {
      window.clearTimeout(timeout);
    }
  }

  function setConnection(mode, title, copy) {
    dom.connectionCard.classList.toggle("is-online", mode === "online");
    dom.connectionCard.classList.toggle("is-offline", mode === "offline");
    dom.connectionTitle.textContent = title;
    dom.connectionCopy.textContent = copy;
  }

  function showToast(message, level, duration) {
    const toast = createElement("div", `toast toast-${level || "info"}`, message);
    dom.toastRegion.appendChild(toast);
    window.setTimeout(() => toast.remove(), duration || 4200);
  }

  function formatDateTime(value, options) {
    if (!value) return "—";
    let date;
    if (typeof value === "number" && value < 100000000000) date = new Date(value * 1000);
    else date = new Date(value);
    if (Number.isNaN(date.getTime())) return safeString(value, "—");
    const opts = options || {
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    };
    return new Intl.DateTimeFormat("zh-CN", opts).format(date);
  }

  function formatClock(value) {
    if (!value) return formatDateTime(Date.now(), { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
    return formatDateTime(value, { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
  }

  function formatDuration(milliseconds) {
    const total = Math.max(0, Math.floor(Number(milliseconds || 0) / 1000));
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    const seconds = total % 60;
    if (hours > 0) {
      return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
    }
    return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
  }

  function formatValue(value) {
    if (value === undefined || value === null || value === "") return "—";
    if (typeof value === "boolean") return value ? "是" : "否";
    if (typeof value === "number") {
      if (!Number.isFinite(value)) return "—";
      if (Math.abs(value) >= 1000) return new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 1 }).format(value);
      if (!Number.isInteger(value)) return Number(value.toFixed(3)).toString();
      return String(value);
    }
    if (Array.isArray(value)) return value.map((item) => formatValue(item)).join(", ");
    if (typeof value === "object") return truncate(safeString(value, "—"), 160);
    return String(value);
  }

  function normalizeWorkflow(raw, index) {
    const source = raw && typeof raw === "object" ? raw : {};
    const id = safeString(source.id || source.workflow_id || source.slug, `workflow-${index + 1}`);
    const nodes = objectEntriesAsArray(source.nodes || source.modules || source.stages, "id").map((node, nodeIndex) =>
      normalizeNode(node, nodeIndex),
    );
    const nodeIds = new Set(nodes.map((node) => node.id));
    const edges = objectEntriesAsArray(source.edges || source.dependencies || source.links, "id")
      .map(normalizeEdge)
      .filter((edge) => edge.source && edge.target && nodeIds.has(edge.source) && nodeIds.has(edge.target));
    return {
      ...source,
      id,
      name: safeString(source.name || source.label || source.title, id),
      description: safeString(source.description || source.summary, "该流程暂无说明。"),
      runnable: source.runnable !== false && Boolean(source.runner || source.runnable),
      nodes,
      edges,
      parameterGroups: normalizeParameterGroups(source.parameter_groups || source.parameterGroups || source.parameters),
      presets: normalizePresets(source.presets || source.parameter_presets),
    };
  }

  function normalizeNode(raw, index) {
    const source = raw && typeof raw === "object" ? raw : {};
    const id = safeString(source.id || source.node_id || source.name || source.key, `node-${index + 1}`);
    const position = source.position && typeof source.position === "object" ? source.position : {};
    return {
      ...source,
      id,
      label: safeString(source.label || source.name || source.title, id),
      category: categoryKey(source.category || source.kind || source.type || id),
      description: safeString(source.description || source.summary || source.help, "该模块暂无详细说明。"),
      statusHint: safeString(source.status_hint || source.statusHint || source.hint, "运行时状态会在流程图和控制台中同步更新。"),
      x: finiteNumber(source.x !== undefined ? source.x : position.x),
      y: finiteNumber(source.y !== undefined ? source.y : position.y),
      metrics: source.metrics || source.metric_hints || {},
      inputs: source.inputs || source.input || [],
      outputs: source.outputs || source.output || [],
      parameters: source.parameters || source.parameter_ids || [],
    };
  }

  function normalizeEdge(raw) {
    const source = raw && typeof raw === "object" ? raw : {};
    const from = source.source || source.from || source.upstream || source.parent;
    const to = source.target || source.to || source.downstream || source.child;
    return {
      source: safeString(from && typeof from === "object" ? from.id : from, ""),
      target: safeString(to && typeof to === "object" ? to.id : to, ""),
      kind: safeString(source.kind || source.type, "data").toLowerCase(),
      label: safeString(source.label || source.description, ""),
    };
  }

  function normalizeParameterGroups(rawGroups) {
    if (!rawGroups) return [];
    let groups = objectEntriesAsArray(rawGroups, "id");
    const looksLikeFields = groups.length > 0 && groups.every((item) => {
      return item && (item.name || item.key || item.path || item.type) && !item.parameters && !item.fields && !item.items;
    });
    if (looksLikeFields) groups = [{ id: "general", name: "通用参数", parameters: groups }];
    return groups.map((group, groupIndex) => {
      const source = group && typeof group === "object" ? group : {};
      const fields = objectEntriesAsArray(source.parameters || source.fields || source.items || source.options, "name");
      return {
        id: safeString(source.id || source.key || source.name, `group-${groupIndex + 1}`),
        name: safeString(source.label || source.name || source.title, `参数组 ${groupIndex + 1}`),
        description: safeString(source.description || source.help, ""),
        nodeId: safeString(source.node_id || source.nodeId || source.module_id, ""),
        parameters: fields.map((field, fieldIndex) => normalizeParameter(field, fieldIndex)),
      };
    });
  }

  function normalizeParameter(raw, index) {
    const source = raw && typeof raw === "object" ? raw : {};
    const key = safeString(source.name || source.key || source.id || source.path || source.flag, `parameter_${index + 1}`);
    const rawType = safeString(source.type || source.kind || source.value_type, "string").toLowerCase();
    let type = rawType;
    if (["bool", "flag", "switch"].includes(type)) type = "boolean";
    if (["int", "long"].includes(type)) type = "integer";
    if (["float", "double", "decimal"].includes(type)) type = "number";
    if (["enum", "choice", "dropdown"].includes(type)) type = "select";
    if (["str", "text", "path", "file", "directory"].includes(type)) type = "string";
    const options = objectEntriesAsArray(source.options || source.choices || source.enum, "value").map((option) => {
      if (option && typeof option === "object") {
        const value = option.value !== undefined ? option.value : option.id !== undefined ? option.id : option.name;
        return { value, label: safeString(option.label || option.name, formatValue(value)) };
      }
      return { value: option, label: formatValue(option) };
    });
    if (options.length && type === "string") type = "select";
    let defaultValue = source.default;
    if (defaultValue === undefined) defaultValue = source.default_value;
    if (defaultValue === undefined) defaultValue = source.value;
    if (defaultValue === undefined) defaultValue = type === "boolean" ? false : "";
    return {
      ...source,
      key,
      label: safeString(source.label || source.title || source.name, key),
      description: safeString(source.description || source.help || source.hint, ""),
      type,
      defaultValue,
      options,
      min: finiteNumber(source.min !== undefined ? source.min : source.minimum),
      max: finiteNumber(source.max !== undefined ? source.max : source.maximum),
      step: finiteNumber(source.step),
      placeholder: safeString(source.placeholder || source.example, ""),
      required: Boolean(source.required),
    };
  }

  function normalizePresets(rawPresets) {
    return objectEntriesAsArray(rawPresets, "id").map((preset, index) => {
      const source = preset && typeof preset === "object" ? preset : {};
      return {
        ...source,
        id: safeString(source.id || source.key || source.name, `preset-${index + 1}`),
        name: safeString(source.label || source.name || source.title, `方案 ${index + 1}`),
        description: safeString(source.description || source.summary, ""),
        parameters: source.parameters || source.values || source.overrides || source.config || {},
        isDefault: Boolean(source.default || source.is_default || source.recommended),
      };
    });
  }

  async function loadWorkflows() {
    dom.canvasLoading.classList.remove("is-hidden");
    dom.canvasEmpty.classList.add("is-hidden");
    try {
      const payload = await apiRequest("/api/workflows");
      const rawWorkflows = objectEntriesAsArray(payload.workflows || payload.items || payload, "id");
      state.workflows = rawWorkflows.map(normalizeWorkflow);
      if (!state.workflows.length) throw new Error("服务端没有返回可用的构建流程");
      populateWorkflowSelect();
      const defaultId = safeString(payload.default_workflow || payload.defaultWorkflow, "");
      const initial = state.workflows.find((workflow) => workflow.id === defaultId) || state.workflows[0];
      setConnection("online", "服务连接正常", `${state.workflows.length} 个流程可用`);
      await activateWorkflow(initial.id);
    } catch (error) {
      console.error(error);
      setConnection("offline", "服务连接失败", truncate(error.message, 72));
      dom.canvasLoading.classList.add("is-hidden");
      dom.canvasEmpty.classList.remove("is-hidden");
      dom.canvasEmpty.querySelector("strong").textContent = "无法载入构建流程";
      dom.canvasEmpty.querySelector("span").textContent = error.message;
      dom.workflowSelect.innerHTML = "";
      const option = createElement("option", "", "服务不可用");
      dom.workflowSelect.appendChild(option);
      dom.workflowSelect.disabled = true;
      showToast(`载入失败：${error.message}`, "error", 6500);
    }
  }

  function populateWorkflowSelect() {
    dom.workflowSelect.innerHTML = "";
    state.workflows.forEach((workflow) => {
      const option = createElement("option", "", workflow.name);
      option.value = workflow.id;
      dom.workflowSelect.appendChild(option);
    });
    dom.workflowSelect.disabled = false;
  }

  async function activateWorkflow(workflowId) {
    const workflow = state.workflows.find((item) => item.id === workflowId);
    if (!workflow) return;
    stopPolling();
    state.activeWorkflow = workflow;
    state.activeRun = null;
    state.activeRunId = null;
    state.processId = null;
    state.snapshot = null;
    state.selectedNodeId = null;
    state.nodeRuntime = new Map();
    state.events = [];
    state.eventKeys = new Set();
    state.eventCursor = 0;
    resetProgressReminder(false);
    resetDepthPreview(null);
    state.activeCategories = new Set(workflow.nodes.map((node) => node.category));
    state.view.manual = false;
    dom.workflowSelect.value = workflow.id;
    dom.workflowDescription.textContent = workflow.description;
    dom.workflowNodeCount.textContent = String(workflow.nodes.length);
    dom.workflowEdgeCount.textContent = String(workflow.edges.length);
    dom.activeWorkflowName.textContent = workflow.name;
    dom.canvasWorkflowName.textContent = workflow.name;
    restoreDocumentTitle();
    initializeParameters();
    renderPresets();
    renderParameterForm();
    renderCategoryFilters();
    renderGraph();
    renderNodeDetail();
    renderLogs();
    renderQualityGates([]);
    updateRunUi("idle");
    if (workflow.runnable) scheduleCommandPreview(0);
    else {
      state.command = "此流程为只读参考流程，当前版本不会从面板启动这些后处理节点。";
      state.commandWarnings = normalizeWarnings(workflow.warnings);
      renderCommandPreview();
      dom.previewStatus.textContent = "只读流程";
    }
    closePanels();
    await loadRuns({ selectLatest: true });
  }

  function initializeParameters(presetId) {
    if (!state.activeWorkflow) return;
    state.parameterDefinitions = state.activeWorkflow.parameterGroups.flatMap((group) => group.parameters);
    const defaults = {};
    state.parameterDefinitions.forEach((definition) => {
      defaults[definition.key] = cloneValue(definition.defaultValue);
    });
    let preset = null;
    if (presetId) preset = state.activeWorkflow.presets.find((item) => item.id === presetId);
    if (!preset) preset = state.activeWorkflow.presets.find((item) => item.isDefault) || state.activeWorkflow.presets[0] || null;
    state.selectedPresetId = preset ? preset.id : null;
    state.parameters = { ...defaults, ...(preset ? flattenPresetParameters(preset.parameters) : {}) };
    state.parametersDirty = false;
    dom.parameterDirtyDot.classList.add("is-hidden");
  }

  function cloneValue(value) {
    if (!value || typeof value !== "object") return value;
    try {
      return JSON.parse(JSON.stringify(value));
    } catch (_error) {
      return value;
    }
  }

  function flattenPresetParameters(value) {
    if (!value || typeof value !== "object" || Array.isArray(value)) return {};
    const known = new Set(state.parameterDefinitions.map((definition) => definition.key));
    const result = {};
    Object.entries(value).forEach(([key, item]) => {
      if (known.has(key) || !item || typeof item !== "object" || Array.isArray(item)) {
        result[key] = cloneValue(item);
      } else {
        Object.entries(item).forEach(([nestedKey, nestedValue]) => {
          if (known.has(nestedKey)) result[nestedKey] = cloneValue(nestedValue);
          else if (known.has(`${key}.${nestedKey}`)) result[`${key}.${nestedKey}`] = cloneValue(nestedValue);
        });
      }
    });
    return result;
  }

  function renderPresets() {
    dom.presetList.innerHTML = "";
    const presets = state.activeWorkflow ? state.activeWorkflow.presets : [];
    dom.presetCount.textContent = String(presets.length);
    if (!presets.length) {
      dom.presetList.appendChild(createElement("p", "empty-compact", "当前流程没有预设方案。"));
      return;
    }
    presets.forEach((preset) => {
      const button = createElement("button", "preset-item");
      button.type = "button";
      if (preset.id === state.selectedPresetId) button.classList.add("is-active");
      button.dataset.presetId = preset.id;
      const top = createElement("div", "preset-item-top");
      top.appendChild(createElement("strong", "", preset.name));
      top.appendChild(createElement("span", "preset-check", "✓"));
      button.appendChild(top);
      if (preset.description) button.appendChild(createElement("p", "", preset.description));
      button.addEventListener("click", () => applyPreset(preset.id));
      dom.presetList.appendChild(button);
    });
  }

  function applyPreset(presetId) {
    if (isCurrentRunActive()) {
      showToast("运行期间不能切换参数方案", "warning");
      return;
    }
    initializeParameters(presetId);
    renderPresets();
    renderParameterForm();
    scheduleCommandPreview(0);
    showToast(`已应用参数方案：${getSelectedPreset()?.name || presetId}`, "info", 2600);
  }

  function getSelectedPreset() {
    if (!state.activeWorkflow) return null;
    return state.activeWorkflow.presets.find((item) => item.id === state.selectedPresetId) || null;
  }

  function renderParameterForm() {
    dom.parameterForm.innerHTML = "";
    const groups = state.activeWorkflow ? state.activeWorkflow.parameterGroups : [];
    const hasFields = groups.some((group) => group.parameters.length);
    dom.noParameters.classList.toggle("is-hidden", hasFields);
    dom.parameterForm.classList.toggle("is-hidden", !hasFields);
    if (!hasFields) return;

    groups.forEach((group) => {
      if (!group.parameters.length) return;
      const section = createElement("section", "parameter-group");
      const heading = createElement("div", "parameter-group-heading");
      heading.appendChild(createElement("h3", "", group.name));
      if (group.description) heading.appendChild(createElement("p", "", group.description));
      section.appendChild(heading);
      group.parameters.forEach((definition) => section.appendChild(renderParameterField(definition)));
      dom.parameterForm.appendChild(section);
    });
  }

  function renderParameterField(definition) {
    const wrapper = createElement("div", "parameter-field");
    const id = `parameter-${definition.key.replace(/[^a-zA-Z0-9_-]/g, "-")}`;
    const labelRow = createElement("div", "parameter-label-row");
    const label = createElement("label", "", definition.label);
    label.htmlFor = id;
    labelRow.appendChild(label);
    const labelMeta = createElement("span", "parameter-label-meta");
    if (definition.hard_gate) labelMeta.appendChild(createElement("em", "is-gate", "硬门"));
    if (definition.advanced) labelMeta.appendChild(createElement("em", "", "高级"));
    labelMeta.appendChild(createElement("code", "", `${safeString(definition.source, "CLI")} · ${definition.key}`));
    labelRow.appendChild(labelMeta);
    wrapper.appendChild(labelRow);
    const control = createElement("div", "parameter-control");
    const value = state.parameters[definition.key];
    const disabled = isCurrentRunActive();

    if (definition.type === "boolean") {
      const row = createElement("div", "boolean-control");
      row.appendChild(createElement("span", "", value ? "已启用" : "已关闭"));
      const switchLabel = createElement("label", "toggle-switch");
      const input = document.createElement("input");
      input.id = id;
      input.type = "checkbox";
      input.checked = Boolean(value);
      input.disabled = disabled;
      input.addEventListener("change", () => {
        updateParameter(definition, input.checked);
        row.firstElementChild.textContent = input.checked ? "已启用" : "已关闭";
      });
      switchLabel.appendChild(input);
      switchLabel.appendChild(createElement("span", "toggle-track"));
      row.appendChild(switchLabel);
      control.appendChild(row);
    } else if (definition.type === "select") {
      const select = document.createElement("select");
      select.id = id;
      select.disabled = disabled;
      definition.options.forEach((option) => {
        const optionElement = createElement("option", "", option.label);
        optionElement.value = serializeOptionValue(option.value);
        optionElement.dataset.encodedValue = serializeOptionValue(option.value);
        if (valuesEqual(value, option.value)) optionElement.selected = true;
        select.appendChild(optionElement);
      });
      select.addEventListener("change", () => {
        const selectedIndex = Math.max(0, select.selectedIndex);
        const selected = definition.options[selectedIndex];
        updateParameter(definition, selected ? cloneValue(selected.value) : select.value);
      });
      control.appendChild(select);
    } else if (["object", "array", "json", "textarea", "multiline"].includes(definition.type)) {
      const textarea = document.createElement("textarea");
      textarea.id = id;
      textarea.disabled = disabled;
      textarea.placeholder = definition.placeholder;
      textarea.value = typeof value === "string" ? value : JSON.stringify(value, null, 2);
      textarea.addEventListener("input", () => {
        if (["object", "array", "json"].includes(definition.type)) {
          try {
            const parsed = JSON.parse(textarea.value);
            textarea.removeAttribute("aria-invalid");
            updateParameter(definition, parsed);
          } catch (_error) {
            textarea.setAttribute("aria-invalid", "true");
          }
        } else {
          updateParameter(definition, textarea.value);
        }
      });
      control.appendChild(textarea);
    } else {
      const input = document.createElement("input");
      input.id = id;
      input.type = definition.type === "integer" || definition.type === "number" ? "number" : "text";
      input.disabled = disabled;
      input.required = definition.required;
      input.placeholder = definition.placeholder;
      if (definition.min !== null) input.min = String(definition.min);
      if (definition.max !== null) input.max = String(definition.max);
      if (definition.step !== null) input.step = String(definition.step);
      else if (definition.type === "integer") input.step = "1";
      input.value = value === undefined || value === null ? "" : String(value);
      input.addEventListener("input", () => {
        let nextValue = input.value;
        if (definition.type === "integer") nextValue = input.value === "" ? null : Number.parseInt(input.value, 10);
        if (definition.type === "number") nextValue = input.value === "" ? null : Number(input.value);
        updateParameter(definition, nextValue);
      });
      control.appendChild(input);
    }
    wrapper.appendChild(control);
    if (definition.description) wrapper.appendChild(createElement("p", "parameter-help", definition.description));
    return wrapper;
  }

  function serializeOptionValue(value) {
    try {
      return JSON.stringify(value);
    } catch (_error) {
      return String(value);
    }
  }

  function valuesEqual(left, right) {
    if (left === right) return true;
    return serializeOptionValue(left) === serializeOptionValue(right);
  }

  function updateParameter(definition, value) {
    state.parameters[definition.key] = value;
    state.parametersDirty = true;
    dom.parameterDirtyDot.classList.remove("is-hidden");
    scheduleCommandPreview(320);
  }

  function resetParameters() {
    if (isCurrentRunActive()) {
      showToast("运行期间不能重置参数", "warning");
      return;
    }
    initializeParameters(state.selectedPresetId);
    renderParameterForm();
    renderPresets();
    scheduleCommandPreview(0);
    showToast("参数已恢复到当前方案", "info", 2500);
  }

  function commandPayload() {
    return {
      workflow_id: state.activeWorkflow ? state.activeWorkflow.id : null,
      preset_id: state.selectedPresetId,
      parameters: { ...state.parameters },
    };
  }

  function scheduleCommandPreview(delay) {
    window.clearTimeout(state.previewTimer);
    state.previewTimer = window.setTimeout(refreshCommandPreview, delay === undefined ? 300 : delay);
  }

  async function refreshCommandPreview() {
    if (!state.activeWorkflow) return;
    if (!state.activeWorkflow.runnable) {
      state.command = "此流程为只读参考流程，当前版本不会从面板启动这些后处理节点。";
      state.commandWarnings = normalizeWarnings(state.activeWorkflow.warnings);
      renderCommandPreview();
      dom.previewStatus.textContent = "只读流程";
      return;
    }
    const sequence = ++state.previewSequence;
    dom.previewStatus.textContent = "正在生成命令…";
    try {
      const payload = await apiRequest("/api/commands/preview", {
        method: "POST",
        body: commandPayload(),
      });
      if (sequence !== state.previewSequence) return;
      let command = safeString(payload.command || payload.shell_command, "");
      if (!command && Array.isArray(payload.argv)) command = payload.argv.map(shellQuote).join(" ");
      if (!command) command = "服务端未返回命令文本";
      state.command = command;
      state.commandWarnings = normalizeWarnings(payload.warnings || payload.warning);
      renderCommandPreview();
      dom.previewStatus.textContent = `已按当前参数生成 · ${formatClock(Date.now())}`;
    } catch (error) {
      if (sequence !== state.previewSequence) return;
      state.command = `命令预览不可用：${error.message}`;
      state.commandWarnings = [];
      renderCommandPreview();
      dom.previewStatus.textContent = "预览生成失败，但仍可尝试启动";
    }
  }

  function shellQuote(value) {
    const text = String(value);
    if (/^[a-zA-Z0-9_./:=+-]+$/.test(text)) return text;
    return `'${text.replace(/'/g, `'"'"'`)}'`;
  }

  function normalizeWarnings(raw) {
    if (!raw) return [];
    if (Array.isArray(raw)) return raw.map((item) => safeString(item && item.message ? item.message : item, "")).filter(Boolean);
    if (typeof raw === "object") return Object.values(raw).map((item) => safeString(item && item.message ? item.message : item, "")).filter(Boolean);
    return [String(raw)];
  }

  function renderCommandPreview() {
    const inlineCode = dom.inlineCommand.querySelector("code");
    inlineCode.textContent = state.command || "等待命令预览…";
    dom.dialogCommand.textContent = state.command || "等待命令预览…";
    renderWarnings(dom.inlineCommandWarnings, state.commandWarnings);
    renderWarnings(dom.dialogCommandWarnings, state.commandWarnings);
  }

  function renderWarnings(container, warnings) {
    container.innerHTML = "";
    container.classList.toggle("is-hidden", !warnings.length);
    warnings.forEach((warning) => container.appendChild(createElement("div", "command-warning", warning)));
  }

  function showCommandDialog() {
    refreshCommandPreview();
    if (typeof dom.commandDialog.showModal === "function") dom.commandDialog.showModal();
    else dom.commandDialog.setAttribute("open", "");
  }

  function closeCommandDialog() {
    if (typeof dom.commandDialog.close === "function") dom.commandDialog.close();
    else dom.commandDialog.removeAttribute("open");
  }

  async function copyCommand() {
    if (!state.command) return;
    try {
      if (navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(state.command);
      else {
        const textarea = document.createElement("textarea");
        textarea.value = state.command;
        textarea.style.position = "fixed";
        textarea.style.opacity = "0";
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand("copy");
        textarea.remove();
      }
      showToast("命令已复制到剪贴板", "info", 2200);
    } catch (error) {
      showToast(`复制失败：${error.message}`, "error");
    }
  }

  function normalizeRun(raw, index) {
    const source = raw && typeof raw === "object" ? raw : {};
    const id = safeString(source.id || source.run_id || source.name, `run-${index + 1}`);
    return {
      ...source,
      id,
      processId: safeString(source.process_id || (source.process && source.process.id), ""),
      status: normalizeStatus(source.status || source.state || source.phase || "unknown"),
      startedAt: source.started_at || source.start_time || source.created_at || source.timestamp || source.modified_at,
      endedAt: source.ended_at || source.end_time || source.completed_at,
      progress: extractProgress(source),
    };
  }

  async function loadRuns(options) {
    if (!state.activeWorkflow) return;
    const workflowId = state.activeWorkflow.id;
    if (!state.activeRunId) {
      dom.runList.innerHTML = "";
      const skeleton = createElement("div", "skeleton-stack");
      skeleton.setAttribute("aria-hidden", "true");
      skeleton.append(createElement("span"), createElement("span"), createElement("span"));
      dom.runList.appendChild(skeleton);
    }
    try {
      const payload = await apiRequest(`/api/runs?workflow_id=${encodeURIComponent(workflowId)}`);
      if (!state.activeWorkflow || state.activeWorkflow.id !== workflowId) return;
      const runs = objectEntriesAsArray(payload.runs || payload.items || payload, "id").map(normalizeRun);
      if (
        state.activeRun &&
        state.activeRun.workflow_id !== workflowId &&
        state.activeWorkflow.id === workflowId
      ) {
        state.activeRun.workflow_id = workflowId;
      }
      if (state.activeRun && !runs.some((run) => run.id === state.activeRun.id)) runs.unshift(state.activeRun);
      state.activeWorkflow.runs = runs;
      renderRuns();
      const shouldSelectLatest = options && options.selectLatest && !state.activeRunId;
      if (shouldSelectLatest && runs.length) await selectRun(runs[0].id, { silent: true });
    } catch (error) {
      dom.runList.innerHTML = "";
      dom.runList.appendChild(createElement("p", "runs-error", `无法读取运行记录：${error.message}`));
    }
  }

  function renderRuns() {
    dom.runList.innerHTML = "";
    const runs = (state.activeWorkflow && state.activeWorkflow.runs) || [];
    if (!runs.length) {
      dom.runList.appendChild(createElement("p", "empty-compact", "暂无运行记录。启动后会显示在这里。"));
      return;
    }
    runs.forEach((run) => {
      const button = createElement("button", "run-item");
      button.type = "button";
      button.setAttribute("role", "option");
      button.setAttribute("aria-selected", String(run.id === state.activeRunId));
      button.dataset.runId = run.id;
      if (run.id === state.activeRunId) button.classList.add("is-active");
      const top = createElement("div", "run-item-top");
      top.appendChild(createElement("strong", "", safeString(run.name || run.label, shortId(run.id))));
      top.appendChild(createElement("span", `run-mini-status status-${statusClass(run.status)}`));
      button.appendChild(top);
      const meta = createElement("span", "run-item-meta", `${statusLabel(run.status)} · ${formatDateTime(run.startedAt)}`);
      button.appendChild(meta);
      button.addEventListener("click", () => selectRun(run.id));
      dom.runList.appendChild(button);
    });
  }

  async function selectRun(runId, options) {
    if (!state.activeWorkflow) return;
    const run = ((state.activeWorkflow && state.activeWorkflow.runs) || []).find((item) => item.id === runId) || {
      id: runId,
      status: "unknown",
    };
    stopPolling();
    state.activeRunId = run.id;
    state.activeRun = run;
    state.processId = run.processId || null;
    state.snapshot = null;
    state.nodeRuntime = new Map();
    state.events = [];
    state.eventKeys = new Set();
    state.eventCursor = 0;
    resetProgressReminder(false);
    resetDepthPreview(run.id);
    renderRuns();
    renderLogs();
    updateRunUi(run.status);
    try {
      const payload = await apiRequest(
        `/api/runs/${encodeURIComponent(run.id)}?workflow_id=${encodeURIComponent(state.activeWorkflow.id)}`,
      );
      if (state.activeRunId !== run.id) return;
      applySnapshot(payload);
      if (isCurrentRunActive()) {
        if (state.processId) startPolling();
        else scheduleSnapshotPolling();
      }
      if (!(options && options.silent)) showToast(`已载入运行 ${shortId(run.id)}`, "info", 2000);
    } catch (error) {
      addLocalEvent("warning", `无法读取运行快照：${error.message}`);
      showToast(`运行快照读取失败：${error.message}`, "error");
    }
  }

  function snapshotBody(payload) {
    if (!payload || typeof payload !== "object") return {};
    return payload.snapshot && typeof payload.snapshot === "object" ? { ...payload, ...payload.snapshot } : payload;
  }

  function applySnapshot(payload) {
    const snapshot = snapshotBody(payload);
    state.snapshot = snapshot;
    const runSource = snapshot.run && typeof snapshot.run === "object" ? snapshot.run : snapshot;
    const processSource = snapshot.process && typeof snapshot.process === "object" ? snapshot.process : {};
    if (state.activeRun) {
      state.activeRun = {
        ...state.activeRun,
        ...runSource,
        id: state.activeRun.id,
        status: normalizeStatus(runSource.status || runSource.state || state.activeRun.status),
        startedAt:
          processSource.started_at ||
          runSource.started_at ||
          runSource.start_time ||
          state.activeRun.startedAt,
        endedAt:
          runSource.ended_at ||
          runSource.end_time ||
          runSource.completed_at ||
          processSource.ended_at ||
          state.activeRun.endedAt,
      };
    }
    const processId =
      snapshot.process_id ||
      (snapshot.process && (snapshot.process.id || snapshot.process.process_id)) ||
      (state.activeRun && (state.activeRun.process_id || state.activeRun.processId));
    if (processId) state.processId = String(processId);
    if (processId && isRunningStatus(runSource.status || processSource.status || state.activeRun?.status)) {
      armProgressReminder();
    }
    state.nodeRuntime = extractNodeRuntime(snapshot);
    const historicalEvents = snapshot.events || snapshot.logs || snapshot.timeline;
    if (historicalEvents) appendEvents(objectEntriesAsArray(historicalEvents, "id"));
    appendSnapshotWarnings(snapshot);
    const cursor = finiteNumber(snapshot.next_cursor || snapshot.event_cursor || snapshot.cursor);
    if (cursor !== null) state.eventCursor = cursor;
    const gates = extractQualityGates(snapshot);
    renderQualityGates(gates, snapshot);
    refreshGraphRuntime();
    renderNodeDetail();
    renderLogs();
    state.lastSync = new Date();
    updateLastSync();
    updateRunUi(state.activeRun ? state.activeRun.status : snapshot.status);
    renderRuns();
    syncDepthPreviewVisibility();
  }

  function extractNodeRuntime(snapshot) {
    const runtime = new Map();
    const candidates = [
      snapshot.node_states,
      snapshot.node_statuses,
      snapshot.modules,
      snapshot.stages,
      snapshot.nodes,
      snapshot.module_status,
      snapshot.pipeline && snapshot.pipeline.nodes,
    ];
    candidates.forEach((candidate) => {
      objectEntriesAsArray(candidate, "id").forEach((item) => {
        const source = item && typeof item === "object" ? item : { value: item };
        const id = safeString(source.id || source.node_id || source.module_id || source.stage_id || source.name, "");
        if (!id) return;
        const existing = runtime.get(id) || {};
        runtime.set(id, {
          ...existing,
          ...source,
          id,
          status: normalizeStatus(source.status || source.state || source.phase || source.value || existing.status || "pending"),
          metrics: source.metrics || source.stats || existing.metrics || {},
        });
      });
    });
    return runtime;
  }

  function extractProgress(source) {
    if (!source || typeof source !== "object") return null;
    const raw = source.progress;
    if (typeof raw === "number") return raw <= 1 ? raw * 100 : raw;
    if (raw && typeof raw === "object") {
      const percent = finiteNumber(raw.percent || raw.percentage || raw.value);
      if (percent !== null) return percent <= 1 ? percent * 100 : percent;
      const current = finiteNumber(raw.current || raw.completed || raw.done || raw.processed);
      const total = finiteNumber(raw.total || raw.expected);
      if (current !== null && total) return (current / total) * 100;
    }
    const current = finiteNumber(
      source.completed_frames || source.frames_completed || source.processed_frames || source.frames_processed || source.completed,
    );
    const total = finiteNumber(source.total_frames || source.frames_total || source.expected_frames || source.total);
    if (current !== null && total) return (current / total) * 100;
    return null;
  }

  function currentProgress() {
    const snapshotProgress = extractProgress(state.snapshot || {});
    if (snapshotProgress !== null) return snapshotProgress;
    return extractProgress(state.activeRun || {});
  }

  function currentElapsedMilliseconds() {
    if (!state.activeRun) return null;
    const process = state.snapshot && state.snapshot.process && typeof state.snapshot.process === "object"
      ? state.snapshot.process
      : {};
    const startRaw =
      state.activeRun.startedAt ||
      state.activeRun.started_at ||
      process.started_at ||
      state.snapshot?.started_at;
    const endRaw =
      state.activeRun.endedAt ||
      state.activeRun.ended_at ||
      process.ended_at ||
      state.snapshot?.ended_at;
    const start = parseTimestamp(startRaw);
    const end = endRaw ? parseTimestamp(endRaw) : Date.now();
    if (start) return Math.max(0, end - start);
    const duration = finiteNumber(
      state.snapshot?.duration_s ||
      state.snapshot?.elapsed_s ||
      process.elapsed_s ||
      state.activeRun.duration_s,
    );
    return duration === null ? null : Math.max(0, duration * 1000);
  }

  function resetProgressReminder(armed) {
    state.progressReminder = {
      runKey: state.activeRunId || "",
      armed: Boolean(armed),
      nextHeartbeatAt: 0,
      milestones: new Set(),
      terminalNotified: false,
      wasLong: false,
      titlePending: false,
    };
    restoreDocumentTitle();
    updateProgressReminderCopy();
  }

  function armProgressReminder() {
    if (!state.activeRunId || state.progressReminderMinutes <= 0) return;
    if (state.progressReminder.runKey !== state.activeRunId) resetProgressReminder(true);
    state.progressReminder.armed = true;
    updateProgressReminderCopy();
  }

  function restoreDocumentTitle() {
    if (state.activeWorkflow) document.title = `${state.activeWorkflow.name} · DAAAM 控制台`;
    else document.title = "DAAAM · 语义地图构建控制台";
    if (state.progressReminder) state.progressReminder.titlePending = false;
  }

  function readProgressReminderMinutes() {
    try {
      const stored = window.localStorage.getItem(PROGRESS_REMINDER_STORAGE_KEY);
      if (stored === null) return DEFAULT_PROGRESS_REMINDER_MINUTES;
      const value = Number(stored);
      if ([0, 5, 10, 20, 30, 60].includes(value)) return value;
    } catch (_error) {
      // localStorage can be unavailable under strict browser privacy settings.
    }
    return DEFAULT_PROGRESS_REMINDER_MINUTES;
  }

  function initializeProgressReminder() {
    state.progressReminderMinutes = readProgressReminderMinutes();
    dom.progressReminderSelect.value = String(state.progressReminderMinutes);
    resetProgressReminder(false);
  }

  function updateProgressReminderCopy() {
    if (!dom.progressReminderCopy) return;
    const minutes = state.progressReminderMinutes;
    const armed = Boolean(state.progressReminder.armed && isCurrentRunActive());
    dom.progressReminderIndicator.classList.toggle("is-enabled", minutes > 0);
    dom.progressReminderIndicator.classList.toggle("is-armed", minutes > 0 && armed);
    if (minutes <= 0) {
      dom.progressReminderCopy.textContent = "已关闭长任务进度提醒。";
      return;
    }
    if (!armed) {
      dom.progressReminderCopy.textContent = `运行超过 ${minutes} 分钟后提醒；之后按里程碑或每 ${Math.max(10, minutes)} 分钟更新。`;
      return;
    }
    const elapsed = currentElapsedMilliseconds();
    const threshold = minutes * 60 * 1000;
    if (elapsed === null || elapsed < threshold) {
      const remaining = elapsed === null ? threshold : Math.max(0, threshold - elapsed);
      dom.progressReminderCopy.textContent = `提醒已启用，首次提示还需 ${formatDuration(remaining)}。`;
      return;
    }
    const untilNext = state.progressReminder.nextHeartbeatAt
      ? Math.max(0, state.progressReminder.nextHeartbeatAt - Date.now())
      : 0;
    dom.progressReminderCopy.textContent = `长任务提醒中；下次周期提示约 ${formatDuration(untilNext)} 后。`;
  }

  function activeModuleSummary() {
    if (!state.activeWorkflow) return "等待模块状态";
    const labels = [];
    state.nodeRuntime.forEach((runtime, nodeId) => {
      if (!isRunningStatus(runtime.status)) return;
      const node = state.activeWorkflow.nodes.find((item) => item.id === nodeId);
      const label = node ? node.label : nodeId;
      if (label && !labels.includes(label)) labels.push(label);
    });
    if (!labels.length) return "等待模块状态";
    if (labels.length <= 2) return labels.join("、");
    return `${labels.slice(0, 2).join("、")} 等 ${labels.length} 个模块`;
  }

  function progressReminderMessage(prefix) {
    const elapsed = currentElapsedMilliseconds();
    const progress = currentProgress();
    const pieces = [prefix, `已运行 ${formatDuration(elapsed || 0)}`];
    if (progress !== null) pieces.push(`完成 ${clamp(progress, 0, 100).toFixed(progress >= 10 ? 0 : 1)}%`);
    pieces.push(`当前：${activeModuleSummary()}`);
    return pieces.join(" · ");
  }

  function emitProgressReminder(message, level) {
    showToast(message, level || "info", 8000);
    addLocalEvent(level === "error" ? "error" : level === "warning" ? "warning" : "info", message);
    if (document.hidden) {
      const progress = currentProgress();
      document.title = `⏱ ${progress === null ? "运行中" : `${Math.round(progress)}%`} · ${state.activeWorkflow?.name || "DAAAM"}`;
      state.progressReminder.titlePending = true;
    }
  }

  function maybeNotifyProgress(statusValue) {
    const reminder = state.progressReminder;
    const minutes = state.progressReminderMinutes;
    const status = normalizeStatus(statusValue || state.activeRun?.status || "idle");
    if (!reminder.armed || minutes <= 0 || !state.activeRunId) {
      updateProgressReminderCopy();
      return;
    }
    const elapsed = currentElapsedMilliseconds();
    if (elapsed === null) {
      updateProgressReminderCopy();
      return;
    }
    const threshold = minutes * 60 * 1000;
    if (isTerminalStatus(status)) {
      if (elapsed >= threshold && !reminder.terminalNotified) {
        reminder.terminalNotified = true;
        reminder.wasLong = true;
        const completed = statusClass(status) === "completed";
        emitProgressReminder(
          progressReminderMessage(`长任务${completed ? "已完成" : `已${statusLabel(status)}`}`),
          completed ? "info" : "warning",
        );
      }
      reminder.armed = false;
      updateProgressReminderCopy();
      return;
    }
    if (!isRunningStatus(status) || elapsed < threshold) {
      updateProgressReminderCopy();
      return;
    }

    reminder.wasLong = true;
    const now = Date.now();
    if (!reminder.nextHeartbeatAt) reminder.nextHeartbeatAt = now;
    const progress = currentProgress();
    let milestone = null;
    if (progress !== null) {
      PROGRESS_MILESTONES.forEach((value) => {
        if (progress >= value && !reminder.milestones.has(value)) milestone = value;
      });
    }
    const heartbeatDue = now >= reminder.nextHeartbeatAt;
    if (milestone === null && !heartbeatDue) {
      updateProgressReminderCopy();
      return;
    }
    if (milestone !== null) {
      PROGRESS_MILESTONES.filter((value) => value <= milestone).forEach((value) => reminder.milestones.add(value));
    }
    const prefix = milestone === null ? "长任务进度提醒" : `长任务达到 ${milestone}% 里程碑`;
    emitProgressReminder(progressReminderMessage(prefix), "info");
    reminder.nextHeartbeatAt = now + Math.max(10, minutes) * 60 * 1000;
    updateProgressReminderCopy();
  }

  function updateRunUi(statusValue) {
    const status = normalizeStatus(statusValue || (state.activeRun && state.activeRun.status) || "idle");
    const cssStatus = statusClass(status);
    dom.globalStatus.className = `status-pill status-${cssStatus}`;
    dom.globalStatusLabel.textContent = statusLabel(status);
    dom.headerRunId.textContent = state.activeRunId ? shortId(state.activeRunId) : "—";
    dom.headerRunId.title = state.activeRunId || "";
    const progress = currentProgress();
    dom.headerProgress.textContent = progress === null ? "—" : `${clamp(progress, 0, 100).toFixed(progress >= 10 ? 0 : 1)}%`;
    const active = isRunningStatus(status);
    state.starting = state.starting && active;
    const runnable = Boolean(state.activeWorkflow && state.activeWorkflow.runnable);
    dom.startButton.disabled = !runnable || active || state.starting;
    dom.startButton.title = runnable ? "" : "该流程当前为只读参考流程";
    dom.previewButton.disabled = !state.activeWorkflow;
    dom.stopButton.classList.toggle("is-hidden", !active);
    dom.stopButton.disabled = state.stopping || status === "stopping";
    dom.workflowSelect.disabled =
      state.workflowSwitching || state.starting || !state.workflows.length;
    dom.workflowSelect.title = active
      ? "切换只改变当前查看的流程，不会停止正在运行的构建。"
      : "";
    dom.liveIndicator.classList.toggle("is-live", active);
    dom.liveIndicator.innerHTML = "";
    dom.liveIndicator.appendChild(createElement("i"));
    dom.liveIndicator.appendChild(document.createTextNode(active ? "实时接收" : state.activeRunId ? statusLabel(status) : "等待运行"));
    renderParameterForm();
    updateElapsed();
  }

  function isCurrentRunActive() {
    return Boolean(state.starting || (state.activeRun && isRunningStatus(state.activeRun.status)));
  }

  function updateElapsed() {
    if (!state.activeRun) {
      dom.headerElapsed.textContent = "00:00";
      updateProgressReminderCopy();
      return;
    }
    const elapsed = currentElapsedMilliseconds();
    dom.headerElapsed.textContent = elapsed === null ? "00:00" : formatDuration(elapsed);
    maybeNotifyProgress(state.activeRun.status);
  }

  function parseTimestamp(value) {
    if (!value) return 0;
    if (typeof value === "number") return value < 100000000000 ? value * 1000 : value;
    const parsed = new Date(value).getTime();
    return Number.isNaN(parsed) ? 0 : parsed;
  }

  async function startRun() {
    if (!state.activeWorkflow || state.starting || isCurrentRunActive()) return;
    state.starting = true;
    dom.startButton.disabled = true;
    dom.startButton.querySelector("span").textContent = "启动中…";
    try {
      const payload = await apiRequest("/api/runs", {
        method: "POST",
        body: commandPayload(),
        timeout: 30000,
      });
      const runId = safeString(payload.run_id || payload.id || (payload.run && payload.run.id), "");
      const processId = safeString(payload.process_id || (payload.process && payload.process.id), "");
      if (!runId && !processId) throw new Error("服务端未返回 run_id 或 process_id");
      state.activeRunId = runId || `process-${processId}`;
      state.processId = processId || null;
      state.activeRun = normalizeRun(
        {
          ...(payload.run || {}),
          id: state.activeRunId,
          process_id: processId,
          status: payload.status || "running",
          started_at: payload.started_at || new Date().toISOString(),
        },
        0,
      );
      state.snapshot = null;
      state.nodeRuntime = new Map();
      state.events = [];
      state.eventKeys = new Set();
      state.eventCursor = 0;
      resetProgressReminder(true);
      resetDepthPreview(state.activeRunId);
      state.parametersDirty = false;
      dom.parameterDirtyDot.classList.add("is-hidden");
      closeCommandDialog();
      addLocalEvent("info", `构建任务已提交，运行编号 ${state.activeRunId}`);
      updateRunUi("running");
      renderRunsWithOptimisticRun();
      showToast(`已启动构建 ${shortId(state.activeRunId)}`, "info");
      if (state.processId) startPolling();
      else {
        addLocalEvent("warning", "服务端未返回进程编号，将通过运行快照刷新状态");
        scheduleSnapshotPolling();
      }
      loadRuns({ selectLatest: false });
    } catch (error) {
      state.starting = false;
      updateRunUi("idle");
      addLocalEvent("error", `启动失败：${error.message}`);
      showToast(`启动失败：${error.message}`, "error", 6000);
    } finally {
      state.starting = false;
      dom.startButton.querySelector("span").textContent = "启动构建";
      updateRunUi(state.activeRun ? state.activeRun.status : "idle");
    }
  }

  function renderRunsWithOptimisticRun() {
    if (!state.activeWorkflow || !state.activeRun) return;
    const runs = state.activeWorkflow.runs || [];
    const index = runs.findIndex((run) => run.id === state.activeRun.id);
    if (index >= 0) runs[index] = state.activeRun;
    else runs.unshift(state.activeRun);
    state.activeWorkflow.runs = runs;
    renderRuns();
  }

  async function stopRun() {
    if (!state.processId || state.stopping) {
      if (!state.processId) showToast("当前运行没有可停止的进程编号", "warning");
      return;
    }
    state.stopping = true;
    dom.stopButton.disabled = true;
    dom.stopButton.querySelector("span").textContent = "停止中…";
    if (state.activeRun) state.activeRun.status = "stopping";
    updateRunUi("stopping");
    addLocalEvent("warning", "正在请求安全停止当前构建…");
    try {
      const payload = await apiRequest(`/api/processes/${encodeURIComponent(state.processId)}/stop`, { method: "POST" });
      const status = normalizeStatus(payload.status || "stopping");
      if (state.activeRun) state.activeRun.status = status;
      updateRunUi(status);
      showToast("停止请求已发送，正在等待进程退出", "warning");
      startPolling();
    } catch (error) {
      state.stopping = false;
      if (state.activeRun) state.activeRun.status = "running";
      updateRunUi("running");
      addLocalEvent("error", `停止失败：${error.message}`);
      showToast(`停止失败：${error.message}`, "error");
    } finally {
      dom.stopButton.querySelector("span").textContent = "停止";
    }
  }

  function startPolling() {
    if (!state.processId) return;
    armProgressReminder();
    stopPolling();
    const generation = ++state.pollGeneration;
    state.pollFailures = 0;
    state.pollCount = 0;
    pollProcess(generation);
  }

  function stopPolling() {
    state.pollGeneration += 1;
    window.clearTimeout(state.pollTimer);
    state.pollTimer = null;
  }

  async function pollProcess(generation) {
    if (generation !== state.pollGeneration || !state.processId) return;
    const processId = state.processId;
    try {
      const payload = await apiRequest(
        `/api/processes/${encodeURIComponent(processId)}/events?after=${encodeURIComponent(state.eventCursor)}`,
        { timeout: 12000 },
      );
      if (generation !== state.pollGeneration || processId !== state.processId) return;
      setConnection("online", "服务连接正常", "运行事件实时同步中");
      state.pollFailures = 0;
      state.pollCount += 1;
      appendEvents(objectEntriesAsArray(payload.events || payload.items || payload.logs, "id"));
      const nextCursor = finiteNumber(payload.next_cursor || payload.cursor);
      if (nextCursor !== null) state.eventCursor = Math.max(state.eventCursor, nextCursor);
      const processStatus = normalizeStatus(
        payload.status || payload.state || (payload.process && payload.process.status) || state.activeRun?.status || "running",
      );
      if (state.activeRun) {
        state.activeRun.status = processStatus;
        if (payload.process && typeof payload.process === "object") {
          state.activeRun.startedAt = payload.process.started_at || state.activeRun.startedAt;
          state.activeRun.endedAt = payload.process.ended_at || state.activeRun.endedAt;
        }
      }
      updateRunUi(processStatus);
      state.lastSync = new Date();
      updateLastSync();

      const embeddedSnapshot = payload.snapshot || payload.run_snapshot;
      if (embeddedSnapshot) applySnapshot(embeddedSnapshot);
      else if (state.pollCount % 4 === 0 && state.activeRunId) await refreshActiveSnapshot(false);

      if (isTerminalStatus(processStatus)) {
        state.stopping = false;
        addLocalEvent(
          statusClass(processStatus) === "completed" ? "info" : "warning",
          `构建进程已结束：${statusLabel(processStatus)}`,
        );
        stopPolling();
        await refreshActiveSnapshot(false);
        await refreshDepthPreviewIndex(true);
        await loadRuns({ selectLatest: false });
        return;
      }
    } catch (error) {
      if (generation !== state.pollGeneration) return;
      state.pollFailures += 1;
      if (state.pollFailures === 1 || state.pollFailures % 5 === 0) {
        addLocalEvent("warning", `状态轮询暂时失败（${state.pollFailures}）：${error.message}`);
      }
      if (state.pollFailures >= 3) setConnection("offline", "事件连接不稳定", "正在自动重试");
    }
    if (generation === state.pollGeneration) {
      const delay = Math.min(6000, 1100 + Math.max(0, state.pollFailures - 1) * 900);
      state.pollTimer = window.setTimeout(() => pollProcess(generation), delay);
    }
  }

  function scheduleSnapshotPolling() {
    armProgressReminder();
    stopPolling();
    const generation = ++state.pollGeneration;
    const tick = async () => {
      if (generation !== state.pollGeneration) return;
      await refreshActiveSnapshot(false);
      if (state.activeRun && isTerminalStatus(state.activeRun.status)) {
        stopPolling();
        loadRuns({ selectLatest: false });
        return;
      }
      state.pollTimer = window.setTimeout(tick, 2200);
    };
    state.pollTimer = window.setTimeout(tick, 1000);
  }

  async function refreshActiveSnapshot(notify) {
    if (!state.activeWorkflow || !state.activeRunId) return;
    try {
      const payload = await apiRequest(
        `/api/runs/${encodeURIComponent(state.activeRunId)}?workflow_id=${encodeURIComponent(state.activeWorkflow.id)}`,
      );
      applySnapshot(payload);
      if (notify) showToast("运行状态已刷新", "info", 1800);
    } catch (error) {
      if (notify) showToast(`刷新失败：${error.message}`, "error");
    }
  }

  function normalizeEvent(raw, index) {
    if (typeof raw === "string") {
      return {
        id: `text-${state.eventCursor}-${index}-${raw}`,
        timestamp: new Date().toISOString(),
        level: "info",
        type: "log",
        message: raw,
        data: {},
      };
    }
    const source = raw && typeof raw === "object" ? raw : {};
    const data = source.data && typeof source.data === "object" ? source.data : source.payload && typeof source.payload === "object" ? source.payload : {};
    const sequence = source.sequence ?? source.seq ?? source.cursor ?? source.event_id ?? source.id;
    const timestamp = source.timestamp || source.time || source.created_at || data.timestamp || new Date().toISOString();
    const type = safeString(source.type || source.kind || source.event || source.name, "log").toLowerCase();
    let level = safeString(source.level || source.severity || data.level, "").toLowerCase();
    if (!level) {
      const stream = safeString(source.stream || data.stream, "").toLowerCase();
      if (stream === "stderr") level = "error";
    }
    if (!level) {
      if (type.includes("error") || type.includes("fail")) level = "error";
      else if (type.includes("warn") || type.includes("stop") || type.includes("cancel")) level = "warning";
      else if (type.includes("debug")) level = "debug";
      else level = "info";
    }
    if (level === "warn") level = "warning";
    if (["fatal", "critical"].includes(level)) level = "error";
    let message = source.message || source.text || source.log || source.description || data.message || data.text;
    if (!message) message = describeEvent(type, { ...data, ...source });
    const id = safeString(sequence, `${timestamp}-${type}-${safeString(message, "").slice(0, 60)}`);
    return { ...source, id, sequence, timestamp, level, type, message: safeString(message, type), data };
  }

  function describeEvent(type, source) {
    const nodeId = source.node_id || source.module_id || source.stage_id;
    const status = source.status || source.state;
    if (nodeId && status) return `${nodeId} → ${statusLabel(status)}`;
    if (nodeId) return `模块 ${nodeId} 产生事件 ${type}`;
    if (status) return `运行状态更新为 ${statusLabel(status)}`;
    return type.replace(/_/g, " ");
  }

  function appendEvents(rawEvents) {
    if (!rawEvents || !rawEvents.length) return;
    let changed = false;
    rawEvents.forEach((raw, index) => {
      const event = normalizeEvent(raw, index);
      const key = `${event.id}|${event.timestamp}|${event.type}`;
      if (state.eventKeys.has(key)) return;
      state.eventKeys.add(key);
      state.events.push(event);
      changed = true;
      const numericSequence = finiteNumber(event.sequence);
      if (numericSequence !== null) state.eventCursor = Math.max(state.eventCursor, numericSequence);
      applyEventToState(event);
    });
    if (!changed) return;
    if (state.events.length > MAX_LOG_EVENTS) {
      const removed = state.events.splice(0, state.events.length - MAX_LOG_EVENTS);
      removed.forEach((event) => state.eventKeys.delete(`${event.id}|${event.timestamp}|${event.type}`));
    }
    renderLogs();
    refreshGraphRuntime();
    renderNodeDetail();
  }

  function applyEventToState(event) {
    const source = { ...(event.data || {}), ...event };
    const nodeId = safeString(source.node_id || source.module_id || source.stage_id || source.component_id, "");
    const eventStatus = source.status || source.state || source.phase;
    if (nodeId && eventStatus) {
      const existing = state.nodeRuntime.get(nodeId) || { id: nodeId, metrics: {} };
      state.nodeRuntime.set(nodeId, {
        ...existing,
        ...source,
        id: nodeId,
        status: normalizeStatus(eventStatus),
        metrics: source.metrics || existing.metrics || {},
      });
    }
    if (nodeId && source.metrics && !eventStatus) {
      const existing = state.nodeRuntime.get(nodeId) || { id: nodeId, status: "running" };
      state.nodeRuntime.set(nodeId, { ...existing, metrics: { ...(existing.metrics || {}), ...source.metrics } });
    }
    const runStatus = source.run_status || source.process_status;
    if (runStatus && state.activeRun) {
      state.activeRun.status = normalizeStatus(runStatus);
      updateRunUi(state.activeRun.status);
    }
    const eventSnapshot = source.snapshot;
    if (eventSnapshot && typeof eventSnapshot === "object") applySnapshot(eventSnapshot);
    if (source.quality_gates || source.gates || event.type.includes("quality")) {
      const gates = extractQualityGates(source);
      if (gates.length) renderQualityGates(gates, source);
    }
  }

  function addLocalEvent(level, message) {
    appendEvents([
      {
        id: `local-${Date.now()}-${Math.random().toString(16).slice(2)}`,
        timestamp: new Date().toISOString(),
        type: "client",
        level,
        message,
      },
    ]);
  }

  function appendSnapshotWarnings(snapshot) {
    const warnings = [
      ...normalizeWarnings(snapshot.warnings),
      ...normalizeWarnings(snapshot.process && snapshot.process.warnings),
    ];
    if (!warnings.length) return;
    const timestamp = snapshot.updated_at || state.activeRun?.startedAt || new Date().toISOString();
    appendEvents(
      Array.from(new Set(warnings)).map((warning, index) => ({
        id: `snapshot-warning-${index}-${warning}`,
        timestamp,
        type: "snapshot_warning",
        level: "warning",
        message: warning,
      })),
    );
  }

  function renderLogs() {
    const fragment = document.createDocumentFragment();
    const visibleEvents = state.events.filter((event) => state.logLevel === "all" || event.level === state.logLevel);
    dom.logStream.innerHTML = "";
    if (!visibleEvents.length) {
      const placeholder = createElement("div", "log-placeholder");
      placeholder.id = "log-placeholder";
      placeholder.appendChild(createElement("span", "prompt-sign", "$"));
      placeholder.appendChild(
        createElement(
          "span",
          "",
          state.events.length ? "当前筛选条件下没有日志。" : "选择历史运行，或配置参数并启动一次新的构建。",
        ),
      );
      dom.logStream.appendChild(placeholder);
    } else {
      visibleEvents.forEach((event) => {
        const line = createElement("div", `log-line log-${event.level}`);
        line.appendChild(createElement("span", "log-time", formatClock(event.timestamp)));
        line.appendChild(createElement("span", "log-level", logLevelLabel(event.level)));
        line.appendChild(createElement("span", "log-message", event.message));
        fragment.appendChild(line);
      });
      dom.logStream.appendChild(fragment);
    }
    dom.eventCountLabel.textContent = state.events.length ? `${state.events.length} 条事件` : "尚无事件";
    if (dom.autoscrollCheckbox.checked) dom.logStream.scrollTop = dom.logStream.scrollHeight;
  }

  function logLevelLabel(level) {
    return { info: "INFO", warning: "WARN", error: "ERROR", debug: "DEBUG" }[level] || String(level).toUpperCase();
  }

  function clearLogs() {
    state.events = [];
    state.eventKeys = new Set();
    renderLogs();
  }

  function extractQualityGates(source) {
    if (!source || typeof source !== "object") return [];
    let raw =
      source.quality_gates ||
      source.gates ||
      (source.quality && source.quality.gates) ||
      (source.quality_report && (source.quality_report.gates || source.quality_report.hard_gates)) ||
      (source.report && source.report.gates);
    if (!raw && Array.isArray(source.checks)) raw = source.checks;
    return objectEntriesAsArray(raw, "name").map((item, index) => {
      const gate = item && typeof item === "object" ? item : { value: item };
      let status = gate.status || gate.state || gate.result;
      if (!status && gate.passed !== undefined) status = gate.passed ? "passed" : "failed";
      if (!status && gate.ok !== undefined) status = gate.ok ? "passed" : "failed";
      if (!status && gate.value === true) status = "passed";
      if (!status && gate.value === false) status = "failed";
      return {
        name: safeString(gate.label || gate.title || gate.name || gate.code || gate.id, `质量门 ${index + 1}`),
        status: normalizeStatus(status || "pending"),
        value:
          gate.actual !== undefined
            ? gate.actual
            : gate.observed !== undefined
              ? gate.observed
              : gate.metric !== undefined
                ? gate.metric
                : gate.metrics !== undefined
                  ? gate.metrics
                  : gate.value,
        threshold: gate.threshold ?? gate.expected ?? gate.limit ?? gate.requirement ?? gate.thresholds,
        message: safeString(gate.message || gate.description || gate.reason || gate.detail, ""),
        hard: gate.hard !== false && gate.required !== false,
      };
    });
  }

  function renderQualityGates(gates, source) {
    dom.qualityList.innerHTML = "";
    const artifacts = objectEntriesAsArray(source && source.artifacts, "id");
    if ((!gates || !gates.length) && !artifacts.length) {
      const placeholder = createElement("div", "quality-placeholder");
      placeholder.appendChild(createElement("span", "shield-glyph", "✓"));
      const copy = createElement("div");
      copy.appendChild(createElement("strong", "", "构建完成后自动验收"));
      copy.appendChild(createElement("span", "", "几何、语义、队列、时延和产物完整性会显示在这里。"));
      placeholder.appendChild(copy);
      dom.qualityList.appendChild(placeholder);
      dom.qualitySummaryLabel.textContent = "等待验收数据";
      dom.qualityScore.textContent = "—";
      dom.qualityScore.className = "quality-score";
      return;
    }
    if (!gates || !gates.length) {
      dom.qualitySummaryLabel.textContent = "暂无质量门数据";
      dom.qualityScore.textContent = "—";
      dom.qualityScore.className = "quality-score";
    }
    let passed = 0;
    let failed = 0;
    (gates || []).forEach((gate) => {
      const css = gateStatusClass(gate.status);
      if (css === "pass") passed += 1;
      if (css === "fail") failed += 1;
      const item = createElement("div", `quality-item gate-${css}`);
      item.appendChild(createElement("span", "gate-icon", css === "pass" ? "✓" : css === "fail" ? "×" : css === "warning" ? "!" : "·"));
      const copy = createElement("div", "gate-copy");
      copy.appendChild(createElement("strong", "", gate.name));
      copy.appendChild(createElement("span", "", gate.message || (gate.hard ? "硬质量门" : "参考指标")));
      item.appendChild(copy);
      const valueParts = [];
      if (gate.value !== undefined) valueParts.push(formatValue(gate.value));
      if (gate.threshold !== undefined) valueParts.push(`阈值 ${formatValue(gate.threshold)}`);
      item.appendChild(createElement("span", "gate-value", valueParts.join(" / ") || statusLabel(gate.status)));
      dom.qualityList.appendChild(item);
    });
    const overall =
      source &&
      (source.quality_passed ?? source.passed ?? source.ok ?? source.quality?.passed ?? source.quality_report?.passed);
    if (gates && gates.length) {
      const allPassed = overall !== undefined ? Boolean(overall) : failed === 0 && passed === gates.length;
      dom.qualitySummaryLabel.textContent = failed ? `${failed} 项未通过` : `${passed}/${gates.length} 项已通过`;
      dom.qualityScore.textContent = `${passed}/${gates.length}`;
      dom.qualityScore.className = `quality-score ${allPassed ? "is-pass" : failed ? "is-fail" : ""}`.trim();
    }
    renderArtifacts(artifacts);
  }

  function renderArtifacts(artifacts) {
    if (!artifacts.length) return;
    dom.qualityList.appendChild(createElement("div", "quality-subheading", `运行产物 · ${artifacts.filter((item) => item.exists).length}/${artifacts.length}`));
    artifacts.forEach((artifact, index) => {
      const exists = artifact.exists !== false;
      const tag = exists && artifact.url ? "a" : "div";
      const item = createElement(tag, `artifact-item ${exists ? "is-ready" : "is-missing"}`);
      if (tag === "a") {
        item.href = artifact.url;
        item.target = "_blank";
        item.rel = "noopener";
      }
      item.appendChild(createElement("span", "artifact-icon", exists ? "↗" : "·"));
      const copy = createElement("span", "artifact-copy");
      copy.appendChild(createElement("strong", "", safeString(artifact.label || artifact.name, `产物 ${index + 1}`)));
      copy.appendChild(createElement("span", "", truncate(artifact.relative_path || artifact.path || (exists ? "可读取" : "尚未生成"), 88)));
      item.appendChild(copy);
      item.appendChild(createElement("span", "artifact-state", exists ? "就绪" : "缺失"));
      dom.qualityList.appendChild(item);
    });
  }

  function gateStatusClass(status) {
    const normalized = normalizeStatus(status);
    if (["completed", "passed", "success"].includes(normalized)) return "pass";
    if (["failed", "error", "blocked"].includes(normalized)) return "fail";
    if (normalized === "warning") return "warning";
    return "pending";
  }

  function computeGraphLayout(nodes, edges) {
    const positions = new Map();
    if (!nodes.length) return { positions, width: 600, height: 300 };
    const explicitNodes = nodes.filter((node) => node.x !== null && node.y !== null);
    let explicitScaleX = 1;
    let explicitScaleY = 1;
    if (explicitNodes.length) {
      const xs = explicitNodes.map((node) => node.x);
      const ys = explicitNodes.map((node) => node.y);
      if (Math.max(...xs) - Math.min(...xs) <= 10 && Math.max(...xs) <= 20) explicitScaleX = 250;
      if (Math.max(...ys) - Math.min(...ys) <= 10 && Math.max(...ys) <= 20) explicitScaleY = 130;
      if (explicitScaleX === 1) {
        const uniqueXs = Array.from(new Set(xs)).sort((left, right) => left - right);
        const gaps = uniqueXs.slice(1).map((value, index) => value - uniqueXs[index]).filter((gap) => gap > 0);
        const sortedGaps = gaps.sort((left, right) => left - right);
        const medianGap = sortedGaps.length ? sortedGaps[Math.floor(sortedGaps.length / 2)] : 230;
        if (medianGap < 220) explicitScaleX = 220 / medianGap;
      }
    }

    const adjacency = new Map(nodes.map((node) => [node.id, []]));
    const indegree = new Map(nodes.map((node) => [node.id, 0]));
    edges.forEach((edge) => {
      if (!adjacency.has(edge.source) || !indegree.has(edge.target)) return;
      adjacency.get(edge.source).push(edge.target);
      indegree.set(edge.target, indegree.get(edge.target) + 1);
    });
    const queue = nodes.filter((node) => indegree.get(node.id) === 0).map((node) => node.id);
    const depth = new Map(nodes.map((node) => [node.id, 0]));
    const visited = new Set();
    while (queue.length) {
      const id = queue.shift();
      visited.add(id);
      adjacency.get(id).forEach((target) => {
        depth.set(target, Math.max(depth.get(target), depth.get(id) + 1));
        indegree.set(target, indegree.get(target) - 1);
        if (indegree.get(target) === 0) queue.push(target);
      });
    }
    nodes.forEach((node, index) => {
      if (!visited.has(node.id)) depth.set(node.id, Math.floor(index / 3));
    });
    const layers = new Map();
    nodes.forEach((node) => {
      const layer = depth.get(node.id) || 0;
      if (!layers.has(layer)) layers.set(layer, []);
      layers.get(layer).push(node);
    });
    layers.forEach((layerNodes, layer) => {
      const totalHeight = (layerNodes.length - 1) * 138;
      layerNodes.forEach((node, row) => {
        positions.set(node.id, {
          x: 54 + layer * 248,
          y: 50 + row * 138 - totalHeight / 2,
        });
      });
    });
    explicitNodes.forEach((node) => {
      positions.set(node.id, { x: node.x * explicitScaleX, y: node.y * explicitScaleY });
    });

    const values = Array.from(positions.values());
    const minX = Math.min(...values.map((item) => item.x));
    const minY = Math.min(...values.map((item) => item.y));
    positions.forEach((position) => {
      position.x = position.x - minX + 54;
      position.y = position.y - minY + 46;
    });
    const shifted = Array.from(positions.values());
    const width = Math.max(500, Math.max(...shifted.map((item) => item.x)) + NODE_WIDTH + 54);
    const height = Math.max(260, Math.max(...shifted.map((item) => item.y)) + NODE_HEIGHT + 46);
    return { positions, width, height };
  }

  function renderGraph() {
    dom.edgeLayer.innerHTML = "";
    dom.nodeLayer.innerHTML = "";
    if (!state.activeWorkflow || !state.activeWorkflow.nodes.length) {
      dom.canvasLoading.classList.add("is-hidden");
      dom.canvasEmpty.classList.remove("is-hidden");
      state.graphLayout = null;
      return;
    }
    dom.canvasLoading.classList.add("is-hidden");
    dom.canvasEmpty.classList.add("is-hidden");
    const workflow = state.activeWorkflow;
    state.graphLayout = computeGraphLayout(workflow.nodes, workflow.edges);
    const { positions, width, height } = state.graphLayout;
    dom.dagScene.style.width = `${width}px`;
    dom.dagScene.style.height = `${height}px`;
    dom.edgeLayer.setAttribute("width", String(width));
    dom.edgeLayer.setAttribute("height", String(height));
    dom.edgeLayer.setAttribute("viewBox", `0 0 ${width} ${height}`);
    dom.nodeLayer.style.width = `${width}px`;
    dom.nodeLayer.style.height = `${height}px`;
    appendEdgeMarkers();
    workflow.edges.forEach((edge, index) => renderEdge(edge, index, positions));
    workflow.nodes.forEach((node) => renderNode(node, positions.get(node.id)));
    renderStageSummary();
    window.requestAnimationFrame(() => fitGraphToView(false));
  }

  function appendEdgeMarkers() {
    const defs = createSvgElement("defs");
    [
      ["data", "#48899a"],
      ["control", "#7966aa"],
      ["quality", "#489a69"],
      ["default", "#617185"],
    ].forEach(([name, color]) => {
      const marker = createSvgElement("marker", {
        id: `arrow-${name}`,
        markerWidth: 8,
        markerHeight: 8,
        refX: 7,
        refY: 4,
        orient: "auto",
        markerUnits: "strokeWidth",
      });
      marker.appendChild(createSvgElement("path", { d: "M 0 1 L 7 4 L 0 7 z", fill: color }));
      defs.appendChild(marker);
    });
    dom.edgeLayer.appendChild(defs);
  }

  function edgeKind(kind) {
    const value = safeString(kind, "data").toLowerCase();
    if (value.includes("control") || value.includes("semantic")) return "control";
    if (value.includes("quality") || value.includes("gate")) return "quality";
    if (value.includes("data") || value.includes("flow")) return "data";
    return "default";
  }

  function renderEdge(edge, index, positions) {
    const source = positions.get(edge.source);
    const target = positions.get(edge.target);
    if (!source || !target) return;
    const sx = source.x + NODE_WIDTH;
    const sy = source.y + NODE_HEIGHT / 2;
    const tx = target.x;
    const ty = target.y + NODE_HEIGHT / 2;
    let d;
    if (tx > sx + 25) {
      const bend = Math.max(52, (tx - sx) * 0.48);
      d = `M ${sx} ${sy} C ${sx + bend} ${sy}, ${tx - bend} ${ty}, ${tx} ${ty}`;
    } else {
      const offset = 58 + (index % 3) * 14;
      const vertical = ty >= sy ? 1 : -1;
      d = `M ${sx} ${sy} C ${sx + offset} ${sy}, ${sx + offset} ${sy + vertical * 70}, ${sx + offset / 2} ${sy + vertical * 70} S ${tx - offset} ${ty}, ${tx} ${ty}`;
    }
    const kind = edgeKind(edge.kind);
    const path = createSvgElement("path", {
      d,
      class: `dag-edge edge-${kind}`,
      "data-source": edge.source,
      "data-target": edge.target,
      "data-kind": kind,
      "marker-end": `url(#arrow-${kind})`,
    });
    dom.edgeLayer.appendChild(path);
  }

  function nodeInitial(node) {
    const known = {
      depth: "D",
      segmentation: "SAM",
      tracking: "ID",
      assignment: "KF",
      grounding: "DAM",
      semantic: "AI",
      mapping: "3D",
      hydra: "H",
      embedding: "E",
      room: "R",
      region: "R",
      quality: "QA",
      input: "IN",
      output: "OUT",
    };
    if (known[node.category]) return known[node.category];
    const ascii = node.label.match(/[A-Za-z0-9]+/g);
    if (ascii && ascii.length) return ascii[0].slice(0, 3).toUpperCase();
    return Array.from(node.label).slice(0, 1).join("");
  }

  function renderNode(node, position) {
    const meta = categoryMeta(node.category);
    const runtime = state.nodeRuntime.get(node.id) || {};
    const status = normalizeStatus(runtime.status || "pending");
    const button = createElement("button", `dag-node status-${statusClass(status)}`);
    button.type = "button";
    button.dataset.nodeId = node.id;
    button.dataset.category = node.category;
    button.style.left = `${position.x}px`;
    button.style.top = `${position.y}px`;
    button.style.setProperty("--node-accent", meta.color);
    button.setAttribute("aria-label", `${node.label}，${meta.label}，${statusLabel(status)}`);
    if (node.id === state.selectedNodeId) button.classList.add("is-selected");

    button.appendChild(createElement("span", "port port-in"));
    const inner = createElement("span", "node-inner");
    inner.appendChild(createElement("span", "node-icon", nodeInitial(node)));
    const heading = createElement("span", "node-heading");
    heading.appendChild(createElement("span", "node-category", meta.label));
    heading.appendChild(createElement("span", "node-label", node.label));
    inner.appendChild(heading);
    inner.appendChild(createElement("span", `node-status-dot status-${statusClass(status)}`));
    const foot = createElement("span", "node-foot");
    foot.appendChild(createElement("span", "node-status-label", statusLabel(status)));
    const primaryMetric = pickPrimaryMetric(runtime.metrics || node.metrics);
    foot.appendChild(createElement("span", "node-metric", primaryMetric ? `${primaryMetric.label} ${primaryMetric.value}` : "等待指标"));
    inner.appendChild(foot);
    button.appendChild(inner);
    button.appendChild(createElement("span", "port port-out"));
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      selectNode(node.id);
    });
    button.addEventListener("mouseenter", () => highlightConnections(node.id));
    button.addEventListener("mouseleave", () => highlightConnections(state.selectedNodeId));
    button.addEventListener("focus", () => highlightConnections(node.id));
    button.addEventListener("blur", () => highlightConnections(state.selectedNodeId));
    dom.nodeLayer.appendChild(button);
  }

  function pickPrimaryMetric(metrics) {
    const pairs = normalizeMetricPairs(metrics);
    if (!pairs.length) return null;
    const preferred = ["progress", "frames", "fps", "latency", "objects", "tracks", "rooms", "nodes"];
    const pair = pairs.find((item) => preferred.some((key) => item.key.toLowerCase().includes(key))) || pairs[0];
    return { label: pair.label, value: formatValue(pair.value) };
  }

  function normalizeMetricPairs(metrics) {
    if (!metrics) return [];
    if (Array.isArray(metrics)) {
      return metrics.map((metric, index) => {
        if (metric && typeof metric === "object") {
          return {
            key: safeString(metric.key || metric.name || metric.label, `metric-${index}`),
            label: safeString(metric.label || metric.name || metric.key, `指标 ${index + 1}`),
            value: metric.value ?? metric.actual ?? metric.current ?? "—",
          };
        }
        const key = safeString(metric, `metric-${index}`);
        return { key, label: humanizeKey(key), value: "—" };
      });
    }
    if (typeof metrics === "object") {
      return Object.entries(metrics).map(([key, value]) => {
        if (value && typeof value === "object" && !Array.isArray(value) && ("value" in value || "label" in value)) {
          return { key, label: safeString(value.label || value.name, humanizeKey(key)), value: value.value ?? value.actual ?? "—" };
        }
        return { key, label: humanizeKey(key), value };
      });
    }
    return [{ key: "value", label: "指标", value: metrics }];
  }

  function humanizeKey(value) {
    const labels = {
      fps: "处理帧率",
      latency_ms: "时延",
      p95_ms: "P95 时延",
      frames: "帧数",
      frames_processed: "已处理帧",
      completed_frames: "完成帧",
      objects: "对象数",
      tracks: "轨迹数",
      rooms: "房间数",
      nodes: "节点数",
      edges: "边数",
      queue_depth: "队列深度",
      progress: "进度",
    };
    const key = safeString(value, "");
    return labels[key.toLowerCase()] || key.replace(/_/g, " ");
  }

  function refreshGraphRuntime() {
    if (!state.activeWorkflow) return;
    dom.nodeLayer.querySelectorAll(".dag-node").forEach((element) => {
      const id = element.dataset.nodeId;
      const runtime = state.nodeRuntime.get(id) || {};
      const status = normalizeStatus(runtime.status || "pending");
      ["pending", "running", "completed", "failed", "warning", "stopping"].forEach((name) => {
        element.classList.remove(`status-${name}`, `is-${name}`);
      });
      element.classList.add(`status-${statusClass(status)}`);
      if (["running", "failed"].includes(statusClass(status))) element.classList.add(`is-${statusClass(status)}`);
      const dot = element.querySelector(".node-status-dot");
      dot.className = `node-status-dot status-${statusClass(status)}`;
      element.querySelector(".node-status-label").textContent = statusLabel(status);
      const workflowNode = state.activeWorkflow.nodes.find((node) => node.id === id);
      const metric = pickPrimaryMetric(runtime.metrics || workflowNode?.metrics);
      element.querySelector(".node-metric").textContent = metric ? `${metric.label} ${metric.value}` : "等待指标";
    });
    applyGraphFilters();
    renderStageSummary();
    syncDepthPreviewVisibility();
  }

  function renderCategoryFilters() {
    dom.categoryFilters.innerHTML = "";
    if (!state.activeWorkflow) return;
    const categories = Array.from(new Set(state.activeWorkflow.nodes.map((node) => node.category)));
    categories.slice(0, 7).forEach((category) => {
      const meta = categoryMeta(category);
      const button = createElement("button", "category-chip is-active");
      button.type = "button";
      button.dataset.category = category;
      button.style.setProperty("--category-color", meta.color);
      button.appendChild(createElement("i"));
      button.appendChild(document.createTextNode(meta.label));
      button.addEventListener("click", () => {
        if (state.activeCategories.has(category)) state.activeCategories.delete(category);
        else state.activeCategories.add(category);
        button.classList.toggle("is-active", state.activeCategories.has(category));
        applyGraphFilters();
      });
      dom.categoryFilters.appendChild(button);
    });
  }

  function applyGraphFilters() {
    const visibleNodeIds = new Set();
    dom.nodeLayer.querySelectorAll(".dag-node").forEach((element) => {
      const visible = state.activeCategories.has(element.dataset.category);
      element.classList.toggle("is-dimmed", !visible);
      if (visible) visibleNodeIds.add(element.dataset.nodeId);
    });
    dom.edgeLayer.querySelectorAll(".dag-edge").forEach((edge) => {
      const visible = visibleNodeIds.has(edge.dataset.source) && visibleNodeIds.has(edge.dataset.target);
      edge.classList.toggle("is-dimmed", !visible);
    });
    highlightConnections(state.selectedNodeId);
  }

  function highlightConnections(nodeId) {
    dom.edgeLayer.querySelectorAll(".dag-edge").forEach((edge) => {
      const connected = nodeId && (edge.dataset.source === nodeId || edge.dataset.target === nodeId);
      edge.classList.toggle("is-highlighted", Boolean(connected));
    });
  }

  function clearDepthPreviewTimer() {
    window.clearTimeout(state.depthPreview.refreshTimer);
    state.depthPreview.refreshTimer = null;
  }

  function revokeDepthPreviewUrl() {
    if (!state.depthPreview.objectUrl) return;
    URL.revokeObjectURL(state.depthPreview.objectUrl);
    state.depthPreview.objectUrl = null;
  }

  function resetDepthPreview(runId) {
    clearDepthPreviewTimer();
    revokeDepthPreviewUrl();
    state.depthPreview = {
      runId: runId || null,
      frames: [],
      index: -1,
      followLatest: true,
      available: false,
      complete: false,
      live: false,
      source: null,
      minimumDepthM: 0.25,
      maximumDepthM: 5.0,
      loading: false,
      indexLoading: false,
      error: "",
      dismissedRunId: null,
      requestSequence: state.depthPreview.requestSequence + 1,
      indexSequence: state.depthPreview.indexSequence + 1,
      refreshTimer: null,
      objectUrl: null,
      displayedFrame: null,
      frameStats: null,
    };
    if (dom.depthPreviewImage) dom.depthPreviewImage.removeAttribute("src");
    if (dom.depthPreviewWindow) dom.depthPreviewWindow.classList.add("is-hidden");
  }

  function foundationDepthExpected() {
    if (!state.activeWorkflow) return false;
    if (state.activeWorkflow.id === "offline_hq") return true;
    const processParameters = state.snapshot?.process?.parameters || {};
    const runParameters = state.activeRun?.parameters || {};
    const backend =
      processParameters.depth_backend ||
      runParameters.depth_backend ||
      state.parameters.depth_backend;
    return backend === "foundation-worker";
  }

  function depthNodeIsRunning() {
    return isRunningStatus(state.nodeRuntime.get("depth")?.status);
  }

  function openDepthPreview(options) {
    if (state.depthPreview.runId !== state.activeRunId) resetDepthPreview(state.activeRunId);
    if (!(options && options.automatic)) state.depthPreview.dismissedRunId = null;
    dom.depthPreviewWindow.classList.remove("is-hidden");
    renderDepthPreview();
    refreshDepthPreviewIndex(true);
  }

  function closeDepthPreview(manual) {
    if (manual && state.activeRunId) state.depthPreview.dismissedRunId = state.activeRunId;
    dom.depthPreviewWindow.classList.add("is-hidden");
    clearDepthPreviewTimer();
  }

  function syncDepthPreviewVisibility() {
    if (!state.activeWorkflow) return;
    if (
      state.selectedNodeId === "depth" &&
      state.depthPreview.dismissedRunId !== state.activeRunId
    ) {
      openDepthPreview({ automatic: true });
      return;
    }
    if (
      foundationDepthExpected() &&
      depthNodeIsRunning() &&
      state.depthPreview.dismissedRunId !== state.activeRunId
    ) {
      openDepthPreview({ automatic: true });
    }
  }

  function scheduleDepthPreviewRefresh(delay) {
    clearDepthPreviewTimer();
    if (dom.depthPreviewWindow.classList.contains("is-hidden") || state.depthPreview.complete) return;
    state.depthPreview.refreshTimer = window.setTimeout(
      () => refreshDepthPreviewIndex(false),
      delay === undefined ? DEPTH_REFRESH_INTERVAL_MS : delay,
    );
  }

  function normalizeDepthPreviewError(error) {
    const message = safeString(error && error.message, "深度预览暂不可用");
    const staleBackend = error && error.status === 404 && /^not found$/i.test(message.trim());
    return {
      message: staleBackend
        ? "仪表盘后端版本过旧，请重启服务；窗口会自动重试。"
        : message,
      staleBackend,
    };
  }

  async function refreshDepthPreviewIndex(force) {
    const preview = state.depthPreview;
    const runId = state.activeRunId;
    if (!runId || preview.runId !== runId) return;
    if (dom.depthPreviewWindow.classList.contains("is-hidden") && !force) return;
    if (preview.indexLoading) return;
    preview.indexLoading = true;
    const sequence = ++preview.indexSequence;
    const after = preview.frames.length ? preview.frames[preview.frames.length - 1] : -1;
    try {
      const payload = await apiRequest(
        `/api/runs/${encodeURIComponent(runId)}/depth-frames?after=${encodeURIComponent(after)}&limit=5000`,
        { timeout: 12000 },
      );
      if (sequence !== preview.indexSequence || runId !== state.activeRunId) return;
      preview.error = "";
      preview.available = Boolean(payload.available);
      preview.complete = Boolean(payload.complete);
      preview.live = Boolean(payload.live);
      preview.source = payload.source || null;
      const minimum = finiteNumber(payload.minimum_depth_m);
      const maximum = finiteNumber(payload.maximum_depth_m);
      if (minimum !== null) preview.minimumDepthM = minimum;
      if (maximum !== null && maximum > preview.minimumDepthM) preview.maximumDepthM = maximum;
      const additions = objectEntriesAsArray(payload.frames).map((item) => {
        if (item && typeof item === "object") return finiteNumber(item.value ?? item.frame_index ?? item.id);
        return finiteNumber(item);
      }).filter((value) => value !== null);
      if (additions.length) {
        preview.frames = Array.from(new Set([...preview.frames, ...additions])).sort((a, b) => a - b);
      }
      if (preview.frames.length && (preview.followLatest || preview.index < 0)) {
        preview.index = preview.frames.length - 1;
      }
      renderDepthPreview();
      const selectedFrame = preview.frames[preview.index];
      if (selectedFrame !== undefined && selectedFrame !== preview.displayedFrame) {
        await showDepthFrame(preview.index);
      }
      if (payload.has_more) scheduleDepthPreviewRefresh(0);
      else if (!preview.complete && (preview.live || isCurrentRunActive())) scheduleDepthPreviewRefresh();
    } catch (error) {
      if (sequence !== preview.indexSequence || runId !== state.activeRunId) return;
      const normalizedError = normalizeDepthPreviewError(error);
      if (!normalizedError.staleBackend && error.status === 404 && isCurrentRunActive()) {
        preview.available = false;
        preview.error = "";
      } else {
        preview.error = normalizedError.message;
      }
      renderDepthPreview();
      if (normalizedError.staleBackend || preview.live || isCurrentRunActive()) {
        scheduleDepthPreviewRefresh(normalizedError.staleBackend ? 5000 : 2500);
      }
    } finally {
      if (sequence === preview.indexSequence) preview.indexLoading = false;
    }
  }

  async function fetchDepthPreviewImage(url) {
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), 15000);
    try {
      const response = await fetch(url, {
        headers: { Accept: "image/png" },
        signal: controller.signal,
      });
      if (!response.ok) {
        let message = `深度预览请求失败（HTTP ${response.status}）`;
        try {
          const payload = await response.json();
          message = safeString(payload.detail || payload.message, message);
        } catch (_error) {
          // Keep the HTTP status message for non-JSON errors.
        }
        const error = new Error(message);
        error.status = response.status;
        throw error;
      }
      return { blob: await response.blob(), headers: response.headers };
    } catch (error) {
      if (error && error.name === "AbortError") throw new Error("深度图加载超时");
      throw error;
    } finally {
      window.clearTimeout(timer);
    }
  }

  async function showDepthFrame(position) {
    const preview = state.depthPreview;
    if (!preview.frames.length) return;
    const target = clamp(position, 0, preview.frames.length - 1);
    const frameIndex = preview.frames[target];
    const runId = state.activeRunId;
    preview.index = target;
    preview.loading = true;
    preview.error = "";
    renderDepthPreview();
    const sequence = ++preview.requestSequence;
    const query = new URLSearchParams({
      minimum_m: String(preview.minimumDepthM),
      maximum_m: String(preview.maximumDepthM),
    });
    try {
      const result = await fetchDepthPreviewImage(
        `/api/runs/${encodeURIComponent(runId)}/depth-frames/${encodeURIComponent(frameIndex)}.png?${query}`,
      );
      if (sequence !== preview.requestSequence || runId !== state.activeRunId) return;
      const objectUrl = URL.createObjectURL(result.blob);
      const previousUrl = preview.objectUrl;
      dom.depthPreviewImage.src = objectUrl;
      try {
        await dom.depthPreviewImage.decode();
      } catch (_error) {
        // The load event may already have decoded the image; retain it when src is set.
      }
      if (sequence !== preview.requestSequence || runId !== state.activeRunId) {
        URL.revokeObjectURL(objectUrl);
        return;
      }
      preview.objectUrl = objectUrl;
      if (previousUrl) URL.revokeObjectURL(previousUrl);
      preview.displayedFrame = frameIndex;
      preview.frameStats = {
        validRatio: finiteNumber(result.headers.get("X-Depth-Valid-Ratio")),
        observedMinimumM: finiteNumber(result.headers.get("X-Depth-Observed-Min-M")),
        observedMaximumM: finiteNumber(result.headers.get("X-Depth-Observed-Max-M")),
      };
    } catch (error) {
      if (sequence !== preview.requestSequence || runId !== state.activeRunId) return;
      preview.error = normalizeDepthPreviewError(error).message;
    } finally {
      if (sequence === preview.requestSequence) {
        preview.loading = false;
        renderDepthPreview();
      }
    }
  }

  function stepDepthFrame(delta) {
    const preview = state.depthPreview;
    if (!preview.frames.length) return;
    const target = clamp(preview.index + delta, 0, preview.frames.length - 1);
    preview.followLatest = target === preview.frames.length - 1;
    dom.depthPreviewFollow.checked = preview.followLatest;
    showDepthFrame(target);
  }

  function renderDepthPreview() {
    if (!dom.depthPreviewWindow) return;
    const preview = state.depthPreview;
    const count = preview.frames.length;
    const hasImage = Boolean(preview.objectUrl);
    const active = (preview.live || isCurrentRunActive()) && !preview.complete;
    dom.depthPreviewStatus.classList.toggle("is-live", active && preview.followLatest);
    if (preview.complete) dom.depthPreviewStatus.textContent = "估计完成";
    else if (active && preview.followLatest) dom.depthPreviewStatus.textContent = "实时跟随";
    else if (preview.available) dom.depthPreviewStatus.textContent = "深度产物";
    else dom.depthPreviewStatus.textContent = "等待产物";

    dom.depthPreviewLoading.classList.toggle("is-hidden", !preview.loading || hasImage);
    dom.depthPreviewError.classList.toggle("is-hidden", !preview.error || hasImage);
    dom.depthPreviewEmpty.classList.toggle("is-hidden", preview.loading || Boolean(preview.error) || count > 0);
    if (!count && !preview.loading && !preview.error) {
      const title = dom.depthPreviewEmpty.querySelector("strong");
      const copy = dom.depthPreviewEmpty.querySelector("span:last-child");
      if (!state.activeRunId) {
        title.textContent = "选择运行记录后查看深度结果";
        copy.textContent = "深度预览不会读取运行目录之外的文件。";
      } else if (isCurrentRunActive() && foundationDepthExpected()) {
        title.textContent = "等待 FoundationStereo 生成第一帧";
        copy.textContent = "完整写入的深度图会自动显示在这里。";
      } else {
        title.textContent = "该运行没有可浏览的深度图";
        copy.textContent = "仅显示本次 FoundationStereo 估计生成的原始深度产物。";
      }
    }
    if (preview.error) {
      dom.depthPreviewError.querySelector("span:last-child").textContent = truncate(preview.error, 120);
      if (hasImage) dom.depthPreviewStatus.textContent = "刷新重试中";
    }

    const frameIndex = count && preview.index >= 0 ? preview.frames[preview.index] : null;
    dom.depthPreviewCounter.textContent = count ? `${preview.index + 1} / ${count}` : "0 / 0";
    dom.depthPreviewMeta.textContent = frameIndex === null ? "帧 —" : `帧 ${String(frameIndex).padStart(8, "0")}`;
    const valid = preview.frameStats?.validRatio;
    dom.depthPreviewValid.textContent = valid === null || valid === undefined
      ? "有效深度 —"
      : `有效深度 ${(valid * 100).toFixed(1)}%`;
    dom.depthLegendMinimum.textContent = `${preview.minimumDepthM.toFixed(2)} m`;
    dom.depthLegendMaximum.textContent = `${preview.maximumDepthM.toFixed(2)} m`;
    dom.depthPreviewPrevious.disabled = !count || preview.index <= 0 || preview.loading;
    dom.depthPreviewNext.disabled = !count || preview.index >= count - 1 || preview.loading;
    dom.depthPreviewFollow.checked = preview.followLatest;
    dom.depthPreviewFollow.disabled = !count;
  }

  function selectNode(nodeId) {
    state.selectedNodeId = nodeId;
    dom.nodeLayer.querySelectorAll(".dag-node").forEach((element) => {
      element.classList.toggle("is-selected", element.dataset.nodeId === nodeId);
    });
    highlightConnections(nodeId);
    renderNodeDetail();
    if (nodeId === "depth") openDepthPreview({ automatic: false });
    else if (!dom.depthPreviewWindow.classList.contains("is-hidden")) closeDepthPreview(true);
    switchDetailsTab("detail");
    if (window.matchMedia("(max-width: 1120px)").matches) openPanel("right");
  }

  function renderNodeDetail() {
    const node =
      state.activeWorkflow && state.activeWorkflow.nodes.find((item) => item.id === state.selectedNodeId);
    dom.nodeDetailEmpty.classList.toggle("is-hidden", Boolean(node));
    dom.nodeDetailContent.classList.toggle("is-hidden", !node);
    dom.nodeDetailContent.innerHTML = "";
    if (!node) return;
    const meta = categoryMeta(node.category);
    const runtime = state.nodeRuntime.get(node.id) || {};
    const status = normalizeStatus(runtime.status || "pending");
    const container = dom.nodeDetailContent;
    container.style.setProperty("--detail-accent", meta.color);
    const hero = createElement("div", "detail-hero");
    hero.appendChild(createElement("div", "detail-icon", nodeInitial(node)));
    const title = createElement("div", "detail-title");
    title.appendChild(createElement("span", "", meta.label));
    title.appendChild(createElement("h2", "", node.label));
    hero.appendChild(title);
    const pill = createElement("span", `status-pill status-${statusClass(status)}`);
    pill.appendChild(createElement("span", "status-dot"));
    pill.appendChild(createElement("span", "", statusLabel(status)));
    hero.appendChild(pill);
    container.appendChild(hero);
    container.appendChild(createElement("p", "detail-description", node.description));
    const runtimeHint = safeString(runtime.message || runtime.description, node.statusHint);
    if (runtimeHint) container.appendChild(createElement("div", "status-hint", runtimeHint));

    const metrics = mergeMetricPairs(node.metrics, runtime.metrics);
    const metricSection = createElement("section", "detail-section");
    metricSection.appendChild(createElement("h3", "", "实时指标"));
    const metricGrid = createElement("div", "metric-grid");
    if (metrics.length) {
      metrics.slice(0, 8).forEach((metric) => {
        const card = createElement("div", "metric-card");
        card.appendChild(createElement("span", "", metric.label));
        card.appendChild(createElement("strong", "", formatMetricValue(metric)));
        metricGrid.appendChild(card);
      });
    } else {
      const card = createElement("div", "metric-card");
      card.appendChild(createElement("span", "", "状态"));
      card.appendChild(createElement("strong", "", statusLabel(status)));
      metricGrid.appendChild(card);
    }
    metricSection.appendChild(metricGrid);
    container.appendChild(metricSection);

    const ioSection = createElement("section", "detail-section");
    ioSection.appendChild(createElement("h3", "", "数据接口"));
    const ioGrid = createElement("div", "io-grid");
    ioGrid.appendChild(renderIoColumn("输入", node.inputs));
    ioGrid.appendChild(renderIoColumn("输出", node.outputs));
    ioSection.appendChild(ioGrid);
    container.appendChild(ioSection);

    const parameterIds = normalizeIoItems(node.parameters);
    const definitions = parameterIds
      .map((parameterId) => state.parameterDefinitions.find((definition) => definition.key === parameterId))
      .filter(Boolean);
    if (definitions.length) {
      const parameterSection = createElement("section", "detail-section");
      parameterSection.appendChild(createElement("h3", "", "关联参数"));
      const parameterList = createElement("div", "node-parameter-list");
      definitions.forEach((definition) => {
        const row = createElement("button", "node-parameter-row");
        row.type = "button";
        const copy = createElement("span", "node-parameter-copy");
        copy.appendChild(createElement("strong", "", definition.label));
        copy.appendChild(createElement("code", "", `${safeString(definition.source, "CLI")} · ${definition.key}`));
        row.appendChild(copy);
        row.appendChild(createElement("span", "node-parameter-value", formatValue(state.parameters[definition.key])));
        row.addEventListener("click", () => {
          switchDetailsTab("parameters");
          window.setTimeout(() => {
            const fieldId = `parameter-${definition.key.replace(/[^a-zA-Z0-9_-]/g, "-")}`;
            const control = document.getElementById(fieldId);
            if (control) {
              control.scrollIntoView({ behavior: "smooth", block: "center" });
              control.focus({ preventScroll: true });
            }
          }, 50);
        });
        parameterList.appendChild(row);
      });
      parameterSection.appendChild(parameterList);
      container.appendChild(parameterSection);
    }
  }

  function mergeMetricPairs(base, runtime) {
    const merged = new Map();
    normalizeMetricPairs(base).forEach((item) => merged.set(item.key, item));
    normalizeMetricPairs(runtime).forEach((item) => merged.set(item.key, item));
    return Array.from(merged.values());
  }

  function formatMetricValue(metric) {
    const value = formatValue(metric.value);
    const key = metric.key.toLowerCase();
    if ((key.endsWith("_ms") || key.includes("latency")) && !String(value).includes("ms")) return `${value} ms`;
    if ((key.includes("ratio") || key.includes("percent")) && typeof metric.value === "number") {
      const percent = metric.value <= 1 ? metric.value * 100 : metric.value;
      return `${Number(percent.toFixed(1))}%`;
    }
    return value;
  }

  function normalizeIoItems(value) {
    if (!value) return [];
    if (Array.isArray(value)) return value.map((item) => safeString(item && (item.label || item.name || item.type) ? item.label || item.name || item.type : item, "")).filter(Boolean);
    if (typeof value === "object") return Object.entries(value).map(([key, item]) => safeString(item && item.label ? item.label : key, key));
    return [String(value)];
  }

  function renderIoColumn(title, values) {
    const column = createElement("div", "io-column");
    column.appendChild(createElement("span", "", title));
    const list = createElement("ul", "io-list");
    const items = normalizeIoItems(values);
    if (!items.length) list.appendChild(createElement("li", "is-empty", "未声明"));
    else items.forEach((item) => list.appendChild(createElement("li", "", item)));
    column.appendChild(list);
    return column;
  }

  function renderStageSummary() {
    dom.stageSummary.innerHTML = "";
    if (!state.activeWorkflow) return;
    const counts = new Map();
    state.activeWorkflow.nodes.forEach((node) => {
      const status = statusClass(state.nodeRuntime.get(node.id)?.status || "pending");
      counts.set(status, (counts.get(status) || 0) + 1);
    });
    const ordered = [
      ["completed", "完成", "#56d58c"],
      ["running", "运行", "#56d5c3"],
      ["pending", "等待", "#64748b"],
      ["failed", "异常", "#ff7485"],
    ];
    ordered.forEach(([status, label, color]) => {
      const count = counts.get(status) || 0;
      if (!count && status !== "pending") return;
      const item = createElement("span", "stage-summary-item");
      item.style.setProperty("--stage-color", color);
      item.appendChild(createElement("i"));
      item.appendChild(document.createTextNode(`${label} ${count}`));
      dom.stageSummary.appendChild(item);
    });
  }

  function applyViewTransform() {
    const { x, y, scale } = state.view;
    dom.dagScene.style.transform = `translate3d(${x}px, ${y}px, 0) scale(${scale})`;
    dom.zoomResetButton.textContent = `${Math.round(scale * 100)}%`;
    dom.gridBackdrop.style.setProperty("--grid-size", `${22 * scale}px`);
    dom.gridBackdrop.style.setProperty("--grid-x", `${x % (22 * scale)}px`);
    dom.gridBackdrop.style.setProperty("--grid-y", `${y % (22 * scale)}px`);
  }

  function fitGraphToView(markManual) {
    if (!state.graphLayout) return;
    const viewportWidth = dom.dagViewport.clientWidth;
    const viewportHeight = dom.dagViewport.clientHeight;
    if (!viewportWidth || !viewportHeight) return;
    const margin = viewportWidth < 600 ? 30 : 70;
    const scale = clamp(
      Math.min(
        (viewportWidth - margin * 2) / state.graphLayout.width,
        (viewportHeight - margin * 1.35) / state.graphLayout.height,
      ),
      0.35,
      1.12,
    );
    state.view.scale = scale;
    state.view.x = (viewportWidth - state.graphLayout.width * scale) / 2;
    state.view.y = (viewportHeight - state.graphLayout.height * scale) / 2;
    state.view.manual = Boolean(markManual);
    applyViewTransform();
  }

  function setZoom(nextScale, anchorX, anchorY) {
    const previous = state.view.scale;
    const scale = clamp(nextScale, 0.32, 1.9);
    const x = anchorX === undefined ? dom.dagViewport.clientWidth / 2 : anchorX;
    const y = anchorY === undefined ? dom.dagViewport.clientHeight / 2 : anchorY;
    const worldX = (x - state.view.x) / previous;
    const worldY = (y - state.view.y) / previous;
    state.view.x = x - worldX * scale;
    state.view.y = y - worldY * scale;
    state.view.scale = scale;
    state.view.manual = true;
    applyViewTransform();
  }

  function bindCanvasInteractions() {
    dom.dagViewport.addEventListener(
      "wheel",
      (event) => {
        if (event.target.closest(".depth-preview-window")) return;
        event.preventDefault();
        const rect = dom.dagViewport.getBoundingClientRect();
        const factor = Math.exp(-event.deltaY * 0.0012);
        setZoom(state.view.scale * factor, event.clientX - rect.left, event.clientY - rect.top);
      },
      { passive: false },
    );
    dom.dagViewport.addEventListener("pointerdown", (event) => {
      if (event.button !== 0 || event.target.closest(".dag-node, .depth-preview-window")) return;
      state.drag = { pointerId: event.pointerId, startX: event.clientX, startY: event.clientY, x: state.view.x, y: state.view.y };
      dom.dagViewport.classList.add("is-dragging");
      dom.dagViewport.setPointerCapture(event.pointerId);
    });
    dom.dagViewport.addEventListener("pointermove", (event) => {
      if (!state.drag || state.drag.pointerId !== event.pointerId) return;
      state.view.x = state.drag.x + event.clientX - state.drag.startX;
      state.view.y = state.drag.y + event.clientY - state.drag.startY;
      state.view.manual = true;
      applyViewTransform();
    });
    const endDrag = (event) => {
      if (!state.drag || state.drag.pointerId !== event.pointerId) return;
      state.drag = null;
      dom.dagViewport.classList.remove("is-dragging");
    };
    dom.dagViewport.addEventListener("pointerup", endDrag);
    dom.dagViewport.addEventListener("pointercancel", endDrag);
    dom.dagViewport.addEventListener("dblclick", (event) => {
      if (!event.target.closest(".dag-node, .depth-preview-window")) fitGraphToView(true);
    });
    dom.dagViewport.addEventListener("keydown", (event) => {
      if (["+", "="].includes(event.key)) {
        event.preventDefault();
        setZoom(state.view.scale * 1.15);
      } else if (event.key === "-") {
        event.preventDefault();
        setZoom(state.view.scale / 1.15);
      } else if (["0", "f", "F"].includes(event.key)) {
        event.preventDefault();
        fitGraphToView(true);
      }
    });
  }

  function switchDetailsTab(panel) {
    const detail = panel === "detail";
    dom.detailTab.classList.toggle("is-active", detail);
    dom.parametersTab.classList.toggle("is-active", !detail);
    dom.detailTab.setAttribute("aria-selected", String(detail));
    dom.parametersTab.setAttribute("aria-selected", String(!detail));
    dom.detailTabPanel.classList.toggle("is-hidden", !detail);
    dom.parametersTabPanel.classList.toggle("is-hidden", detail);
  }

  function openPanel(side) {
    if (side === "left") {
      dom.leftPanel.classList.add("is-open");
      dom.rightPanel.classList.remove("is-open");
    } else {
      dom.rightPanel.classList.add("is-open");
      dom.leftPanel.classList.remove("is-open");
    }
    dom.panelScrim.classList.add("is-visible");
  }

  function closePanels() {
    dom.leftPanel.classList.remove("is-open");
    dom.rightPanel.classList.remove("is-open");
    dom.panelScrim.classList.remove("is-visible");
  }

  function toggleDock() {
    dom.workspaceGrid.classList.toggle("is-dock-collapsed");
    const collapsed = dom.workspaceGrid.classList.contains("is-dock-collapsed");
    dom.collapseDockButton.setAttribute("aria-label", collapsed ? "展开运行控制台" : "折叠运行控制台");
    window.setTimeout(() => fitGraphToView(false), 190);
  }

  function updateLastSync() {
    dom.lastUpdatedTime.textContent = state.lastSync ? formatClock(state.lastSync) : "—";
    if (state.lastSync) dom.lastUpdatedTime.dateTime = state.lastSync.toISOString();
  }

  function bindEvents() {
    dom.workflowSelect.addEventListener("change", async () => {
      if (state.workflowSwitching) return;
      const targetWorkflowId = dom.workflowSelect.value;
      const backgroundRunId = isCurrentRunActive() ? state.activeRunId : null;
      state.workflowSwitching = true;
      dom.workflowSelect.disabled = true;
      try {
        await activateWorkflow(targetWorkflowId);
        if (backgroundRunId) {
          showToast(
            `已切换查看流程；运行 ${shortId(backgroundRunId)} 仍在后台继续。`,
            "info",
            4200,
          );
        }
      } finally {
        state.workflowSwitching = false;
        updateRunUi(state.activeRun?.status || "idle");
      }
    });
    dom.refreshRunsButton.addEventListener("click", () => loadRuns({ selectLatest: false }));
    dom.progressReminderSelect.addEventListener("change", () => {
      const selected = Number(dom.progressReminderSelect.value);
      state.progressReminderMinutes = [0, 5, 10, 20, 30, 60].includes(selected)
        ? selected
        : DEFAULT_PROGRESS_REMINDER_MINUTES;
      try {
        window.localStorage.setItem(PROGRESS_REMINDER_STORAGE_KEY, String(state.progressReminderMinutes));
      } catch (_error) {
        // Keep the in-memory setting when localStorage is unavailable.
      }
      resetProgressReminder(state.progressReminderMinutes > 0 && isCurrentRunActive());
      if (state.progressReminder.armed) maybeNotifyProgress(state.activeRun?.status);
    });
    dom.previewButton.addEventListener("click", showCommandDialog);
    dom.closeCommandDialog.addEventListener("click", closeCommandDialog);
    dom.commandDialog.addEventListener("click", (event) => {
      if (event.target === dom.commandDialog) closeCommandDialog();
    });
    dom.copyCommandInline.addEventListener("click", copyCommand);
    dom.copyCommandDialog.addEventListener("click", copyCommand);
    dom.startButton.addEventListener("click", startRun);
    dom.startFromDialog.addEventListener("click", startRun);
    dom.stopButton.addEventListener("click", stopRun);
    dom.closeDepthPreview.addEventListener("click", () => closeDepthPreview(true));
    dom.depthPreviewPrevious.addEventListener("click", () => stepDepthFrame(-1));
    dom.depthPreviewNext.addEventListener("click", () => stepDepthFrame(1));
    dom.depthPreviewFollow.addEventListener("change", () => {
      state.depthPreview.followLatest = dom.depthPreviewFollow.checked;
      if (state.depthPreview.followLatest && state.depthPreview.frames.length) {
        showDepthFrame(state.depthPreview.frames.length - 1);
      } else {
        renderDepthPreview();
      }
    });
    dom.depthPreviewWindow.addEventListener("keydown", (event) => {
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        stepDepthFrame(-1);
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        stepDepthFrame(1);
      } else if (event.key === "Home" && state.depthPreview.frames.length) {
        event.preventDefault();
        state.depthPreview.followLatest = false;
        showDepthFrame(0);
      } else if (event.key === "End" && state.depthPreview.frames.length) {
        event.preventDefault();
        state.depthPreview.followLatest = true;
        showDepthFrame(state.depthPreview.frames.length - 1);
      }
    });
    dom.resetParametersButton.addEventListener("click", resetParameters);
    dom.detailTab.addEventListener("click", () => switchDetailsTab("detail"));
    dom.parametersTab.addEventListener("click", () => switchDetailsTab("parameters"));
    dom.zoomOutButton.addEventListener("click", () => setZoom(state.view.scale / 1.18));
    dom.zoomInButton.addEventListener("click", () => setZoom(state.view.scale * 1.18));
    dom.zoomResetButton.addEventListener("click", () => setZoom(1));
    dom.fitViewButton.addEventListener("click", () => fitGraphToView(true));
    dom.clearLogsButton.addEventListener("click", clearLogs);
    dom.collapseDockButton.addEventListener("click", toggleDock);
    dom.logFilters.addEventListener("click", (event) => {
      const button = event.target.closest("[data-level]");
      if (!button) return;
      state.logLevel = button.dataset.level;
      dom.logFilters.querySelectorAll("[data-level]").forEach((item) => item.classList.toggle("is-active", item === button));
      renderLogs();
    });
    dom.openLeftPanel.addEventListener("click", () => openPanel("left"));
    dom.openRightPanel.addEventListener("click", () => openPanel("right"));
    dom.closeLeftPanel.addEventListener("click", closePanels);
    dom.closeRightPanel.addEventListener("click", closePanels);
    dom.panelScrim.addEventListener("click", closePanels);
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") closePanels();
    });
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden && state.progressReminder.titlePending) restoreDocumentTitle();
    });
    window.addEventListener("resize", () => {
      window.clearTimeout(state.resizeTimer);
      state.resizeTimer = window.setTimeout(() => {
        if (!state.view.manual) fitGraphToView(false);
        if (window.innerWidth > 1120) closePanels();
      }, 150);
    });
    bindCanvasInteractions();
  }

  function initialize() {
    cacheDom();
    initializeProgressReminder();
    bindEvents();
    state.elapsedTimer = window.setInterval(updateElapsed, 1000);
    loadWorkflows();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initialize, { once: true });
  else initialize();
})();
