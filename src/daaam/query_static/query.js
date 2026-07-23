(function () {
  "use strict";

  const STORAGE_PATH = "daaam.semanticQuery.lastMap";
  const STORAGE_HISTORY = "daaam.semanticQuery.history";
  const MAX_HISTORY = 12;
  const state = {
    map: null,
    result: null,
    selectedIndex: -1,
    maps: [],
    view: { scale: 1, x: 0, y: 0 },
    drag: null,
    history: readHistory(),
  };
  const dom = {};

  function bindDom() {
    const ids = [
      "map-loader", "map-path", "available-maps", "load-map-button",
      "service-state", "service-state-title", "service-state-copy", "map-name",
      "map-state", "map-summary-copy", "metric-objects", "metric-mesh",
      "metric-spatial", "metric-evidence", "query-form", "query-input", "top-k",
      "require-mesh", "query-button", "result-count", "decision-banner", "result-list",
      "history-list", "clear-history", "map-query-caption", "map-stage", "map-image",
      "map-placeholder", "map-loading", "map-loading-title", "map-loading-copy",
      "map-legend", "zoom-out", "zoom-reset", "zoom-in", "fullscreen-map",
      "download-map", "selected-node", "selected-position", "download-report",
      "evidence-state", "evidence-image", "evidence-placeholder", "evidence-badge",
      "open-evidence", "download-evidence", "object-rank", "object-id",
      "object-geometry", "score-ring", "object-score", "object-description",
      "meta-position", "meta-dimensions", "meta-label", "meta-source", "meta-frame",
      "meta-time", "meta-mask", "meta-camera", "meta-hash", "toast-region",
    ];
    for (const id of ids) dom[toCamel(id)] = document.getElementById(id);
  }

  function toCamel(value) {
    return value.replace(/-([a-z])/g, (_match, character) => character.toUpperCase());
  }

  async function api(path, options) {
    const requestOptions = { ...(options || {}) };
    const timeout = requestOptions.timeout || 120000;
    delete requestOptions.timeout;
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), timeout);
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
      const contentType = response.headers.get("content-type") || "";
      const payload = contentType.includes("application/json")
        ? await response.json()
        : await response.text();
      if (!response.ok) {
        const detail = payload && typeof payload === "object" ? payload.detail : payload;
        throw new Error(detail || `请求失败（HTTP ${response.status}）`);
      }
      return payload;
    } catch (error) {
      if (error && error.name === "AbortError") throw new Error("请求超时，请检查查询服务");
      throw error;
    } finally {
      window.clearTimeout(timer);
    }
  }

  function setService(mode, title, copy) {
    dom.serviceState.classList.toggle("is-online", mode === "online");
    dom.serviceState.classList.toggle("is-error", mode === "error");
    dom.serviceStateTitle.textContent = title;
    dom.serviceStateCopy.textContent = copy;
  }

  function setBusy(button, busy, label) {
    button.classList.toggle("is-busy", busy);
    button.disabled = busy;
    const labelElement = button.querySelector(".button-label");
    if (labelElement && label) labelElement.textContent = label;
  }

  function toast(message, level) {
    const element = document.createElement("div");
    element.className = `toast toast-${level || "info"}`;
    element.textContent = message;
    dom.toastRegion.appendChild(element);
    window.setTimeout(() => element.remove(), 4600);
  }

  function number(value, digits) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed.toFixed(digits === undefined ? 3 : digits) : "—";
  }

  function vector(value, suffix) {
    if (!Array.isArray(value) || value.length < 3) return "—";
    return `${number(value[0])}, ${number(value[1])}, ${number(value[2])}${suffix || ""}`;
  }

  function geometryLabel(value) {
    const labels = {
      mesh_bound: "Mesh 绑定",
      spatial_only: "空间实体",
      image_only: "仅图像",
    };
    return labels[value] || value || "未知几何";
  }

  function rejectionLabel(value) {
    const labels = {
      below_min_similarity: "相似度低于接受阈值",
      below_min_margin: "候选过于相似，无法唯一判断",
    };
    return labels[value] || value || "未找到可信匹配";
  }

  function resetView() {
    state.view = { scale: 1, x: 0, y: 0 };
    applyView();
  }

  function applyView() {
    const { scale, x, y } = state.view;
    dom.mapImage.style.transform = `translate(calc(-50% + ${x}px), calc(-50% + ${y}px)) scale(${scale})`;
    dom.zoomReset.textContent = `${Math.round(scale * 100)}%`;
  }

  function setZoom(nextScale) {
    state.view.scale = Math.min(5, Math.max(0.5, nextScale));
    applyView();
  }

  function showMapImage(url, caption, marked) {
    dom.mapPlaceholder.classList.add("is-hidden");
    dom.mapLoading.classList.remove("is-hidden");
    dom.mapLoadingTitle.textContent = marked ? "正在载入查询定位图" : "正在生成 Mesh 俯视图";
    dom.mapLoadingCopy.textContent = marked
      ? "物体 ID、位置和证据相机将叠加到地图中。"
      : "读取当前 DSG 中的 RGB mesh 顶点。";
    dom.mapImage.style.display = "none";
    dom.mapImage.onload = () => {
      dom.mapLoading.classList.add("is-hidden");
      dom.mapImage.style.display = "block";
      resetView();
    };
    dom.mapImage.onerror = () => {
      dom.mapLoading.classList.add("is-hidden");
      toast("俯视图加载失败", "error");
    };
    dom.mapImage.src = url;
    dom.mapQueryCaption.textContent = caption;
    dom.downloadMap.href = url;
    dom.downloadMap.classList.remove("is-disabled");
    dom.downloadMap.setAttribute("download", marked ? "semantic_query_topdown.png" : "mesh_topdown.png");
    dom.mapLegend.classList.toggle("is-hidden", !marked);
  }

  function resetResults() {
    state.result = null;
    state.selectedIndex = -1;
    dom.resultCount.textContent = "0";
    dom.decisionBanner.classList.add("is-hidden");
    dom.resultList.replaceChildren(emptyResult("等待查询", "候选物体会按语义相似度排列。"));
    dom.downloadReport.classList.add("is-disabled");
    dom.downloadReport.removeAttribute("href");
    dom.downloadReport.textContent = "等待查询";
    resetEvidence();
  }

  function emptyResult(title, copy) {
    const root = document.createElement("div");
    root.className = "empty-state small";
    const orbit = document.createElement("span");
    orbit.className = "empty-orbit";
    orbit.setAttribute("aria-hidden", "true");
    orbit.append(document.createElement("i"), document.createElement("i"), document.createElement("i"));
    const strong = document.createElement("strong");
    strong.textContent = title;
    const paragraph = document.createElement("p");
    paragraph.textContent = copy;
    root.append(orbit, strong, paragraph);
    return root;
  }

  function resetEvidence() {
    dom.evidenceImage.style.display = "none";
    dom.evidenceImage.removeAttribute("src");
    dom.evidencePlaceholder.classList.remove("is-hidden");
    dom.evidencePlaceholder.querySelector("strong").textContent = "查询后显示证据";
    dom.evidencePlaceholder.querySelector("p").textContent = "图片来自对应对象的精确 FastSAM mask 帧。";
    dom.evidenceBadge.classList.add("is-hidden");
    dom.evidenceState.textContent = "未选择";
    dom.evidenceState.classList.remove("is-ready");
    for (const link of [dom.openEvidence, dom.downloadEvidence]) {
      link.classList.add("is-disabled");
      link.removeAttribute("href");
    }
    dom.objectRank.textContent = "—";
    dom.objectId.textContent = "尚未选择物体";
    dom.objectGeometry.textContent = "—";
    dom.scoreRing.style.setProperty("--score", 0);
    dom.objectScore.textContent = "—";
    dom.objectDescription.textContent = "查询候选的描述、坐标与证据元数据会显示在这里。";
    for (const key of [
      "metaPosition", "metaDimensions", "metaLabel", "metaSource", "metaFrame",
      "metaTime", "metaMask", "metaCamera", "metaHash",
    ]) dom[key].textContent = "—";
    dom.selectedNode.textContent = "—";
    dom.selectedPosition.textContent = "x — · y — · z —";
  }

  function setControlsEnabled(enabled) {
    dom.queryInput.disabled = !enabled;
    dom.topK.disabled = !enabled;
    dom.requireMesh.disabled = !enabled;
    dom.queryButton.disabled = !enabled;
    for (const button of document.querySelectorAll("[data-query]")) button.disabled = !enabled;
  }

  function renderMapSummary(info) {
    dom.mapName.textContent = info.run_name;
    dom.mapState.textContent = "已就绪";
    dom.mapState.classList.add("is-ready");
    dom.mapSummaryCopy.textContent = `${info.relative_run} · ${info.encoder_device} 编码器`;
    dom.metricObjects.textContent = String(info.queryable_objects);
    dom.metricMesh.textContent = String(info.geometry_counts.mesh_bound || 0);
    dom.metricSpatial.textContent = String(info.geometry_counts.spatial_only || 0);
    dom.metricEvidence.textContent = `${Math.round(info.evidence_coverage * 100)}%`;
  }

  async function openMap(pathValue, options) {
    const runPath = String(pathValue || "").trim();
    if (!runPath) {
      toast("请输入语义地图输出路径", "error");
      dom.mapPath.focus();
      return;
    }
    setBusy(dom.loadMapButton, true, "正在加载");
    setControlsEnabled(false);
    dom.mapPlaceholder.classList.add("is-hidden");
    dom.mapLoading.classList.remove("is-hidden");
    dom.mapLoadingTitle.textContent = "正在加载语义地图";
    dom.mapLoadingCopy.textContent = "首次加载需要初始化多语种编码器，请稍候。";
    try {
      const info = await api("/api/map/open", {
        method: "POST",
        body: { run_path: runPath },
      });
      state.map = info;
      dom.mapPath.value = info.run_path;
      window.localStorage.setItem(STORAGE_PATH, info.run_path);
      renderMapSummary(info);
      resetResults();
      setControlsEnabled(true);
      showMapImage(info.mesh_preview_url, "完整 RGB mesh", false);
      setService("online", "查询服务就绪", `${info.queryable_objects} 个可查询对象`);
      if (!(options && options.silent)) toast(`已加载 ${info.run_name}`, "success");
      dom.queryInput.focus();
    } catch (error) {
      state.map = null;
      setControlsEnabled(false);
      dom.mapLoading.classList.add("is-hidden");
      dom.mapPlaceholder.classList.remove("is-hidden");
      dom.mapState.textContent = "加载失败";
      dom.mapState.classList.remove("is-ready");
      setService("error", "地图加载失败", error.message);
      toast(error.message, "error");
    } finally {
      setBusy(dom.loadMapButton, false, "加载地图");
    }
  }

  function buildResultCard(match, index) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "result-card";
    card.dataset.index = String(index);

    const head = document.createElement("div");
    head.className = "result-card-head";
    const identity = document.createElement("div");
    const rank = document.createElement("span");
    rank.className = "rank-chip";
    rank.textContent = `#${match.rank}`;
    const id = document.createElement("strong");
    id.textContent = match.node_id;
    identity.append(rank, id);
    const score = document.createElement("span");
    score.className = "result-score";
    score.textContent = number(match.score, 4);
    head.append(identity, score);

    const description = document.createElement("p");
    description.textContent = match.description || "无物体描述";
    const foot = document.createElement("div");
    foot.className = "result-card-foot";
    const geometry = document.createElement("span");
    geometry.className = "geometry-chip";
    geometry.textContent = geometryLabel(match.geometry_status);
    const evidence = document.createElement("span");
    evidence.className = `evidence-chip${match.evidence ? "" : " is-missing"}`;
    evidence.textContent = match.evidence ? "有图片证据" : "无精确证据";
    foot.append(geometry, evidence);
    card.append(head, description, foot);
    card.addEventListener("click", () => selectCandidate(index));
    return card;
  }

  function renderResult(result) {
    dom.resultCount.textContent = String(result.matches.length);
    dom.decisionBanner.classList.remove("is-hidden", "is-rejected");
    if (result.found) {
      const margin = result.top1_margin === null ? "—" : number(result.top1_margin, 4);
      dom.decisionBanner.textContent = `查询已接受 · Top score ${number(result.top_score, 4)} · Margin ${margin}`;
    } else {
      dom.decisionBanner.classList.add("is-rejected");
      dom.decisionBanner.textContent = `未接受：${rejectionLabel(result.rejection_reason)} · Top score ${number(result.top_score, 4)}`;
    }
    dom.resultList.replaceChildren();
    if (!result.matches.length) {
      dom.resultList.appendChild(emptyResult("没有可信候选", "可以补充颜色、材质或物体类别后重试。"));
      resetEvidence();
      return;
    }
    result.matches.forEach((match, index) => dom.resultList.appendChild(buildResultCard(match, index)));
    selectCandidate(0);
  }

  function selectCandidate(index) {
    if (!state.result || !state.result.matches[index]) return;
    state.selectedIndex = index;
    const match = state.result.matches[index];
    for (const card of dom.resultList.querySelectorAll(".result-card")) {
      card.classList.toggle("is-selected", Number(card.dataset.index) === index);
    }
    dom.objectRank.textContent = `#${match.rank}`;
    dom.objectId.textContent = match.node_id;
    dom.objectGeometry.textContent = geometryLabel(match.geometry_status);
    dom.objectScore.textContent = number(match.score, 3);
    dom.scoreRing.style.setProperty("--score", Math.min(1, Math.max(0, Number(match.score) || 0)));
    dom.objectDescription.textContent = match.description || "无物体描述";
    dom.metaPosition.textContent = vector(match.position_m, " m");
    dom.metaDimensions.textContent = vector(match.dimensions_m, " m");
    dom.metaLabel.textContent = String(match.semantic_label ?? "—");
    dom.metaSource.textContent = match.source || "—";
    dom.selectedNode.textContent = match.node_id;
    if (Array.isArray(match.position_m)) {
      dom.selectedPosition.textContent = `x ${number(match.position_m[0])} · y ${number(match.position_m[1])} · z ${number(match.position_m[2])} m`;
    } else {
      dom.selectedPosition.textContent = "x — · y — · z —";
    }

    const evidence = match.evidence;
    if (!evidence) {
      dom.evidenceImage.style.display = "none";
      dom.evidenceImage.removeAttribute("src");
      dom.evidencePlaceholder.classList.remove("is-hidden");
      dom.evidencePlaceholder.querySelector("strong").textContent = "没有精确图片证据";
      dom.evidencePlaceholder.querySelector("p").textContent = "该物体仍可按可靠三维坐标定位，不会使用其他物体图片替代。";
      dom.evidenceBadge.classList.add("is-hidden");
      dom.evidenceState.textContent = "证据缺失";
      dom.evidenceState.classList.remove("is-ready");
      for (const link of [dom.openEvidence, dom.downloadEvidence]) {
        link.classList.add("is-disabled");
        link.removeAttribute("href");
      }
      dom.metaFrame.textContent = "—";
      dom.metaTime.textContent = "—";
      dom.metaMask.textContent = "—";
      dom.metaCamera.textContent = "—";
      dom.metaHash.textContent = "—";
      return;
    }

    dom.evidencePlaceholder.classList.add("is-hidden");
    dom.evidenceImage.style.display = "block";
    dom.evidenceImage.src = evidence.image_url;
    dom.evidenceImage.onerror = () => toast(`证据图片读取失败：${match.node_id}`, "error");
    dom.evidenceBadge.classList.remove("is-hidden");
    dom.evidenceState.textContent = "已验证";
    dom.evidenceState.classList.add("is-ready");
    for (const link of [dom.openEvidence, dom.downloadEvidence]) {
      link.href = evidence.image_url;
      link.classList.remove("is-disabled");
    }
    dom.downloadEvidence.setAttribute("download", `${match.node_id.replace(/[^A-Za-z0-9_-]+/g, "_")}_evidence.png`);
    dom.metaFrame.textContent = String(evidence.frame_index);
    dom.metaTime.textContent = evidence.observed_s === null ? "—" : `${number(evidence.observed_s)} s`;
    dom.metaMask.textContent = Number(evidence.mask_pixels).toLocaleString("zh-CN");
    dom.metaCamera.textContent = vector(evidence.camera_position_m, " m");
    dom.metaHash.textContent = evidence.image_sha256 || "—";
  }

  async function runQuery(queryValue) {
    if (!state.map) {
      toast("请先加载语义地图", "error");
      return;
    }
    const query = String(queryValue || dom.queryInput.value).trim();
    if (!query) {
      toast("请输入要查询的物体", "error");
      dom.queryInput.focus();
      return;
    }
    dom.queryInput.value = query;
    setBusy(dom.queryButton, true, "正在查询");
    dom.mapLoading.classList.remove("is-hidden");
    dom.mapLoadingTitle.textContent = "正在执行语义查询";
    dom.mapLoadingCopy.textContent = "检索候选、复制证据并生成标注俯视图。";
    try {
      const result = await api("/api/query", {
        method: "POST",
        body: {
          run_path: state.map.run_path,
          query,
          top_k: Number(dom.topK.value),
          require_mesh: dom.requireMesh.checked,
        },
      });
      state.result = result;
      renderResult(result);
      showMapImage(result.topdown_image_url, `“${query}” · ${result.matches.length} 个候选`, true);
      dom.downloadReport.href = result.report_url;
      dom.downloadReport.textContent = "下载 query_result.json";
      dom.downloadReport.classList.remove("is-disabled");
      addHistory(query);
      toast(result.found ? `找到 ${result.matches[0].node_id}` : "没有找到可信匹配", result.found ? "success" : "info");
    } catch (error) {
      dom.mapLoading.classList.add("is-hidden");
      toast(error.message, "error");
    } finally {
      setBusy(dom.queryButton, false, "在地图中查询");
      if (state.map) dom.queryButton.disabled = false;
    }
  }

  function readHistory() {
    try {
      const value = JSON.parse(window.localStorage.getItem(STORAGE_HISTORY) || "[]");
      return Array.isArray(value) ? value.filter((item) => typeof item === "string").slice(0, MAX_HISTORY) : [];
    } catch (_error) {
      return [];
    }
  }

  function addHistory(query) {
    state.history = [query, ...state.history.filter((item) => item !== query)].slice(0, MAX_HISTORY);
    window.localStorage.setItem(STORAGE_HISTORY, JSON.stringify(state.history));
    renderHistory();
  }

  function renderHistory() {
    dom.historyList.replaceChildren();
    if (!state.history.length) {
      const empty = document.createElement("span");
      empty.className = "history-empty";
      empty.textContent = "暂无历史";
      dom.historyList.appendChild(empty);
      return;
    }
    for (const query of state.history) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "history-item";
      button.textContent = query;
      button.title = `再次查询：${query}`;
      button.addEventListener("click", () => {
        dom.queryInput.value = query;
        if (state.map) runQuery(query);
      });
      dom.historyList.appendChild(button);
    }
  }

  async function initializeService() {
    try {
      const [health, discovered] = await Promise.all([api("/api/health"), api("/api/maps")]);
      state.maps = Array.isArray(discovered.maps) ? discovered.maps : [];
      dom.availableMaps.replaceChildren();
      for (const item of state.maps) {
        const option = document.createElement("option");
        option.value = item.run_path;
        option.label = `${item.run_name}${item.queryable_objects ? ` · ${item.queryable_objects} objects` : ""}`;
        dom.availableMaps.appendChild(option);
      }
      setService("online", "独立查询服务在线", health.output_root);

      const params = new URLSearchParams(window.location.search);
      const requested = params.get("path") || window.localStorage.getItem(STORAGE_PATH);
      const fallback = state.maps.find((item) => item.run_name.includes("fast_semantic_10cm")) || state.maps[0];
      const initial = requested || (fallback && fallback.run_path);
      if (initial) {
        dom.mapPath.value = initial;
        await openMap(initial, { silent: true });
        const initialQuery = params.get("q");
        if (state.map && initialQuery) {
          dom.requireMesh.checked = params.get("require_mesh") === "1";
          dom.queryInput.value = initialQuery;
          await runQuery(initialQuery);
        }
      }
    } catch (error) {
      setService("error", "查询服务不可用", error.message);
      toast(error.message, "error");
    }
  }

  function bindEvents() {
    dom.mapLoader.addEventListener("submit", (event) => {
      event.preventDefault();
      openMap(dom.mapPath.value);
    });
    dom.queryForm.addEventListener("submit", (event) => {
      event.preventDefault();
      runQuery();
    });
    dom.queryInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
        event.preventDefault();
        runQuery();
      }
    });
    for (const button of document.querySelectorAll("[data-query]")) {
      button.addEventListener("click", () => {
        dom.queryInput.value = button.dataset.query || "";
        runQuery(button.dataset.query);
      });
    }
    dom.clearHistory.addEventListener("click", () => {
      state.history = [];
      window.localStorage.removeItem(STORAGE_HISTORY);
      renderHistory();
    });

    dom.zoomIn.addEventListener("click", () => setZoom(state.view.scale * 1.2));
    dom.zoomOut.addEventListener("click", () => setZoom(state.view.scale / 1.2));
    dom.zoomReset.addEventListener("click", resetView);
    dom.mapStage.addEventListener("dblclick", resetView);
    dom.mapStage.addEventListener("wheel", (event) => {
      event.preventDefault();
      setZoom(state.view.scale * (event.deltaY < 0 ? 1.12 : 0.89));
    }, { passive: false });
    dom.mapStage.addEventListener("pointerdown", (event) => {
      if (event.button !== 0 || !dom.mapImage.src) return;
      state.drag = { startX: event.clientX, startY: event.clientY, x: state.view.x, y: state.view.y };
      dom.mapStage.classList.add("is-dragging");
      dom.mapStage.setPointerCapture(event.pointerId);
    });
    dom.mapStage.addEventListener("pointermove", (event) => {
      if (!state.drag) return;
      state.view.x = state.drag.x + event.clientX - state.drag.startX;
      state.view.y = state.drag.y + event.clientY - state.drag.startY;
      applyView();
    });
    const endDrag = () => {
      state.drag = null;
      dom.mapStage.classList.remove("is-dragging");
    };
    dom.mapStage.addEventListener("pointerup", endDrag);
    dom.mapStage.addEventListener("pointercancel", endDrag);
    dom.fullscreenMap.addEventListener("click", async () => {
      try {
        if (document.fullscreenElement) await document.exitFullscreen();
        else await dom.mapPanel?.requestFullscreen?.();
      } catch (_error) {
        await dom.mapStage.requestFullscreen();
      }
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "/" && document.activeElement !== dom.queryInput) {
        event.preventDefault();
        dom.queryInput.focus();
      }
      if (event.key === "Escape" && !document.fullscreenElement) resetView();
    });
  }

  function initialize() {
    bindDom();
    dom.mapPanel = document.querySelector(".map-panel");
    bindEvents();
    renderHistory();
    resetEvidence();
    setControlsEnabled(false);
    initializeService();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialize, { once: true });
  } else {
    initialize();
  }
})();
