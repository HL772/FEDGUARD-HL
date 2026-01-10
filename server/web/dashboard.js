if (typeof Chart === "undefined") {
  console.error("Chart.js is not loaded.");
}

// Dashboard 前端逻辑（AGENT.md 3.1.G / 5.x）：
// - 消费 MetricsModule 推送的轮次指标
// - 渲染客户端/服务端曲线、告警与管理视图
const state = {
  rounds: [],
  loss: [],
  accuracy: [],
  epsilonMax: [],
  epsilonAvg: [],
  epsilonAccountant: [],
  epsilonTarget: [],
  noiseMultiplier: [],
  clipRate: [],
  uploadBytes: [],
  downloadBytes: [],
  compressedRatio: [],
  serverEvalLoss: [],
  serverEvalAcc: [],
  serverUpdateNorm: [],
  fairnessAvg: [],
  fairnessMin: [],
  fairnessStd: [],
  fairnessJain: [],
  lastRound: 0,
  metricsByRound: {},
  clientUpdatesRound: 0,
  clientUpdatesPinned: false,
  excludedClients: new Set(),
  processedRounds: new Set(),
  latestMetric: null,
  detectionTotals: {
    detected: 0,
    malicious: 0,
    truePositive: 0,
  },
  attackMethods: new Set(),
};

function shortId(value) {
  if (!value) {
    return "-";
  }
  return value.slice(0, 8);
}

function formatClientLabel(clientId, clientName) {
  if (clientName) {
    return clientName;
  }
  return shortId(clientId);
}

function formatMetric(value, digits) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) {
    return "-";
  }
  return Number(value).toFixed(digits);
}

function resetSummaryCards() {
  const mapping = {
    "summary-round": "-",
    "summary-participants": "-",
    "summary-round-time": "-",
    "summary-dp": "-",
    "summary-agg": "-",
    "summary-sampling": "-",
    "summary-epsilon": "-",
    "summary-upload": "-",
    "summary-compress": "-",
    "summary-excluded": "-",
    "summary-timeout": "-",
  };
  Object.keys(mapping).forEach((key) => {
    const el = document.getElementById(key);
    if (el) {
      el.textContent = mapping[key];
    }
  });
}

function resetUiState() {
  state.rounds = [];
  state.loss = [];
  state.accuracy = [];
  state.epsilonMax = [];
  state.epsilonAvg = [];
  state.epsilonAccountant = [];
  state.epsilonTarget = [];
  state.noiseMultiplier = [];
  state.clipRate = [];
  state.uploadBytes = [];
  state.downloadBytes = [];
  state.compressedRatio = [];
  state.serverEvalLoss = [];
  state.serverEvalAcc = [];
  state.serverUpdateNorm = [];
  state.fairnessAvg = [];
  state.fairnessMin = [];
  state.fairnessStd = [];
  state.fairnessJain = [];
  state.lastRound = 0;
  state.metricsByRound = {};
  state.clientUpdatesRound = 0;
  state.clientUpdatesPinned = false;
  state.excludedClients = new Set();
  state.processedRounds = new Set();
  state.latestMetric = null;
  state.detectionTotals = { detected: 0, malicious: 0, truePositive: 0 };
  state.attackMethods = new Set();

  const roundEl = document.getElementById("round-id");
  if (roundEl) {
    roundEl.textContent = "-";
  }
  const clientCountEl = document.getElementById("clients-count");
  if (clientCountEl) {
    clientCountEl.textContent = "0";
  }
  updateClientUpdates([]);
  updateSummary([]);
  updateSecuritySummary({});
  updateSimilarityRank({}, new Map());
  const alertBody = document.getElementById("alert-body");
  if (alertBody) {
    alertBody.innerHTML = "";
  }
  resetSummaryCards();
  rebuildSeries();
}

function updateSpinner(active) {
  const spinner = document.getElementById("spinner");
  if (spinner) {
    spinner.style.display = active ? "inline-block" : "none";
  }
}

function setLaunchStatus(text, isError) {
  const el = document.getElementById("launch-status");
  const dot = document.getElementById("status-dot");
  const startBtn = document.getElementById("launch-start");
  const stopBtn = document.getElementById("launch-stop");
  if (!el) {
    return;
  }
  // Remove "状态：" prefix if present for cleaner display in header
  el.textContent = text.replace(/^状态：/, "");
  
  if (startBtn && stopBtn) {
    const running =
      text.includes("运行中") || text.includes("启动中") || text.includes("停止中");
    startBtn.classList.toggle("is-active", !running);
    stopBtn.classList.toggle("is-active", running);
  }

  if (dot) {
    if (isError) {
      dot.style.color = "#ef4444";
      updateSpinner(false);
    } else if (text.includes("运行中") || text.includes("启动中") || text.includes("停止中")) {
      dot.style.color = "#10b981";
      updateSpinner(true);
    } else {
      dot.style.color = "#94a3b8";
      updateSpinner(false);
    }
  }
}

function getLaunchElements() {
  return {
    numClients: document.getElementById("launch-num-clients"),
    clientsPerRound: document.getElementById("launch-clients-per-round"),
    clientBatch: document.getElementById("launch-client-batch"),
    maxRounds: document.getElementById("launch-max-rounds"),
    onlineTtl: document.getElementById("launch-online-ttl"),
    dpEnabled: document.getElementById("launch-dp-enabled"),
    dpMode: document.getElementById("launch-dp-mode"),
    dpClip: document.getElementById("launch-dp-clip"),
    dpNoise: document.getElementById("launch-dp-noise"),
    dpDelta: document.getElementById("launch-dp-delta"),
    dpTarget: document.getElementById("launch-dp-target"),
    dpSchedule: document.getElementById("launch-dp-schedule"),
    dpSigmaRatio: document.getElementById("launch-dp-sigma-ratio"),
    dpAdaptive: document.getElementById("launch-dp-adaptive"),
    dpPercentile: document.getElementById("launch-dp-percentile"),
    dpEma: document.getElementById("launch-dp-ema"),
    compressionEnabled: document.getElementById("launch-compression-enabled"),
    topkRatio: document.getElementById("launch-topk-ratio"),
    quantBits: document.getElementById("launch-quant-bits"),
    errorFeedback: document.getElementById("launch-error-feedback"),
    secureAgg: document.getElementById("launch-secure-agg"),
    maliciousDetect: document.getElementById("launch-malicious-detect"),
    robustMethod: document.getElementById("launch-robust-method"),
    trimRatio: document.getElementById("launch-trim-ratio"),
    byzantineF: document.getElementById("launch-byzantine-f"),
    lossThreshold: document.getElementById("launch-loss-threshold"),
    normThreshold: document.getElementById("launch-norm-threshold"),
    requireBoth: document.getElementById("launch-require-both"),
    minMad: document.getElementById("launch-min-mad"),
    cosineEnabled: document.getElementById("launch-cosine-enabled"),
    cosineThreshold: document.getElementById("launch-cosine-threshold"),
    cosineTopk: document.getElementById("launch-cosine-topk"),
    attackEnabled: document.getElementById("launch-attack-enabled"),
    attackMethod: document.getElementById("launch-attack-method"),
    attackScale: document.getElementById("launch-attack-scale"),
    attackFraction: document.getElementById("launch-attack-fraction"),
    attackRanks: document.getElementById("launch-attack-ranks"),
    attackLabelFlip: document.getElementById("launch-attack-label-flip"),
    attackLossScale: document.getElementById("launch-attack-loss-scale"),
    attackAccScale: document.getElementById("launch-attack-acc-scale"),
    trainAlgo: document.getElementById("launch-train-algo"),
    trainLr: document.getElementById("launch-train-lr"),
    trainEpochs: document.getElementById("launch-train-epochs"),
    fedproxMu: document.getElementById("launch-fedprox-mu"),
    deadlineMs: document.getElementById("launch-deadline-ms"),
    serverLr: document.getElementById("launch-server-lr"),
    serverClip: document.getElementById("launch-server-clip"),
    privateHeadLr: document.getElementById("launch-private-head-lr"),
    privateHeadEpochs: document.getElementById("launch-private-head-epochs"),
    dataAlpha: document.getElementById("launch-data-alpha"),
    samplingEnabled: document.getElementById("launch-sampling-enabled"),
    samplingStrategy: document.getElementById("launch-sampling-strategy"),
    samplingEpsilon: document.getElementById("launch-sampling-epsilon"),
    samplingEma: document.getElementById("launch-sampling-ema"),
    samplingTimeoutPenalty: document.getElementById("launch-sampling-timeout-penalty"),
    samplingAnomalyPenalty: document.getElementById("launch-sampling-anomaly-penalty"),
    samplingFairness: document.getElementById("launch-sampling-fairness"),
    samplingSoftmax: document.getElementById("launch-sampling-softmax"),
    timeoutEnabled: document.getElementById("launch-timeout-enabled"),
    timeoutRanks: document.getElementById("launch-timeout-ranks"),
    timeoutCooldown: document.getElementById("launch-timeout-cooldown"),
    downloadData: document.getElementById("launch-download-data"),
    stayOnline: document.getElementById("launch-stay-online"),
  };
}

function parseIntField(el, fallback) {
  if (!el) {
    return fallback;
  }
  const value = parseInt(el.value, 10);
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

function parseFloatField(el) {
  if (!el || el.value === "") {
    return null;
  }
  const value = parseFloat(el.value);
  return Number.isFinite(value) ? value : null;
}

function parseIntOptional(el) {
  if (!el || el.value === "") {
    return null;
  }
  const value = parseInt(el.value, 10);
  return Number.isFinite(value) ? value : null;
}

function parseCsvInts(el) {
  if (!el || el.value.trim() === "") {
    return null;
  }
  return el.value
    .split(",")
    .map((value) => parseInt(value.trim(), 10))
    .filter((value) => Number.isFinite(value));
}

function setFieldValue(el, value) {
  if (!el) {
    return;
  }
  if (value === null || value === undefined) {
    el.value = "";
    return;
  }
  el.value = value;
}

function setChecked(el, value) {
  if (!el) {
    return;
  }
  el.checked = !!value;
}

function getServerUrl() {
  let origin = window.location.origin || "";
  if (!origin) {
    return "";
  }
  origin = origin.replace("0.0.0.0", "127.0.0.1");
  origin = origin.replace("[::]", "127.0.0.1");
  origin = origin.replace("://localhost", "://127.0.0.1");
  return origin;
}

async function loadDefaultConfig() {
  try {
    const resp = await fetch("/api/v1/config/default");
    if (!resp.ok) {
      return;
    }
    const payload = await resp.json();
    const cfg = payload.config || {};
    const runtime = payload.runtime || {};
    const elements = getLaunchElements();

    const defaultClients = runtime.num_clients ?? 14;
    setFieldValue(elements.numClients, defaultClients);
    setFieldValue(
      elements.clientsPerRound,
      runtime.clients_per_round ?? 8
    );
    setFieldValue(
      elements.clientBatch,
      runtime.client_batch_size ?? defaultClients
    );
    setFieldValue(elements.maxRounds, runtime.max_rounds ?? 20);
    setFieldValue(elements.onlineTtl, runtime.online_ttl_sec ?? 60);

    const train = cfg.train || {};
    setFieldValue(elements.trainAlgo, train.algo || "fedavg");
    setFieldValue(elements.trainLr, train.lr);
    setFieldValue(elements.trainEpochs, train.epochs);
    setFieldValue(elements.fedproxMu, train.fedprox_mu);
    setFieldValue(elements.deadlineMs, train.deadline_ms);
    setFieldValue(elements.serverLr, train.server_lr);
    setFieldValue(elements.serverClip, train.server_update_clip);
    setFieldValue(elements.privateHeadLr, train.private_head_lr);
    setFieldValue(elements.privateHeadEpochs, train.private_head_epochs);

    const data = cfg.data || {};
    setFieldValue(elements.dataAlpha, data.alpha);

    const sampling = cfg.sampling || {};
    setChecked(elements.samplingEnabled, sampling.enabled);
    setFieldValue(elements.samplingStrategy, sampling.strategy || "random");
    setFieldValue(elements.samplingEpsilon, sampling.epsilon);
    setFieldValue(elements.samplingEma, sampling.score_ema);
    setFieldValue(elements.samplingTimeoutPenalty, sampling.timeout_penalty);
    setFieldValue(elements.samplingAnomalyPenalty, sampling.anomaly_penalty);
    setFieldValue(elements.samplingFairness, sampling.fairness_window);
    setFieldValue(elements.samplingSoftmax, sampling.softmax_temp);

    const dp = cfg.dp || {};
    setChecked(elements.dpEnabled, dp.enabled);
    setFieldValue(elements.dpMode, dp.mode || "off");
    setFieldValue(elements.dpClip, dp.clip_norm);
    setFieldValue(elements.dpNoise, dp.noise_multiplier);
    setFieldValue(elements.dpDelta, dp.delta);
    setFieldValue(elements.dpTarget, dp.target_epsilon);
    const schedule = dp.schedule || {};
    setFieldValue(elements.dpSchedule, schedule.type || "");
    setFieldValue(elements.dpSigmaRatio, schedule.sigma_end_ratio);
    const adaptive = dp.adaptive_clip || {};
    setChecked(elements.dpAdaptive, adaptive.enabled);
    setFieldValue(elements.dpPercentile, adaptive.percentile);
    setFieldValue(elements.dpEma, adaptive.ema);

    const compression = cfg.compression || {};
    setChecked(elements.compressionEnabled, compression.enabled);
    setFieldValue(elements.topkRatio, compression.topk_ratio);
    setFieldValue(elements.quantBits, compression.quant_bits || 16);
    setChecked(elements.errorFeedback, compression.error_feedback);

    const security = cfg.security || {};
    setChecked(elements.secureAgg, security.secure_aggregation);
    setChecked(elements.maliciousDetect, security.malicious_detection);
    setFieldValue(elements.robustMethod, security.robust_aggregation || "fedavg");
    setFieldValue(elements.trimRatio, security.trim_ratio);
    setFieldValue(elements.byzantineF, security.byzantine_f);
    setFieldValue(elements.lossThreshold, security.loss_threshold);
    setFieldValue(elements.normThreshold, security.norm_threshold);
    setChecked(elements.requireBoth, security.require_both);
    setFieldValue(elements.minMad, security.min_mad);
    const cosine = security.cosine_detection || {};
    setChecked(elements.cosineEnabled, cosine.enabled);
    setFieldValue(elements.cosineThreshold, cosine.threshold);
    setFieldValue(elements.cosineTopk, cosine.top_k);

    const attack = security.attack_simulation || {};
    setChecked(elements.attackEnabled, attack.enabled);
    setFieldValue(elements.attackMethod, attack.method || "scale");
    setFieldValue(elements.attackScale, attack.scale);
    setFieldValue(elements.attackFraction, attack.malicious_fraction);
    setFieldValue(elements.attackLossScale, attack.loss_scale);
    setFieldValue(elements.attackAccScale, attack.accuracy_scale);
    setChecked(elements.attackLabelFlip, attack.label_flip);
    if (elements.attackRanks && Array.isArray(attack.malicious_ranks)) {
      elements.attackRanks.value = attack.malicious_ranks.join(",");
    }

    const timeout = train.timeout_simulation || {};
    setChecked(elements.timeoutEnabled, timeout.enabled);
    if (elements.timeoutRanks) {
      const ranks = Array.isArray(timeout.client_ranks)
        ? timeout.client_ranks
        : Array.isArray(timeout.malicious_ranks)
          ? timeout.malicious_ranks
          : [];
      elements.timeoutRanks.value = ranks.map((value) => String(value)).join(",");
    }
    setFieldValue(elements.timeoutCooldown, timeout.cooldown_rounds);
  } catch (err) {
    console.warn("load default config failed", err);
  }
}

async function loadSessionStatus() {
  try {
    const resp = await fetch("/api/v1/session/status");
    if (!resp.ok) {
      return;
    }
    const payload = await resp.json();
    const running = !!payload.running;
    const status = String(payload.status || "");
    if (running) {
      setLaunchStatus("状态：运行中", false);
    } else if (status === "completed") {
      setLaunchStatus("状态：已结束", false);
    } else if (status === "stopped") {
      setLaunchStatus("状态：已停止", false);
    } else if (status === "error") {
      setLaunchStatus("状态：异常", true);
    } else {
      setLaunchStatus("状态：未启动", false);
    }
    const params = payload.params || {};
    const elements = getLaunchElements();
    if (elements.numClients && params.num_clients) {
      elements.numClients.value = params.num_clients;
    }
    if (elements.clientsPerRound && params.clients_per_round) {
      elements.clientsPerRound.value = params.clients_per_round;
    }
    if (elements.clientBatch && params.client_batch_size) {
      elements.clientBatch.value = params.client_batch_size;
    }
    if (elements.maxRounds && params.max_rounds) {
      elements.maxRounds.value = params.max_rounds;
    }
    if (elements.dpEnabled && params.dp_enabled !== undefined && params.dp_enabled !== null) {
      elements.dpEnabled.checked = !!params.dp_enabled;
    }
    if (elements.dpMode && params.dp_mode) {
      elements.dpMode.value = params.dp_mode;
    }
    if (elements.trainAlgo && params.train_algo) {
      elements.trainAlgo.value = params.train_algo;
    }
  } catch (err) {
    console.warn("load session status failed", err);
  }
}

async function startSession() {
  const elements = getLaunchElements();
  const numClients = parseIntField(elements.numClients, 3);
  const payload = {
    server_url: getServerUrl(),
    num_clients: numClients,
    clients_per_round: parseIntField(elements.clientsPerRound, numClients),
    client_batch_size: parseIntField(elements.clientBatch, 1),
    max_rounds: parseIntField(elements.maxRounds, 3),
    online_ttl_sec: parseFloatField(elements.onlineTtl),
    dp_enabled: elements.dpEnabled ? elements.dpEnabled.checked : undefined,
    dp_mode: elements.dpMode ? elements.dpMode.value : "",
    dp_clip_norm: parseFloatField(elements.dpClip),
    dp_noise_multiplier: parseFloatField(elements.dpNoise),
    dp_delta: parseFloatField(elements.dpDelta),
    dp_target_epsilon: parseFloatField(elements.dpTarget),
    dp_schedule_type: elements.dpSchedule ? elements.dpSchedule.value : "",
    dp_sigma_end_ratio: parseFloatField(elements.dpSigmaRatio),
    dp_adaptive_clip_enabled: elements.dpAdaptive ? elements.dpAdaptive.checked : undefined,
    dp_adaptive_clip_percentile: parseFloatField(elements.dpPercentile),
    dp_adaptive_clip_ema: parseFloatField(elements.dpEma),
    compression_enabled: elements.compressionEnabled
      ? elements.compressionEnabled.checked
      : undefined,
    compression_topk_ratio: parseFloatField(elements.topkRatio),
    compression_quant_bits: parseIntOptional(elements.quantBits),
    compression_error_feedback: elements.errorFeedback
      ? elements.errorFeedback.checked
      : undefined,
    secure_aggregation: elements.secureAgg ? elements.secureAgg.checked : undefined,
    malicious_detection: elements.maliciousDetect ? elements.maliciousDetect.checked : undefined,
    robust_aggregation: elements.robustMethod ? elements.robustMethod.value : "",
    trim_ratio: parseFloatField(elements.trimRatio),
    byzantine_f: parseIntOptional(elements.byzantineF),
    loss_threshold: parseFloatField(elements.lossThreshold),
    norm_threshold: parseFloatField(elements.normThreshold),
    require_both: elements.requireBoth ? elements.requireBoth.checked : undefined,
    min_mad: parseFloatField(elements.minMad),
    cosine_enabled: elements.cosineEnabled ? elements.cosineEnabled.checked : undefined,
    cosine_threshold: parseFloatField(elements.cosineThreshold),
    cosine_top_k: parseIntOptional(elements.cosineTopk),
    attack_enabled: elements.attackEnabled ? elements.attackEnabled.checked : undefined,
    attack_method: elements.attackMethod ? elements.attackMethod.value : "",
    attack_scale: parseFloatField(elements.attackScale),
    attack_malicious_fraction: parseFloatField(elements.attackFraction),
    attack_malicious_ranks: parseCsvInts(elements.attackRanks),
    attack_label_flip: elements.attackLabelFlip ? elements.attackLabelFlip.checked : undefined,
    attack_loss_scale: parseFloatField(elements.attackLossScale),
    attack_accuracy_scale: parseFloatField(elements.attackAccScale),
    train_algo: elements.trainAlgo ? elements.trainAlgo.value : "",
    train_lr: parseFloatField(elements.trainLr),
    train_epochs: parseIntOptional(elements.trainEpochs),
    fedprox_mu: parseFloatField(elements.fedproxMu),
    deadline_ms: parseFloatField(elements.deadlineMs),
    server_lr: parseFloatField(elements.serverLr),
    server_update_clip: parseFloatField(elements.serverClip),
    private_head_lr: parseFloatField(elements.privateHeadLr),
    private_head_epochs: parseIntOptional(elements.privateHeadEpochs),
    data_alpha: parseFloatField(elements.dataAlpha),
    sampling_enabled: elements.samplingEnabled ? elements.samplingEnabled.checked : undefined,
    sampling_strategy: elements.samplingStrategy ? elements.samplingStrategy.value : "",
    sampling_epsilon: parseFloatField(elements.samplingEpsilon),
    sampling_score_ema: parseFloatField(elements.samplingEma),
    sampling_timeout_penalty: parseFloatField(elements.samplingTimeoutPenalty),
    sampling_anomaly_penalty: parseFloatField(elements.samplingAnomalyPenalty),
    sampling_fairness_window: parseIntOptional(elements.samplingFairness),
    sampling_softmax_temp: parseFloatField(elements.samplingSoftmax),
    timeout_simulation_enabled: elements.timeoutEnabled
      ? elements.timeoutEnabled.checked
      : undefined,
    timeout_client_ranks: parseCsvInts(elements.timeoutRanks),
    timeout_cooldown_rounds: parseIntOptional(elements.timeoutCooldown),
    download_data: elements.downloadData ? elements.downloadData.checked : false,
    stay_online_on_not_selected: elements.stayOnline ? elements.stayOnline.checked : false,
  };
  setLaunchStatus("状态：启动中...", false);
  try {
    const resp = await fetch("/api/v1/session/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) {
      const text = await resp.text();
      setLaunchStatus(`启动失败：${text || resp.status}`, true);
      return;
    }
    resetUiState();
    await loadHistory();
    await refreshOnlineClients();
    setLaunchStatus("状态：运行中", false);
  } catch (err) {
    setLaunchStatus(`启动失败：${err}`, true);
  }
}

async function stopSession() {
  setLaunchStatus("状态：停止中...", false);
  try {
    const resp = await fetch("/api/v1/session/stop", { method: "POST" });
    if (!resp.ok) {
      const text = await resp.text();
      setLaunchStatus(`停止失败：${text || resp.status}`, true);
      return;
    }
    setLaunchStatus("状态：未启动", false);
  } catch (err) {
    setLaunchStatus(`停止失败：${err}`, true);
  }
}

function createLineChart(ctx, label, color) {
  return new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label,
          data: [],
          borderColor: color,
          backgroundColor: "transparent",
          tension: 0.3,
          borderWidth: 2,
          pointRadius: 2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      layout: {
        padding: { top: 6, right: 10, bottom: 12, left: 6 },
      },
      scales: {
        x: {
          ticks: { color: "#94a3b8", padding: 6, maxRotation: 0, autoSkip: true },
          grid: { color: "rgba(148, 163, 184, 0.1)" },
        },
        y: {
          ticks: { color: "#94a3b8", padding: 6 },
          grid: { color: "rgba(148, 163, 184, 0.1)" },
        },
      },
      plugins: {
        legend: { labels: { color: "#f1f5f9" } },
      },
    },
  });
}

function createDualChart(ctx) {
  return new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "上传 KB",
          data: [],
          borderColor: "#3b82f6",
          backgroundColor: "transparent",
          tension: 0.3,
          borderWidth: 2,
          pointRadius: 2,
        },
        {
          label: "压缩比",
          data: [],
          borderColor: "#06b6d4",
          backgroundColor: "transparent",
          tension: 0.3,
          borderWidth: 2,
          pointRadius: 2,
          yAxisID: "y1",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      layout: {
        padding: { top: 6, right: 10, bottom: 12, left: 6 },
      },
      scales: {
        x: {
          ticks: { color: "#94a3b8", padding: 6, maxRotation: 0, autoSkip: true },
          grid: { color: "rgba(148, 163, 184, 0.1)" },
        },
      y: {
        ticks: { color: "#94a3b8", padding: 6 },
        grid: { color: "rgba(148, 163, 184, 0.1)" },
      },
      y1: {
        position: "right",
        ticks: { color: "#94a3b8", padding: 6 },
        grid: { display: false },
      },
      },
      plugins: {
        legend: { labels: { color: "#f1f5f9" } },
      },
    },
  });
}

// 客户端侧曲线
const lossChart = createLineChart(document.getElementById("loss-chart"), "损失", "#f59e0b");
const accChart = createLineChart(document.getElementById("acc-chart"), "准确率", "#3b82f6");
const epsilonChart = new Chart(document.getElementById("epsilon-chart"), {
  type: "line",
  data: {
    labels: [],
    datasets: [
      {
        label: "ε 最大值",
        data: [],
        borderColor: "#10b981",
        backgroundColor: "transparent",
        tension: 0.3,
        borderWidth: 2,
        pointRadius: 2,
      },
      {
        label: "ε 平均值",
        data: [],
        borderColor: "#f59e0b",
        backgroundColor: "transparent",
        tension: 0.3,
        borderWidth: 2,
        pointRadius: 2,
      },
      {
        label: "ε 记账器",
        data: [],
        borderColor: "#06b6d4",
        backgroundColor: "transparent",
        tension: 0.3,
        borderWidth: 2,
        pointRadius: 2,
      },
      {
        label: "目标 ε",
        data: [],
        borderColor: "#ffffff",
        backgroundColor: "transparent",
        tension: 0.3,
        borderWidth: 1,
        pointRadius: 0,
        borderDash: [4, 4],
      },
    ],
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    layout: {
      padding: { top: 6, right: 10, bottom: 12, left: 6 },
    },
    scales: {
      x: {
        ticks: { color: "#94a3b8", padding: 6, maxRotation: 0, autoSkip: true },
        grid: { color: "rgba(148, 163, 184, 0.1)" },
      },
      y: { ticks: { color: "#94a3b8", padding: 6 }, grid: { color: "rgba(148, 163, 184, 0.1)" } },
    },
    plugins: {
      legend: { labels: { color: "#f1f5f9" } },
    },
  },
});
// 通信负载（上传 KB + 压缩比）
const commChart = createDualChart(document.getElementById("comm-chart"));
const dpControlChart = new Chart(document.getElementById("dp-control-chart"), {
  type: "line",
  data: {
    labels: [],
    datasets: [
      {
        label: "噪声乘子",
        data: [],
        borderColor: "#f59e0b",
        backgroundColor: "transparent",
        tension: 0.3,
        borderWidth: 2,
        pointRadius: 2,
      },
      {
        label: "裁剪率",
        data: [],
        borderColor: "#3b82f6",
        backgroundColor: "transparent",
        tension: 0.3,
        borderWidth: 2,
        pointRadius: 2,
        yAxisID: "y1",
      },
    ],
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    layout: {
      padding: { top: 6, right: 10, bottom: 12, left: 6 },
    },
    scales: {
      x: {
        ticks: { color: "#94a3b8", padding: 6, maxRotation: 0, autoSkip: true },
        grid: { color: "rgba(148, 163, 184, 0.1)" },
      },
      y: {
        ticks: { color: "#94a3b8", padding: 6 },
        grid: { color: "rgba(148, 163, 184, 0.1)" },
      },
      y1: {
        position: "right",
        ticks: { color: "#94a3b8", padding: 6, min: 0, max: 1 },
        grid: { display: false },
      },
    },
    plugins: {
      legend: { labels: { color: "#f1f5f9" } },
    },
  },
});
const serverLossChart = createLineChart(
  document.getElementById("server-loss-chart"),
  "评估损失",
  "#ef4444"
);
const serverAccChart = createLineChart(
  document.getElementById("server-acc-chart"),
  "评估准确率",
  "#10b981"
);
const serverUpdateChart = createLineChart(
  document.getElementById("server-update-chart"),
  "更新范数",
  "#06b6d4"
);
const fairnessChart = new Chart(document.getElementById("fairness-chart"), {
  type: "line",
  data: {
    labels: [],
    datasets: [
      {
        label: "个体准确率均值",
        data: [],
        borderColor: "#f59e0b",
        backgroundColor: "transparent",
        tension: 0.3,
        borderWidth: 2,
        pointRadius: 2,
      },
      {
        label: "个体准确率最小值",
        data: [],
        borderColor: "#ef4444",
        backgroundColor: "transparent",
        tension: 0.3,
        borderWidth: 2,
        pointRadius: 2,
      },
      {
        label: "个体准确率标准差",
        data: [],
        borderColor: "#3b82f6",
        backgroundColor: "transparent",
        tension: 0.3,
        borderWidth: 2,
        pointRadius: 2,
      },
      {
        label: "Jain 指数",
        data: [],
        borderColor: "#10b981",
        backgroundColor: "transparent",
        tension: 0.3,
        borderWidth: 2,
        pointRadius: 2,
        borderDash: [4, 4],
      },
    ],
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    layout: {
      padding: { top: 6, right: 10, bottom: 12, left: 6 },
    },
    scales: {
      x: {
        ticks: { color: "#94a3b8", padding: 6, maxRotation: 0, autoSkip: true },
        grid: { color: "rgba(148, 163, 184, 0.1)" },
      },
      y: {
        ticks: { color: "#94a3b8", padding: 6, min: 0, max: 1 },
        grid: { color: "rgba(148, 163, 184, 0.1)" },
      },
    },
    plugins: {
      legend: { labels: { color: "#f1f5f9" } },
    },
  },
});

function updateList(el, items) {
  el.innerHTML = "";
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = formatClientLabel(item.client_id || "", item.client_name || "");
    el.appendChild(li);
  });
}

function updateOnlineClients(el, clients, timedOut) {
  if (!el) {
    return;
  }
  el.innerHTML = "";
  const timedOutSet = timedOut || new Set();
  clients.forEach((client) => {
    const li = document.createElement("li");
    const isExcluded = !!client.blacklisted;
    const isTimeout = timedOutSet.has(String(client.client_id || ""));
    const isCooldown = !!client.cooldown;
    const isOnline = client.online !== false;
    li.classList.add("client-item");
    if (isExcluded) {
      li.classList.add("is-excluded");
    } else if (isTimeout || isCooldown) {
      li.classList.add("is-timeout");
    } else if (!isOnline) {
      li.classList.add("is-offline");
    } else {
      li.classList.add("is-online");
    }
    const label = document.createElement("span");
    label.textContent = formatClientLabel(client.client_id || "", client.client_name || "");
    const status = document.createElement("span");
    status.classList.add("client-status");
    if (isExcluded) {
      status.textContent = "已拉黑";
    } else if (isTimeout || isCooldown) {
      status.textContent = "超时";
    } else if (!isOnline) {
      status.textContent = "离线";
    } else {
      status.textContent = "在线";
    }
    li.appendChild(label);
    li.appendChild(status);
    el.appendChild(li);
  });
  if (clients.length === 0) {
    const li = document.createElement("li");
    li.textContent = "-";
    el.appendChild(li);
  }
}

function updateClientManagement(clients, timedOut) {
  // 客户端管理：在线列表 + 手动拉黑/解除
  const body = document.getElementById("client-manage-body");
  if (!body) {
    return;
  }
  body.innerHTML = "";
  const timedOutSet = timedOut || new Set();
  clients.forEach((client) => {
    const row = document.createElement("tr");
    const label = formatClientLabel(client.client_id || "", client.client_name || "");
    const isBlacklisted = !!client.blacklisted;
    const isTimeout =
      timedOutSet.has(String(client.client_id || "")) || !!client.cooldown;
    const isOnline = client.online !== false;
    let statusLabel = "正常";
    if (isBlacklisted) {
      statusLabel = "已拉黑";
    } else if (isTimeout) {
      statusLabel = "超时";
    } else if (!isOnline) {
      statusLabel = "离线";
    }

    const labelCell = document.createElement("td");
    labelCell.textContent = label;
    row.appendChild(labelCell);

    const statusCell = document.createElement("td");
    statusCell.textContent = statusLabel;
    if (isBlacklisted) {
      statusCell.classList.add("status-excluded");
    } else if (isTimeout) {
      statusCell.classList.add("status-timeout");
    } else if (!isOnline) {
      statusCell.classList.add("status-not-selected");
    } else {
      statusCell.classList.add("status-included");
    }
    row.appendChild(statusCell);

    const actionCell = document.createElement("td");
    const actionBtn = document.createElement("button");
    actionBtn.type = "button";
    actionBtn.classList.add("action-btn");
    actionBtn.classList.add(isBlacklisted ? "action-restore" : "action-ban");
    actionBtn.dataset.clientId = String(client.client_id || "");
    actionBtn.dataset.action = isBlacklisted ? "unblacklist" : "blacklist";
    actionBtn.textContent = isBlacklisted ? "解除" : "拉黑";
    actionCell.appendChild(actionBtn);
    row.appendChild(actionCell);

    body.appendChild(row);
  });
  if (clients.length === 0) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 3;
    cell.textContent = "-";
    row.appendChild(cell);
    body.appendChild(row);
  }
}

async function postClientAction(action, clientId) {
  const path =
    action === "unblacklist" ? "/api/v1/clients/unblacklist" : "/api/v1/clients/blacklist";
  try {
    const resp = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ client_id: clientId }),
    });
    return resp.ok;
  } catch (err) {
    console.warn("client action failed", err);
    return false;
  }
}

async function refreshOnlineClients() {
  // 主动刷新在线客户端（管理操作后即时更新）
  try {
    const resp = await fetch("/api/v1/clients");
    if (!resp.ok) {
      return;
    }
    const payload = await resp.json();
    const clients = Array.isArray(payload.clients) ? payload.clients : [];
    const timedOut = new Set(
      (state.latestMetric?.dropped_clients || []).map((value) => String(value))
    );
    const onlineCount = clients.filter((client) => client && client.online !== false).length;
    document.getElementById("clients-count").textContent = onlineCount;
    updateOnlineClients(document.getElementById("client-list"), clients, timedOut);
    updateClientManagement(clients, timedOut);
  } catch (err) {
    console.warn("refresh clients failed", err);
  }
}

function formatLabelHistogram(histogram) {
  if (!histogram || typeof histogram !== "object") {
    return "-";
  }
  const keys = Object.keys(histogram).sort((a, b) => Number(a) - Number(b));
  if (keys.length === 0) {
    return "-";
  }
  return keys.map((key) => `${key}:${histogram[key]}`).join(" ");
}

function updateSummaryCards(metric) {
  // 概览卡片：轮次与稳定属性（聚合/DP/采样）
  const roundId = metric.round_id || 0;
  const participants = metric.participants || [];
  const counts = metric.client_counts || {};
  const eligible = counts.eligible ?? (metric.online_clients || []).length;
  const roundTime = metric.timing?.round_wall_time_ms || 0;
  const dpMode = metric.privacy?.mode || "off";
  const dpEnabled = metric.privacy?.enabled ? dpMode : "关闭";
  const aggMethod = metric.robust?.agg_method || "fedavg";
  const samplingEnabled = metric.sampling?.enabled;
  const samplingStrategy = metric.sampling?.strategy || "random";
  const samplingMode = metric.sampling?.selection_mode || "pre";
  const samplingLabel = samplingEnabled
    ? `${samplingStrategy}/${samplingMode}`
    : "关闭";
  const epsilonMax = metric.privacy?.epsilon_max || 0;
  const uploadKb = (metric.comm?.upload_bytes_total || 0) / 1024.0;
  const compressedRatio = metric.comm?.compressed_ratio || 1;
  const excluded = metric.client_counts?.blacklisted ?? state.excludedClients.size;
  const timeoutCount = (metric.dropped_clients || []).length;

  document.getElementById("summary-round").textContent = roundId;
  document.getElementById("summary-participants").textContent = `${participants.length}/${eligible}`;
  document.getElementById("summary-round-time").textContent = `${roundTime.toFixed(0)} ms`;
  document.getElementById("summary-dp").textContent = dpEnabled;
  document.getElementById("summary-agg").textContent = aggMethod;
  document.getElementById("summary-sampling").textContent = samplingLabel;
  document.getElementById("summary-epsilon").textContent = epsilonMax.toFixed(4);
  document.getElementById("summary-upload").textContent = uploadKb.toFixed(1);
  document.getElementById("summary-compress").textContent = compressedRatio.toFixed(2);
  document.getElementById("summary-excluded").textContent = excluded;
  document.getElementById("summary-timeout").textContent = timeoutCount;
}

function updateClientUpdates(updates) {
  // 客户端更新表（本轮回传统计）
  const body = document.getElementById("client-update-body");
  if (!body) {
    return;
  }
  body.innerHTML = "";
  updates.forEach((update) => {
    const row = document.createElement("tr");
    const uploadKb =
      update.upload_bytes === null || update.upload_bytes === undefined
        ? null
        : (update.upload_bytes || 0) / 1024.0;
    const labelHist = formatLabelHistogram(update.label_histogram);
    const status = String(update.status || "").toLowerCase();
    let statusLabel = update.status || "-";
    if (status === "included") {
      statusLabel = "参与聚合";
    } else if (status === "excluded") {
      statusLabel = "已拉黑";
    } else if (status === "not_selected") {
      statusLabel = "未入选";
    } else if (status === "timeout") {
      statusLabel = "超时";
    }
    const fields = [
      formatClientLabel(update.client_id || "", update.client_name || ""),
      statusLabel,
      formatMetric(update.train_loss, 4),
      formatMetric(update.train_accuracy, 4),
      formatMetric(update.epsilon, 4),
      uploadKb === null ? "-" : uploadKb.toFixed(1),
      labelHist,
    ];
    fields.forEach((value, idx) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      if (idx === 1) {
        if (status === "included") {
          cell.classList.add("status-included");
        } else if (status === "excluded") {
          cell.classList.add("status-excluded");
        } else if (status === "not_selected") {
          cell.classList.add("status-not-selected");
        } else if (status === "timeout") {
          cell.classList.add("status-timeout");
        }
      }
      if (idx === fields.length - 1) {
        cell.classList.add("label-cell");
        if (labelHist !== "-") {
          cell.title = labelHist;
        }
      }
      row.appendChild(cell);
    });
    body.appendChild(row);
  });
}

function updateClientUpdatesView(roundId) {
  const metric = state.metricsByRound[roundId];
  const label = document.getElementById("client-updates-label");
  if (label) {
    label.textContent = roundId ? `轮次 ${roundId}` : "轮次 -";
  }
  if (metric && Array.isArray(metric.client_updates)) {
    updateClientUpdates(metric.client_updates);
  }
  updateClientUpdatesControls();
}

function updateClientUpdatesControls() {
  const prevBtn = document.getElementById("client-updates-prev");
  const nextBtn = document.getElementById("client-updates-next");
  if (!prevBtn || !nextBtn) {
    return;
  }
  const rounds = getRoundIds();
  let current = state.clientUpdatesRound || state.lastRound;
  if (!rounds.includes(current)) {
    current = rounds[rounds.length - 1] || 0;
  }
  const idx = rounds.indexOf(current);
  prevBtn.disabled = idx <= 0;
  nextBtn.disabled = idx < 0 || idx >= rounds.length - 1;
}

function navigateClientUpdates(step) {
  const rounds = getRoundIds();
  if (!rounds.length) {
    return;
  }
  let current = state.clientUpdatesRound || state.lastRound || rounds[rounds.length - 1];
  let idx = rounds.indexOf(current);
  if (idx < 0) {
    idx = rounds.length - 1;
  }
  const nextIdx = Math.min(Math.max(0, idx + step), rounds.length - 1);
  state.clientUpdatesRound = rounds[nextIdx];
  state.clientUpdatesPinned = nextIdx < rounds.length - 1;
  updateClientUpdatesView(state.clientUpdatesRound);
}

function getRoundIds() {
  return Object.keys(state.metricsByRound)
    .map((value) => Number(value))
    .filter((value) => Number.isFinite(value) && value > 0)
    .sort((a, b) => a - b);
}

function rebuildSeries() {
  // 由 metricsByRound 重建曲线序列
  const roundIds = getRoundIds();

  state.rounds = roundIds;
  state.loss = [];
  state.accuracy = [];
  state.epsilonMax = [];
  state.epsilonAvg = [];
  state.epsilonAccountant = [];
  state.epsilonTarget = [];
  state.noiseMultiplier = [];
  state.clipRate = [];
  state.uploadBytes = [];
  state.downloadBytes = [];
  state.compressedRatio = [];
  state.serverEvalLoss = [];
  state.serverEvalAcc = [];
  state.serverUpdateNorm = [];
  state.fairnessAvg = [];
  state.fairnessMin = [];
  state.fairnessStd = [];
  state.fairnessJain = [];

  roundIds.forEach((roundId) => {
    const metric = state.metricsByRound[roundId] || {};
    state.loss.push(metric.global_loss || 0);
    state.accuracy.push(metric.global_accuracy || 0);
    state.epsilonMax.push(metric.privacy?.epsilon_max || 0);
    state.epsilonAvg.push(metric.privacy?.epsilon_avg || 0);
    state.epsilonAccountant.push(metric.privacy?.epsilon_accountant || 0);
    state.epsilonTarget.push(metric.privacy?.target_epsilon || 0);
    state.noiseMultiplier.push(metric.privacy?.noise_multiplier || 0);
    state.clipRate.push(metric.privacy?.clip_rate || 0);
    state.uploadBytes.push((metric.comm?.upload_bytes_total || 0) / 1024.0);
    state.downloadBytes.push((metric.comm?.download_bytes_total || 0) / 1024.0);
    state.compressedRatio.push(metric.comm?.compressed_ratio || 1);
    state.serverEvalLoss.push(metric.server?.eval_loss || 0);
    state.serverEvalAcc.push(metric.server?.eval_accuracy || 0);
    state.serverUpdateNorm.push(metric.server?.update_norm || 0);
    state.fairnessAvg.push(metric.fairness?.avg_accuracy || 0);
    state.fairnessMin.push(metric.fairness?.min_accuracy || 0);
    state.fairnessStd.push(metric.fairness?.std_accuracy || 0);
    state.fairnessJain.push(metric.fairness?.jain_index || 0);
  });

  lossChart.data.labels = state.rounds.slice();
  lossChart.data.datasets[0].data = state.loss.slice();
  lossChart.update();

  accChart.data.labels = state.rounds.slice();
  accChart.data.datasets[0].data = state.accuracy.slice();
  accChart.update();

  epsilonChart.data.labels = state.rounds.slice();
  epsilonChart.data.datasets[0].data = state.epsilonMax.slice();
  epsilonChart.data.datasets[1].data = state.epsilonAvg.slice();
  epsilonChart.data.datasets[2].data = state.epsilonAccountant.slice();
  epsilonChart.data.datasets[3].data = state.epsilonTarget.slice();
  epsilonChart.update();

  commChart.data.labels = state.rounds.slice();
  commChart.data.datasets[0].data = state.uploadBytes.slice();
  commChart.data.datasets[1].data = state.compressedRatio.slice();
  const commMax = state.uploadBytes.length
    ? Math.max(...state.uploadBytes, 0)
    : 0;
  const commSpan = commMax > 0 ? commMax * 0.5 : 1;
  commChart.options.scales.y.suggestedMin = 0;
  commChart.options.scales.y.suggestedMax = commMax + commSpan;
  commChart.update();

  dpControlChart.data.labels = state.rounds.slice();
  dpControlChart.data.datasets[0].data = state.noiseMultiplier.slice();
  dpControlChart.data.datasets[1].data = state.clipRate.slice();
  dpControlChart.update();

  serverLossChart.data.labels = state.rounds.slice();
  serverLossChart.data.datasets[0].data = state.serverEvalLoss.slice();
  serverLossChart.update();

  serverAccChart.data.labels = state.rounds.slice();
  serverAccChart.data.datasets[0].data = state.serverEvalAcc.slice();
  serverAccChart.update();

  serverUpdateChart.data.labels = state.rounds.slice();
  serverUpdateChart.data.datasets[0].data = state.serverUpdateNorm.slice();
  serverUpdateChart.update();

  fairnessChart.data.labels = state.rounds.slice();
  fairnessChart.data.datasets[0].data = state.fairnessAvg.slice();
  fairnessChart.data.datasets[1].data = state.fairnessMin.slice();
  fairnessChart.data.datasets[2].data = state.fairnessStd.slice();
  fairnessChart.data.datasets[3].data = state.fairnessJain.slice();
  fairnessChart.update();
}

async function loadHistory() {
  try {
    const resp = await fetch("/api/v1/metrics/all");
    if (!resp.ok) {
      return;
    }
    const payload = await resp.json();
    const metrics = Array.isArray(payload.metrics) ? payload.metrics : [];
    metrics.sort((a, b) => (a.round_id || 0) - (b.round_id || 0));
    metrics.forEach((metric) => {
      if (metric && metric.round_id) {
        applyMetric(metric);
      }
    });
  } catch (err) {
    console.warn("load history failed", err);
  }
}

function appendAlerts(alerts) {
  // 告警日志表格（异常/超时/拉黑）
  const body = document.getElementById("alert-body");
  const nameMap = arguments.length > 1 ? arguments[1] : null;
  alerts.forEach((alert) => {
    const row = document.createElement("tr");
    ["round_id", "time", "client_id", "reason", "action"].forEach((key) => {
      const cell = document.createElement("td");
      let value = alert[key] || "-";
      if (key === "client_id" && nameMap) {
        const name = nameMap.get(String(value));
        value = formatClientLabel(String(value || ""), name || "");
      }
      if (key === "reason") {
        const text = String(value);
        value = text
          .replace(/anomalous/gi, "异常")
          .replace(/timeout/gi, "超时")
          .replace(/\bloss\b/gi, "损失")
          .replace(/\bnorm\b/gi, "范数")
          .replace(/\bupdate\b/gi, "更新")
          .replace(/\bor\b/gi, "或");
      }
      if (key === "action") {
        const text = String(value).toLowerCase();
        if (text === "excluded") {
          value = "已拉黑";
        } else if (text === "timeout") {
          value = "超时";
        }
      }
      cell.textContent = value;
      row.appendChild(cell);
    });
    body.prepend(row);
  });
  while (body.rows.length > 20) {
    body.deleteRow(body.rows.length - 1);
  }
}

function updateSummaryList(list, items) {
  list.innerHTML = "";
  items.filter((item) => item).forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    list.appendChild(li);
  });
}

function updateSummary(items) {
  const list = document.getElementById("server-summary");
  if (!list) {
    return;
  }
  updateSummaryList(list, items);
}

function updateSecuritySummary(metric) {
  // 检测摘要：P/R/F1、阈值、攻击模拟
  const list = document.getElementById("security-summary");
  if (!list) {
    return;
  }
  const detector = metric.security?.detector || {};
  const detected = state.detectionTotals.detected;
  const malicious = state.detectionTotals.malicious;
  const truePositive = state.detectionTotals.truePositive;
  const precision = detected > 0 ? truePositive / detected : 0.0;
  const recall = malicious > 0 ? truePositive / malicious : 0.0;
  const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0.0;
  let methodLabel = "off";
  if (state.attackMethods.size === 1) {
    methodLabel = [...state.attackMethods][0] || "on";
  } else if (state.attackMethods.size > 1) {
    methodLabel = "mixed";
  }
  const items = [
    `攻击: ${methodLabel} (恶意 ${malicious})`,
    `检测 P/R/F1: ${precision.toFixed(2)} / ${recall.toFixed(2)} / ${f1.toFixed(2)}`,
    `余弦 z 阈值: ${(detector.cosine_threshold || 0).toFixed(2)}`,
  ];
  updateSummaryList(list, items);
}

function updateSimilarityRank(metric, nameMap) {
  // 相似度排名：基于 cosine 分数的 z-score
  const list = document.getElementById("similarity-rank");
  if (!list) {
    return;
  }
  list.innerHTML = "";
  const cosineScores = metric.security?.cosine_scores || {};
  const rank = metric.security?.similarity_rank || [];
  let entries = Object.entries(cosineScores);
  if ((!entries || entries.length === 0) && Array.isArray(rank) && rank.length > 0) {
    entries = rank.map((entry) => [entry[0], entry[1]]);
  }
  if (!entries || entries.length === 0) {
    const li = document.createElement("li");
    li.textContent = "-";
    list.appendChild(li);
    return;
  }

  const values = entries
    .map((entry) => Number(entry[1]))
    .filter((value) => Number.isFinite(value));
  values.sort((a, b) => a - b);
  const mid = Math.floor(values.length / 2);
  const median =
    values.length === 0
      ? 0
      : values.length % 2 === 0
        ? 0.5 * (values[mid - 1] + values[mid])
        : values[mid];
  const deviations = values.map((value) => Math.abs(value - median)).sort((a, b) => a - b);
  const mad =
    deviations.length === 0
      ? 0
      : deviations.length % 2 === 0
        ? 0.5 * (deviations[mid - 1] + deviations[mid])
        : deviations[mid];
  const denom = mad > 0 ? mad : 1.0;

  const entryMap = new Map(entries.map((entry) => [String(entry[0] || ""), Number(entry[1])])); 
  let displayOrder = Array.isArray(rank) && rank.length > 0 ? rank : entries;
  displayOrder.forEach((entry) => {
    const clientId = entry[0] || "-";
    const raw = entryMap.has(String(clientId)) ? entryMap.get(String(clientId)) : Number(entry[1]);
    const zScore = Number.isFinite(raw) ? (median - raw) / denom : 0.0;
    let label = String(clientId || "-");
    if (nameMap) {
      const name = nameMap.get(String(clientId));
      label = formatClientLabel(String(clientId || ""), name || "");
    }
    const li = document.createElement("li");
    const zText = Number.isFinite(zScore) ? zScore.toFixed(2) : "-";
    li.textContent = `${label}: z=${zText}`;
    if (Number.isFinite(raw)) {
      li.title = `cos=${raw.toFixed(4)}`;
    }
    list.appendChild(li);
  });
}

function applyMetric(metric) {
  // 单条指标入库并驱动 UI 更新
  const roundId = Number(metric.round_id || 0);
  if (!Number.isFinite(roundId) || roundId <= 0) {
    return;
  }
  const isNewRound = !state.processedRounds.has(roundId);
  state.metricsByRound[roundId] = metric;
  state.lastRound = Math.max(state.lastRound, roundId);
  state.latestMetric = metric;
  document.getElementById("round-id").textContent = roundId;

  const clients = metric.online_clients || [];
  const timedOut = new Set((metric.dropped_clients || []).map((value) => String(value)));
  if (clients.length > 0) {
    const onlineCount = clients.filter((client) => client && client.online !== false).length;
    document.getElementById("clients-count").textContent = onlineCount;
    updateOnlineClients(document.getElementById("client-list"), clients, timedOut);
    updateClientManagement(clients, timedOut);
  } else {
    void refreshOnlineClients();
  }

  const nameMap = new Map();
  if (Array.isArray(metric.online_clients)) {
    metric.online_clients.forEach((client) => {
      if (client && client.client_id) {
        nameMap.set(String(client.client_id), String(client.client_name || ""));
      }
    });
  }
  if (Array.isArray(metric.client_updates)) {
    metric.client_updates.forEach((update) => {
      if (update && update.client_id) {
        nameMap.set(String(update.client_id), String(update.client_name || ""));
      }
    });
  }

  if (isNewRound && Array.isArray(metric.robust?.excluded_clients)) {
    metric.robust.excluded_clients.forEach((clientId) => {
      state.excludedClients.add(String(clientId));
    });
  }

  if (isNewRound) {
    const detector = metric.security?.detector || {};
    state.detectionTotals.detected += Number(detector.detected || 0);
    state.detectionTotals.malicious += Number(detector.malicious || 0);
    state.detectionTotals.truePositive += Number(detector.true_positive || 0);
    if (metric.security?.attack_simulation?.enabled) {
      const method = metric.security?.attack_simulation?.method;
      if (method) {
        state.attackMethods.add(String(method));
      } else {
        state.attackMethods.add("on");
      }
    }
    state.processedRounds.add(roundId);
  }

  if (Array.isArray(metric.alerts) && metric.alerts.length > 0) {
    appendAlerts(metric.alerts, nameMap);
  }
  if (!state.clientUpdatesPinned || state.clientUpdatesRound === 0) {
    state.clientUpdatesRound = roundId;
    state.clientUpdatesPinned = false;
    updateClientUpdatesView(roundId);
  } else {
    updateClientUpdatesControls();
  }
  updateSummaryCards(metric);
  rebuildSeries();

  updateSummary([
    `公平性 (Jain): ${(metric.fairness?.jain_index || 0).toFixed(3)}`,
    `个体准确率 最小/标准差: ${(metric.fairness?.min_accuracy || 0).toFixed(3)} / ${(metric.fairness?.std_accuracy || 0).toFixed(3)}`,
    `超时率: ${(metric.sampling?.timeout_rate || 0).toFixed(2)}`,
  ]);
  updateSecuritySummary(metric);
  updateSimilarityRank(metric, nameMap);
}

let pollTimer = null;
let uiInitialized = false;
let sessionPoller = null;
let wsClient = null;
let wsRetryTimer = null;

function setupHelpModal() {
  const modal = document.getElementById("help-modal");
  const openBtn = document.getElementById("help-btn");
  const closeBtn = document.getElementById("help-close");
  if (!modal || !openBtn || !closeBtn) {
    return;
  }
  const openModal = () => {
    modal.classList.remove("hidden");
    modal.hidden = false;
    modal.setAttribute("aria-hidden", "false");
    document.body.classList.add("modal-open");
  };
  const closeModal = () => {
    modal.classList.add("hidden");
    modal.hidden = true;
    modal.setAttribute("aria-hidden", "true");
    document.body.classList.remove("modal-open");
  };
  openBtn.addEventListener("click", openModal);
  closeBtn.addEventListener("click", closeModal);
  modal.addEventListener("click", (event) => {
    if (event.target === modal) {
      closeModal();
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !modal.classList.contains("hidden")) {
      closeModal();
    }
  });
}

function startPolling() {
  if (pollTimer) {
    return;
  }
  pollTimer = setInterval(async () => {
    try {
      const resp = await fetch("/api/v1/metrics/latest");
      if (!resp.ok) {
        return;
      }
      const metric = await resp.json();
      if (metric && metric.round_id) {
        applyMetric(metric);
      }
    } catch (err) {
      console.warn("polling metrics failed", err);
    }
  }, 2000);
}

function connectWebSocket() {
  if (wsClient && wsClient.readyState <= 1) {
    return;
  }
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const wsUrl = `${protocol}://${window.location.host}/ws/metrics`;
  wsClient = new WebSocket(wsUrl);

  wsClient.onmessage = (event) => {
    try {
      const metric = JSON.parse(event.data);
      applyMetric(metric);
    } catch (err) {
      console.error("invalid metric payload", err);
    }
  };

  wsClient.onclose = () => {
    console.warn("metrics websocket closed, retrying...");
    startPolling();
    if (wsRetryTimer) {
      return;
    }
    wsRetryTimer = setTimeout(() => {
      wsRetryTimer = null;
      connectWebSocket();
    }, 1500);
  };

  wsClient.onerror = () => {
    startPolling();
  };
}

async function init() {
  if (!uiInitialized) {
    document.querySelectorAll(".tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach((btn) => btn.classList.remove("active"));
        document.querySelectorAll(".panel").forEach((panel) => panel.classList.remove("active"));
        tab.classList.add("active");
        const target = tab.getAttribute("data-target");
        if (target) {
          const panel = document.getElementById(target);
          if (panel) {
            panel.classList.add("active");
          }
        }
      });
    });

    const prevBtn = document.getElementById("client-updates-prev");
    const nextBtn = document.getElementById("client-updates-next");
    if (prevBtn) {
      prevBtn.addEventListener("click", () => navigateClientUpdates(-1));
    }
    if (nextBtn) {
      nextBtn.addEventListener("click", () => navigateClientUpdates(1));
    }

    const manageBody = document.getElementById("client-manage-body");
    if (manageBody) {
      manageBody.addEventListener("click", async (event) => {
        const target = event.target;
        if (!(target instanceof HTMLElement)) {
          return;
        }
        const button = target.closest("button[data-action]");
        if (!button) {
          return;
        }
        const clientId = String(button.dataset.clientId || "");
        const action = String(button.dataset.action || "");
        if (!clientId || !action) {
          return;
        }
        button.disabled = true;
        const ok = await postClientAction(action, clientId);
        button.disabled = false;
        if (ok) {
          await refreshOnlineClients();
        }
      });
    }

    const launchStart = document.getElementById("launch-start");
    const launchStop = document.getElementById("launch-stop");
    if (launchStart) {
      launchStart.addEventListener("click", startSession);
    }
    if (launchStop) {
      launchStop.addEventListener("click", stopSession);
    }

    setupHelpModal();
    setLaunchStatus("状态：未启动", false);
    await loadDefaultConfig();
    await loadSessionStatus();
    await loadHistory();
    await refreshOnlineClients();
    if (!sessionPoller) {
      sessionPoller = window.setInterval(() => {
        loadSessionStatus();
      }, 4000);
    }
    uiInitialized = true;
  }

  connectWebSocket();
}

window.addEventListener("load", init);
