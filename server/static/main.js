const state = {
  tasks: [],
  taskId: "",
  obs: null,
  done: false,
  stepNo: 0,
  prices: [],
  equities: [],
  rewards: [],
};

const el = {
  taskSelect: document.getElementById("task-select"),
  decisionSelect: document.getElementById("decision-select"),
  qtyInput: document.getElementById("qty-input"),
  confidenceInput: document.getElementById("confidence-input"),
  resetBtn: document.getElementById("reset-btn"),
  stepBtn: document.getElementById("step-btn"),
  clearLogBtn: document.getElementById("clear-log-btn"),
  statusText: document.getElementById("status-text"),
  objective: document.getElementById("task-objective"),
  symbolChip: document.getElementById("symbol-chip"),
  taskScore: document.getElementById("task-score"),
  cumReward: document.getElementById("cum-reward"),
  currentPrice: document.getElementById("current-price"),
  portfolioValue: document.getElementById("portfolio-value"),
  cashValue: document.getElementById("cash-value"),
  positionValue: document.getElementById("position-value"),
  drawdownValue: document.getElementById("drawdown-value"),
  progressValue: document.getElementById("progress-value"),
  priceChart: document.getElementById("price-chart"),
  equityChart: document.getElementById("equity-chart"),
  rewardChart: document.getElementById("reward-chart"),
  logList: document.getElementById("log-list"),
};

function setStatus(message, tone = "neutral") {
  el.statusText.textContent = message;
  const color = tone === "error" ? "#ff6b6b" : tone === "success" ? "#1fd17c" : "#9ab5d4";
  el.statusText.style.color = color;
}

function fmtMoney(value) {
  return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 }).format(
    Number(value || 0),
  );
}

function fmtNum(value, digits = 2) {
  return Number(value || 0).toFixed(digits);
}

async function fetchTasks() {
  const res = await fetch("/tasks");
  const payload = await res.json();
  state.tasks = payload.tasks || [];
  el.taskSelect.innerHTML = "";

  for (const task of state.tasks) {
    const option = document.createElement("option");
    option.value = task.id;
    option.textContent = `${task.name} (${task.difficulty})`;
    el.taskSelect.appendChild(option);
  }

  if (state.tasks.length > 0) {
    state.taskId = state.tasks[0].id;
  }
}

function selectedTaskMeta() {
  return state.tasks.find((t) => t.id === state.taskId) || null;
}

function updateMetrics(obs) {
  if (!obs) {
    return;
  }
  const metadata = obs.metadata || {};
  el.taskScore.textContent = fmtNum(metadata.task_score ?? metadata.score ?? 0);
  el.cumReward.textContent = fmtNum(metadata.cumulative_reward ?? 0);

  el.currentPrice.textContent = fmtMoney(obs.current_price);
  el.portfolioValue.textContent = fmtMoney(obs.portfolio_value);
  el.cashValue.textContent = fmtMoney(obs.cash);
  el.positionValue.textContent = String(obs.position ?? 0);
  el.drawdownValue.textContent = `${fmtNum((obs.drawdown || 0) * 100)}%`;
  el.progressValue.textContent = `${obs.day_index ?? 0} / ${obs.max_days ?? 0}`;

  el.symbolChip.textContent = obs.symbol || "-";
  el.objective.textContent = obs.objective || "";
}

function drawLineChart(svg, values, color) {
  svg.innerHTML = "";
  const width = svg.viewBox.baseVal.width || 600;
  const height = svg.viewBox.baseVal.height || 240;
  const pad = 24;

  if (!values || values.length === 0) {
    return;
  }

  let min = Math.min(...values);
  let max = Math.max(...values);
  if (Math.abs(max - min) < 1e-9) {
    min -= 1;
    max += 1;
  }

  for (let i = 0; i < 4; i += 1) {
    const y = pad + ((height - pad * 2) * i) / 3;
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", String(pad));
    line.setAttribute("x2", String(width - pad));
    line.setAttribute("y1", String(y));
    line.setAttribute("y2", String(y));
    line.setAttribute("stroke", "rgba(179,213,255,0.18)");
    line.setAttribute("stroke-width", "1");
    svg.appendChild(line);
  }

  const points = values.map((v, i) => {
    const x = pad + ((width - pad * 2) * i) / Math.max(values.length - 1, 1);
    const yNorm = (v - min) / (max - min);
    const y = height - pad - yNorm * (height - pad * 2);
    return [x, y];
  });

  const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
  const d = points.map((p, i) => `${i === 0 ? "M" : "L"}${p[0]},${p[1]}`).join(" ");
  path.setAttribute("d", d);
  path.setAttribute("fill", "none");
  path.setAttribute("stroke", color);
  path.setAttribute("stroke-width", "3");
  svg.appendChild(path);

  const end = points[points.length - 1];
  const marker = document.createElementNS("http://www.w3.org/2000/svg", "circle");
  marker.setAttribute("cx", String(end[0]));
  marker.setAttribute("cy", String(end[1]));
  marker.setAttribute("r", "4.5");
  marker.setAttribute("fill", color);
  svg.appendChild(marker);

  const minText = document.createElementNS("http://www.w3.org/2000/svg", "text");
  minText.setAttribute("x", "6");
  minText.setAttribute("y", String(height - 8));
  minText.setAttribute("fill", "#9ab5d4");
  minText.setAttribute("font-size", "11");
  minText.textContent = min.toFixed(2);
  svg.appendChild(minText);

  const maxText = document.createElementNS("http://www.w3.org/2000/svg", "text");
  maxText.setAttribute("x", "6");
  maxText.setAttribute("y", "16");
  maxText.setAttribute("fill", "#9ab5d4");
  maxText.setAttribute("font-size", "11");
  maxText.textContent = max.toFixed(2);
  svg.appendChild(maxText);
}

function drawRewardBars(svg, values) {
  svg.innerHTML = "";
  const width = svg.viewBox.baseVal.width || 1200;
  const height = svg.viewBox.baseVal.height || 180;
  const pad = 18;

  if (!values || values.length === 0) {
    return;
  }

  const barW = (width - pad * 2) / values.length;
  for (let i = 0; i < values.length; i += 1) {
    const v = Number(values[i] || 0);
    const h = Math.max(2, ((height - pad * 2) * v) / 1.0);
    const x = pad + i * barW + 4;
    const y = height - pad - h;

    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", String(x));
    rect.setAttribute("y", String(y));
    rect.setAttribute("width", String(Math.max(2, barW - 8)));
    rect.setAttribute("height", String(h));
    rect.setAttribute("rx", "4");
    rect.setAttribute("fill", v >= 0.5 ? "#1fd17c" : "#ffd166");
    svg.appendChild(rect);
  }
}

function renderCharts() {
  drawLineChart(el.priceChart, state.prices, "#81b9ff");
  drawLineChart(el.equityChart, state.equities, "#1fd17c");
  drawRewardBars(el.rewardChart, state.rewards);
}

function addLog(action, reward, done) {
  const node = document.createElement("div");
  node.className = "log-item";
  node.innerHTML = `
    <span class="step">#${state.stepNo}</span>
    <span class="action">${action}</span>
    <span class="reward">${Number(reward).toFixed(2)}${done ? " • done" : ""}</span>
  `;
  el.logList.prepend(node);
}

async function resetTask() {
  state.taskId = el.taskSelect.value;
  state.done = false;
  state.stepNo = 0;

  const response = await fetch(`/reset?task_id=${encodeURIComponent(state.taskId)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
  const result = await response.json();
  state.obs = result.observation;

  state.prices = [Number(state.obs.current_price || 0)];
  state.equities = [Number(state.obs.portfolio_value || 0)];
  state.rewards = [Number(result.reward || 0)];

  updateMetrics(state.obs);
  renderCharts();

  el.stepBtn.disabled = false;
  setStatus(`Reset ${state.taskId}`, "success");
}

async function executeStep() {
  if (!state.obs || state.done) {
    return;
  }

  const decision = el.decisionSelect.value;
  const quantity = Math.max(0, Number(el.qtyInput.value || 0));
  const confidence = Math.max(0, Math.min(1, Number(el.confidenceInput.value || 0.5)));

  const payload = {
    action: {
      symbol: state.obs.symbol,
      decision,
      quantity,
      confidence,
      rationale: "dashboard_manual",
    },
  };

  const response = await fetch("/step", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const result = await response.json();

  state.stepNo += 1;
  state.done = Boolean(result.done);
  state.obs = result.observation;

  state.prices.push(Number(state.obs.current_price || 0));
  state.equities.push(Number(state.obs.portfolio_value || 0));
  state.rewards.push(Number(result.reward || 0));

  updateMetrics(state.obs);
  renderCharts();
  addLog(`${decision.toUpperCase()} ${quantity}`, result.reward, state.done);

  if (state.done) {
    el.stepBtn.disabled = true;
    setStatus("Episode complete. Reset to run again.", "success");
  } else {
    setStatus(`Step ${state.stepNo} executed`, "neutral");
  }
}

function bindEvents() {
  el.resetBtn.addEventListener("click", () => {
    resetTask().catch((err) => setStatus(`reset_error_${err}`, "error"));
  });

  el.stepBtn.addEventListener("click", () => {
    executeStep().catch((err) => setStatus(`step_error_${err}`, "error"));
  });

  el.clearLogBtn.addEventListener("click", () => {
    el.logList.innerHTML = "";
    setStatus("Log cleared", "neutral");
  });

  el.taskSelect.addEventListener("change", () => {
    const task = selectedTaskMeta();
    if (task) {
      el.objective.textContent = task.objective;
      state.taskId = task.id;
    }
  });
}

async function bootstrap() {
  try {
    setStatus("Loading tasks...", "neutral");
    await fetchTasks();
    bindEvents();

    if (state.tasks.length > 0) {
      state.taskId = state.tasks[0].id;
      el.taskSelect.value = state.taskId;
      el.objective.textContent = state.tasks[0].objective;
      await resetTask();
      setStatus("Dashboard ready", "success");
    } else {
      setStatus("No tasks found", "error");
    }
  } catch (err) {
    setStatus(`bootstrap_error_${err}`, "error");
  }
}

bootstrap();
