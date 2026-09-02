/* Routing Desk - talks to the FastAPI wrapper in api.py.
   Every number shown here comes from the server; nothing is recomputed
   client-side except formatting. */

const $ = (id) => document.getElementById(id);

const el = {
  form: $("route-form"),
  ticket: $("ticket"),
  submit: $("submit"),
  samples: $("samples"),
  slipNo: $("slip-no"),
  empty: $("slip-empty"),
  body: $("slip-body"),
  error: $("slip-error"),
  label: $("verdict-label"),
  dept: $("verdict-dept"),
  conf: $("verdict-conf"),
  priority: $("verdict-priority"),
  stamp: $("stamp"),
  bandSub: $("band-sub"),
  plot: $("band-plot"),
  checks: $("checks"),
  tableBody: $("table-body"),
};

let thresholds = { confidence: 0.5, ambiguity: 0.2 };

const pct = (n, digits = 1) => `${(n * 100).toFixed(digits)}%`;
const pts = (n) => `${(n * 100).toFixed(1)} pts`;
const INITIALISMS = new Set(["HR"]);
const title = (s) => (INITIALISMS.has(s) ? s : s.charAt(0) + s.slice(1).toLowerCase());

/* ---------------------------------------------------------------- boot */
async function boot() {
  try {
    const [health, samples] = await Promise.all([
      fetch("/api/health").then(r => r.json()),
      fetch("/api/samples").then(r => r.json()),
    ]);

    thresholds = { confidence: health.confidence_threshold, ambiguity: health.ambiguity_threshold };
    $("meta-model").textContent = health.model;
    $("meta-classes").textContent = health.categories.length;
    $("meta-floor").textContent = pct(health.confidence_threshold, 0);
    $("meta-margin").textContent = pct(health.ambiguity_threshold, 0);

    el.samples.innerHTML = "";
    for (const text of samples) {
      const li = document.createElement("li");
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "samples__btn";
      btn.textContent = text;
      btn.title = text;
      btn.addEventListener("click", () => {
        el.ticket.value = text;
        el.ticket.focus();
        el.form.requestSubmit();
      });
      li.append(btn);
      el.samples.append(li);
    }
  } catch {
    showError("Can't reach the router. Start it with: uvicorn api:app --reload");
    return;
  }

  // ?ticket=... routes on load, so a particular slip can be linked to directly.
  const linked = new URLSearchParams(location.search).get("ticket");
  if (linked) {
    el.ticket.value = linked;
    el.form.requestSubmit();
  }
}

/* ---------------------------------------------------------------- submit */
el.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = el.ticket.value.trim();
  if (!text) return;

  el.submit.disabled = true;
  el.submit.textContent = "Routing…";
  try {
    const response = await fetch("/api/classify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ticket_text: text }),
    });
    if (!response.ok) throw new Error(String(response.status));
    render(await response.json());
  } catch {
    showError("The router didn't answer. Check the server is still running.");
  } finally {
    el.submit.disabled = false;
    el.submit.textContent = "Route ticket";
  }
});

el.ticket.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") el.form.requestSubmit();
});

/* ---------------------------------------------------------------- render */
function render(slip) {
  el.error.hidden = true;
  el.empty.hidden = true;
  el.body.hidden = false;

  const held = slip.needs_review;
  el.slipNo.textContent = slip.id;
  el.label.textContent = held ? "Held for review · model guessed" : "Routed to";
  el.dept.textContent = title(slip.model_prediction);
  el.conf.textContent = pct(slip.confidence);
  el.priority.textContent = title(slip.priority);
  el.priority.dataset.priority = slip.priority;
  el.stamp.textContent = held ? "Hold for review" : "Routed";

  renderLegend(held);
  renderPlot(slip, held);
  renderChecks(slip);
  renderTable(slip);
}

function renderLegend(held) {
  el.bandSub.innerHTML = "";
  const legend = document.createElement("span");
  legend.className = "legend";
  legend.innerHTML =
    `<span><i class="${held ? "is-pick-held" : "is-pick"}"></i>` +
    `${held ? "held for a person" : "cleared for routing"}</span>` +
    `<span><i></i>not selected</span>`;
  el.bandSub.append(legend);
}

function renderPlot(slip, held) {
  const ranked = Object.entries(slip.distribution).sort((a, b) => b[1] - a[1]);

  el.plot.innerHTML = "";
  for (const [dept, probability] of ranked) {
    const isTop = dept === slip.model_prediction;
    const row = document.createElement("div");
    row.className = `row${isTop ? " row--top" : ""}${isTop && held ? " row--held" : ""}`;
    // The exact figure is on screen; the tooltip adds the precision it rounds off.
    row.title = `${dept}: ${probability.toFixed(4)}`;
    row.innerHTML =
      `<span class="row__name">${dept}</span>` +
      `<span class="row__track"><span class="row__bar"></span></span>` +
      `<span class="row__value">${pct(probability)}</span>`;
    row.querySelector(".row__bar").style.width = `${probability * 100}%`;
    el.plot.append(row);
  }

  const floor = document.createElement("div");
  floor.className = "band__floor";
  el.plot.append(floor);
}

function renderChecks(slip) {
  const margin = slip.confidence - slip.runner_up.confidence;
  const rows = [
    {
      pass: slip.confidence >= thresholds.confidence,
      label: `Top pick clears the ${pct(thresholds.confidence, 0)} confidence floor`,
      figure: `${pct(slip.confidence)} vs ${pct(thresholds.confidence, 0)}`,
    },
    {
      pass: margin >= thresholds.ambiguity,
      label: `Lead over ${title(slip.runner_up.department)} clears the ${pct(thresholds.ambiguity, 0)} margin`,
      figure: `${pts(margin)} vs ${pts(thresholds.ambiguity)}`,
    },
  ];

  el.checks.innerHTML = "";
  for (const row of rows) {
    const li = document.createElement("li");
    li.className = `check ${row.pass ? "check--pass" : "check--fail"}`;
    li.innerHTML =
      `<span class="check__mark" aria-hidden="true">${row.pass ? "✓" : "✗"}</span>` +
      `<span>${row.label}</span>` +
      `<span class="check__figure">${row.figure}</span>`;
    li.prepend(hiddenText(row.pass ? "Passed: " : "Failed: "));
    el.checks.append(li);
  }
}

function hiddenText(text) {
  const span = document.createElement("span");
  span.textContent = text;
  span.style.cssText = "position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%)";
  return span;
}

function renderTable(slip) {
  const ranked = Object.entries(slip.distribution).sort((a, b) => b[1] - a[1]);
  el.tableBody.innerHTML = ranked
    .map(([dept, p]) => `<tr><th scope="row">${dept}</th><td>${pct(p, 2)}</td></tr>`)
    .join("");
}

function showError(message) {
  el.error.textContent = message;
  el.error.hidden = false;
  el.empty.hidden = true;
  el.body.hidden = true;
}

boot();
