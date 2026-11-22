import {badge, btn, e, img, n, option, selectField, stat, table, textField, numField, areaField, api} from "./widgets.js";

const STATE_DOT = {
  connected: "#5ecd9a",
  severed: "#e8745c",
  obscured: "#c4b27a",
};
const OBSERVATION_DOT = {
  tracer: "#e85c7e",
  clear: "#78c4dc",
};
const KIND_STYLE = {
  inlet: {color: "#f5e282", label: "Inlet (1 point)"},
  segment: {color: "#9ad6ff", label: "Fine crack (polyline, 2)"},
  exit: {color: "#e8a07c", label: "Margin outlets (2–8 points)"},
  candidate: {color: "#d2c28a", label: "Baffle positions (2 points)"},
};
const KIND_LIMIT = {inlet: 1, exit: 8, candidate: 2};
const SEGMENT_VERTEX_LIMIT = 12;

let planState = {markers: null, mode: "inlet", segIndex: 0};

export function resetPlanDrawing() {
  planState = {
    markers: {inlet: [], exit: [], candidate: [], segments: [[], []]},
    mode: "inlet",
    segIndex: 0,
  };
}

function svgPoints(point) {
  return [point[0] * 1000, point[1] * 1000];
}

function planSvg() {
  const m = planState.markers;
  const circles = [];
  const push = (point, kind, label) => {
    const [x, y] = svgPoints(point);
    circles.push(
      `<circle class="plan-point" cx="${x}" cy="${y}" r="11" fill="${KIND_STYLE[kind].color}"/>
       <text class="plan-label" x="${x + 15}" y="${y + 6}">${e(label)}</text>`,
    );
  };
  m.inlet.forEach((p, i) => push(p, "inlet", "I" + (i + 1)));
  m.exit.forEach((p, i) => push(p, "exit", "E" + (i + 1)));
  m.candidate.forEach((p, i) => push(p, "candidate", "C" + (i + 1)));
  const lines = m.segments.map((points, index) => {
    if (!points.length) return "";
    const coords = points.map(p => svgPoints(p).join(",")).join(" ");
    const tag = `L${index + 1}`;
    points.forEach((p, i) => {
      const [x, y] = svgPoints(p);
      circles.push(
        `<circle class="plan-point" cx="${x}" cy="${y}" r="9" fill="${KIND_STYLE.segment.color}"/>
         <text class="plan-label" x="${x + 12}" y="${y - 10}">${tag}.${i + 1}</text>`,
      );
    });
    return `<polyline class="plan-line" points="${coords}" stroke="${KIND_STYLE.segment.color}"/>`;
  }).join("");
  return lines + circles.join("");
}

export function planDrawUpdate() {
  const artist = document.querySelector("#plan-draw-overlay");
  if (!artist) return;
  artist.innerHTML = planSvg();
  const counter = document.querySelector("#plan-draw-count");
  if (counter) {
    const m = planState.markers;
    counter.textContent =
      `Inlet ${m.inlet.length}/1 · cracks ${m.segments.map(p => p.length).join("/")} of ${SEGMENT_VERTEX_LIMIT} pts · ` +
      `outlets ${m.exit.length}/8 · baffle positions ${m.candidate.length}/2`;
  }
}

export function planClick(u, v) {
  const point = [Number(u.toFixed(6)), Number(v.toFixed(6))];
  const {mode, markers} = planState;
  if (mode === "segment") {
    const active = markers.segments[planState.segIndex];
    if (active.length >= SEGMENT_VERTEX_LIMIT) return;
    active.push(point);
  } else {
    if (markers[mode].length >= KIND_LIMIT[mode]) return;
    markers[mode].push(point);
  }
  planDrawUpdate();
}

export function planUndo() {
  const {mode, markers} = planState;
  if (mode === "segment") {
    const active = markers.segments[planState.segIndex];
    active.pop();
  } else {
    markers[mode].pop();
  }
  planDrawUpdate();
}

function validateMarkers() {
  const m = planState.markers;
  if (m.inlet.length !== 1) throw Error("Mark exactly one injection point.");
  if (m.exit.length < 2) throw Error("Mark at least two margin outlets.");
  if (m.candidate.length !== 2) throw Error("Mark the two possible baffle positions.");
  if (m.segments.some(points => points.length < 2)) {
    throw Error("Draw both observed fine cracks as polylines (at least two vertices each).");
  }
}

function crazeChoices(state, selected) {
  const runs = state.results.filter(item => item.kind === "craze");
  return option("", "Re-trace with current photo and region", selected) +
    runs.map(item => option(item.id, `${item.label} · ${item.created_at.slice(0, 10)}`, selected)).join("");
}

export function planForm(state, onSaved) {
  resetPlanDrawing();
  const photo = state.tile.photos.find(item => item.registration_state === "base");
  if (!photo) throw Error("Set a base photograph before drawing a baffle plan.");
  const dialog = document.querySelector("#dialog");
  const radius = 14;
  dialog.className = "wide";
  dialog.innerHTML = `<div class="dialog-head"><h2>Tracer baffle plan</h2>
      ${btn("Close", "plan-close", "tiny ghost", "x")}</div>
    <form id="dialog-form" class="form-stack"><div class="dialog-error" role="alert"></div>
      <p class="notice">Mark the injection point, the two fine cracks to watch (each a polyline), the
        margin outlets and the two positions where one baffle can physically sit. The plan reuses the
        Fine-line network trace and chooses one baffle that leaves the fewest routes to the margin.</p>
      <div class="form-columns">${textField("Plan title", "title", "Single-baffle sealing plan", 'required maxlength="110"')}
        ${numField("Baffle radius (registered px)", "baffle_radius_px", radius, 2, 120, 1)}
        ${selectField("Fine-line network", "craze_result_id", crazeChoices(state, ""))}
        ${selectField("Sampling region", "region_id", state.regionChoices())}</div>
      <div class="actions" id="plan-kind-bar">
        ${Object.entries(KIND_STYLE).map(([kind, style]) =>
          btn(style.label, "plan-kind", "tiny secondary", null, `data-kind="${kind}"`)).join("")}
        ${btn("Switch crack (L1/L2)", "plan-seg-switch", "tiny ghost")}
        ${btn("Undo last point", "plan-undo", "tiny ghost")}
        ${btn("Clear current group", "plan-clear", "tiny ghost")}
      </div>
      <div class="stage"><div class="capture-drawing" id="plan-draw-stage">
        ${img(photo.registered_url, "Registered plane for baffle planning")}
        <svg viewBox="0 0 1000 1000" preserveAspectRatio="none" aria-hidden="true" id="plan-draw-overlay"></svg>
      </div></div>
      <p class="soft" id="plan-draw-count"></p>
      <div class="actions dialog-controls"><button type="submit" class="button primary">Analyze and save plan</button></div>
    </form>`;
  if (!dialog.open) dialog.showModal();
  planDrawUpdate();
  const controller = new AbortController();
  const refreshBar = () => {
    document.querySelectorAll("[data-do=plan-kind]").forEach(button => {
      button.className = "button tiny " + (button.dataset.kind === planState.mode ? "primary" : "secondary");
    });
    const switchButton = document.querySelector("[data-do=plan-seg-switch]");
    if (switchButton) switchButton.textContent = `Editing crack L${planState.segIndex + 1} — switch`;
  };
  refreshBar();
  dialog.addEventListener("click", async event => {
    const target = event.target.closest("[data-do]");
    if (!target || !dialog.contains(target)) return;
    if (target.dataset.do === "plan-close") { dialog.close(); return; }
    if (target.dataset.do === "plan-kind") { planState.mode = target.dataset.kind; refreshBar(); return; }
    if (target.dataset.do === "plan-seg-switch") {
      planState.mode = "segment";
      planState.segIndex = 1 - planState.segIndex;
      refreshBar();
      return;
    }
    if (target.dataset.do === "plan-undo") { planUndo(); return; }
    if (target.dataset.do === "plan-clear") {
      if (planState.mode === "segment") planState.markers.segments[planState.segIndex] = [];
      else planState.markers[planState.mode] = [];
      planDrawUpdate();
    }
  }, {signal: controller.signal});
  dialog.addEventListener("close", () => controller.abort(), {once: true});
  const form = dialog.querySelector("#dialog-form");
  form.dataset.planForm = "1";
  form.addEventListener("submit", async submit => {
    submit.preventDefault();
    const button = submit.submitter || form.querySelector("button[type=submit]");
    button.disabled = true;
    const error = dialog.querySelector(".dialog-error");
    error.textContent = "";
    try {
      validateMarkers();
      const data = new FormData(form);
      const m = planState.markers;
      const payload = {
        title: data.get("title"),
        photo_id: photo.id,
        craze_result_id: data.get("craze_result_id") || null,
        region_id: data.get("region_id") || null,
        baffle_radius_px: Number(data.get("baffle_radius_px")),
        inlets: m.inlet,
        exits: m.exit,
        candidates: m.candidate,
        segments: [
          {label: "Observed crack L1", points: m.segments[0]},
          {label: "Observed crack L2", points: m.segments[1]},
        ],
      };
      const saved = await api(`/api/specimens/${state.tile.id}/baffle-plans`, {method: "POST", body: payload});
      dialog.close();
      await onSaved(saved);
    } catch (err) {
      error.textContent = err.message;
    } finally {
      button.disabled = false;
    }
  });
}

function statePill(state) {
  return `<span><i class="dot" style="background:${STATE_DOT[state]}"></i>${e(state)}</span>`;
}

function scenarioTable(plan, scenario, title) {
  const exitRows = scenario.exits.map(row => [
    row.key, "Outlet", statePill(row.state), row.marked_at_visible_margin ? "visible margin" : "interior trace",
  ]);
  const segmentRows = scenario.segments.map(row => [
    row.key, e(row.label), statePill(row.state),
    `inlet ${Math.round(row.inlet_trace_fraction * 100)}% · hidden ${Math.round(row.hidden_fraction * 100)}%`,
  ]);
  return `<h3>${e(title)}</h3>
    ${table(["Marker", "Role", "Surface state", "Basis"], [...segmentRows, ...exitRows])}`;
}

function candidateTable(plan) {
  const rows = plan.candidates.map((candidate, index) => [
    candidate.candidate_key + (index === plan.selected_candidate_index ? " (chosen)" : ""),
    candidate.edge_routes_open,
    n(candidate.inlet_connected_trace_pixels, 0),
    n(candidate.covered_trace_pixels, 0),
  ]);
  return `<h3>Baffle candidates compared</h3>
    ${table(["Candidate", "Inlet trace px still reaching margin", "Inlet-connected trace px", "Trace covered by disk px"], rows)}`;
}

function observationRow(plan, row, hints) {
  const observed = plan.observations[row.key];
  const hint = hints?.markers?.[row.key];
  const choices = [["", "Not photographed — leave blank"], ["tracer", "Tracer color visible"], ["clear", "No tracer visible"]];
  const select = `<select name="obs-${row.key}">
    ${choices.map(([value, label]) => option(value, label, observed?.state || "")).join("")}</select>`;
  const expected = STATE_DOT[row.expected_state]
    ? `<span><i class="dot" style="background:${STATE_DOT[row.expected_state]}"></i>${e(row.expected_state)}</span>`
    : e(row.expected_state);
  const evidence = observed
    ? `<span><i class="dot" style="background:${OBSERVATION_DOT[observed.state]}"></i>${e(observed.state)}</span>`
    : `<span class="soft">blank</span>`;
  const hintText = hint
    ? `saturated ${Math.round(hint.saturated_fraction * 100)}% of ${hint.sampled_pixels} px`
    : "";
  return `<tr>
    <td><strong>${e(row.key)}</strong><br><small>${e(row.label)}</small></td>
    <td>${expected}</td>
    <td>${evidence}<br><small class="soft">${e(hintText)}</small></td>
    <td>${select}<input type="text" name="note-${row.key}" maxlength="280"
      placeholder="Evidence note" value="${e(observed?.notes || "")}"></td></tr>`;
}

export function verificationPanel(state, plan, hints, actions) {
  const photoChoices = state.tile.photos.filter(item => item.registered_url);
  const selected = plan.observation_photo_id || photoChoices[0]?.id || "";
  const rows = plan.expected_rows.map(row => observationRow(plan, row, hints)).join("");
  return `<section class="card stack" id="verification-panel">
    <h3>Post-experiment verification</h3>
    <p class="plot-note">Expectations are frozen from the pre-experiment plan. Observed states stay blank
      until you set them from the aligned follow-up photograph; saturation figures are viewing aids only.</p>
    <div class="form-columns">
      ${selectField("Follow-up capture", "verify_photo", photoChoices.map(item => option(item.id, item.role_label + " · " + item.name, selected)).join(""))}
      <div class="field"><label>Tracer color hint</label>${btn("Compute hint", "plan-hints", "secondary", "plot")}</div>
    </div>
    <div class="table-wrap"><table><thead><tr><th>Marker</th><th>Expected (plan)</th><th>Observed evidence</th><th>Record / note</th></tr></thead>
      <tbody>${rows}</tbody></table></div>
    ${areaField("Verification note", "verify_note", plan.observation_note || "", 600)}
    <div class="actions">${btn("Save verification", "plan-save-verify", "primary", "check")}
      ${btn("Clear all observations", "plan-clear-verify", "ghost")}</div>
  </section>`;
}

export function planDetail(state, plan, layer, hints) {
  const base = `/api/baffle-plans/${plan.id}`;
  const chosen = plan.candidates[plan.selected_candidate_index];
  const layers = [["plan_overlay", "Plan overlay"]];
  if (plan.pictures.verification_overlay) layers.push(["verification_overlay", "Verification overlay"]);
  const stats = [
    stat("Chosen baffle", chosen.candidate_key),
    stat("Margin trace left open (px)", n(chosen.edge_routes_open, 0)),
    stat("Inlet-connected px", n(chosen.inlet_connected_trace_pixels, 0)),
    stat("Observations recorded", n(Object.keys(plan.observations || {}).length, 0)),
  ].join("");
  return `<section class="card stack">
    <div class="title-row"><div><h3>${e(plan.title)}</h3>
      <small>Plan revision ${plan.version} · ${e(plan.created_at.slice(0, 10))}</small></div>
      ${plan.observation_photo_id ? badge("Verification recorded", "ready") : badge("Plan only", "caution")}</div>
    <div class="actions">${layers.map(([key, label]) =>
      btn(label, "plan-layer", layer === key ? "tiny primary" : "tiny secondary", null, `data-value="${key}"`)).join("")}</div>
    <figure class="stage">${img(`${base}/image/${layer}?v=${plan.version}`, "Baffle plan map")}</figure>
    <div class="legend"><div class="swatches">
      ${Object.entries(plan.state_labels).map(([key, label]) =>
        `<span><i class="dot" style="background:${STATE_DOT[key]}"></i>${e(label)}</span>`).join("")}
    </div></div>
    <p class="notice">${e(plan.selection_reason)}</p>
    <div class="metrics">${stats}</div>
    ${candidateTable(plan)}
    ${scenarioTable(plan, plan.baseline, "Before any baffle (baseline)")}
    ${scenarioTable(plan, chosen, `With baffle ${chosen.candidate_key} placed`)}
  </section>`;
}

export function planListCard(state, plans) {
  if (!plans.length) {
    return `<section class="card"><h3>Baffle plans</h3>
      <p class="soft">Build a sealing plan from a registered photo and its Fine-line network.</p>
      <div class="actions">${btn("New baffle plan", "plan-new", "primary", "plus")}</div></section>`;
  }
  return `<section class="card"><h3>Baffle plans</h3><div class="divider-list scroll-list">
    ${plans.map(plan => `<div class="row"><div class="grow">
      <strong class="row-title">${e(plan.title)}</strong><br>
      <small>${e(plan.selected_candidate)} · ${plan.edge_routes_open} inlet trace px still at margin ·
        ${plan.observation_count} observation(s)</small></div>
      ${btn("Open", "plan-open", "tiny secondary", null, `data-id="${plan.id}"`)}</div>`).join("")}
    </div><div class="actions">${btn("New baffle plan", "plan-new", "primary", "plus")}</div></section>`;
}
