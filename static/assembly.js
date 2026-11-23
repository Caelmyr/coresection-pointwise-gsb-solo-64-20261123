import {api, btn, e, img, link, n, option, selectField, textField, numField, areaField, field, table, badge, empty} from "./widgets.js";

let assembly = null;
let assemblies = [];

export function assemblyState() {
  return {assembly, assemblies};
}

export function setAssemblyData(list, active) {
  assemblies = list || [];
  assembly = active || null;
}

function gapFields(prefix, spec = {}) {
  return `<div class="form-columns">
    ${selectField(`${prefix}gap_end`, [
      option("", "No documented gap", spec.gap_end || ""),
      option("uphole", "Gap at uphole end", spec.gap_end),
      option("downhole", "Gap at downhole end", spec.gap_end),
    ].join(""))}
    ${numField(`${prefix}gap_length_mm`, spec.gap_length_mm ?? "", 0.1, 100000, 0.1)}
  </div>`;
}

function segmentCard(key, index) {
  return `<section class="card stack segment-card">
    <h3>Segment ${key}${index === 0 ? " · fixed uphole" : ""}</h3>
    <div class="form-columns">
      ${numField(`${key}length_mm`, "", 0.1, 100000, 0.1)}
      ${selectField(`${key}downhole_arrow`, [
        option("down", "Arrow points downhole (down)", "down"),
        option("up", "Arrow points uphole (up)", ""),
      ].join(""))}
    </div>
    <div class="form-columns">
      ${textField(`${key}old_label`, "", "", 'maxlength="60"')}
      ${numField(`${key}old_depth_top_m`, "", -100000, 100000, 0.001)}
    </div>
    <div class="form-columns">
      ${numField(`${key}old_depth_bottom_m`, "", -100000, 100000, 0.001)}
    </div>
    ${gapFields(`${key}`)}
    ${areaField(`${key}note`, "", 400)}
    ${field(`Segment ${key} fracture-face photo`, `photo-${key}`,
      `<input id="field-photo-${key}" name="photo-${key}" type="file" accept="image/png,image/jpeg,image/tiff,image/webp" required>`)}
    <p class="soft">The photographed face only. Documented gaps stay missing — no other image is copied in.</p>
  </section>`;
}

export function assemblyWizard(modal, onDone) {
  draftFiles = {A: null, B: null, C: null};
  const content = `
    <section class="card stack">
      <h3>Assembly record</h3>
      <div class="form-columns">
        ${textField("name", "", "", 'required maxlength="120"')}
        ${textField("tray", "", "", 'maxlength="120"')}
        ${numField("core_diameter_mm", 84, 1, 100000, 0.1)}
        ${numField("origin_depth_m", "", -100000, 100000, 0.001)}
        ${numField("shelf_length_mm", "", 1, 100000, 0.1)}
      </div>
      <p class="soft">Origin depth is the uphole end of the adopted string. Both field orderings must fit the same shelf.</p>
    </section>
    ${["A", "B", "C"].map((key, i) => segmentCard(key, i)).join("")}
    <section class="card stack">
      <h3>Two field orderings</h3>
      <p class="notice">Segment A is fixed at the uphole end; the two sheets can only swap B and C.</p>
      <div class="form-columns">
        ${textField("order1_label", "Field sheet 1", "Field sheet 1", 'maxlength="120"')}
        ${textField("order2_label", "Field sheet 2", "Field sheet 2", 'maxlength="120"')}
      </div>
      <p class="soft">Order 1: A → B → C · Order 2: A → C → B</p>
    </section>`;
  modal("Reassemble a broken core string", content, "Evaluate and archive one order", async (form) => {
    const value = (name) => form.get(name);
    const number = (name) => (value(name) === "" || value(name) == null ? null : Number(value(name)));
    const segment = (key) => ({
      segment_key: key,
      length_mm: Number(value(`${key}length_mm`)),
      downhole_arrow: value(`${key}downhole_arrow`),
      old_label: value(`${key}old_label`) || "",
      old_depth_top_m: number(`${key}old_depth_top_m`),
      old_depth_bottom_m: number(`${key}old_depth_bottom_m`),
      gap_end: value(`${key}gap_end`) || null,
      gap_length_mm: value(`${key}gap_end`) ? Number(value(`${key}gap_length_mm`)) : null,
      note: value(`${key}note`) || "",
    });
    const payload = {
      name: value("name"),
      tray: value("tray") || "",
      core_diameter_mm: Number(value("core_diameter_mm")),
      origin_depth_m: number("origin_depth_m"),
      shelf_length_mm: number("shelf_length_mm"),
      segments: ["A", "B", "C"].map(segment),
      orders: [
        {label: value("order1_label") || "Field sheet 1", segment_keys: ["A", "B", "C"]},
        {label: value("order2_label") || "Field sheet 2", segment_keys: ["A", "C", "B"]},
      ],
    };
    const body = new FormData();
    body.append("document", JSON.stringify(payload));
    for (const key of ["A", "B", "C"]) {
      const file = form.get(`photo-${key}`);
      if (!file || !file.size) throw Error(`Choose the fracture-face photograph for segment ${key}.`);
      body.append(`photo-${key}`, file);
    }
    const created = await api("/api/assemblies", {method: "POST", body});
    onDone(created);
  }, true);
}

function joinRows(evaluation) {
  return evaluation.joins.map((join) => {
    const lamina = join.lamina;
    const depth = join.depth;
    return [
      `${join.uphole_segment} → ${join.downhole_segment}`,
      n(lamina.score, 3),
      n(lamina.tone_delta_linear, 3),
      `${n(lamina.lamina_angle_delta_deg, 1)}°`,
      n(lamina.edge_profile_correlation, 3),
      depth.evaluable
        ? `${n(depth.delta_m, 3)} m${depth.within_tolerance ? " ✓" : ""}`
        : "no legacy depth",
    ];
  });
}

function orderCard(evaluation, adoptedId) {
  const adopted = evaluation.order_id === adoptedId;
  return `<article class="card stack ${adopted ? "adopted" : "rejected"}">
    <div class="title-row"><h3>${e(evaluation.label)}</h3>
      ${adopted ? badge("Archived order", "ready") : badge("Not adopted", "caution")}</div>
    <p class="soft">Sequence ${evaluation.segment_keys.join(" → ")} · lamina score <strong>${n(evaluation.lamina_score, 4)}</strong>
      · legacy depth score <strong>${evaluation.depth_score == null ? "—" : n(evaluation.depth_score, 4)}</strong></p>
    ${table(["Fracture join", "Lamina", "Tone Δ", "Angle Δ", "Edge r", "Legacy depth step"], joinRows(evaluation))}
  </article>`;
}

function segmentRows(record) {
  return record.segments.map((segment) => {
    const photoUrl = `/api/assemblies/${record.id}/segments/${segment.segment_key}/photo/working`;
    return [
      segment.segment_key,
      img(photoUrl, `Segment ${segment.segment_key} face`, 'class="item-image" loading="lazy"'),
      `${n(segment.length_mm, 1)} mm`,
      segment.downhole_arrow,
      segment.old_label || "—",
      segment.old_depth_top_m == null ? "—" : `${n(segment.old_depth_top_m, 3)} m`,
      segment.old_depth_bottom_m == null ? "—" : `${n(segment.old_depth_bottom_m, 3)} m`,
      segment.gap_end ? `${segment.gap_end} · ${n(segment.gap_length_mm, 1)} mm missing` : "—",
    ];
  });
}

export function assemblyPage(active) {
  assembly = active;
  if (!assembly) {
    return `<header class="page-head"><div><h1>Core segment assembly</h1>
      <p>Reassemble three photographed fracture faces, keep documented gaps honest, and archive exactly one of the two field orderings.</p></div>
      <div class="actions">${btn("Load demo assembly", "assembly-demo", "light", "sample")}${btn("New assembly", "assembly-new", "primary", "plus")}</div></header>
      ${assemblies.length ? `<div class="collection">${assemblies.map(item => `<button class="tile-card" type="button" data-do="assembly-open" data-id="${item.id}">
        <div class="tile-info"><h3>${e(item.name)}</h3><p>${e(item.tray || "—")}</p>
        <div class="tile-meta"><span>${n(item.occupied_length_mm, 1)} mm string</span><span>${n(item.total_gap_length_mm, 1)} mm gaps</span></div>
        ${item.duplicate_old_labels.length ? badge("Duplicate legacy number", "caution") : ""}</div></button>`).join("")}</div>`
        : empty("No assembly archived yet. Load the practice assembly or start a new record.")}`;
  }
  const adoptedId = assembly.decision.order_id;
  const adoptedOrder = assembly.field_orders.find((o) => o.order_id === adoptedId);
  return `<header class="page-head"><div><p class="eyebrow">Archived core string</p><h1>${e(assembly.name)}</h1>
    <p>${e(assembly.tray || "—")} · ${n(assembly.occupied_length_mm, 1)} mm on the shelf · ${n(assembly.total_gap_length_mm, 1)} mm in documented gaps</p></div>
    <div class="actions">${btn("New assembly", "assembly-new", "secondary", "plus")}${link("Open adopted string", `#tile:${assembly.specimen_id}`, "primary", "tiles")}</div></header>
  ${assembly.duplicate_old_labels.length ? `<p class="notice">Duplicate legacy number <strong>${e(assembly.duplicate_old_labels.map(d => d[2]).join(", "))}</strong> on segments ${e(assembly.duplicate_old_labels.map(d => d[1]).join(" and "))}; old numbering cannot be trusted on its own.</p>` : ""}
  <section class="card stack"><h3>Segments and evidence</h3>
    ${table(["Key", "Photographed face", "Length", "Downhole arrow", "Old number", "Legacy top", "Legacy bottom", "Documented gap"], segmentRows(assembly))}
  </section>
  <section class="card stack"><h3>Adopted string preview</h3>
    <figure class="stage"><img src="/api/specimens/${assembly.specimen_id}/viewer/${assembly.stitched_photo_id}?layer=raw" alt="Adopted stitched core string with transparent gaps"></figure>
    <p class="soft">Archived sequence: ${adoptedOrder.segment_keys.join(" → ")} · transparent bands are the known missing intervals, never borrowed imagery.</p>
  </section>
  <div class="grid-detail">${assembly.evaluations.map((item) => orderCard(item, adoptedId)).join("")}</div>
  <section class="card stack"><h3>Archive decision</h3>
    <div class="metrics">
      <div class="stat"><strong>${e(assembly.decision.basis)}</strong><span>Deciding evidence</span></div>
      <div class="stat"><strong>${e(adoptedOrder.label)}</strong><span>Adopted ordering</span></div>
    </div>
    <p>${e(assembly.decision.rationale)}</p>
    <p class="notice"><strong>Set aside:</strong> ${e(assembly.decision.defeated_reason)}</p>
    ${assembly.decision.manual_note ? `<p><strong>Curator note:</strong> ${e(assembly.decision.manual_note)}</p>` : ""}
    <div class="actions">
      ${assembly.field_orders.filter((o) => o.order_id !== adoptedId).map((o) =>
        btn(`Switch to '${o.label}'`, "assembly-override", "ghost", "edit", `data-id="${o.order_id}"`)).join("")}
      ${link("Line scan the adopted string", `#tile:${assembly.specimen_id}`, "secondary", "measure")}
    </div>
  </section>`;
}
