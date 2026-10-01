const $ = (s) => document.querySelector(s);
const show = (el, on = true) => el.classList.toggle("hidden", !on);
const fmt = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;

let pid = null;
let project = null;
let selected = null;
let pollTimer = null;
let lastVideoSrc = null;
let lastResultSrc = null;

const fileUrl = (path) => `/files/${pid}/${path}`;

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  let body = null;
  try { body = await r.json(); } catch { /* empty */ }
  if (!r.ok) throw new Error((body && body.detail) || `Request failed (${r.status})`);
  return { status: r.status, body };
}

function setError(msg) {
  $("#error").textContent = msg || "";
  show($("#error"), !!msg);
}

// ------------------------------------------------------------ source
const drop = $("#drop");
$("#file").addEventListener("change", (e) => e.target.files[0] && uploadVideo(e.target.files[0]));
["dragover", "dragenter"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, () => drop.classList.remove("over")));
drop.addEventListener("drop", (e) => { e.preventDefault(); e.dataTransfer.files[0] && uploadVideo(e.dataTransfer.files[0]); });

async function uploadVideo(file) {
  setError();
  const fd = new FormData();
  fd.append("file", file);
  showJob({ status: "running", progress: 0, message: `Uploading ${file.name}` }, "Uploading");
  try {
    const { body } = await api("/api/projects/upload", { method: "POST", body: fd });
    openProject(body.project);
  } catch (e) { show($("#job"), false); setError(e.message); }
}

$("#yt-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  setError();
  try {
    const { body } = await api("/api/projects/youtube", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: $("#yt").value }),
    });
    openProject(body.project);
  } catch (err) { setError(err.message); }
});

$("#restart").addEventListener("click", () => {
  location.hash = "";
  location.reload();
});

// ------------------------------------------------------------ project lifecycle
function openProject(id) {
  pid = id;
  if (location.hash !== `#p=${id}`) history.replaceState(null, "", `#p=${id}`);
  refresh();
}

async function refresh() {
  clearTimeout(pollTimer);
  try {
    const { body } = await api(`/api/projects/${pid}`);
    project = body;
    render();
    const j = project.job_state;
    if (j && (j.status === "queued" || j.status === "running")) pollTimer = setTimeout(refresh, 1000);
  } catch (e) {
    setError(e.message);
  }
}

const JOB_TITLES = { detect: "Finding objects", process: "Replacing the object" };

function showJob(j, title) {
  show($("#job"), true);
  $("#job-title").textContent = title;
  $("#job-pct").textContent = `${Math.round((j.progress || 0) * 100)}%`;
  $("#job-bar").style.width = `${(j.progress || 0) * 100}%`;
  $("#job-msg").textContent = j.message || "";
}

function render() {
  const p = project;
  const j = p.job_state;
  show($("#restart"), true);
  show($("#step-source"), false);

  const busy = j && (j.status === "queued" || j.status === "running");
  if (busy) showJob(j, JOB_TITLES[j.kind] || "Working");
  else show($("#job"), false);
  if (j && j.status === "error") setError(j.error);
  else if (p.status === "error" && !j) setError(p.error);
  else setError();

  // Step 2: objects
  const haveObjects = p.objects && p.objects.length;
  show($("#step-objects"), !!p.video && !(busy && j.kind === "detect"));
  if (p.video) {
    const src = fileUrl(p.video);
    if (lastVideoSrc !== src) { $("#source-video").src = src; lastVideoSrc = src; }
  }
  if (!selected && p.selected) selected = p.selected;
  if (!selected && haveObjects && p.objects.length === 1) selected = p.objects[0].id;
  show($("#no-objects"), p.status === "no_objects");
  renderObjects();

  // Step 3: product
  const sel = haveObjects && p.objects.find((o) => o.id === selected);
  show($("#step-product"), !!sel);
  if (sel) {
    $("#sel-thumb").src = fileUrl(sel.thumb);
    $("#sel-label").textContent = `Original ${sel.label}`;
    const prod = p.product && p.product.object_id === sel.id ? p.product : null;
    renderCheck(prod, sel);
  }

  // Step 4: result
  const done = p.status === "done" && p.result;
  show($("#step-result"), !!done);
  if (done) renderResult(p.result);
  document.querySelectorAll("#go, #pdrop").forEach((el) => el.classList.toggle("disabled", !!busy));
  $("#go").disabled = !!busy;
}

function renderObjects() {
  const box = $("#objects");
  box.innerHTML = "";
  (project.objects || []).forEach((o) => {
    const el = document.createElement("button");
    el.className = "obj" + (o.id === selected ? " selected" : "");
    el.innerHTML = `
      <div class="thumb" style="background-image:url('${fileUrl(o.thumb)}')"></div>
      <div class="meta">
        <div class="name">${o.label}</div>
        <div class="muted">${o.visible_seconds}s on screen</div>
        <div class="chips">${o.segments.map(([a, b]) => `<span class="chip" data-t="${a}">${fmt(a)}–${fmt(b)}</span>`).join("")}</div>
      </div>`;
    el.addEventListener("click", (e) => {
      const t = e.target.dataset && e.target.dataset.t;
      const v = $("#source-video");
      v.currentTime = t !== undefined ? parseFloat(t) : o.best_frame / project.info.fps;
      if (selected !== o.id) { selected = o.id; render(); }
    });
    box.appendChild(el);
  });
}

// ------------------------------------------------------------ product upload
const pdrop = $("#pdrop");
$("#pfile").addEventListener("change", (e) => e.target.files[0] && uploadProduct(e.target.files[0]));
["dragover", "dragenter"].forEach((ev) => pdrop.addEventListener(ev, (e) => { e.preventDefault(); pdrop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => pdrop.addEventListener(ev, () => pdrop.classList.remove("over")));
pdrop.addEventListener("drop", (e) => { e.preventDefault(); e.dataTransfer.files[0] && uploadProduct(e.dataTransfer.files[0]); });

async function uploadProduct(file) {
  setError();
  const fd = new FormData();
  fd.append("object_id", selected);
  fd.append("file", file);
  const c = $("#check");
  c.className = "alert";
  c.textContent = "Checking the image…";
  show(c, true);
  show($("#go"), false);
  try {
    await api(`/api/projects/${pid}/product`, { method: "POST", body: fd });
    await refresh();
  } catch (e) { show(c, false); setError(e.message); }
}

function renderCheck(prod, sel) {
  const c = $("#check");
  const img = $("#cutout");
  if (!prod) {
    show(c, false); show($("#go"), false); show(img, false); show($("#pdrop-text"), true);
    return;
  }
  c.className = "alert " + (prod.check.ok ? "good" : "bad");
  c.textContent = prod.check.message;
  show(c, true);
  const src = fileUrl(prod.cutout || prod.file);
  if (img.getAttribute("src") !== src) img.src = src;
  show(img, true);
  show($("#pdrop-text"), false);
  const processing = project.job_state && project.job_state.kind === "process" &&
    ["queued", "running"].includes(project.job_state.status);
  show($("#go"), prod.check.ok && !processing);
  $("#go").textContent = project.result ? "Run again" : "Replace in video";
}

$("#go").addEventListener("click", async () => {
  setError();
  try {
    await api(`/api/projects/${pid}/process`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ object_id: selected }),
    });
    show($("#step-result"), false);
    refresh();
  } catch (e) { setError(e.message); }
});

// ------------------------------------------------------------ result + compare
function renderResult(r) {
  const after = fileUrl(r.video) + `?v=${r.stamp || ""}`;
  const before = fileUrl(project.video);
  if (lastResultSrc === after) return;
  lastResultSrc = after;
  $("#cmp-after").src = after;
  $("#cmp-before").src = before;
  $("#result-video").src = after;
  $("#side-after").src = after;
  $("#side-before").src = before;
  $("#download").href = after;
  $("#download").setAttribute("download", r.download_name || "vid-bid.mp4");
  $("#result-stats").textContent = r.stats || "";
}

document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === t));
  ["compare", "result", "side"].forEach((n) => show($(`#tab-${n}`), n === t.dataset.tab));
  document.querySelectorAll("video").forEach((v) => v.pause());
}));

const ca = $("#cmp-after"), cb = $("#cmp-before"), cmp = $("#cmp");
function setSplit(x) {
  const r = cmp.getBoundingClientRect();
  const f = Math.min(1, Math.max(0, (x - r.left) / r.width));
  cb.style.clipPath = `inset(0 ${(1 - f) * 100}% 0 0)`;
  $("#cmp-handle").style.left = `${f * 100}%`;
}
let dragging = false;
cmp.addEventListener("pointerdown", (e) => { dragging = true; cmp.setPointerCapture(e.pointerId); setSplit(e.clientX); });
cmp.addEventListener("pointermove", (e) => dragging && setSplit(e.clientX));
cmp.addEventListener("pointerup", () => { dragging = false; });

$("#cmp-play").addEventListener("click", () => {
  if (ca.paused) { cb.currentTime = ca.currentTime; ca.play(); cb.play(); $("#cmp-play").textContent = "Pause"; }
  else { ca.pause(); cb.pause(); $("#cmp-play").textContent = "Play"; }
});
ca.addEventListener("timeupdate", () => {
  if (Math.abs(cb.currentTime - ca.currentTime) > 0.08) cb.currentTime = ca.currentTime;
  if (ca.duration) $("#cmp-seek").value = (ca.currentTime / ca.duration) * 1000;
  $("#cmp-time").textContent = `${fmt(ca.currentTime)} / ${fmt(ca.duration || 0)}`;
});
ca.addEventListener("ended", () => { cb.pause(); $("#cmp-play").textContent = "Play"; });
$("#cmp-seek").addEventListener("input", (e) => {
  if (!ca.duration) return;
  ca.currentTime = cb.currentTime = (e.target.value / 1000) * ca.duration;
});

// side-by-side: keep the two in step
const sa = $("#side-after"), sb = $("#side-before");
sa.addEventListener("play", () => { sb.currentTime = sa.currentTime; sb.play(); });
sa.addEventListener("pause", () => sb.pause());
sa.addEventListener("seeked", () => { sb.currentTime = sa.currentTime; });

// ------------------------------------------------------------ boot
const m = location.hash.match(/p=([a-z0-9]+)/);
if (m) openProject(m[1]);
