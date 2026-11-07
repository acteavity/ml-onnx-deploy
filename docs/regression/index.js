// Concrete strength regression – onnxruntime-web
// ONNX graph: input1 [batch, 8] raw values -> output1 [batch, 1] strength in MPa
const MODEL_URL = "./concrete_model.onnx";
const N_FEATURES = 8;
let session = null;
let meta = null;

const PRESETS = {
  high:    [540, 0, 0, 162, 2.5, 1040, 676, 28],          // dataset row 1 (79.99 MPa)
  typical: [280, 75, 55, 180, 6, 970, 775, 28],
  young:   [540, 0, 0, 162, 2.5, 1040, 676, 3],
};

async function getSession() {
  if (!session) {
    session = await ort.InferenceSession.create(MODEL_URL + "?v=" + Date.now());
  }
  return session;
}

function preset(name) {
  PRESETS[name].forEach((v, i) => (document.getElementById("f" + i).value = v));
  runRegression();
}

function readInputs() {
  const x = new Float32Array(N_FEATURES);
  for (let i = 0; i < N_FEATURES; i++) {
    x[i] = parseFloat(document.getElementById("f" + i).value) || 0;
  }
  if (x[7] < 1) x[7] = 1;                         // age must be at least 1 day
  return x;
}

function outOfRange(x) {
  if (!meta || !meta.features || !meta.mins) return [];
  return meta.features.filter((f, i) => x[i] < meta.mins[i] || x[i] > meta.maxs[i]);
}

async function runRegression() {
  const out = document.getElementById("output");
  try {
    const x = readInputs();
    const sess = await getSession();
    const tensorX = new ort.Tensor("float32", x, [1, N_FEATURES]);
    const results = await sess.run({ input1: tensorX });
    const mpa = results.output1.data[0];
    const psi = mpa * 145.038;
    const wc = x[0] > 0 ? (x[3] / x[0]).toFixed(2) : "–";
    const oor = outOfRange(x);
    out.innerHTML = `
      <div class="big">${mpa.toFixed(1)} <small>MPa</small></div>
      <div class="muted">≈ ${Math.round(psi).toLocaleString()} psi · water/cement ratio ${wc} · age ${x[7]} days</div>
      ${oor.length ? `<p class="err">Outside training range: ${oor.join(", ")} – extrapolation, treat with caution.</p>` : ""}
      ${meta ? `<p class="muted">Typical error on unseen mixes: ± ${meta.test_metrics.MAE.toFixed(1)} MPa (test MAE).</p>` : ""}`;
  } catch (e) {
    console.error("ONNX runtime error:", e);
    out.innerHTML = `<p class="err">Error: ${e.message}</p>`;
  }
}

async function loadMeta() {
  try {
    meta = await (await fetch("./metrics.json?v=" + Date.now())).json();
    const m = meta.test_metrics;
    document.getElementById("modelInfo").innerHTML = `
      <table>
        <tr><th>Architecture</th><td class="num">${meta.best_model}</td></tr>
        <tr><th>Test R²</th><td class="num">${m.R2.toFixed(3)}</td></tr>
        <tr><th>Test MAE</th><td class="num">${m.MAE.toFixed(2)} MPa</td></tr>
        <tr><th>Test RMSE</th><td class="num">${m.RMSE.toFixed(2)} MPa</td></tr>
        <tr><th>Train / val / test</th><td class="num">${meta.n_train} / ${meta.n_val} / ${meta.n_test}</td></tr>
      </table>`;
    const rows = meta.summary.map(r => `
      <tr><td>${r.model}${r.model === meta.best_model ? " ★" : ""}</td>
          <td class="num">${r.test_R2_mean.toFixed(3)} ± ${r.test_R2_std.toFixed(3)}</td>
          <td class="num">${r.test_MAE.toFixed(2)}</td><td class="num">${r.test_RMSE.toFixed(2)}</td></tr>`).join("");
    document.getElementById("compare").innerHTML = `
      <table><tr><th>Model</th><th class="num">R²</th><th class="num">MAE (MPa)</th><th class="num">RMSE (MPa)</th></tr>${rows}</table>
      <p class="muted">★ deployed model, selected on the validation set.</p>`;
    document.getElementById("rangeNote").textContent =
      "Inputs outside the ranges seen in training are flagged as extrapolation.";
  } catch (e) {
    document.getElementById("modelInfo").textContent = "metrics.json not found.";
    document.getElementById("compare").textContent = "metrics.json not found.";
  }
}

loadMeta();
