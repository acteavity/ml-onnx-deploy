// Pima diabetes classifier – onnxruntime-web
// ONNX graph: input1 [batch, 8] raw values -> output1 [batch, 2] softmax [P(no diabetes), P(diabetes)]
const MODEL_URL = "./pima_model.onnx";
const N_FEATURES = 8;
let session = null;
let meta = null;

const PRESETS = {
  pos: [6, 148, 72, 35, 0, 33.6, 0.627, 50],   // dataset row 1, outcome = 1
  neg: [1, 85, 66, 29, 0, 26.6, 0.351, 31],    // dataset row 2, outcome = 0
};

async function getSession() {
  if (!session) session = await ort.InferenceSession.create(MODEL_URL + "?v=" + Date.now());
  return session;
}

function preset(name) {
  PRESETS[name].forEach((v, i) => (document.getElementById("f" + i).value = v));
  runClassifier();
}

async function runClassifier() {
  const out = document.getElementById("output");
  try {
    const x = new Float32Array(N_FEATURES);
    for (let i = 0; i < N_FEATURES; i++) x[i] = parseFloat(document.getElementById("f" + i).value) || 0;
    const sess = await getSession();
    const results = await sess.run({ input1: new ort.Tensor("float32", x, [1, N_FEATURES]) });
    const p = results.output1.data[1];                       // P(diabetes)
    const thr = 0.5;
    const tuned = meta ? meta.threshold_tuned : null;
    const positive = p >= thr;
    const imputed = meta && meta.features && meta.zero_as_missing ? meta.features.filter((f, i) => meta.zero_as_missing.includes(f) && x[i] === 0) : [];
    out.innerHTML = `
      <div class="big">${(p * 100).toFixed(1)}<small>%</small></div>
      <div class="muted">probability of diabetes</div>
      <div class="bar"><span style="width:${(p * 100).toFixed(1)}%"></span></div>
      <p><span class="badge ${positive ? "warn" : "good"}">${positive ? "Predicted: diabetes" : "Predicted: no diabetes"}</span>
         <span class="muted"> at threshold ${thr}</span></p>
      ${tuned !== null && tuned !== thr ? `<p class="muted">With the screening threshold ${tuned} (chosen on validation to maximize F1):
         <b>${p >= tuned ? "diabetes" : "no diabetes"}</b>.</p>` : ""}
      ${imputed.length ? `<p class="muted">Imputed with training median: ${imputed.join(", ")}.</p>` : ""}`;
  } catch (e) {
    console.error("ONNX runtime error:", e);
    out.innerHTML = `<p class="err">Error: ${e.message}</p>`;
  }
}

async function loadMeta() {
  try {
    meta = await (await fetch("./metrics.json?v=" + Date.now())).json();
    const m = meta.test_metrics;
    const cm = meta.confusion_matrix;
    document.getElementById("modelInfo").innerHTML = `
      <table>
        <tr><th>Architecture</th><td class="num">${meta.best_model}</td></tr>
        <tr><th>Test F1 (diabetes)</th><td class="num">${m.F1.toFixed(3)}</td></tr>
        <tr><th>Precision / recall</th><td class="num">${m.precision.toFixed(3)} / ${m.recall.toFixed(3)}</td></tr>
        <tr><th>Accuracy</th><td class="num">${m.accuracy.toFixed(3)}</td></tr>
        <tr><th>ROC-AUC</th><td class="num">${m.ROC_AUC.toFixed(3)}</td></tr>
        <tr><th>Confusion [TN FP; FN TP]</th><td class="num">[${cm[0][0]} ${cm[0][1]}; ${cm[1][0]} ${cm[1][1]}]</td></tr>
      </table>`;
    const rows = meta.summary.map(r => `
      <tr><td>${r.model}${r.model === meta.best_model ? " ★" : ""}</td>
          <td class="num">${r.F1.toFixed(3)} ± ${r.F1_std.toFixed(3)}</td>
          <td class="num">${r.precision.toFixed(3)}</td><td class="num">${r.recall.toFixed(3)}</td>
          <td class="num">${r.accuracy.toFixed(3)}</td><td class="num">${r.ROC_AUC.toFixed(3)}</td></tr>`).join("");
    document.getElementById("compare").innerHTML = `
      <table><tr><th>Model</th><th class="num">F1</th><th class="num">Precision</th><th class="num">Recall</th>
        <th class="num">Accuracy</th><th class="num">AUC</th></tr>${rows}</table>
      <p class="muted">★ deployed model, selected on the validation set.</p>`;
  } catch (e) {
    document.getElementById("modelInfo").textContent = "metrics.json not found.";
    document.getElementById("compare").textContent = "metrics.json not found.";
  }
}

loadMeta();
