# Regression & Classification with PyTorch → ONNX → GitHub Pages

ITS 52000 – Applied Machine Learning · Hoang Thien Thanh Le (Tea)

| Task | Dataset | Target | Models (PyTorch) | Main metric |
|---|---|---|---|---|
| Regression | Concrete Compressive Strength (UCI, 1 030 rows, 8 inputs) | strength (MPa) | Linear, MLP, Deep NN, F1+F2, ResidualNet | R² (+ MAE, RMSE) |
| Classification | Pima Indians Diabetes (768 rows, 8 inputs) | diabetes 0/1 | Logistic Regression, MLP, Deep NN, F1+F2, ResidualNet | F-measure (+ precision, recall, MCC, ROC-AUC, confusion matrix, correlation matrix) |

Live demo: `https://<your-username>.github.io/<repo>/`

## Repository layout
```
notebooks/
  1_regression_concrete.ipynb      # full regression pipeline + ONNX export
  2_classification_pima.ipynb      # full classification pipeline + ONNX export
  3_make_report.ipynb              # builds REPORT.md / REPORT.docx from your results
data/                              # local copies of both CSVs (notebooks also download them)
docs/                              # GitHub Pages site (onnxruntime-web)
  index.html  style.css
  regression/      index.html index.js  + concrete_model.onnx + metrics.json   (from notebook 1)
  classification/  index.html index.js  + pima_model.onnx     + metrics.json   (from notebook 2)
```

## Step 1 – Run the notebooks in Google Colab (≈ 10–15 min total, CPU is fine)
1. Open https://colab.research.google.com → *File → Upload notebook* → `notebooks/1_regression_concrete.ipynb`.
2. *Runtime → Run all*. The data are downloaded automatically (or upload `data/concrete.csv` to the Colab file panel).
3. In the **same Colab session**, upload and run `2_classification_pima.ipynb`, then `3_make_report.ipynb`
   (edit `GITHUB_USER` and `REPO` in its first code cell).
4. Colab downloads `ALL_outputs.zip`, containing:
   * `docs/regression/concrete_model.onnx`, `docs/regression/metrics.json`
   * `docs/classification/pima_model.onnx`, `docs/classification/metrics.json`
   * `figures/*.png`, `results/*.csv|json`, `REPORT.md`, `REPORT.docx`

Each notebook prints `max |ONNX - PyTorch|` – it should be ~1e-6, which proves the exported model is identical.

## Step 2 – Put the models into the site
Copy the 4 generated files into this repo:
```
outputs/docs/regression/concrete_model.onnx   → docs/regression/
outputs/docs/regression/metrics.json          → docs/regression/
outputs/docs/classification/pima_model.onnx   → docs/classification/
outputs/docs/classification/metrics.json      → docs/classification/
```

## Step 3 – Deploy on GitHub Pages
1. Create a **public** repo on GitHub (e.g. `ml-onnx-deploy`) and upload everything (drag-and-drop in the browser works:
   *Add file → Upload files*).
2. *Settings → Pages → Build and deployment → Source: Deploy from a branch → Branch: `main`, folder `/docs` → Save.*
3. After ~1 minute the site is live at `https://<username>.github.io/<repo>/`.
   * Regression app: `…/regression/index.html`
   * Classification app: `…/classification/index.html`

Test locally before pushing: `cd docs && python -m http.server 8000` → open http://localhost:8000
(ONNX files cannot be loaded from `file://`).

## How the deployment works (class approach)
* Pre-processing lives **inside** the PyTorch module as buffers – training-set medians (Pima zero-imputation),
  `log(1+age)` (concrete), `x_means / x_deviations`, and for regression `y_means / y_deviations` to de-scale the output.
* `torch.onnx.export(model, dummy_input, "...onnx", input_names=["input1"], output_names=["output1"], opset_version=15, dynamic_axes=...)`
* In the browser: `ort.InferenceSession.create("./model.onnx")` → `session.run({ input1: new ort.Tensor("float32", x, [1, 8]) })` → `results.output1.data`.

## Methodology highlights
* 70 / 15 / 15 train / validation / test split (stratified for classification); all statistics fit on train only.
* Concrete: 25 duplicate rows removed; `log(1+age)` ablation included.
* Pima: physiologically impossible zeros (glucose, BP, skin, insulin, BMI) treated as missing → median imputation; class-weighted cross-entropy; threshold analysis.
* Every architecture trained with 5 seeds (mean ± std); early stopping on validation loss; the deployed model is selected on **validation**, the test set is only reported.
