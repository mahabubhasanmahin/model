# CrossSkinNet — Kaggle research package

## বাংলা: প্রথমে কী করবে

1. `CrossSkinNet-Kaggle.zip` দিয়ে একটি **private Kaggle Dataset** তৈরি করো। Kaggle সাধারণত ZIP extract করে দেয়।
2. নতুন Kaggle Notebook-এ code dataset, ISIC 2019 dataset, এবং saved DINOv3 model **Add Input** দিয়ে attach করো।
3. এই package-এর `Kaggle_Start.ipynb` notebook Kaggle-এ import করো। সেটি download করতে না পারলে নিচের bootstrap cell ব্যবহার করো এবং notebook file-টি Kaggle input থেকে খুলে cells copy করো।
4. GPU এবং dependency install-এর জন্য Internet চালু করো। Local model ব্যবহার করলে Hugging Face token লাগবে না।
5. Notebook-এর config cell-এ paths দেখো। Automatic discovery-তে একাধিক match হলে actual paths বসাও।
6. প্রথমে `SMOKE_RUN=True`; সফল হলে `False` দিয়ে নতুন output directory-তে full training। এটি training data-এর একটি subset নয়—সব selected training samples-এ ১+১ epochs।
7. Image-only baseline → augmentation comparison → gated fusion → loss comparison → optional deeper fine-tuning। প্রত্যেকবার একই fixed split ব্যবহার করো এবং একবারে একটি setting বদলাও।
8. Validation দিয়ে final model নির্বাচন করার পরে notebook-এর `RUN_FINAL_TEST=True` করো। Test দিয়ে বারবার hyperparameter tune করবে না।
9. `Save Version` দিয়ে outputs রাখো। `best.pt`, `processor/`, `schema.json`, split manifests ও config একসঙ্গে রাখবে।

**98% বা 99% accuracy নিশ্চিত নয়।** Synthetic smoke-test accuracy গবেষণার ফল নয়। আগের image-level results এবং নতুন lesion-grouped results সরাসরি তুলনাযোগ্য নয়।

### Download-independent bootstrap after the code is attached

```python
from pathlib import Path
import shutil, sys
roots = [p.parent for p in Path('/kaggle/input').rglob('main.py')
         if (p.parent / 'crossskinnet' / 'engine.py').exists()]
assert len(roots) == 1, roots
shutil.copytree(roots[0], '/kaggle/working/CrossSkinNet_Kaggle', dirs_exist_ok=True)
%cd /kaggle/working/CrossSkinNet_Kaggle
!pip install -r requirements.txt
```

## GitHub থেকে Kaggle-এ চালানো

Code ZIP upload প্রয়োজন নেই। নতুন Kaggle notebook-এ GPU/Internet enable করে চালাও:

```python
!git clone https://github.com/mahabubhasanmahin/model.git /kaggle/working/model
%cd /kaggle/working/model/kaggle_release
!pip install -r requirements.txt
!python smoke_test.py
```

তারপর `configs/isic2019.yaml` copy করে `/kaggle/working/experiment.yaml`-এ actual model/data paths বসাও। প্রথমে prepare, তারপর train:

```python
!python main.py prepare --config /kaggle/working/experiment.yaml
!python main.py train --config /kaggle/working/experiment.yaml
```

Guided notebook: `kaggle_release/Kaggle_Start.ipynb`। সেটি GitHub থেকে notebook হিসেবে import করেও ব্যবহার করা যায়। Setup cell existing `/kaggle/working/model/kaggle_release` clone চিনবে।

## Contents

- `Kaggle_Start.ipynb`: guided cells for setup, paths, preparation, training, validation, final testing, prediction, XAI, output export.
- `main.py`: command-line interface.
- `configs/isic2019.yaml`: complete editable experiment configuration.
- `crossskinnet/`: model, data/splits, training/resume/evaluation, optional XAI.
- `smoke_test.py`: offline synthetic end-to-end software test.
- `tests/`: behavioral and adapter tests.
- `metadata_example.json`: raw metadata example, not normalized/encoded values.

No datasets, pretrained weights, tokens, or real patient records are bundled.

## Runtime

Use Kaggle's existing matched PyTorch/torchvision packages. Requirements pin Transformers 4.57.6 and do not independently replace torch/torchvision. If upgrading Python packages after importing Transformers, restart the session. On a fresh local machine first install the appropriate official matched PyTorch/torchvision pair for your CPU/CUDA, then requirements. CPU tests used Python 3.12, torch 2.5.1+cpu, torchvision 0.20.1+cpu, Transformers 4.57.6. GPU performance and full ISIC training have not been verified by this package's synthetic tests.

DINOv3 uses `use_fast: true` and requires torchvision. The processor must come from the selected checkpoint, including resize and normalization. Supported Transformers architectures:

| backbone | Example checkpoint |
| --- | --- |
| dinov3 | facebook/dinov3-vitb16-pretrain-lvd1689m |
| dinov2 | facebook/dinov2-base |
| efficientnet | google/efficientnet-b0 |
| swin | microsoft/swin-tiny-patch4-window7-224 |

For a local model, `path` points to a folder with `config.json`, safetensors/PyTorch weights (all shards and index if sharded), and `preprocessor_config.json`. A lone `.pth` is not a supported Hugging Face checkpoint. Online gated models require approved account access and authentication; use Kaggle Secrets for HF_TOKEN, never YAML. Weights stay governed by their original license.

`tiny` is an offline software-testing architecture, not a scientific baseline.

## Data and leakage controls

ISIC 2019 uses one-hot `label_columns`. Other datasets can use `label_columns: null`, `label_column: diagnosis`. Image identifiers may be full relative filenames or basenames without extension. A separate metadata CSV is left-joined by image ID, with one-to-one validation; if metadata already exists in the labels CSV set metadata_csv to null. Ambiguous duplicate columns are rejected.

Numeric feature scaling, class mapping, and categorical vocabularies are fitted on training rows only. Missing metadata uses learned tokens. Unknown observed categories have their own ID 0; metadata values must use the dataset's coding/vocabulary. Empty/standard pandas NA CSV values count as missing. Remove label-derived, post-diagnosis, patient/lesion ID, and source leakage variables from feature lists. IDs may be used for grouping but should not be predictive metadata.

For no metadata, set numeric_features and categorical_features to empty lists. The gated model then operates as image-only. Metadata-only mode requires at least one feature.

The ISIC configuration explicitly drops rows with missing lesion IDs. The number and class distribution of exclusions are saved in `split_info.json`. This subset may be biased and must be disclosed; it is not interchangeable with the full dataset. It does not establish patient independence. Provide patient IDs as well when available. With multiple group columns, connected components prevent crossing either identifier. Unknown relationships cannot be verified. Missing IDs are never replaced with invented patient IDs.

New grouped splits use reproducible label-stratified group folds: approximately 70/10/20 by image count. Exact ratios are not guaranteed. Every split must contain all classes. Small datasets may need externally prepared splits via split_column. Split selection must not be driven by model scores. Saved CSVs are reused across every experiment; changed raw CSVs/data settings are rejected for an existing split_dir. Moving data or changing cohort needs a new split_dir and must be reported as a different evaluation.

Image identifiers/paths and configured group IDs are checked for overlap. Corrupt images are checked by default. `check_exact_duplicates: true` hashes image content before splitting and rejects exact duplicates. That option can be costly. Near-duplicate detection and source-domain confounding still need separate review. Cached split signatures hash source CSVs/settings, not the complete image corpus: do not change image contents silently after preparation.

Historical test results already inspected during tuning should be treated as exploratory. A new untouched external/final holdout is preferable for final claims. This package's test split is internal, not the official ISIC challenge evaluation.

## Preparation progress

Run `python -u main.py prepare --config experiment.yaml` for unbuffered status output. Preparation shows progress bars for file scanning, image-path resolution, group-ID linking, image verification, and optional duplicate hashing. File scanning has an unknown total and reports the count; later bars show percentages and ETA. Fold creation is announced by stage because sklearn does not expose per-fold-construction progress. When fixed splits already exist, preparation reuses them and skips image verification, so no image progress bar is expected.

Update a clean Kaggle clone with `git -C /kaggle/working/model pull --ff-only` before using this version. Do not update files while a training/preparation process is still running.

## Training

```bash
python main.py prepare --config configs/isic2019.yaml
python main.py train --config configs/isic2019.yaml
```

Phase 1 freezes vision weights; Phase 2 unfreezes the last two blocks. Optional Phase 3 unfreezes the last four. Swin blocks are counted across stages, EfficientNet uses convolutional blocks. Unfrozen blocks receive gradients while vision dropout/BatchNorm run in eval mode, preventing frozen running-stat drift; this does not disable gradient computation.

Separate backbone/head LRs, AdamW, warmup/cosine decay (floor 10% of base LR), mixed precision on CUDA, exact weighted-loss normalization across accumulation groups, clipping, and phase-level early stopping are implemented. Early stopping advances to the next configured phase. The global best validation macro-F1 checkpoint is retained, regardless of the last epoch. Minimum improvement controls patience, not which best checkpoint is saved. Full training is not an instruction to run more epochs after validation has stopped improving.

Loss choices: ce, weighted_ce, focal (inverse-frequency alpha and configurable gamma). Compare separately; no automatic oversampling is applied. Aggressive augmentation and stronger class weighting are not inherently better. Training-only augmentation uses flips/right-angle rotations. Metadata dropout affects training masks only.

Training history logs both validation accuracy and macro-F1. `best.pt` is for evaluation/inference. `last.pt` stores model, optimizer, scheduler, scaler, epoch/phase/patience counters, history, Python/NumPy/torch/CUDA RNG states. Writes are atomic. Resume is at epoch boundaries; minibatch progress is not saved.

```bash
python main.py train --config experiment.yaml --resume outputs/run/last.pt
```

Resume requires identical configuration/data/splits except output_dir. Keep last.pt, best.pt and processor/ together. For a read-only saved Kaggle input, use a fresh writable output_dir and ensure the original image paths and fixed split directory are restored. If split_dir was under the previous working folder, restore its files to the same path first. Do not change epochs or LRs and call it an exact resume. A completed run does not automatically extend training. CPU uninterrupted-vs-resumed weights are tested bit-for-bit; GPU determinism across devices/drivers is not guaranteed.

## Controlled experiments

Keep `data` identical and choose new `run.output_dir` values. Modes:

- image_only: pretrained vision + classifier; ignores metadata.
- metadata_only: missing-aware metadata MLP; no vision encoder weights.
- naive: zero-imputed metadata projections, simple learned concatenation fusion, no missing tokens/mask input/fallback/residual gate.
- gated: missing-aware encoder + sigmoid gate + residual fusion, exact image fallback per fully missing sample in eval mode.

Use the same feature schema for mode comparisons; image_only ignores it. Use the same split files and seed. Compare augmentation off/on, losses, then two/four-block schedules independently. Several training seeds help quantify stability. Multi-backbone ensembling, lesion segmentation, external data training and higher-resolution preprocessing changes are not implemented automatically.

```bash
python main.py evaluate --checkpoint outputs/A/best.pt --split val --device cuda
python main.py evaluate --checkpoint outputs/B/best.pt --split val --device cuda
python main.py compare outputs/A outputs/B
```

Comparison refuses different split fingerprints.

## Final evaluation and single-image inference

```bash
python main.py evaluate --checkpoint outputs/run/best.pt --split test --device cuda --stress --bootstrap 200
python main.py predict --checkpoint outputs/run/best.pt --image /path/image.jpg --metadata metadata_example.json --device cuda
```

Omit --metadata for all-missing metadata. Predict accepts a new image without a true label; its probability is a model score, not a calibrated probability of a correct clinical diagnosis. Dataset accuracy requires known labels across many unseen examples. Do not interpret one/two images as paper accuracy.

Evaluation reports accuracy, balanced accuracy, macro-F1, class precision/recall/F1/support, confusion matrix, probabilities CSV, plots, and optional percentile bootstrap 95% intervals for accuracy/macro-F1. Group bootstrap resamples connected groups when available; otherwise image bootstrap is used. For final reporting increase bootstrap samples as resources permit and disclose the method.

Stress testing randomly removes each metadata entry with probabilities 0/25/50/75/100%, using one reproducible seed. This is additional masking, not exactly a percentage of originally observed values. Repeat seeds for statistical robustness. Same-model 100% fallback is NOT a separately trained image-only baseline. Accuracy stability is an empirical result, never assumed.

## Optional explainability

```bash
pip install -r requirements-xai.txt
python main.py xai --checkpoint outputs/run/best.pt --image /path/image.jpg --metadata metadata_example.json --output-dir outputs/run/explanation --device cuda --shap
```

Grad-CAM hooks a pre-attention norm in the last transformer block, or a final CNN convolution. It is displayed in processor image coordinates; a processor crop is not the full raw image. Zero/degenerate maps are reported rather than fabricated. The Swin implementation requires square token grids. Metadata-only mode has no image Grad-CAM; this combined XAI command requires an image encoder. A separate metadata-only SHAP entry point is not included.

Kernel SHAP is a single-case metadata explanation for the chosen class with a fixed image and fixed availability mask, using a train-derived encoded background. Numeric attributions use normalized feature values, categorical substitutions come from training values. It does not explain missingness mechanisms or causal effects. Kernel SHAP can be slow. Neither Grad-CAM nor SHAP proves correct lesion localization or clinical safety. Dataset-wide beeswarm analysis and combined publication case figures require further analysis.

## Output retention and privacy

Save Version in Kaggle or download outputs. Keep source, config, split manifests, schema, best.pt, last.pt and processor/. CSVs include metadata and image paths: keep them private. Do not put Hugging Face tokens in saved source, YAML, metadata JSONs or logs. The package does not change external repositories or upload patient data.

## Software verification

```bash
pip install -r requirements-dev.txt
python -m pytest -q
python smoke_test.py
```

Tests exercise grouped splitting/filtering, frozen split identity, all ablation modes, per-sample fallback, exact CPU resume, differential LR groups, four real adapter architectures with random small configurations, local DINOv3 fast-processor round-trip, prediction, evaluation/stress/bootstrap and optional SHAP/Grad-CAM. The optional SHAP test skips if SHAP is not installed. Synthetic tests validate software behavior, not disease accuracy or trained-model clinical quality.
