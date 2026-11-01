# CrossSkinNet

Configurable PyTorch research pipeline for skin-lesion images and optional patient metadata. Supports DINOv3, DINOv2, EfficientNet and Swin; missing-aware metadata encoding, gated fusion and per-sample image-only fallback.

**The complete maintained package is in [`kaggle_release/`](kaggle_release/README.md).** Start with [`Kaggle_Start.ipynb`](kaggle_release/Kaggle_Start.ipynb) or use the commands below.

## Kaggle quick start

Enable GPU and Internet. Attach your skin dataset and saved Hugging Face model, then run:

```python
!git clone https://github.com/mahabubhasanmahin/model.git /kaggle/working/model
%cd /kaggle/working/model/kaggle_release
!pip install -r requirements.txt
!python smoke_test.py
```

Edit a copy of [`configs/isic2019.yaml`](kaggle_release/configs/isic2019.yaml) with the actual attached dataset/model paths and save it as `/kaggle/working/experiment.yaml`. Then:

```python
!python main.py prepare --config /kaggle/working/experiment.yaml
!python main.py train --config /kaggle/working/experiment.yaml
```

See the package README for validation, final testing, checkpoint resume, single-image prediction and optional Grad-CAM/SHAP.

## Included

- Fixed, audited group splits; explicit missing-ID exclusion reports.
- Train-only augmentation and train-fitted metadata preprocessing.
- AdamW, separate backbone/head learning rates, warmup/cosine decay, AMP, accumulation and early stopping.
- Exact epoch-boundary CPU resume, best validation checkpoint, class mapping and saved image processor.
- Image-only, metadata-only, naive fusion and gated fusion modes.
- Per-class metrics, confusion matrices, metadata-removal stress tests and optional bootstrap intervals.
- Single-image inference with optional metadata; optional image/metadata explanations.

## Validation and limits

11 synthetic/software tests passed, including CPU resume and local random DINOv3 fast-processor round-trip. See [`VERIFICATION.md`](kaggle_release/VERIFICATION.md). This is not evidence of real-dataset accuracy; no GPU/full ISIC run or 98%/99% accuracy is claimed. Lesion-grouped splits do not establish patient independence without patient IDs. Model weights, datasets and credentials are not included.
