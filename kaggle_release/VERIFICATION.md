# Verification

Verified in this session:

- `python -m pytest -q`: **12 passed**, including optional SHAP installed.
- Controlled experiment runner: baseline reuse, actual synthetic augmentation training, validation-only summaries, fixed data/seed preservation, rerun protection and error-review gallery verified.
- `python smoke_test.py`: passed synthetic training, checkpoint reload, stress evaluation, and single-image prediction.
- Mid-phase epoch-boundary CPU resume produced identical final weights and history to uninterrupted training.
- Small random-config DINOv3, DINOv2, EfficientNet, Swin adapter forwards and block selection exercised.
- Local random DINOv3 checkpoint + fast image processor saved, trained, reloaded, evaluated and used for inference without remote model access.
- Grad-CAM and single-case Kernel SHAP software paths exercised. A degenerate spatial map is rejected, not fabricated.

Environment: Python 3.12, torch 2.5.1+cpu, torchvision 0.20.1+cpu, Transformers 4.57.6. No GPU run or real ISIC training was executed for this release. No 98%/99% disease-accuracy claim is made. Synthetic results are not paper results. Optional pretrained model access, actual Kaggle GPU availability, runtime storage/quota, and real-data quality remain environment/data-specific.
