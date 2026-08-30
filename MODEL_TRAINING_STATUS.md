# Model Training Status

The project is configured for a compact, balanced CPU-friendly training run.

- Dataset: `mohanty/PlantVillage` official color/RGB split
- Binary classes: `Diseased` and `Healthy`
- Official test split is retained; the downloader caps each class to 500 images for a compact test subset
- Validation is created leaf-safely from the official training split
- Default dataset size: 500/class train, 500/class validation, 500/class test
- Model: MobileNetV2 pretrained on ImageNet
- Default training: 5 epochs, batch size 32, learning rate 0.0003
- The final model must be trained using this pipeline; do not substitute a checkpoint trained on another dataset or preprocessing pipeline.
