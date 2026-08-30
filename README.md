# Plant Disease Classification System

Binary plant-leaf classification system for the internship activity: **Healthy vs Diseased**.

## Activity Alignment

* **Dataset:** Hugging Face `mohanty/PlantVillage`
* **Configuration:** Official RGB/color image split
* **Task:** Binary classification with exactly two output classes:

  * `Healthy`
  * `Diseased`
* **Official test split:** Retained from the PlantVillage dataset
* **Validation:** Created only from the official training split using a **leaf-ID-safe split**
* **Input size:** 224×224
* **Normalization:** ImageNet mean and standard deviation
* **Data augmentation:** Applied only during training
* **Model:** ImageNet-pretrained MobileNetV2 with transfer learning/fine-tuning
* **Evaluation:** Accuracy, precision, recall, F1-score and confusion matrix
* **Interface:** Gradio with Hugging Face Spaces support
* **Backend:** FastAPI support

The activity requires providing the dataset source/link rather than uploading the complete PlantVillage dataset to GitHub.

## Dataset

Official dataset source:

https://huggingface.co/datasets/mohanty/PlantVillage

The project uses the official PlantVillage RGB/color images and retains the official test split.

The validation set is created only from the official training portion. Images belonging to the same leaf are kept within the same split using the provided leaf grouping information, helping prevent data leakage.

### Current experiment

For the final compact training run, the dataset preparation used:

```text
Training:
Healthy   : 500
Diseased  : 500

Validation:
Healthy   : 500
Diseased  : 500

Test:
Healthy   : 500
Diseased  : 500
```

The test set remains separated from training and validation.

## Preprocessing and Augmentation

### Training preprocessing

Training images use:

* Random resized crop to 224×224
* Random horizontal flip
* Random vertical flip
* Random rotation up to 20°
* Color jitter
* Conversion to tensor
* ImageNet normalization

### Validation and test preprocessing

Validation and test images use deterministic preprocessing:

* Resize to 224×224
* Conversion to tensor
* ImageNet normalization

No random augmentation is applied during validation or testing.

## Class Mapping

The project uses the following mapping consistently:

```text
Diseased = 0
Healthy  = 1
```

The class mapping is also stored in the trained model checkpoint to prevent accidental label inversion during deployment.

## Installation

Create and activate a virtual environment:

```bash
python -m venv venv
```

### Windows PowerShell

```powershell
venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Prepare the Dataset

Download and prepare the official Hugging Face dataset:

```bash
python src/data_download.py
```

For the compact final experiment, the dataset can be limited to 500 images per binary class:

```bash
python src/data_download.py --max_per_class 500
```

The preparation process downloads the official dataset files, uses the official train/test split information, and creates a leaf-safe validation split.

The local `data/` directory should not be uploaded to GitHub.

## Train MobileNetV2

The final experiment was trained for 5 epochs using:

```bash
python src/train.py --epochs 5 --batch_size 32 --lr 0.0003
```

The training pipeline:

1. Loads ImageNet-pretrained MobileNetV2.
2. Replaces the classification head for the two-class task.
3. Applies the training augmentation pipeline.
4. Fine-tunes the network.
5. Tracks validation performance.
6. Saves the best validation checkpoint.

The trained model is saved as:

```text
model/plant_classifier.pth
```

### Final training result

```text
Epochs:             5
Batch size:         32
Learning rate:      0.0003
Best validation accuracy: 89.50%
Training time:      approximately 10.75 minutes
```

## Evaluate the Model

Run:

```bash
python src/evaluate.py
```

The evaluation uses the holdout test set and produces classification metrics and a confusion matrix.

### Final Test Results

```text
Test samples:       1000
Accuracy:            93.60%
Macro Precision:     94.33%
Macro Recall:        93.60%
Macro F1-Score:      93.57%
Weighted F1-Score:   93.57%
```

### Per-Class Results

```text
                 Precision   Recall   F1-Score   Support

Diseased           1.0000    0.8720    0.9316      500
Healthy            0.8865    1.0000    0.9398      500
```

Confusion matrix:

```text
                 Predicted
                 Diseased  Healthy

Actual Diseased     436       64
Actual Healthy        0      500
```

The evaluation reports and plots are stored in:

```text
reports/
```

## Run Locally

### Gradio / Hugging Face interface

Run:

```bash
python app_hf.py
```

The application will be available at:

```text
http://localhost:7860
```

or:

```text
http://127.0.0.1:7860
```

The interface allows users to:

* Upload a plant leaf image
* Use webcam input
* Analyze the image
* View Healthy/Diseased probabilities
* View the predicted health status
* View inference latency
* View general health/disease guidance

### FastAPI backend

Run:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Then open:

```text
http://localhost:8000
```

## Model Architecture

The project uses **MobileNetV2** with ImageNet pretrained weights.

MobileNetV2 is used as a transfer-learning backbone and adapted for binary plant-leaf classification.

The final classifier outputs:

```text
Diseased
Healthy
```

## Important Model Limitation

This system performs **binary classification only**.

It predicts whether an uploaded leaf is:

```text
Healthy
```

or:

```text
Diseased
```

It does **not** reliably identify the exact disease species or pathogen.

Therefore, treatment information displayed by the application should be considered general guidance rather than a definitive agricultural diagnosis.

## Deployment

The project supports deployment using Hugging Face Spaces with Docker.

The deployment package should contain:

```text
app/
backend/
model/
src/
reports/
app_hf.py
requirements.txt
Dockerfile
README.md
render.yaml
```

The trained checkpoint should be included:

```text
model/plant_classifier.pth
```

The dataset itself should **not** be uploaded to GitHub.

The application can download/prepare the required dataset using:

```bash
python src/data_download.py
```

when dataset preparation is required.

Do not upload:

```text
data/
venv/
__pycache__/
```

to the repository.

## Repository Structure

```text
plant-disease-classifier/
│
├── app/
│   ├── disease_info.py
│   ├── main.py
│   ├── model_loader.py
│   └── schemas.py
│
├── backend/
│
├── model/
│   ├── network.py
│   ├── plant_classifier.pth
│   ├── class_indices.json
│   └── class_names.json
│
├── reports/
│   ├── metrics_summary.json
│   ├── confusion_matrix.png
│   └── training/evaluation plots
│
├── src/
│   ├── data_download.py
│   ├── dataset.py
│   ├── train.py
│   └── evaluate.py
│
├── tests/
│
├── notebooks/
│
├── app_hf.py
├── Dockerfile
├── requirements.txt
├── render.yaml
├── README.md
└── .gitignore
```

## Reproducibility

To reproduce the final compact experiment:

```bash
python src/data_download.py --max_per_class 500
```

Then:

```bash
python src/train.py --epochs 5 --batch_size 32 --lr 0.0003
```

Then:

```bash
python src/evaluate.py
```

Finally:

```bash
python app_hf.py
```

The expected final test accuracy for the recorded run is:

```text
93.60%
```

## Dataset Attribution

Dataset:

**PlantVillage — `mohanty/PlantVillage`**

Source:

https://huggingface.co/datasets/mohanty/PlantVillage

The dataset is used for academic/internship activity purposes. The complete dataset is not included in this repository.
