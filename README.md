
# CT Mid-L3 Localization with Soft Labeling and Local Context Modeling

This repository contains the implementation of a deep learning-based framework for automatic identification of the mid-L3 vertebral level in axial CT images.

The project focuses on improving mid-L3 localization by combining **distance-aware soft labeling** and **local axial context modeling**. Instead of treating each axial slice only as an independent binary classification target, this study investigates whether anatomical continuity between adjacent CT slices can improve localization performance.

---

## 1. Background

The third lumbar vertebral level (L3) is widely used as a reference slice for CT-based body composition analysis, including skeletal muscle area, adipose tissue distribution, and sarcopenia-related assessment.

However, manual selection of the mid-L3 slice can be time-consuming and observer-dependent. In large-scale retrospective CT studies, automatic identification of the L3 level can reduce manual workload and improve the reproducibility of downstream body composition analysis.

This project addresses the problem as an axial CT slice-level localization task.

---

## 2. Project Objective

The main objective of this project is to automatically identify the axial CT slice corresponding to the mid-L3 vertebral level.

The project investigates the following questions:

- Can a CNN-based model identify L3-related axial CT slices from abdominal CT images?
- Can distance-aware soft labels improve mid-L3 localization compared with binary hard labels?
- Which soft-labeling function is more suitable for axial mid-L3 localization?
- Does local context from adjacent axial slices improve localization performance?
- How accurately can the predicted mid-L3 slice be localized at the patient level?

---

## 3. Dataset

Axial CT slices were generated from volumetric CT data with corresponding vertebral segmentation masks.

The L3 vertebral mask was used to determine the ground-truth mid-L3 level. The center axial slice of the L3 vertebra was defined based on the center of gravity of the L3 mask.

Each saved axial slice was assigned one of the following labels:

```text
L3_mid  : ground-truth mid-L3 slice
L3      : slice within the L3 vertebral region
NL3     : non-L3 slice

Axial slices were sampled at fixed slice intervals to ensure consistent experimental conditions.

Due to data access restrictions, raw CT volumes, segmentation masks, and processed image files are not included in this repository.

---

## 4. Preprocessing

The preprocessing pipeline consists of the following steps:

Load volumetric CT images and vertebral segmentation masks.
Reorient CT volumes into a consistent anatomical orientation.
Identify the L3 vertebral region from the segmentation mask.
Compute the axial center of gravity of the L3 mask.
Define the nearest saved axial slice as the ground-truth mid-L3 slice.
Save axial CT slices as 2D images at fixed slice intervals.
Assign slice-level labels based on their relationship to the L3 region.

The CT images were converted into single-channel axial images and resized for CNN-based training.

---

## 5. Labeling Strategy

Two types of labeling strategies are used in this project.

#5.1 Hard Labeling

In the hard-label setting, slices are assigned binary labels:

L3 → 1
NL3 → 0

This setting treats all L3 slices as positive samples and all non-L3 slices as negative samples.

Hard labeling is suitable for slice-level L3 classification, but it does not explicitly encode how close each slice is to the true mid-L3 level.

#5.2 Soft Labeling

In the soft-label setting, each slice is assigned a continuous target value according to its distance from the ground-truth mid-L3 slice.

The goal is to provide smoother supervision for localization by giving higher target values to slices closer to the mid-L3 level and lower target values to distant slices.

The following soft-labeling functions are used:

Gaussian Soft Label
$$
y = \exp\left(-\frac{d^2}{2\sigma^2}\right)
$$
Laplace Soft Label
$$
y = \exp\left(-\frac{|d|}{\tau}\right)
$$
Sigmoid Soft Label
$$
y = \frac{1}{1 + \exp\left(-\frac{d}{\tau}\right)}
$$

where d is the axial slice distance from the ground-truth mid-L3 slice.

Gaussian and Laplace labels assign the maximum value at the mid-L3 slice and decrease symmetrically as the distance increases. The sigmoid function provides a monotonic transition across the mid-L3 level.

---

## 6. Model Structure

This project includes two main modeling strategies:

Slice-wise single-image classification
Local context-based axial sequence modeling
6.1 Slice-wise CNN Model

The slice-wise baseline model predicts the L3 probability of each axial CT slice independently.

Architecture
Input axial CT slice
        ↓
1-channel ResNet34 backbone
        ↓
Global average pooling
        ↓
Fully connected layer
        ↓
Single output logit
        ↓
Sigmoid probability during inference
Main Characteristics
Backbone: ResNet34
Input channel: 1-channel CT image
Output: one logit per axial slice
Loss function: binary cross-entropy with logits
Prediction target: hard label or soft label

This model does not use neighboring slices. Each axial slice is processed independently.

#6.2 Local Context Model

The local context model uses a sequence of adjacent axial CT slices to predict the target slice.

This approach is motivated by the anatomical continuity of vertebral structures across neighboring axial slices. Since the L3 level changes gradually along the axial direction, adjacent slices may provide useful context for distinguishing the mid-L3 region.

Input Structure
X = [slice t-k, ..., slice t, ..., slice t+k]

where slice t is the center target slice and neighboring slices are used as local anatomical context.

For example, when T = 9, the model receives:

4 previous slices + center slice + 4 next slices
Architecture
Input axial slice sequence
        ↓
Shared ResNet34 feature encoder
        ↓
Slice-level feature tokens
        ↓
Linear projection to embedding dimension
        ↓
Relative positional embedding
        ↓
Transformer encoder
        ↓
Center token selection
        ↓
Linear prediction head
        ↓
Single output logit for the center slice
Main Characteristics
Input: sequence of adjacent axial CT slices
Sequence length: T
Shared feature encoder: ResNet34
Sequence modeling: Transformer encoder
Positional information: learnable relative positional embedding
Output: one logit for the center slice
Loss function: binary cross-entropy with logits

The local context model predicts the label of the center slice while using surrounding slices as contextual information.

---

## 7. Training

All models were trained using binary cross-entropy with logits.

Main training settings:

Optimizer      : AdamW
Learning rate  : 1e-4
Batch size     : 32
Epochs         : 50
Input size     : 256 × 256
Backbone       : ResNet34

For hard-label training, binary targets were used.
For soft-label training, continuous soft targets were used.

The final model checkpoint was selected based on the lowest validation loss.

---

## 8. Evaluation

The models were evaluated at both slice level and patient level.

8.1 Slice-level Evaluation

For hard-label classification, the following metrics were calculated:

Accuracy
Precision
Recall
F1-score

For soft-label prediction, the predicted probability curve was compared with the soft target curve.

#8.2 Patient-level Localization

For each patient, the model outputs a probability score for each saved axial slice.

The predicted mid-L3 slice is determined from the patient-level prediction curve.

Depending on the labeling strategy, the predicted slice can be selected using:

maximum probability
closest point to the midpoint of the sigmoid curve
fitted probability curve peak

Localization error is calculated as the difference between the predicted mid-L3 slice and the ground-truth mid-L3 slice.

#8.3 Localization Error

The following error metrics are used:

Signed error   = predicted slice number - ground-truth slice number
Absolute error = |predicted slice number - ground-truth slice number|

Errors can be reported in original axial slice units or converted into sampled-slice interval units.

If axial slices are saved every six original slices:

sampled interval error = original slice number error / 6

The physical distance can also be estimated using slice spacing.

---

## 9. Repository Structure
.
├── README.md
├── train_ax.py
├── test_ax.py
├── models/
│   ├── __init__.py
│   └── axial_models.py
├── datasets/
│   ├── __init__.py
│   └── axial_dataset.py
├── utils/
│   ├── metrics.py
│   ├── preprocessing.py
│   └── visualization.py
├── configs/
│   └── axial_config.yaml
├── checkpoints/
│   └── README.md
├── test_results/
│   └── README.md
└── requirements.txt

Some files or directories may be excluded depending on data access and experiment settings.

---

## 10. Example Usage
Train Slice-wise Model
python train_ax.py
Evaluate Trained Model
python test_ax.py

Example options can be added depending on the experimental configuration:

python train_ax.py --target_mode soft --soft_type gaussian --context_length 9
python test_ax.py --checkpoint checkpoints/best_model.pth

---

## 11. Experimental Settings

The project compares multiple experimental settings:

Category	Settings
Labeling strategy	Hard label, soft label
Soft-label function	Gaussian, Laplace, Sigmoid
Local context	Single-slice, local-context sequence
Sequence length	T1, T9
Backbone	ResNet34
Evaluation	Slice-level classification, patient-level localization

---

## 12. Key Findings

The main findings of this project can be summarized as follows:

Soft labeling provides distance-aware supervision for mid-L3 localization.
Gaussian and Laplace soft labels are suitable for symmetric mid-L3 localization because they assign the maximum target value at the mid-L3 slice.
Local context modeling allows the model to use anatomical continuity between adjacent axial slices.
Patient-level probability curves can be used to estimate the final mid-L3 slice location.
Localization error analysis provides a more direct evaluation of mid-L3 identification than slice-level classification metrics alone.

---

## 13. Notes on Data Availability

The CT data used in this project are not included in this repository due to dataset access and redistribution restrictions.

This repository is intended to provide the implementation structure, model code, training pipeline, and evaluation logic for research purposes.

---
