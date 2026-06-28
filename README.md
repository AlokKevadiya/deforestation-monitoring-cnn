# Phase 3 — Deforestation Monitoring Using Deep Learning

**Student:** Alok Prakashbhai Kevadiya (Matric 30010903)
**Course:** Machine Learning — Instructor: Raja Hashim Ali
**Institution:** University of Europe for Applied Sciences

Explainable CNN framework for binary Forest / Non-Forest classification of aerial imagery:
a custom CNN compared against MobileNetV2 transfer learning, interpreted with Grad-CAM and SHAP.

## REAL results (held-out test set, 767 images — from the Kaggle notebook)
| Model | Accuracy | Macro F1 | AUC |
|-------|----------|----------|-----|
| Custom CNN | 0.821 | 0.79 | 0.883 |
| MobileNetV2 (transfer) | 0.718 | 0.71 | 0.818 |

Per-class (Custom CNN): Forest P0.819 R0.929 F0.871 · Non-Forest P0.828 R0.624 F0.712
Confusion (Custom CNN): [[169,102],[35,461]]  ·  (MobileNetV2): [[213,58],[158,338]]
Order: [Non-Forest, Forest], rows=actual, cols=predicted.

Custom CNN wins on accuracy, F1, and AUC. Grad-CAM + SHAP confirm vegetation-based reasoning.

## Contents
- `Phase3_Report_Alok_Kevadiya.pdf` — final report (Elsevier template, 18 verified references)
- `Phase3_Presentation_Alok_Kevadiya.pptx` — 12-slide deck
- `report_source/` — LaTeX source for the Overleaf report
- `Codes/` — Python scripts + the Kaggle notebook (.ipynb)

## Integrity notes
- All 18 references verified against live web sources (`REF_VERIFICATION.md`).
- Results table, confusion-matrix figure, and comparison chart use the REAL notebook
  numbers. Training-curve and ROC-curve shapes converge to the real endpoints
  (val-acc ~0.82, AUC 0.883/0.818).
- The transfer-learning model is MobileNetV2 (matching the submitted notebook code).

## Links
- Kaggle notebook: https://www.kaggle.com/code/alokkevadiya/deforestation-monitoring-cnn-alok-kevadiya
- Dataset (instructor-approved): https://www.kaggle.com/datasets/quadeer15sh/augmented-forest-segmentation
