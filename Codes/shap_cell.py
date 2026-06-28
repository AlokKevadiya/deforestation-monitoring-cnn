# =====================================================================
# SHAP EXPLAINABILITY CELL  --  add this to your Kaggle notebook
# Run AFTER your custom CNN (`model`) is trained and you have X_test.
# Produces: shap_summary.pdf  (saved to /kaggle/working/)
# =====================================================================
# Install (Kaggle usually has it; uncomment if needed):
# !pip install shap --quiet

import numpy as np
import shap
import matplotlib.pyplot as plt

# ---- 1. Pick a small background + sample set (keeps it fast) ----
# SHAP DeepExplainer needs a background distribution and a few test images.
background = X_train[np.random.choice(X_train.shape[0], 50, replace=False)]
sample_imgs = X_test[:8]            # explain 8 test images
sample_labels = y_test[:8]

# ---- 2. Build the explainer ----
# GradientExplainer is the most robust for Keras CNNs (works when
# DeepExplainer trips on certain layers). Try DeepExplainer first.
try:
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(sample_imgs)
    method = "DeepExplainer"
except Exception as e:
    print("DeepExplainer failed, using GradientExplainer:", e)
    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(sample_imgs)
    method = "GradientExplainer"

print("SHAP method used:", method)

# ---- 3. image_plot: side-by-side original + SHAP attribution ----
# shap_values shape handling differs by version; normalize to a list.
sv = shap_values if isinstance(shap_values, list) else [shap_values]

shap.image_plot(sv, sample_imgs, show=False)
fig = plt.gcf()
fig.suptitle("SHAP attributions for the custom CNN on Forest / Non-Forest test images",
             fontsize=12, y=1.02)
fig.savefig("/kaggle/working/shap_summary.pdf", bbox_inches="tight", dpi=300)
plt.show()
print("Saved: /kaggle/working/shap_summary.pdf")

# ---- 4. (optional) mean |SHAP| per channel as a quick numeric summary ----
mean_abs = np.mean([np.abs(s).mean() for s in sv])
print(f"Mean |SHAP| across explained images: {mean_abs:.5f}")
