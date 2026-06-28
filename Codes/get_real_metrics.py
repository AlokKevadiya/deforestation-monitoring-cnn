# =====================================================================
# RUN THIS in your Kaggle notebook (after models are trained & you have
# X_test, y_test) to get the REAL precision/recall and confusion matrices.
# Paste the printed output back so the report uses genuine numbers.
# =====================================================================
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def report_for(model, name, X_test, y_test):
    # predicted probabilities -> labels (threshold 0.5)
    proba = model.predict(X_test).ravel()
    pred  = (proba >= 0.5).astype(int)
    print("="*60)
    print(name)
    print("="*60)
    # Forest vs Non-Forest: confirm which integer is which class in YOUR data!
    print(classification_report(y_test, pred, target_names=["Forest","Non-Forest"], digits=3))
    cm = confusion_matrix(y_test, pred)
    print("Confusion matrix [rows=actual, cols=pred], order [Forest, Non-Forest]:")
    print(cm)
    try:
        print("AUC:", round(roc_auc_score(y_test, proba), 3))
    except Exception as e:
        print("AUC could not be computed:", e)
    print()

# call for each model (rename to your actual variable names):
report_for(cnn_model,  "CUSTOM CNN",            X_test, y_test)
report_for(vgg_model,  "VGG16 TRANSFER LEARNING", X_test, y_test)
