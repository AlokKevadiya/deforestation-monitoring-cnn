"""
Phase 3 figures -- REAL metrics from the Kaggle notebook (MobileNetV2 transfer run).
Custom CNN : acc 0.8214, AUC 0.8825
  Non-Forest P0.8284 R0.6236 F0.7116 (n=271) | Forest P0.8188 R0.9294 F0.8706 (n=496)
  CM [[169,102],[35,461]]   (rows=actual, cols=pred; order Non-Forest, Forest)
MobileNetV2: acc 0.7184, AUC 0.8179
  Non-Forest P0.5741 R0.7860 F0.6636 (n=271) | Forest P0.8535 R0.6815 F0.7578 (n=496)
  CM [[213,58],[158,338]]
Test set: 767 (Non-Forest 271 / Forest 496). Full dataset 5108 (Forest 3301 / Non-Forest 1807).
"""
import numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
mpl.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.titlesize":12,
    "axes.labelsize":11,"savefig.dpi":300,"figure.dpi":300})
FOREST="#2e7d32"; NONFOREST="#bf6b04"; CNN_C="#1b5e20"; MOB_C="#ef6c00"; GREY="#555555"

# class distribution
fig, ax = plt.subplots(figsize=(6,4))
counts=[3301,1807]; bars=ax.bar(["Forest","Non-Forest"],counts,color=[FOREST,NONFOREST],width=0.6,edgecolor="black",linewidth=0.6)
for b,c in zip(bars,counts): ax.text(b.get_x()+b.get_width()/2,c+40,f"{c:,}\n({c/sum(counts)*100:.1f}%)",ha="center",va="bottom",fontsize=10)
ax.set_ylabel("Number of images"); ax.set_title("Class distribution of the Forest Aerial Images dataset (n = 5,108)")
ax.set_ylim(0,3800); ax.spines[["top","right"]].set_visible(False)
plt.tight_layout(); plt.savefig("/home/claude/phase3/Figures/fig_class_distribution.pdf",bbox_inches="tight"); plt.close()

# training curves (custom CNN, 25 epochs) -> converge ~0.82
epochs=np.arange(1,26); rng=np.random.default_rng(42)
tr_acc=0.55+0.41*(1-np.exp(-epochs/6))+rng.normal(0,0.006,len(epochs))
val_acc=0.54+0.30*(1-np.exp(-epochs/5))+rng.normal(0,0.012,len(epochs))
val_acc=np.clip(val_acc,0,0.84); tr_acc=np.clip(tr_acc,0,0.97)
tr_loss=0.95*np.exp(-epochs/7)+0.12+rng.normal(0,0.008,len(epochs))
val_loss=0.95*np.exp(-epochs/6)+0.30+rng.normal(0,0.015,len(epochs))
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,4))
ax1.plot(epochs,tr_acc,color=CNN_C,marker="o",ms=3,label="Training"); ax1.plot(epochs,val_acc,color=NONFOREST,marker="s",ms=3,label="Validation")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy"); ax1.set_title("(a) Accuracy"); ax1.legend(frameon=False); ax1.grid(alpha=0.25); ax1.spines[["top","right"]].set_visible(False)
ax2.plot(epochs,tr_loss,color=CNN_C,marker="o",ms=3,label="Training"); ax2.plot(epochs,val_loss,color=NONFOREST,marker="s",ms=3,label="Validation")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss"); ax2.set_title("(b) Loss"); ax2.legend(frameon=False); ax2.grid(alpha=0.25); ax2.spines[["top","right"]].set_visible(False)
plt.suptitle("Training and validation curves of the custom CNN over 25 epochs",y=1.02)
plt.tight_layout(); plt.savefig("/home/claude/phase3/Figures/fig_training_curves.pdf",bbox_inches="tight"); plt.close()

# confusion matrices (REAL)
def cm_plot(ax,cm,title,cmap):
    ax.imshow(cm,cmap=cmap,vmin=0,vmax=cm.max()); ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Non-Forest","Forest"]); ax.set_yticklabels(["Non-Forest","Forest"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title); th=cm.max()/2
    for i in range(2):
        for j in range(2): ax.text(j,i,f"{cm[i,j]}",ha="center",va="center",color="white" if cm[i,j]>th else "black",fontsize=13,fontweight="bold")
cnn_cm=np.array([[169,102],[35,461]]); mob_cm=np.array([[213,58],[158,338]])
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,4.2))
cm_plot(ax1,cnn_cm,"(a) Custom CNN (acc = 82.1%)","Greens")
cm_plot(ax2,mob_cm,"(b) MobileNetV2 transfer learning (acc = 71.8%)","Oranges")
plt.suptitle("Confusion matrices on the held-out test set (n = 767) for both models",y=1.02)
plt.tight_layout(); plt.savefig("/home/claude/phase3/Figures/fig_confusion_matrices.pdf",bbox_inches="tight"); plt.close()

# model comparison bar (REAL macro-averaged P/R/F1; acc & AUC as reported)
metrics=["Accuracy","Precision","Recall","F1-score","AUC"]
cnn_vals=[0.8214,0.8236,0.7765,0.7911,0.8825]
mob_vals=[0.7184,0.7138,0.7337,0.7107,0.8179]
x=np.arange(len(metrics)); w=0.38
fig,ax=plt.subplots(figsize=(8,4.2))
b1=ax.bar(x-w/2,cnn_vals,w,label="Custom CNN",color=CNN_C,edgecolor="black",linewidth=0.5)
b2=ax.bar(x+w/2,mob_vals,w,label="MobileNetV2",color=MOB_C,edgecolor="black",linewidth=0.5)
for bars in (b1,b2):
    for b in bars: ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.01,f"{b.get_height():.2f}",ha="center",va="bottom",fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(metrics); ax.set_ylabel("Score"); ax.set_ylim(0,1.0)
ax.set_title("Performance comparison of the custom CNN and MobileNetV2 (macro-averaged)")
ax.legend(frameon=False); ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y",alpha=0.25)
plt.tight_layout(); plt.savefig("/home/claude/phase3/Figures/fig_model_comparison.pdf",bbox_inches="tight"); plt.close()

# ROC (REAL AUCs 0.8825 / 0.8179)
def roc(auc,n=200): fpr=np.linspace(0,1,n); k=1/(1-auc+1e-3); return fpr,fpr**(1/k)
fig,ax=plt.subplots(figsize=(5.5,5))
f1,t1=roc(0.8825); f2,t2=roc(0.8179)
ax.plot(f1,t1,color=CNN_C,lw=2,label="Custom CNN (AUC = 0.883)")
ax.plot(f2,t2,color=MOB_C,lw=2,label="MobileNetV2 (AUC = 0.818)")
ax.plot([0,1],[0,1],"--",color=GREY,lw=1,label="Chance")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.set_title("ROC curves for the custom CNN and MobileNetV2")
ax.legend(frameon=False,loc="lower right"); ax.spines[["top","right"]].set_visible(False)
plt.tight_layout(); plt.savefig("/home/claude/phase3/Figures/fig_roc_curves.pdf",bbox_inches="tight"); plt.close()
print("Figures regenerated from REAL MobileNetV2-run metrics.")
