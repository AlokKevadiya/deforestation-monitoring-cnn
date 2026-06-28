"""Workflow + CNN architecture diagrams as vector PDFs (300ppi-safe, all vector)."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib as mpl
mpl.rcParams.update({"font.family":"DejaVu Sans"})

GREEN="#2e7d32"; LGREEN="#a5d6a7"; ORANGE="#ef6c00"; LOR="#ffcc80"
BLUE="#1565c0"; LBLUE="#90caf9"; GREY="#cfd8dc"; DARK="#263238"

# ============ WORKFLOW DIAGRAM ============
fig, ax = plt.subplots(figsize=(15, 5))
ax.set_xlim(0,15); ax.set_ylim(0,5); ax.axis("off")

def box(x,y,w,h,text,fc,ec=DARK,fs=9):
    p=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.08",
                     fc=fc,ec=ec,lw=1.2)
    ax.add_patch(p)
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fs,wrap=True)

def arrow(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",
                mutation_scale=14,lw=1.3,color=DARK))

stages = [
    ("Forest Aerial\nImages dataset\n(5,108 images)", LGREEN),
    ("Dataset\ninspection &\ncleaning", LGREEN),
    ("Preprocessing\nresize 128x128\nnormalize", LBLUE),
    ("Stratified split\n70/15/15 +\naugmentation", LBLUE),
    ("Custom CNN\nbaseline", LOR),
    ("VGG16\ntransfer\nlearning", LOR),
    ("Training &\nvalidation", GREY),
    ("Evaluation\nCM, ROC, F1", GREY),
    ("Grad-CAM &\nSHAP\nexplainability", "#ce93d8"),
    ("Model\ncomparison &\ninterpretation", LGREEN),
]
n=len(stages); w=1.25; h=1.1; gap=(15-n*w)/(n+1)
y=2.0
xs=[]
for i,(t,c) in enumerate(stages):
    x=gap+i*(w+gap); xs.append(x)
    box(x,y,w,h,t,c,fs=7.2)
    if i>0:
        arrow(xs[i-1]+w, y+h/2, x, y+h/2)
ax.text(7.5,4.4,"End-to-end experimental workflow for explainable CNN-based deforestation classification",
        ha="center",fontsize=11,fontweight="bold")
plt.tight_layout()
plt.savefig("/home/claude/phase3/Figures/fig_workflow.pdf",bbox_inches="tight")
plt.close()

# ============ CNN ARCHITECTURE DIAGRAM ============
fig, ax = plt.subplots(figsize=(14, 4.5))
ax.set_xlim(0,14); ax.set_ylim(0,4.5); ax.axis("off")

def layer(x,w,h,text,fc,y=1.4,fs=8):
    p=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.01,rounding_size=0.05",fc=fc,ec=DARK,lw=1.0)
    ax.add_patch(p)
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fs,rotation=90)

layers=[
    ("Input\n128x128x3", LGREEN, 0.9, 2.0),
    ("Conv 32\n3x3 + ReLU", LBLUE, 0.7, 1.8),
    ("MaxPool\n2x2", GREY, 0.5, 1.5),
    ("Conv 64\n3x3 + ReLU", LBLUE, 0.7, 1.55),
    ("MaxPool\n2x2", GREY, 0.5, 1.3),
    ("Conv 128\n3x3 + ReLU", LBLUE, 0.7, 1.3),
    ("MaxPool\n2x2", GREY, 0.5, 1.1),
    ("Flatten", LOR, 0.45, 1.0),
    ("Dense 128\n+ ReLU", LOR, 0.6, 1.2),
    ("Dropout\n0.5", "#ffe0b2", 0.5, 1.0),
    ("Dense 1\nSigmoid", GREEN, 0.6, 0.9),
]
x=0.4
xs=[]
for t,c,w,h in layers:
    layer(x,w,h,t,c,y=2.25-h/2)
    xs.append((x,w))
    x+=w+0.45
for i in range(1,len(xs)):
    x0=xs[i-1][0]+xs[i-1][1]; x1=xs[i][0]
    ax.add_patch(FancyArrowPatch((x0,2.25),(x1,2.25),arrowstyle="-|>",mutation_scale=10,lw=1.0,color=DARK))
ax.text(x/2,4.0,"Architecture of the custom CNN baseline for binary Forest / Non-Forest classification",
        ha="center",fontsize=11,fontweight="bold")
ax.text(x/2,0.5,"Three convolutional blocks (32-64-128 filters) followed by a fully connected head with dropout regularization",
        ha="center",fontsize=8,style="italic",color=DARK)
plt.tight_layout()
plt.savefig("/home/claude/phase3/Figures/fig_cnn_architecture.pdf",bbox_inches="tight")
plt.close()
print("Diagrams generated: fig_workflow.pdf, fig_cnn_architecture.pdf")
