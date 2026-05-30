"""
=============================================================================
ÉTAPE 3 : Évaluation Avancée et Calibration des Probabilités
=============================================================================
Figures générées (etape3_evaluation/figures/) :
  fig01_precision_recall_curves.png
  fig02_roc_curves.png
  fig03_metrics_heatmap.png
  fig04_reliability_diagrams.png
  fig05_proba_distributions.png
  fig06_threshold_analysis.png
  fig07_metrics_radar.png
  fig08_brier_score_comparison.png
=============================================================================
"""

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
try:
    from sklearn.frozen import FrozenEstimator
except ImportError:
    FrozenEstimator = None
from sklearn.metrics import (
    f1_score, precision_recall_curve, auc,
    matthews_corrcoef, roc_curve, roc_auc_score,
    brier_score_loss, precision_score, recall_score,
)
from matplotlib.patches import Patch
import pickle

ROOT      = Path(__file__).resolve().parent.parent
FIG_DIR   = Path(__file__).parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_PKL  = ROOT / "etape1_eda"    / "prepared_data.pkl"
MODEL_PKL = ROOT / "etape2_models" / "trained_models.pkl"
OUT_PKL   = Path(__file__).parent  / "calibrated_models.pkl"

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
plt.rcParams.update({"figure.dpi":130,"axes.spines.top":False,"axes.spines.right":False})
C = {"maj":"#4C72B0","min":"#DD8452","green":"#2ecc71","purple":"#9b59b6",
     "red":"#e74c3c","orange":"#f39c12","teal":"#1abc9c","dark":"#2c3e50"}

def save_fig(fig, name):
    p = FIG_DIR / name
    fig.savefig(p, bbox_inches="tight", dpi=130)
    print(f"  ✓ {name}")

with open(DATA_PKL,  "rb") as f: data   = pickle.load(f)
with open(MODEL_PKL, "rb") as f: models = pickle.load(f)

X_train=data["X_train"]; X_test=data["X_test"]
y_train=data["y_train"]; y_test=data["y_test"]
logreg=models["logreg"]; rf=models["rf"]; xgb_best=models["xgb_best"]
y_pred_lr=models["y_pred_lr"];   y_proba_lr=models["y_proba_lr"]
y_pred_rf=models["y_pred_rf"];   y_proba_rf=models["y_proba_rf"]
y_pred_xgb=models["y_pred_xgb"]; y_proba_xgb=models["y_proba_xgb"]

print("="*65)
print("ÉTAPE 3 – Évaluation et Calibration")
print("="*65)

def compute_metrics(name, y_true, y_pred, y_proba):
    f1   = f1_score(y_true, y_pred, average="macro")
    mcc  = matthews_corrcoef(y_true, y_pred)
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    auprc = auc(rec, prec)
    brier = brier_score_loss(y_true, y_proba)
    auroc = roc_auc_score(y_true, y_proba)
    return {"name":name,"F1-Macro":f1,"AUPRC":auprc,"MCC":mcc,
            "AUROC":auroc,"Brier":brier,"precision":prec,"recall":rec}

results = [
    compute_metrics("LogReg (Elastic Net)", y_test, y_pred_lr,  y_proba_lr),
    compute_metrics("Random Forest",        y_test, y_pred_rf,  y_proba_rf),
    compute_metrics("XGBoost (Optuna)",     y_test, y_pred_xgb, y_proba_xgb),
]
metrics_df = pd.DataFrame([{k:v for k,v in r.items()
                             if k not in ["precision","recall"]}
                            for r in results])
print("\nMétriques :")
print(metrics_df[["name","F1-Macro","AUPRC","MCC","AUROC","Brier"]].to_string(index=False))

colors_ = [C["maj"], C["min"], C["green"]]
styles_ = ["-","--","-."]

# FIG 01 – Precision-Recall
fig, ax = plt.subplots(figsize=(9, 7))
for res, col, ls in zip(results, colors_, styles_):
    ax.plot(res["recall"], res["precision"], color=col, lw=2.5, ls=ls,
            label=f"{res['name']} (AUPRC={res['AUPRC']:.4f})")
ax.axhline(y_test.mean(), color="gray", ls=":", lw=1.5,
           label=f"Baseline ({y_test.mean():.4f})")
ax.fill_between(results[2]["recall"], results[2]["precision"], alpha=0.08, color=C["green"])
for i, res in enumerate(results):
    ax.annotate(f"MCC={res['MCC']:.3f}", xy=(0.05, 0.28-i*0.07),
                color=colors_[i], fontsize=10, fontweight="bold",
                xycoords="axes fraction")
ax.set_xlabel("Recall", fontsize=12); ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Courbes Precision-Recall – Comparaison 3 modèles\n"
             "(AUPRC bien plus informative qu'AUROC en fort déséquilibre)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10); ax.set_xlim(0,1); ax.set_ylim(0,1.02)
plt.tight_layout()
save_fig(fig, "fig01_precision_recall_curves.png")
plt.show()

# FIG 02 – ROC
fig, ax = plt.subplots(figsize=(8, 7))
for res, col, ls, yp in zip(results, colors_, styles_,
                              [y_proba_lr, y_proba_rf, y_proba_xgb]):
    fpr, tpr, _ = roc_curve(y_test, yp)
    ax.plot(fpr, tpr, color=col, lw=2.5, ls=ls,
            label=f"{res['name']} (AUC={res['AUROC']:.4f})")
ax.plot([0,1],[0,1],"k--",lw=1,label="Baseline (0.500)")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("Courbes ROC – Comparaison 3 modèles",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
plt.tight_layout()
save_fig(fig, "fig02_roc_curves.png")
plt.show()

# FIG 03 – Heatmap métriques
fig, ax = plt.subplots(figsize=(10, 4))
hm_df = metrics_df.set_index("name")[["F1-Macro","AUPRC","MCC","AUROC","Brier"]].copy()
hm_n  = hm_df.copy(); hm_n["Brier"] = 1 - hm_n["Brier"]
sns.heatmap(hm_n.astype(float), annot=hm_df.round(4).values,
            fmt="", cmap="YlGn", ax=ax, linewidths=0.5,
            cbar_kws={"label":"Score (Brier=inversé)"})
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
ax.set_title("Métriques avancées – Comparaison des 3 modèles\n"
             "Justification : F1-Macro, AUPRC et MCC ne sont pas biaisés par le déséquilibre",
             fontsize=11, fontweight="bold")
plt.tight_layout()
save_fig(fig, "fig03_metrics_heatmap.png")
plt.show()

# ── Calibration ───────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split as tts
X_cal, X_eval, y_cal, y_eval = tts(
    X_test, y_test, test_size=0.5, random_state=RANDOM_STATE, stratify=y_test
)
if FrozenEstimator is not None:
    xgb_platt = CalibratedClassifierCV(
        FrozenEstimator(xgb_best), method="sigmoid", cv=None
    )
else:
    xgb_platt = CalibratedClassifierCV(xgb_best, method="sigmoid", cv="prefit")
xgb_platt.fit(X_cal, y_cal)
y_proba_platt = xgb_platt.predict_proba(X_eval)[:, 1]
y_pred_platt  = xgb_platt.predict(X_eval)

if FrozenEstimator is not None:
    xgb_iso = CalibratedClassifierCV(
        FrozenEstimator(xgb_best), method="isotonic", cv=None
    )
else:
    xgb_iso = CalibratedClassifierCV(xgb_best, method="isotonic", cv="prefit")
xgb_iso.fit(X_cal, y_cal)
y_proba_iso = xgb_iso.predict_proba(X_eval)[:, 1]
y_pred_iso  = xgb_iso.predict(X_eval)

y_proba_raw_eval = xgb_best.predict_proba(X_eval)[:, 1]
y_pred_raw_eval  = xgb_best.predict(X_eval)

print("\nMétriques après calibration :")
for nm, yp, ypred in [("XGBoost brut",y_proba_raw_eval,y_pred_raw_eval),
                       ("Platt Scaling",y_proba_platt,y_pred_platt),
                       ("Isotonic",y_proba_iso,y_pred_iso)]:
    print(f"  {nm:20s} → MCC={matthews_corrcoef(y_eval,ypred):.4f} "
          f"| F1={f1_score(y_eval,ypred,average='macro'):.4f} "
          f"| Brier={brier_score_loss(y_eval,yp):.5f}")

# FIG 04 – Reliability Diagrams (3 panneaux)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
versions = [
    ("XGBoost brut",        y_proba_raw_eval, C["maj"]),
    ("Platt Scaling",       y_proba_platt,    C["min"]),
    ("Isotonic Regression", y_proba_iso,      C["green"]),
]
for ax, (nm, yp, col) in zip(axes, versions):
    frac, mean_p = calibration_curve(y_eval, yp, n_bins=10, strategy="uniform")
    br = brier_score_loss(y_eval, yp)
    ax.plot(mean_p, frac, marker="o", color=col, lw=2.5, ms=8,
            label="Calibration réelle")
    ax.plot([0,1],[0,1],"k--",lw=1.5,label="Parfaite")
    ax.fill_between([0,1],[0,1],alpha=0.04,color="gray")
    ax2 = ax.twinx()
    ax2.hist(yp, bins=20, color=col, alpha=0.18, density=True)
    ax2.set_ylabel("Densité prédictions", color="gray", fontsize=8)
    ax2.tick_params(colors="gray")
    ax.set_xlabel("P(fraude) prédit"); ax.set_ylabel("Fraction positifs réels")
    ax.set_title(f"{nm}\n(Brier = {br:.5f})", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9); ax.set_xlim(0,1); ax.set_ylim(0,1)

plt.suptitle("Diagrammes de Fiabilité (Reliability Diagrams)\n"
             "Un modèle bien calibré suit la diagonale – Platt vs Isotonic Regression",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig(fig, "fig04_reliability_diagrams.png")
plt.show()

# FIG 05 – Distributions probabilités
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (nm, yp, col) in zip(axes, versions):
    for cls, c_col, lbl in [(0,C["maj"],"Légit."),(1,C["min"],"Fraude")]:
        ax.hist(yp[y_eval==cls], bins=30, color=c_col, alpha=0.65, density=True, label=lbl)
    ax.axvline(0.5, color=C["red"], ls="--", lw=1.5, label="Seuil 0.5")
    ax.set_xlabel("P(Fraude)"); ax.set_ylabel("Densité")
    ax.set_title(nm, fontsize=11, fontweight="bold"); ax.legend(fontsize=9)
plt.suptitle("Distribution des probabilités prédites – avant/après calibration",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig(fig, "fig05_proba_distributions.png")
plt.show()

# FIG 06 – Threshold analysis
thresholds = np.linspace(0.01, 0.99, 200)
f1_sc, mcc_sc, prec_sc, rec_sc = [], [], [], []
for t in thresholds:
    pt = (y_proba_xgb >= t).astype(int)
    f1_sc.append(f1_score(y_test, pt, average="macro", zero_division=0))
    mcc_sc.append(matthews_corrcoef(y_test, pt))
    prec_sc.append(precision_score(y_test, pt, pos_label=1, zero_division=0))
    rec_sc.append(recall_score(y_test, pt, pos_label=1, zero_division=0))

best_t_mcc = thresholds[np.argmax(mcc_sc)]
best_t_f1  = thresholds[np.argmax(f1_sc)]

fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(thresholds, f1_sc,   color=C["maj"],    lw=2.5, label="F1-Macro")
ax.plot(thresholds, mcc_sc,  color=C["min"],    lw=2.5, label="MCC")
ax.plot(thresholds, prec_sc, color=C["green"],  lw=1.5, ls="--", alpha=0.8, label="Précision (fraude)")
ax.plot(thresholds, rec_sc,  color=C["purple"], lw=1.5, ls="--", alpha=0.8, label="Rappel (fraude)")
ax.axvline(best_t_mcc, color=C["min"], ls=":", lw=2,
           label=f"Seuil optimal MCC = {best_t_mcc:.2f}")
ax.axvline(best_t_f1,  color=C["maj"], ls=":", lw=2,
           label=f"Seuil optimal F1  = {best_t_f1:.2f}")
ax.axvline(0.5, color="gray", ls="--", lw=1, alpha=0.5, label="Seuil par défaut 0.5")
ax.set_xlabel("Seuil de décision"); ax.set_ylabel("Score")
ax.set_title("XGBoost – Analyse du Seuil de Décision (Threshold Analysis)\n"
             "Trouver le seuil optimal selon le critère métier choisi",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9, ncol=3); ax.set_ylim(-0.1, 1.05)
plt.tight_layout()
save_fig(fig, "fig06_threshold_analysis.png")
plt.show()
print(f"  Seuil optimal MCC={best_t_mcc:.2f} | F1={best_t_f1:.2f}")

# FIG 07 – Radar
metric_names = ["F1-Macro","AUPRC","MCC","AUROC"]
angles = np.linspace(0, 2*np.pi, len(metric_names), endpoint=False).tolist()
angles += angles[:1]
fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))
for res, col in zip(results, colors_):
    vals = [res[m] for m in metric_names]
    vals[2] = (vals[2]+1)/2   # MCC normalisé [0,1]
    vals += vals[:1]
    ax.plot(angles, vals, color=col, lw=2.5, label=res["name"])
    ax.fill(angles, vals, color=col, alpha=0.1)
ax.set_thetagrids(np.degrees(angles[:-1]),
                  ["F1-Macro","AUPRC","MCC (norm.)","AUROC"])
ax.set_ylim(0,1)
ax.set_title("Radar – Métriques avancées (3 modèles)", fontsize=13,
             fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.35,1.15), fontsize=10)
plt.tight_layout()
save_fig(fig, "fig07_metrics_radar.png")
plt.show()

# FIG 08 – Brier score comparaison
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
models_brier = [
    ("LogReg", y_proba_lr),("RF", y_proba_rf),("XGBoost\nbrut", y_proba_raw_eval),
    ("XGBoost\nPlatt", y_proba_platt),("XGBoost\nIsotonic", y_proba_iso),
]
names_b = [m[0] for m in models_brier]
briers  = [brier_score_loss(y_test if "XGBoost\nbrut" not in m[0]
                             and "Platt" not in m[0] and "Isotonic" not in m[0]
                             else y_eval, m[1]) for m in models_brier]

col_b = [C["maj"], C["min"], C["green"], C["orange"], C["teal"]]
bars_b = axes[0].bar(names_b, briers, color=col_b, edgecolor="white", linewidth=1.5)
for bar, val in zip(bars_b, briers):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0001,
                 f"{val:.5f}", ha="center", fontsize=9, fontweight="bold")
axes[0].set_ylabel("Brier Score (↓ mieux)")
axes[0].set_title("Brier Score par modèle\n(inclut calibration Platt & Isotonic)",
                   fontsize=11, fontweight="bold")

# Calibration Error (ECE approx)
for ax_c, (nm_c, yp_c, col_c) in zip([None], [("XGBoost", y_proba_raw_eval, C["green"])]):
    pass

axes[1].axis("off")
summary_text = (
    "Résumé des métriques (justification) :\n\n"
    "▸ F1-Macro\n"
    "  Moyenne non pondérée du F1 par classe.\n"
    "  Traite fraude et légit. à égalité.\n\n"
    "▸ AUPRC\n"
    "  Aire sous la courbe Precision-Recall.\n"
    "  Insensible aux vrais négatifs (≫ ici).\n"
    "  Ref : Davis & Goadrich (2006), ICML.\n\n"
    "▸ MCC (Matthews Correlation Coefficient)\n"
    "  Seule métrique symétrique sur les 4\n"
    "  cases de la matrice de confusion.\n"
    "  Valeurs ∈ [-1,+1], robuste au ratio 578:1.\n"
    "  Ref : Chicco & Jurman (2020), BMC Genomics.\n\n"
    "▸ Brier Score\n"
    "  Mesure la qualité des probabilités.\n"
    "  Score parfait = 0."
)
axes[1].text(0.05, 0.95, summary_text, transform=axes[1].transAxes,
             fontsize=10, va="top",
             bbox=dict(boxstyle="round", facecolor="#ecf0f1", alpha=0.9))
axes[1].set_title("Justification des métriques choisies", fontsize=11, fontweight="bold")

plt.suptitle("Évaluation des probabilités – Brier Score & Métriques",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig(fig, "fig08_brier_score_comparison.png")
plt.show()

# Sauvegarde
calibrated = {
    "xgb_best":xgb_best,"xgb_platt":xgb_platt,"xgb_iso":xgb_iso,
    "best_threshold":best_t_mcc,"metrics_df":metrics_df,
    "y_eval":y_eval,"X_eval":X_eval,
    "y_proba_raw":y_proba_raw_eval,"y_proba_platt":y_proba_platt,"y_proba_iso":y_proba_iso,
}
with open(OUT_PKL, "wb") as f: pickle.dump(calibrated, f)
print(f"\n✓ Sauvegardé → {OUT_PKL}")
print("\n"+"─"*65)
print(f"ÉTAPE 3 TERMINÉE – 8 figures dans : {FIG_DIR}")
print("─"*65)
