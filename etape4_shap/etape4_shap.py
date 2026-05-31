"""
=============================================================================
ÉTAPE 4 : Interprétabilité – SHAP (SHapley Additive Explanations)
=============================================================================
Figures générées (etape4_shap/figures/) :
  fig01_shap_summary_dot.png
  fig02_shap_bar_importance.png
  fig03_shap_waterfall_fraud.png
  fig04_shap_waterfall_legit.png
  fig05_shap_dependence_top4.png
  fig06_shap_fraud_vs_legit.png
  fig07_shap_rf_summary.png
  fig08_shap_force_fraud.png
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
import pickle
import shap

ROOT      = Path(__file__).resolve().parent.parent
FIG_DIR   = Path(__file__).parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_PKL  = ROOT / "etape1_eda"        / "prepared_data.pkl"
MODEL_PKL = ROOT / "etape2_models"     / "trained_models.pkl"
CALIB_PKL = ROOT / "etape3_evaluation" / "calibrated_models.pkl"

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
plt.rcParams.update({"figure.dpi":130,"axes.spines.top":False,"axes.spines.right":False})
C = {"maj":"#4C72B0","min":"#DD8452","green":"#2ecc71","purple":"#9b59b6",
     "red":"#e74c3c","teal":"#1abc9c","dark":"#2c3e50"}

def save_fig(fig, name):
    p = FIG_DIR / name
    fig.savefig(p, bbox_inches="tight", dpi=130)
    print(f"  ✓ {name}")

with open(DATA_PKL,  "rb") as f: data   = pickle.load(f)
with open(MODEL_PKL, "rb") as f: models = pickle.load(f)
with open(CALIB_PKL, "rb") as f: calib  = pickle.load(f)

X_test   = data["X_test"]
y_test   = data["y_test"]
xgb_best = models["xgb_best"]
rf       = models["rf"]

print("="*65)
print("ÉTAPE 4 – Interprétabilité SHAP")
print("="*65)

N_SHAP   = 500
N_SHAP = min(N_SHAP, len(X_test))

fraud_pool = np.where(y_test.values == 1)[0]
legit_pool = np.where(y_test.values == 0)[0]

if len(fraud_pool) > 0 and len(legit_pool) > 0:
  n_fraud = min(len(fraud_pool), max(1, int(0.1 * N_SHAP)))
  n_legit = min(len(legit_pool), N_SHAP - n_fraud)

  idx_fraud = np.random.choice(fraud_pool, n_fraud, replace=False)
  idx_legit = np.random.choice(legit_pool, n_legit, replace=False)

  chosen = np.unique(np.concatenate([idx_fraud, idx_legit]))
  if len(chosen) < N_SHAP:
    remaining = np.setdiff1d(np.arange(len(X_test)), chosen, assume_unique=False)
    fill = np.random.choice(remaining, N_SHAP - len(chosen), replace=False)
    chosen = np.concatenate([chosen, fill])

  shap_idx = np.random.permutation(chosen)
else:
  shap_idx = np.random.choice(len(X_test), N_SHAP, replace=False)

X_shap   = X_test.iloc[shap_idx].reset_index(drop=True)
y_shap   = y_test.iloc[shap_idx].reset_index(drop=True)
print(f"Échantillon SHAP : {N_SHAP} pts | Fraudes : {y_shap.sum()} ({y_shap.mean():.1%})")

# ── TreeExplainer XGBoost ─────────────────────────────────────────────────────
explainer   = shap.TreeExplainer(xgb_best)
shap_vals   = explainer.shap_values(X_shap)
shap_exp    = explainer(X_shap)
print("✓ SHAP values XGBoost calculées.")

fraud_idx = np.where(y_shap.values == 1)[0]
legit_idx = np.where(y_shap.values == 0)[0]

# FIG 01 – Summary dot
fig, ax = plt.subplots(figsize=(11, 9))
shap.summary_plot(shap_vals, X_shap, plot_type="dot",
                  max_display=20, show=False, alpha=0.6)
plt.title("SHAP Summary Plot – XGBoost (Top 20 features)\n"
          "Chaque point = 1 obs. | Position X = impact sur P(fraude) | Couleur = valeur de la feature",
          fontsize=12, fontweight="bold")
plt.tight_layout()
save_fig(plt.gcf(), "fig01_shap_summary_dot.png")
plt.show()

# FIG 02 – Bar importance
fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(shap_vals, X_shap, plot_type="bar",
                  max_display=20, show=False, color=C["min"])
plt.title("SHAP – Importance globale (|SHAP| moyen)\n"
          "= contribution moyenne absolue de chaque feature à toutes les prédictions",
          fontsize=12, fontweight="bold")
plt.tight_layout()
save_fig(plt.gcf(), "fig02_shap_bar_importance.png")
plt.show()

# FIG 03 – Waterfall fraude
if len(fraud_idx) > 0:
    idx_f = fraud_idx[0]
    prob_f = xgb_best.predict_proba(X_shap.iloc[[idx_f]])[0, 1]
    fig, ax = plt.subplots(figsize=(11, 7))
    shap.plots.waterfall(shap_exp[idx_f], max_display=15, show=False)
    plt.title(f"SHAP Waterfall – Transaction FRAUDULEUSE (obs #{idx_f})\n"
              f"P(fraude) prédite = {prob_f:.4f}  |  Vraie classe = FRAUDE ✓",
              fontsize=11, fontweight="bold")
    plt.tight_layout()
    save_fig(plt.gcf(), "fig03_shap_waterfall_fraud.png")
    plt.show()

    top5_f = pd.Series(shap_vals[idx_f], index=X_shap.columns).abs().nlargest(5)
    print(f"\nTop 5 features pour fraude #{idx_f} :")
    for feat, sv_abs in top5_f.items():
        sv = shap_vals[idx_f][list(X_shap.columns).index(feat)]
        print(f"  {feat:8s} val={X_shap[feat].iloc[idx_f]:.3f} | "
              f"SHAP={sv:+.4f} ({'↑fraude' if sv>0 else '↓fraude'})")

# FIG 04 – Waterfall légitime
if len(legit_idx) > 0:
    idx_l = legit_idx[0]
    prob_l = xgb_best.predict_proba(X_shap.iloc[[idx_l]])[0, 1]
    fig, ax = plt.subplots(figsize=(11, 7))
    shap.plots.waterfall(shap_exp[idx_l], max_display=15, show=False)
    plt.title(f"SHAP Waterfall – Transaction LÉGITIME (obs #{idx_l})\n"
              f"P(fraude) prédite = {prob_l:.4f}  |  Vraie classe = LÉGIT. ✓",
              fontsize=11, fontweight="bold")
    plt.tight_layout()
    save_fig(plt.gcf(), "fig04_shap_waterfall_legit.png")
    plt.show()

# FIG 05 – Dependence plots Top 4
mean_abs = np.abs(shap_vals).mean(0)
top4_feat = X_shap.columns[np.argsort(mean_abs)[::-1][:4]].tolist()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for i, feat in enumerate(top4_feat):
    shap.dependence_plot(feat, shap_vals, X_shap,
                         interaction_index="auto", ax=axes[i], show=False, alpha=0.5)
    axes[i].set_title(f"SHAP Dependence : {feat}\n"
                      f"(|SHAP| moyen = {mean_abs[list(X_shap.columns).index(feat)]:.4f})",
                      fontsize=11, fontweight="bold")

plt.suptitle("SHAP Dependence Plots – Top 4 features\n"
             "Couleur = feature d'interaction automatique (corrélation la plus forte)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig(fig, "fig05_shap_dependence_top4.png")
plt.show()

# FIG 06 – Fraude vs Légit (grouped bar)
sv_fraud = shap_vals[y_shap==1]
sv_legit = shap_vals[y_shap==0]
m_fraud  = np.abs(sv_fraud).mean(0) if len(sv_fraud) > 0 else np.zeros(X_shap.shape[1])
m_legit  = np.abs(sv_legit).mean(0) if len(sv_legit) > 0 else np.zeros(X_shap.shape[1])

top15 = np.argsort(m_fraud + m_legit)[::-1][:15]
feat_names15 = X_shap.columns[top15]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

x = np.arange(len(feat_names15))
w = 0.38
b1 = axes[0].bar(x-w/2, m_fraud[top15], width=w, color=C["min"], alpha=0.85, label="Fraude")
b2 = axes[0].bar(x+w/2, m_legit[top15], width=w, color=C["maj"], alpha=0.85, label="Légitime")
axes[0].set_xticks(x)
axes[0].set_xticklabels(feat_names15, rotation=45, ha="right", fontsize=10)
axes[0].set_ylabel("|SHAP| moyen")
axes[0].set_title("Importance SHAP par classe\n(Fraude vs Légitime)", fontsize=11, fontweight="bold")
axes[0].legend(fontsize=11)

# Heatmap SHAP values (fraudes uniquement)
if len(sv_fraud) > 0:
    sv_hm = pd.DataFrame(sv_fraud[:, top15], columns=feat_names15)
    sns.heatmap(sv_hm.T, cmap="RdBu_r", center=0, ax=axes[1],
                cbar_kws={"label": "SHAP value"}, yticklabels=True, xticklabels=False)
    axes[1].set_xlabel("Transactions frauduleuses")
    axes[1].set_title("Heatmap SHAP – Top 15 features sur les fraudes\n"
                      "(rouge=↑fraude, bleu=↓fraude)", fontsize=11, fontweight="bold")

plt.suptitle("Analyse SHAP – Différences d'importance entre Fraude et Légitime",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig(fig, "fig06_shap_fraud_vs_legit.png")
plt.show()

# FIG 07 – SHAP RF
print("\n── SHAP Random Forest ──")
explainer_rf = shap.TreeExplainer(rf)
sv_rf = explainer_rf.shap_values(X_shap)
if isinstance(sv_rf, list): sv_rf = sv_rf[1]
print("✓ SHAP RF calculées.")

fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(sv_rf, X_shap, plot_type="dot", max_display=15, show=False, alpha=0.55)
plt.title("SHAP Summary Plot – Random Forest (Top 15)\n"
          "Comparaison avec XGBoost : convergence des features importantes",
          fontsize=12, fontweight="bold")
plt.tight_layout()
save_fig(plt.gcf(), "fig07_shap_rf_summary.png")
plt.show()

# FIG 08 – Force plot (fraude) → sauvegarde HTML + image de remplacement
if len(fraud_idx) > 0:
    idx_f2 = fraud_idx[min(1, len(fraud_idx)-1)]
    # Matplotlib version du force plot (compatible sans JS)
    fig, ax = plt.subplots(figsize=(14, 3))
    shap_row = shap_vals[idx_f2]
    feat_vals = X_shap.iloc[idx_f2]
    sv_sorted = pd.Series(shap_row, index=X_shap.columns).sort_values()
    colors_fp = [C["min"] if v > 0 else C["maj"] for v in sv_sorted]
    sv_sorted.plot.barh(ax=ax, color=colors_fp, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("SHAP value (impact sur log-odds de fraude)")
    ax.set_title(f"Force Plot (barchart) – Fraude #{idx_f2}\n"
                 f"P(fraude) = {xgb_best.predict_proba(X_shap.iloc[[idx_f2]])[0,1]:.4f}  "
                 f"| valeur de base E[f(x)] = {explainer.expected_value:.4f}",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "fig08_shap_force_fraud.png")
    plt.show()

# ── Synthèse SHAP ─────────────────────────────────────────────────────────────
print("\n" + "─"*65)
print("SYNTHÈSE SHAP – Top 5 features XGBoost")
top5_global = pd.Series(mean_abs, index=X_shap.columns).nlargest(5)
for feat, val in top5_global.items():
    print(f"  {feat:8s} : |SHAP| moyen = {val:.4f}")
print("""
Interprétation globale :
  → V14, V17, V12 sont systématiquement les features les plus discriminantes
    dans le dataset Credit Card (composantes PCA des transactions CB).
  → Une valeur très négative de V14 est le signal le plus fort de fraude.
  → Amount : les fraudes ont des montants atypiques (très bas ou inhabituels).
  → TreeExplainer garantit l'exactitude des SHAP values pour les modèles à arbres.
  Ref : Lundberg & Lee (2017), NeurIPS.
""")
print("─"*65)
print(f"ÉTAPE 4 TERMINÉE – 8 figures dans : {FIG_DIR}")
print("─"*65)
