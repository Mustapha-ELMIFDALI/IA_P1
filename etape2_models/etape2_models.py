"""
=============================================================================
ÉTAPE 2 : Développement des Modèles
=============================================================================
Modèles  : 1) Logistic Regression (Elastic Net)
            2) Random Forest + Analyse de Proximité
            3) XGBoost Cost-Sensitive (scale_pos_weight + focal loss) + Optuna

Figures générées (etape2_models/figures/) :
  fig01_logreg_coefficients.png
  fig02_logreg_pr_curve.png
  fig03_rf_feature_importance.png
  fig04_rf_proximity_matrix.png
  fig05_rf_outliers_pca.png
  fig06_outlier_profiles.png
  fig07_cost_sensitive_comparison.png   ← scale_pos_weight vs focal loss
  fig08_optuna_history.png
  fig09_optuna_hp_importance.png
  fig10_confusion_matrices.png
  fig11_learning_curves.png
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.base import clone
from sklearn.metrics import (
    classification_report, matthews_corrcoef,
    f1_score, precision_recall_curve, auc,
    confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold, learning_curve, train_test_split as tts
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from matplotlib.patches import Patch
import pickle

# ─── Chemins ─────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
FIG_DIR   = Path(__file__).parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_PKL  = ROOT / "etape1_eda" / "prepared_data.pkl"
OUT_PKL   = Path(__file__).parent / "trained_models.pkl"

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False,
                     "axes.spines.right": False})
C = {"maj":"#4C72B0","min":"#DD8452","green":"#2ecc71","purple":"#9b59b6",
     "red":"#e74c3c","orange":"#f39c12","teal":"#1abc9c","dark":"#2c3e50"}

def save_fig(fig, name):
    p = FIG_DIR / name
    fig.savefig(p, bbox_inches="tight", dpi=130)
    print(f"  ✓ {name}")

# ─── Chargement ──────────────────────────────────────────────────────────────
with open(DATA_PKL, "rb") as f: data = pickle.load(f)
X_train = data["X_train"]; X_test  = data["X_test"]
y_train = data["y_train"]; y_test  = data["y_test"]
cw_dict = data["class_weight"]

train_n = min(120000, len(X_train))
if train_n < len(X_train):
    X_train_model, _, y_train_model, _ = tts(
        X_train, y_train, train_size=train_n,
        stratify=y_train, random_state=RANDOM_STATE
    )
else:
    X_train_model, y_train_model = X_train, y_train

print("=" * 65)
print("ÉTAPE 2 – Développement des modèles")
print("=" * 65)
print(f"Train utilisé pour l'entraînement des modèles : {len(X_train_model):,} observations")

# ══════════════════════════════════════════════════════════════════════════════
# MODÈLE 1 : RÉGRESSION LOGISTIQUE ELASTIC NET
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Modèle 1 : Logistic Regression Elastic Net ──")
logreg = LogisticRegression(
    penalty="elasticnet", solver="saga", l1_ratio=0.5,
    C=0.1, class_weight="balanced", max_iter=300, tol=1e-3,
    random_state=RANDOM_STATE, n_jobs=1,
)
lr_sample_n = min(80000, len(X_train_model))
if lr_sample_n < len(X_train_model):
    X_lr, _, y_lr, _ = tts(
        X_train_model, y_train_model, train_size=lr_sample_n,
        stratify=y_train_model, random_state=RANDOM_STATE
    )
else:
    X_lr, y_lr = X_train_model, y_train_model

logreg.fit(X_lr, y_lr)
y_pred_lr  = logreg.predict(X_test)
y_proba_lr = logreg.predict_proba(X_test)[:, 1]
print(f"✓ LogReg entraîné (subset stratifié: {len(X_lr):,} obs).")
print(classification_report(y_test, y_pred_lr, target_names=["Légit.", "Fraude"]))

# FIG 01 – Coefficients
coefs = pd.Series(logreg.coef_[0], index=X_train.columns).sort_values(key=abs, ascending=False)
top20 = coefs.head(20).sort_values()

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
colors_c = [C["min"] if v > 0 else C["maj"] for v in top20]
axes[0].barh(top20.index, top20.values, color=colors_c)
axes[0].axvline(0, color="black", lw=0.8)
axes[0].set_xlabel("Coefficient (Elastic Net)")
axes[0].set_title("Top 20 coefficients\n(rouge=↑ risque fraude, bleu=↓ risque fraude)",
                   fontsize=11, fontweight="bold")
for i, val in enumerate(top20.values):
    axes[0].text(val+(0.008 if val>=0 else -0.008), i,
                 f"{val:.3f}", va="center",
                 ha="left" if val>=0 else "right", fontsize=8)

# L1 vs L2 proportion (l1_ratio=0.5)
axes[1].axis("off")
info_text = (
    "Hyperparamètres retenus :\n\n"
    "• solver = 'saga'\n"
    "  Seul solver supportant Elastic Net\n"
    "  sur grands datasets (>100K obs)\n\n"
    "• penalty = 'elasticnet'\n"
    "  L1 + L2 : sélection + stabilité\n\n"
    "• l1_ratio = 0.5\n"
    "  Équilibre L1/L2 (50% chacun)\n\n"
    "• C = 0.1\n"
    "  Régularisation modérée\n"
    "  (C=1/λ, faible C = forte régul.)\n\n"
    "• class_weight = 'balanced'\n"
    "  w(fraude) ≈ 290×  w(légit.)"
)
axes[1].text(0.1, 0.9, info_text, transform=axes[1].transAxes,
             fontsize=10, va="top",
             bbox=dict(boxstyle="round", facecolor="#ecf0f1", alpha=0.9))
axes[1].set_title("Justification des hyperparamètres", fontsize=11, fontweight="bold")

plt.suptitle("Régression Logistique – Elastic Net",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig(fig, "fig01_logreg_coefficients.png")
plt.show()

# FIG 02 – PR curve LogReg
prec_lr, rec_lr, _ = precision_recall_curve(y_test, y_proba_lr)
auprc_lr = auc(rec_lr, prec_lr)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(rec_lr, prec_lr, color=C["maj"], lw=2.5,
        label=f"LogReg Elastic Net (AUPRC={auprc_lr:.4f})")
ax.axhline(y_test.mean(), color="gray", ls=":", label=f"Baseline ({y_test.mean():.4f})")
ax.fill_between(rec_lr, prec_lr, alpha=0.1, color=C["maj"])
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Courbe Precision-Recall – Logistic Regression")
ax.legend()
plt.tight_layout()
save_fig(fig, "fig02_logreg_pr_curve.png")
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# MODÈLE 2 : RANDOM FOREST + PROXIMITÉ
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Modèle 2 : Random Forest ──")
rf = RandomForestClassifier(
    n_estimators=120, max_depth=10, min_samples_leaf=6,
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE, n_jobs=-1,
)
rf.fit(X_train_model, y_train_model)
y_pred_rf  = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]
print("✓ RF entraîné.")
print(classification_report(y_test, y_pred_rf, target_names=["Légit.", "Fraude"]))

# FIG 03 – Feature importance RF
imp = pd.Series(rf.feature_importances_, index=X_train.columns)
imp = imp.sort_values(ascending=False).head(20)

fig, ax = plt.subplots(figsize=(10, 7))
palette_imp = sns.color_palette("viridis", 20)
ax.barh(imp.index[::-1], imp.values[::-1], color=palette_imp)
ax.set_xlabel("Importance (Mean Decrease Impurity)")
ax.set_title("Random Forest – Importance des features (Top 20)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
save_fig(fig, "fig03_rf_feature_importance.png")
plt.show()

# ── Matrice de proximité ──────────────────────────────────────────────────────
def proximity_matrix(model, X_arr, n=2000, batch=500):
    idx  = np.random.choice(len(X_arr), min(n, len(X_arr)), replace=False)
    Xs   = X_arr[idx]
    lv   = model.apply(Xs)          # (n, n_trees)
    n_t  = lv.shape[1]
    P    = np.zeros((len(idx), len(idx)), dtype=np.float32)
    for s in range(0, len(idx), batch):
        e = min(s+batch, len(idx))
        P[s:e] += (lv[s:e, np.newaxis, :] == lv[np.newaxis, :, :]).sum(2).astype(np.float32)
    P /= n_t
    return P, idx

def outlierness(P):
    return 1.0 / (np.sum(P**2, axis=1) + 1e-8)

print("\nCalcul de la matrice de proximité…")
X_test_arr   = X_test.values
P, s_idx     = proximity_matrix(rf, X_test_arr, n=2000)
out_sc       = outlierness(P)
y_sub        = y_test.iloc[s_idx].values
pred_sub     = rf.predict(X_test_arr[s_idx])
thresh       = np.percentile(out_sc, 95)
is_out       = out_sc > thresh
print(f"Outliers (top 5%) : {is_out.sum()}")

# FIG 04 – Heatmap proximité + distribution outlierness
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Heatmap 200×200
n200 = 200
sub200 = np.random.choice(len(P), n200, replace=False)
P200  = P[np.ix_(sub200, sub200)]
im = axes[0].imshow(P200, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
plt.colorbar(im, ax=axes[0], shrink=0.8, label="Proximité [0-1]")
axes[0].set_title("Matrice de proximité RF (200×200)\n"
                  "Valeur = fréquence de co-feuille terminale",
                  fontsize=11, fontweight="bold")
axes[0].set_xlabel("Observation"); axes[0].set_ylabel("Observation")

# Distribution outlierness
for cls, col, lbl in [(0, C["maj"],"Légitime"),(1, C["min"],"Fraude")]:
    axes[1].hist(out_sc[y_sub==cls], bins=40, color=col, alpha=0.65, density=True, label=lbl)
axes[1].axvline(thresh, color=C["red"], ls="--", lw=2, label=f"Seuil P95={thresh:.2f}")
axes[1].set_xlabel("Score d'outlierness")
axes[1].set_ylabel("Densité")
axes[1].set_title("Distribution du score d'outlierness par classe\n"
                  "Score élevé = point isolé dans l'espace de décision",
                  fontsize=11, fontweight="bold")
axes[1].legend()

plt.suptitle("Random Forest – Matrice de Proximité & Score d'Outlierness",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig(fig, "fig04_rf_proximity_matrix.png")
plt.show()

# FIG 05 – PCA 2D outliers
pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca2.fit_transform(X_test_arr[s_idx])
correct = (y_sub == pred_sub)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

ax = axes[0]
ax.scatter(X_pca[~is_out,0], X_pca[~is_out,1], c=C["maj"], s=6, alpha=0.4, label="Normal")
ax.scatter(X_pca[is_out,0],  X_pca[is_out,1],  c=C["red"], s=40, alpha=0.9,
           marker="X", label=f"Outlier de prédiction (N={is_out.sum()})")
ax.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]:.1%})")
ax.set_title("PCA 2D – Outliers de prédiction")
ax.legend(markerscale=1.5)

ax2 = axes[1]
for cls, col, mk, lbl in [(0,C["maj"],"o","Légitime"),(1,C["min"],"^","Fraude")]:
    m = y_sub == cls
    ax2.scatter(X_pca[m,0], X_pca[m,1], c=col, s=8, alpha=0.5, marker=mk, label=lbl)
ax2.scatter(X_pca[~correct,0], X_pca[~correct,1],
            edgecolors=C["red"], facecolors="none",
            s=60, lw=1.5, label=f"Erreurs ({(~correct).sum()})", zorder=5)
ax2.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]:.1%})")
ax2.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]:.1%})")
ax2.set_title("PCA 2D – Vraies classes + erreurs RF")
ax2.legend(markerscale=1.5)

plt.suptitle("Outliers de Prédiction – Random Forest\n"
             "Points à fort outlierness = zones ambiguës de la frontière de décision",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig(fig, "fig05_rf_outliers_pca.png")
plt.show()

# FIG 06 – Profil features des outliers
X_sub_df = X_test.iloc[s_idx].copy()
X_sub_df["is_outlier"] = is_out
X_sub_df["true_class"] = y_sub

top_feats = ["V14","V17","V12","V10","V16","Amount"]
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
for i, feat in enumerate(top_feats):
    ax = axes[i]
    for grp, col, lbl in [(False, C["maj"],"Non-outlier"), (True, C["red"],"Outlier")]:
        v = X_sub_df[X_sub_df["is_outlier"]==grp][feat]
        ax.hist(v, bins=30, color=col, alpha=0.6, density=True, label=lbl)
    ax.set_title(feat, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)

plt.suptitle("Profil des features – Outliers vs Non-outliers de prédiction\n"
             "(les outliers occupent des zones atypiques de l'espace de features)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
save_fig(fig, "fig06_outlier_profiles.png")
plt.show()

# ── Analyse textuelle des outliers de prédiction ─────────────────────────────
print("\n── Analyse des outliers de prédiction RF ──")
out_mask     = is_out
in_mask      = ~is_out
n_out_fraud  = ((y_sub == 1) & out_mask).sum()
n_out_legit  = ((y_sub == 0) & out_mask).sum()
n_err_out    = ((pred_sub != y_sub) & out_mask).sum()
n_err_in     = ((pred_sub != y_sub) & in_mask).sum()
err_rate_out = n_err_out / max(out_mask.sum(), 1)
err_rate_in  = n_err_in  / max(in_mask.sum(), 1)

print(f"""
Outliers (top 5% outlierness) : {out_mask.sum()} observations
  • Fraudes  parmi outliers : {n_out_fraud} ({n_out_fraud/max(out_mask.sum(),1):.1%})
  • Légitimes parmi outliers : {n_out_legit} ({n_out_legit/max(out_mask.sum(),1):.1%})
  • Taux d'erreur RF – outliers : {err_rate_out:.1%}
  • Taux d'erreur RF – non-outliers : {err_rate_in:.1%}

Pourquoi le RF hésite sur ces points ?
  1. ZONE DE CHEVAUCHEMENT : les outliers se situent à la frontière des deux classes
     dans l'espace PCA. Le RF agrège les votes des 120 arbres, mais dans ces zones
     le vote est quasi-partagé (ex: 55% légitime / 45% fraude).
  2. FEATURES ATYPIQUES : les profils montrent que les outliers ont des valeurs
     de V14, V12 et V17 hors de leur distribution habituelle (queue de distribution).
     Ces valeurs rares apparaissent peu dans l'entraînement → les arbres n'ont
     pas assez d'exemples pour construire des frontières fiables.
  3. DÉSÉQUILIBRE LOCAL : même avec class_weight='balanced_subsample', les nœuds
     proches de la frontière peuvent être peu peuplés en fraudes, réduisant la
     confiance du modèle.
  → Conclusion : Ces outliers correspondent aux transactions "ambiguës" qui
     ressemblent à la fois aux fraudes et aux légitimes, ou qui présentent des
     combinaisons de features rares jamais vues ensemble en entraînement.
""")
# ─────────────────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════════
# MODÈLE 3 : XGBOOST COST-SENSITIVE + OPTUNA
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Modèle 3 : XGBoost Cost-Sensitive + Optuna ──")
neg_count = (y_train_model == 0).sum()
pos_count = (y_train_model == 1).sum()
spw_base  = neg_count / pos_count
print(f"scale_pos_weight naturel : {spw_base:.1f}")

# ── Stratégie A : scale_pos_weight ──────────────────────────────────────────
xgb_spw = xgb.XGBClassifier(
    scale_pos_weight=spw_base, n_estimators=150,
    max_depth=6, learning_rate=0.1,
    random_state=RANDOM_STATE,
    tree_method="hist", n_jobs=-1, eval_metric="aucpr",
)
xgb_spw.fit(X_train_model, y_train_model, verbose=False)
y_pred_spw  = xgb_spw.predict(X_test)
y_proba_spw = xgb_spw.predict_proba(X_test)[:, 1]
mcc_spw = matthews_corrcoef(y_test, y_pred_spw)
print(f"[A] scale_pos_weight → MCC={mcc_spw:.4f}")

# ── Stratégie B : Focal Loss ─────────────────────────────────────────────────
def focal_loss(y_pred, dtrain, gamma=2.0, alpha=0.25):
    y_true = dtrain.get_label()
    p = 1.0 / (1.0 + np.exp(-y_pred))
    p_t = np.where(y_true==1, p, 1-p)
    a_t = np.where(y_true==1, alpha, 1-alpha)
    grad = -a_t*(1-p_t)**gamma*(gamma*p_t*np.log(p_t+1e-7)+p_t-1)
    hess = np.abs(a_t*(1-p_t)**gamma*(
        2*gamma*(1-p_t)*np.log(p_t+1e-7)+gamma*(gamma-1)*p_t+2*(1-p_t)+p_t-1
    )) + 1e-6
    return grad, hess

xgb_fl = xgb.XGBClassifier(
    n_estimators=150, max_depth=6, learning_rate=0.1,
    random_state=RANDOM_STATE,
    tree_method="hist", n_jobs=-1, eval_metric="aucpr",
)
dtrain_fl, X_val_fl, ytrain_fl, yval_fl = tts(
    X_train_model, y_train_model, test_size=0.20,
    stratify=y_train_model, random_state=RANDOM_STATE
)
dtrain = xgb.DMatrix(dtrain_fl, label=ytrain_fl)
dval   = xgb.DMatrix(X_val_fl, label=yval_fl)
dtest  = xgb.DMatrix(X_test,  label=y_test)
params_fl = dict(max_depth=6, learning_rate=0.1, eval_metric="aucpr",
                 tree_method="hist", seed=RANDOM_STATE, verbosity=0)
booster_fl = xgb.train(params_fl, dtrain, num_boost_round=150, obj=focal_loss)

# Seuil optimisé sur validation pour éviter l'effondrement à 0.5
y_proba_val_fl = booster_fl.predict(dval)
th_grid = np.linspace(0.02, 0.98, 97)
mcc_grid = [matthews_corrcoef(yval_fl, (y_proba_val_fl >= t).astype(int)) for t in th_grid]
best_th_fl = th_grid[int(np.argmax(mcc_grid))]

y_proba_fl = booster_fl.predict(dtest)
y_pred_fl  = (y_proba_fl >= best_th_fl).astype(int)
mcc_fl = matthews_corrcoef(y_test, y_pred_fl)
print(f"[B] Focal Loss        → MCC={mcc_fl:.4f} (seuil opt={best_th_fl:.2f})")

# FIG 07 – Comparaison scale_pos_weight vs focal loss
prec_spw, rec_spw, _ = precision_recall_curve(y_test, y_proba_spw)
prec_fl,  rec_fl,  _ = precision_recall_curve(y_test, y_proba_fl)
auprc_spw = auc(rec_spw, prec_spw)
auprc_fl  = auc(rec_fl,  prec_fl)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# PR curves
ax = axes[0]
ax.plot(rec_spw, prec_spw, color=C["maj"], lw=2.5,
        label=f"scale_pos_weight (AUPRC={auprc_spw:.4f})")
ax.plot(rec_fl, prec_fl, color=C["orange"], lw=2.5, ls="--",
        label=f"Focal Loss (AUPRC={auprc_fl:.4f})")
ax.axhline(y_test.mean(), color="gray", ls=":", label="Baseline")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall\nscale_pos_weight vs Focal Loss")
ax.legend(fontsize=9)

# Confusion matrices côte à côte
for ax_i, (name_m, y_pred_m, col_m) in zip([axes[1], axes[2]], [
    ("scale_pos_weight", y_pred_spw, C["maj"]),
    ("Focal Loss",       y_pred_fl,  C["orange"]),
]):
    cm   = confusion_matrix(y_test, y_pred_m)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Légit.", "Fraude"])
    disp.plot(ax=ax_i, cmap="Blues", colorbar=False)
    mcc_m = matthews_corrcoef(y_test, y_pred_m)
    f1_m  = f1_score(y_test, y_pred_m, average="macro")
    ax_i.set_title(f"Matrice confusion – {name_m}\n"
                   f"MCC={mcc_m:.3f} | F1-macro={f1_m:.3f}",
                   fontsize=10, fontweight="bold")

plt.suptitle("XGBoost Cost-Sensitive : Comparaison des deux stratégies\n"
             "Stratégie A : scale_pos_weight  |  Stratégie B : Focal Loss personnalisée",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig(fig, "fig07_cost_sensitive_comparison.png")
plt.show()

# ── Optuna (TPE, 10 trials) ───────────────────────────────────────────────────
def optuna_obj(trial):
    params = {
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "lambda":           trial.suggest_float("lambda", 1e-3, 10.0, log=True),
        "alpha":            trial.suggest_float("alpha",  1e-3, 10.0, log=True),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight",
                                                  spw_base*0.5, spw_base*1.5),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "n_estimators": 200,
        "random_state": RANDOM_STATE, "tree_method": "hist", "n_jobs": -1,
    }
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for fold, (ti, vi) in enumerate(skf.split(X_train_model, y_train_model)):
        m = xgb.XGBClassifier(**params)
        m.fit(X_train_model.iloc[ti], y_train_model.iloc[ti],
              eval_set=[(X_train_model.iloc[vi], y_train_model.iloc[vi])], verbose=False)
        scores.append(matthews_corrcoef(y_train_model.iloc[vi], m.predict(X_train_model.iloc[vi])))
        trial.report(np.mean(scores), step=fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return np.mean(scores)

# ── Justifications théoriques de l'espace de recherche Optuna ────────────────
print("\n── Justification de l'espace de recherche Optuna ──")
print("""
  max_depth [3, 10] :
    • min=3 : arbres trop superficiels → underfitting sur données complexes PCA.
    • max=10 : au-delà, risque d'overfitting sur ~100K obs avec 30+ features.
    • Zone optimale typique pour XGBoost sur données tabulaires : 4-8.

  lambda (L2) et alpha (L1) [1e-3, 10], log-uniforme :
    • Régularisation essentielle face au déséquilibre 577:1.
    • Échelle logarithmique : plage de 4 ordres de grandeur
      pour explorer aussi bien les petites (0.001) que grandes (10) valeurs.
    • lambda=L2 stabilise les gradients ; alpha=L1 induit la sparsité.

  learning_rate [0.01, 0.3], log-uniforme :
    • 0.01 : convergence lente mais régulière (besoin de +rounds).
    • 0.3 : apprentissage rapide mais risque d'oscillations.
    • Couplé à n_estimators=200 : permet une exploration équilibrée.

  scale_pos_weight [spw×0.5, spw×1.5] :
    • Centré sur le ratio naturel (~577) ± 50%.
    • Variation modérée : le ratio naturel est déjà un point de départ solide.

  subsample et colsample_bytree [0.6, 1.0] :
    • 0.6 : assure suffisamment de diversité pour réduire la variance.
    • 1.0 : utilise toutes les données/features (sans sous-échantillonnage).
    • En-dessous de 0.5 : risque de perte d'information sur la classe minoritaire.
""")

print("\nOptuna (50 trials TPE)…")
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(seed=RANDOM_STATE),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
)
study.optimize(optuna_obj, n_trials=50, show_progress_bar=True)
print(f"✓ Meilleur MCC Optuna : {study.best_value:.4f}")
print(f"  Params : {study.best_params}")

# FIG 08 – Convergence Optuna
completed = [t for t in study.trials if t.state.name == "COMPLETE"]
t_nums = [t.number for t in completed]
t_vals = [t.value  for t in completed]
best_sf = pd.Series(t_vals).cummax().tolist()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
ax.scatter(t_nums, t_vals, color=C["maj"], s=70, alpha=0.85, zorder=3,
           label="MCC par trial")
ax.plot(t_nums, best_sf, color=C["red"], lw=2.5, label="Meilleur cumulatif")
ax.fill_between(t_nums, t_vals, best_sf, alpha=0.1, color=C["min"])
ax.set_xlabel("Trial"); ax.set_ylabel("MCC (3-fold CV)")
ax.set_title("Optimization History\n(convergence du MCC)")
ax.legend()

ax = axes[1]
ax.hist(t_vals, bins=8, color=C["teal"], edgecolor="white", alpha=0.85)
ax.axvline(study.best_value, color=C["red"], ls="--", lw=2,
           label=f"Best = {study.best_value:.4f}")
ax.set_xlabel("MCC"); ax.set_ylabel("Nombre de trials")
ax.set_title("Distribution des valeurs objectif")
ax.legend()

# Parallel coordinate (hyperparams vs MCC)
ax = axes[2]
trials_df = pd.DataFrame([
    {**t.params, "MCC": t.value}
    for t in completed if t.value is not None
])
if len(trials_df) > 0:
    from pandas.plotting import parallel_coordinates
    cols_show = ["max_depth", "learning_rate", "MCC"]
    cols_show = [c for c in cols_show if c in trials_df.columns]
    trials_norm = trials_df[cols_show].copy()
    for col in cols_show:
        rng = trials_norm[col].max() - trials_norm[col].min()
        if rng > 0:
            trials_norm[col] = (trials_norm[col] - trials_norm[col].min()) / rng
    trials_norm["_label"] = (trials_df["MCC"] > trials_df["MCC"].median()).astype(str)
    parallel_coordinates(trials_norm[cols_show + ["_label"]], "_label",
                         color=[C["maj"], C["min"]], alpha=0.7, ax=ax)
    ax.set_title("Coordonnées parallèles\n(hyperparams vs MCC)")
    ax.legend(["MCC ≤ médiane", "MCC > médiane"], fontsize=9)

plt.suptitle("Optuna – Recherche Bayésienne TPE (50 trials)\n"
             "Maximisation du MCC via validation croisée stratifiée 3-fold",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig(fig, "fig08_optuna_history.png")
plt.show()

# FIG 09 – Importance des hyperparamètres
try:
    hp_imp = optuna.importance.get_param_importances(study)
    fig, ax = plt.subplots(figsize=(9, 5))
    names_h = list(hp_imp.keys())[::-1]
    vals_h  = list(hp_imp.values())[::-1]
    pal_h   = sns.color_palette("magma", len(names_h))
    bars_h  = ax.barh(names_h, vals_h, color=pal_h)
    for bar, v in zip(bars_h, vals_h):
        ax.text(bar.get_width()+0.004, bar.get_y()+bar.get_height()/2,
                f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("Importance relative (Fanova)")
    ax.set_title("Optuna – Importance des hyperparamètres\n"
                 "Quel hyperparamètre influence le plus le MCC ?",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "fig09_optuna_hp_importance.png")
    plt.show()
except Exception as e:
    print(f"  (HP importance : {e})")

# ── XGBoost final (meilleurs params) ─────────────────────────────────────────
best_p = study.best_params.copy()
best_p.update({"n_estimators":300,
               "random_state":RANDOM_STATE,"tree_method":"hist","n_jobs":-1})
xgb_best = xgb.XGBClassifier(**best_p)
xgb_best.fit(X_train_model, y_train_model, verbose=False)
y_pred_xgb  = xgb_best.predict(X_test)
y_proba_xgb = xgb_best.predict_proba(X_test)[:, 1]
print("\n✓ XGBoost final entraîné.")
print(classification_report(y_test, y_pred_xgb, target_names=["Légit.", "Fraude"]))

# ── Validation croisée (rigueur) ─────────────────────────────────────────────
cv_n = min(30000, len(X_train_model))
if cv_n < len(X_train_model):
    X_cv, _, y_cv, _ = tts(
        X_train_model, y_train_model, train_size=cv_n,
        stratify=y_train_model, random_state=RANDOM_STATE
    )
else:
    X_cv, y_cv = X_train_model, y_train_model

cv_models = [
    ("LogReg Elastic Net", clone(logreg)),
    ("Random Forest", clone(rf)),
    ("XGBoost (Optuna)", clone(xgb_best)),
]
skf_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
cv_rows = []

print(f"\nValidation croisée stratifiée (3-fold) sur {len(X_cv):,} observations…")
for model_name, model_obj in cv_models:
    f1_fold, mcc_fold = [], []
    for tr_idx, va_idx in skf_cv.split(X_cv, y_cv):
        X_tr, X_va = X_cv.iloc[tr_idx], X_cv.iloc[va_idx]
        y_tr, y_va = y_cv.iloc[tr_idx], y_cv.iloc[va_idx]
        model_obj.fit(X_tr, y_tr)
        y_hat = model_obj.predict(X_va)
        f1_fold.append(f1_score(y_va, y_hat, average="macro", zero_division=0))
        mcc_fold.append(matthews_corrcoef(y_va, y_hat))
    cv_rows.append({
        "name": model_name,
        "cv_f1_macro_mean": float(np.mean(f1_fold)),
        "cv_f1_macro_std": float(np.std(f1_fold)),
        "cv_mcc_mean": float(np.mean(mcc_fold)),
        "cv_mcc_std": float(np.std(mcc_fold)),
    })

cv_df = pd.DataFrame(cv_rows)
print(cv_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# FIG 10 – Matrices de confusion (3 modèles)
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, (name_m, y_pred_m) in zip(axes, [
    ("LogReg Elastic Net", y_pred_lr),
    ("Random Forest",      y_pred_rf),
    ("XGBoost (Optuna)",   y_pred_xgb),
]):
    cm   = confusion_matrix(y_test, y_pred_m)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Légit.", "Fraude"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    mcc_m = matthews_corrcoef(y_test, y_pred_m)
    f1_m  = f1_score(y_test, y_pred_m, average="macro")
    ax.set_title(f"{name_m}\nMCC={mcc_m:.3f} | F1-macro={f1_m:.3f}",
                 fontsize=10, fontweight="bold")

plt.suptitle("Matrices de Confusion – Comparaison des 3 modèles",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig(fig, "fig10_confusion_matrices.png")
plt.show()

# FIG 11 – Learning curves XGBoost
print("\nCalcul des learning curves (peut prendre quelques minutes)…")
train_sizes, train_sc, val_sc = learning_curve(
    xgb_best, X_train_model, y_train_model,
    cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE),
    scoring="f1_macro", n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 6),
)
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(train_sizes, train_sc.mean(1), "o-", color=C["maj"], lw=2.5, label="Score train")
ax.plot(train_sizes, val_sc.mean(1),   "s--", color=C["min"], lw=2.5, label="Score val.")
ax.fill_between(train_sizes,
                train_sc.mean(1)-train_sc.std(1),
                train_sc.mean(1)+train_sc.std(1), alpha=0.12, color=C["maj"])
ax.fill_between(train_sizes,
                val_sc.mean(1)-val_sc.std(1),
                val_sc.mean(1)+val_sc.std(1), alpha=0.12, color=C["min"])
ax.set_xlabel("Taille du train set"); ax.set_ylabel("F1-Macro")
ax.set_title("Learning Curves – XGBoost (meilleurs hyperparamètres)\n"
             "Analyse de la généralisation selon la taille du dataset",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=11)
plt.tight_layout()
save_fig(fig, "fig11_learning_curves.png")
plt.show()

# ── Sauvegarde ────────────────────────────────────────────────────────────────
models = {
    "logreg":      logreg,
    "rf":          rf,
    "xgb_best":    xgb_best,
    "xgb_spw":     xgb_spw,
    "y_pred_lr":   y_pred_lr,   "y_proba_lr":  y_proba_lr,
    "y_pred_rf":   y_pred_rf,   "y_proba_rf":  y_proba_rf,
    "y_pred_xgb":  y_pred_xgb,  "y_proba_xgb": y_proba_xgb,
    "y_pred_spw":  y_pred_spw,  "y_proba_spw": y_proba_spw,
    "y_proba_fl":  y_proba_fl,
    "best_threshold_fl": best_th_fl,
    "study":       study,
    "spw_base":    spw_base,
    "cv_results":  cv_df,
    "sample_idx":  s_idx,
    "out_scores":  out_sc,
}
with open(OUT_PKL, "wb") as f:
    pickle.dump(models, f)

print(f"\n✓ Modèles sauvegardés → {OUT_PKL}")
print("\n" + "─"*65)
print(f"ÉTAPE 2 TERMINÉE – 11 figures dans : {FIG_DIR}")
print("─"*65)
