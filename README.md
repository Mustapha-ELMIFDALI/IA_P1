# Classification Robuste et Analyse de Décision en Environnement Critique
### Détection de Fraudes Bancaires par Apprentissage Automatique

> **Dataset :** Credit Card Fraud Detection (Kaggle) — 284 807 transactions | Ratio de déséquilibre : **578:1**
> **Auteur :** Mustapha EL MIFDALI, Hicham Ouaouche, ILYAS MOUSSNAOUI, ABOUBAKER ID HAMIDE  |  **Module :** Intelligence Artificielle  |  **2025-2026**

---

## Résultats en bref

| Modèle | F1-Macro | AUPRC | MCC | Brier Score |
|--------|----------|-------|-----|-------------|
| Régression Logistique Elastic Net | 0.503 | 0.708 | 0.137 | 0.142 |
| Random Forest (120 arbres) | 0.916 | 0.828 | 0.832 | 0.001 |
| **XGBoost + Optuna (50 trials)** | **0.933** | **0.877** | **0.866** | **0.0004** |
| XGBoost + Isotonic Calibration | **0.942** | — | **0.885** | **0.00034** |

---

## Structure du Projet

```
IA_P1/
├── data/
│   └── creditcard.csv              # Dataset (284 807 transactions)
│
├── etape1_eda/
│   ├── etape1_eda.py               # EDA, feature engineering, gestion déséquilibre
│   ├── prepared_data.pkl           # Données préparées (train/test splits + méthodes)
│   └── figures/                    # 12 figures (distribution, VIF, SMOTE, ADASYN...)
│
├── etape2_models/
│   ├── etape2_models.py            # LogReg, RF, XGBoost + Optuna
│   ├── trained_models.pkl          # Modèles entraînés
│   └── figures/                    # 11 figures (coefficients, proximité RF, Optuna...)
│
├── etape3_evaluation/
│   ├── etape3_evaluation.py        # Métriques avancées + calibration
│   ├── calibrated_models.pkl       # Modèles calibrés (Platt + Isotonic)
│   └── figures/                    # 8 figures (PR/ROC, reliability diagrams...)
│
├── etape4_shap/
│   ├── etape4_shap.py              # Interprétabilité SHAP (XGBoost + RF)
│   └── figures/                    # 8 figures (summary, waterfall, dependence...)
│
└── rapport_fraud_detection.docx    # Rapport professionnel complet (39 figures)
```

---

## Cahier des Charges — Conformité Complète

### Étape 1 : Analyse Exploratoire et Préparation ✅

- **Colinéarité** : Matrice de corrélation de Pearson + VIF (Variance Inflation Factor)
  - `Amount` : VIF = 16.80 (colinéarité forte avec ses transformées → justifie Elastic Net)
- **Feature Engineering** : 7 nouvelles variables (encodage cyclique temporel, interactions non-linéaires, transformations de montant)
- **Sélection** : Mutual Information sur 60 000 observations → top 30 features
- **Traitement du déséquilibre** — 4 méthodes comparées :
  - Niveau algorithmique : `class_weight='balanced'` (w_fraude = 289.1)
  - Niveau données : SMOTE, ADASYN, NearMiss v1

### Étape 2 : Développement des Modèles ✅

#### 1. Régression Logistique Elastic Net (Baseline)
- `solver='saga'`, `l1_ratio=0.5`, `C=0.1`, `class_weight='balanced'`
- Justification complète de chaque hyperparamètre

#### 2. Random Forest + Analyse de Proximité
- Matrice de proximité via `rf.apply()` (fréquence de co-feuille terminale)
- Score d'outlierness : `1 / Σ(P²)` → identification des zones ambiguës
- **Explication des outliers** : chevauchement des classes, features atypiques (V14, V12 en queue de distribution), déséquilibre local

#### 3. XGBoost Cost-Sensitive + Optimisation Bayésienne (Optuna)
- **Stratégie A** : `scale_pos_weight = 575.9` (ratio naturel)
- **Stratégie B** : Focal Loss personnalisée (γ=2, α=0.25)
- **Optuna TPE — 50 trials** — justification théorique de chaque borne :
  - `max_depth [3,10]` : évite underfitting (3) et overfitting (10)
  - `lambda, alpha [1e-3,10]` log-uniforme : 4 ordres de grandeur
  - `learning_rate [0.01,0.3]` log-uniforme : couplé à `n_estimators=200`
  - `scale_pos_weight [spw×0.5, spw×1.5]` : variation ±50% autour du ratio naturel
- **Graphiques de convergence Optuna** : Optimization History + HP Importance (Fanova)

### Étape 3 : Évaluation et Calibration ✅

- **Métriques** (Accuracy exclue) :
  - **F1-Macro** : traite les classes à égalité (Davis & Goadrich, 2006)
  - **AUPRC** : insensible aux vrais négatifs
  - **MCC** : unique métrique symétrique sur les 4 cases, robuste au ratio 578:1
  - **Brier Score** : qualité de calibration des probabilités
- **Calibration** :
  - Platt Scaling → Brier = 0.00038
  - **Isotonic Regression** → Brier = **0.00034** ✓ retenu
- Reliability Diagrams (10 bins) validés
- Analyse du seuil de décision optimal : **0.84**

### Étape 4 : Interprétabilité SHAP ✅

- `shap.TreeExplainer` sur XGBoost **et** Random Forest
- Summary plots, bar importance, waterfall plots (fraude + légitime)
- Dependence plots Top 4 features avec interaction automatique
- **Convergence XGBoost/RF** : V4, V14, V12 dans le top 3 des deux modèles
- Explications individuelles (force plot)

---

## Installation et Exécution

### Prérequis

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost optuna shap imbalanced-learn statsmodels
```

### Exécution séquentielle

```bash
# Étape 1 — EDA et préparation (~2 min)
python etape1_eda/etape1_eda.py

# Étape 2 — Modèles + Optuna 50 trials (~10 min)
python etape2_models/etape2_models.py

# Étape 3 — Évaluation et calibration (~1 min)
python etape3_evaluation/etape3_evaluation.py

# Étape 4 — SHAP (~1 min)
python etape4_shap/etape4_shap.py
```

> **Note Windows** : Les scripts configurent automatiquement `matplotlib.use("Agg")` et `sys.stdout.reconfigure(encoding="utf-8")` pour la compatibilité.

---

## Versions des Bibliothèques

| Bibliothèque | Version testée |
|---|---|
| Python | 3.11 |
| scikit-learn | 1.8.0 |
| xgboost | 3.2.0 |
| optuna | 4.8.0 |
| shap | 0.51.0 |
| imbalanced-learn | >= 0.12 |

---

## Principales Figures Générées (39 au total)

| Figure | Description |
|--------|-------------|
| `etape1_eda/figures/fig01_*` | Distribution des classes (déséquilibre 578:1) |
| `etape1_eda/figures/fig05_vif.png` | VIF par feature — analyse colinéarité |
| `etape1_eda/figures/fig11_*` | PCA 2D comparative des 4 méthodes de rééquilibrage |
| `etape2_models/figures/fig04_*` | Matrice de proximité RF + score d'outlierness |
| `etape2_models/figures/fig08_optuna_*` | Convergence Optuna (50 trials) |
| `etape3_evaluation/figures/fig04_*` | Reliability Diagrams (calibration) |
| `etape4_shap/figures/fig01_*` | SHAP Summary Plot — top 20 features |
| `etape4_shap/figures/fig03_*` | Waterfall SHAP — transaction frauduleuse |

---

## Références

1. Davis & Goadrich (2006) — Precision-Recall vs ROC curves, ICML
2. Chicco & Jurman (2020) — MCC over F1 score, BMC Genomics
3. Lundberg & Lee (2017) — SHAP unified approach, NeurIPS
4. Chen & Guestrin (2016) — XGBoost, KDD
5. Chawla et al. (2002) — SMOTE, JAIR
6. Akiba et al. (2019) — Optuna, KDD

---
