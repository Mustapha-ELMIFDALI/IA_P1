"""
=============================================================================
MAIN – Exécution séquentielle du pipeline complet
=============================================================================
Usage : python main.py
Ou bien, exécuter chaque étape indépendamment depuis son dossier.
=============================================================================
"""
import runpy, time
from pathlib import Path

ROOT  = Path(__file__).resolve().parent
steps = [
    ("ÉTAPE 1 – EDA & Préparation",         ROOT/"etape1_eda"/"etape1_eda.py"),
    ("ÉTAPE 2 – Développement des modèles", ROOT/"etape2_models"/"etape2_models.py"),
    ("ÉTAPE 3 – Évaluation & Calibration",  ROOT/"etape3_evaluation"/"etape3_evaluation.py"),
    ("ÉTAPE 4 – Interprétabilité SHAP",     ROOT/"etape4_shap"/"etape4_shap.py"),
]
t0_total = time.time()
for i, (label, script) in enumerate(steps, 1):
    print("\n"+"█"*70)
    print(f"  [{i}/4] {label}")
    print("█"*70)
    t0 = time.time()
    runpy.run_path(str(script), run_name="__main__")
    print(f"\n  ✓ {label} – {time.time()-t0:.1f}s")
print(f"\n{'='*70}")
print(f"  PIPELINE COMPLET – {time.time()-t0_total:.1f}s")
print(f"  Figures par étape :")
for _, script in steps:
    fig_dir = script.parent / "figures"
    n = len(list(fig_dir.glob("*.png"))) if fig_dir.exists() else 0
    print(f"    {script.parent.name}/figures/  → {n} figure(s)")
print("="*70)
