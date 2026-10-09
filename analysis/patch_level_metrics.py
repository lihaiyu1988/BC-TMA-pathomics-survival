# -*- coding: utf-8 -*-
"""Table 2: patch-level classification of the 5-year overall-survival label by the four encoders. AUC with a
300-sample patch bootstrap 95% CI; accuracy, sensitivity, specificity, PPV and NPV at the Youden threshold of the
training patches, applied unchanged to the test patches (the patch-level CIs ignore clustering of patches within
patients, as stated in the Table 2 caption). Output -> analysis_outputs/Table2_patch_level_4backbones.csv"""
import os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
sys.stdout.reconfigure(encoding='utf-8')
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root
DATA = 'pipeline_outputs'; OUT = 'analysis_outputs'

alldl = pd.read_csv(f'{DATA}/results/ALL_DL_PREDICTIONS.csv'); gt = alldl.groupby('ID')['gt'].first()
rng = np.random.default_rng(42)
rows = []
for m, disp in [('resnet18', 'ResNet18'), ('resnet50', 'ResNet50'), ('densenet121', 'DenseNet121'), ('CrossFormer', 'CrossFormer')]:
    thr = None
    for c in ['train', 'test']:
        d = pd.read_csv(f'{DATA}/results/Pathomics_Slice_{m}_{c}.csv'); y = d['ID'].map(gt).values; s = d['label-1'].values
        auc = roc_auc_score(y, s)
        fpr, tpr, th = roc_curve(y, s)
        if c == 'train':
            thr = th[np.argmax(tpr - fpr)]                 # Youden threshold of the training patches, locked for the test patches
        pred = (s >= thr).astype(int)
        tp = ((pred == 1) & (y == 1)).sum(); tn = ((pred == 0) & (y == 0)).sum(); fp = ((pred == 1) & (y == 0)).sum(); fn = ((pred == 0) & (y == 1)).sum()
        v = []; n = len(y)
        for _ in range(300):                               # patch bootstrap of the AUC
            b = rng.choice(n, n, replace=True); v.append(roc_auc_score(y[b], s[b]))
        lo, hi = np.percentile(v, [2.5, 97.5])
        rows.append({'Backbone': disp, 'Cohort': c, 'n patches': n, 'Accuracy': round((tp + tn) / n, 3), 'AUC': round(auc, 3), '95% CI': f'{lo:.3f}-{hi:.3f}',
                     'Sensitivity': round(tp / (tp + fn), 3), 'Specificity': round(tn / (tn + fp), 3), 'PPV': round(tp / (tp + fp), 3), 'NPV': round(tn / (tn + fn), 3),
                     'threshold(Youden,train)': round(float(thr), 3)})
t = pd.DataFrame(rows); t.to_csv(f'{OUT}/Table2_patch_level_4backbones.csv', index=False); print(t.to_string())
