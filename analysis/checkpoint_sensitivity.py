# -*- coding: utf-8 -*-
"""Sensitivity of the deployed signature to the ResNet50 checkpoint (Supplementary Table S11).
During encoder training, the framework computed the patch-level AUC of its validation set (the hold-out block of the
split: the 97 hold-out patients and the one training patient without a defined label) after every epoch; the deployed
ResNet50 checkpoint (epoch index 1, i.e. the second of 12 epochs) is the epoch with the highest validation AUC.
For each of the 12 saved checkpoints (pipeline_outputs/results/resnet50_checkpoints.csv.gz), this script reports
  - patch-level AUCs: training patches (193 patients), validation set as monitored during training (98 patients;
    patients without a defined label carry the placeholder gt = 1, as in the training framework), and the 81 labelled
    test-cohort patients;
  - the deployed three-feature signature (BoW_prob(0.50), BoW_prob(0.58), BoW_pred(1)) refitted on the training cohort
    (features standardized with the training mean and SD, unpenalized Cox): C-index in the training and test cohorts;
  - the training-only selection pipeline (pathomics_pipeline.run_pipeline) rerun at matched complexity (k = 3):
    selected features and C-indices.
Patch probabilities are taken exactly as in the deployed pipeline, which wrote the probability of the poor-prognosis
class to results/ALL_DL_PREDICTIONS.csv and read it back with the default pandas CSV parser before rounding to two
decimals (the parser can move a value by one unit in the last place, which decides the rounding of exact ties such as
0.145); with this, the deployed checkpoint reproduces the deployed features and the deployed test C-index exactly
(checked below)."""
import io, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pathomics_pipeline as pp                                       # changes to the repository root
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score
DATA, OUT = 'pipeline_outputs', 'analysis_outputs'
DEPLOYED = 1                                                          # 0-based epoch index of the deployed checkpoint
SIG = ['BoW_prob_0.50', 'BoW_prob_0.58', 'BoW_pred_1']
ck = pd.read_csv(f'{DATA}/results/resnet50_checkpoints.csv.gz')
sur = pp.SUR
lab = np.where((sur.event == 1) & (sur.duration <= 60), 1, np.where(sur.duration > 60, 0, -1))
test81 = set(sur.index[(sur.group == 'test').values & (lab >= 0)]); assert len(test81) == 81
pf = pd.read_csv(f'{DATA}/features/path_features.csv').set_index('ID')
rows = []
for e in range(12):
    d = ck[ck.epoch == e]
    p1 = np.where(d.pred_label == 1, d.pred_score, 1 - d.pred_score)
    p1 = pd.read_csv(io.StringIO('p\n' + '\n'.join(map(repr, p1.tolist()))))['p'].values   # written and read back as deployed
    feat = pp.aggregate(pd.DataFrame({'ID': d.ID.values, 'prob': pd.Series(p1).round(2).values, 'pred': d.pred_label.values}))
    tr_b, va_b = d.block.values == 'train', d.block.values == 'valid'
    t81 = va_b & d.ID.isin(test81).values
    df = feat[SIG].join(sur, how='inner'); tr = df[df.group == 'train'].copy(); te = df[df.group == 'test'].copy()
    mu, sd = tr[SIG].mean(), tr[SIG].std()
    tr[SIG] = (tr[SIG] - mu) / sd; te[SIG] = (te[SIG] - mu) / sd
    cph = CoxPHFitter().fit(tr[SIG + ['duration', 'event']], 'duration', 'event')
    c_tr = concordance_index(tr.duration, -cph.predict_partial_hazard(tr[SIG]).values, tr.event)
    c_te = concordance_index(te.duration, -cph.predict_partial_hazard(te[SIG]).values, te.event)
    if e == DEPLOYED:
        dev = max(float(np.abs(feat.loc[pf.index, a].values - pf[b].values).max()) for a, b in zip(SIG, ['prob05', 'prob058', 'pred1']))
        assert dev < 1e-9, dev                                                         # deployed features reproduced
        saved = pd.read_csv(f'{DATA}/results/Pathomics_cox_predictions_test.csv').set_index('ID').loc[te.index]
        assert round(c_te, 3) == round(concordance_index(te.duration, -saved.HR.values, te.event), 3) == 0.765
    res = pp.run_pipeline(feat, f'epoch {e}', force_k=3, verbose=False)
    rows.append({'epoch_0based': e, 'epoch': e + 1, 'deployed': e == DEPLOYED,
                 'patch_AUC_training': roc_auc_score(d['gt'].values[tr_b], p1[tr_b]),
                 'patch_AUC_validation_monitored': roc_auc_score(d['gt'].values[va_b], p1[va_b]),
                 'patch_AUC_test81': roc_auc_score(d['gt'].values[t81], p1[t81]),
                 'signature_C_training': c_tr, 'signature_C_test': c_te,
                 'k3_features': ', '.join(res['selected']), 'k3_n_features': len(res['selected']),
                 'k3_C_training': res['C_train'], 'k3_C_test': res['C_test']})
t = pd.DataFrame(rows)
assert t.loc[t.patch_AUC_validation_monitored.idxmax(), 'epoch_0based'] == DEPLOYED     # deployed = best monitored AUC
t.to_csv(f'{OUT}/TableS11_checkpoint_sensitivity.csv', index=False)
pd.set_option('display.width', 250); print(t.round(3).to_string(index=False))
print(f"\nsignature test C-index over the 12 checkpoints: {t.signature_C_test.min():.3f}-{t.signature_C_test.max():.3f} "
      f"(deployed {t.loc[t.deployed, 'signature_C_test'].item():.3f}); pipeline rerun at k = 3: {t.k3_C_test.min():.3f}-{t.k3_C_test.max():.3f}")
