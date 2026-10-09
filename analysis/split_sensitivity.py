# -*- coding: utf-8 -*-
"""Random-split sensitivity of the original Clinical model (age, N category, AJCC stage), Section 3.3 and
Supplementary Figure S1: 1000 random 7:3 patient splits stratified on the event (55 deaths and 139 censored patients in
each training set of 194, as in the study split); the Cox model is refitted on every training set and its C-index is
computed in the training and in the test set. Outputs -> analysis_outputs/split_sens_clinical.npy (rows = splits;
columns = training C-index, test C-index) and Table_split_sensitivity_summary.csv."""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
DATA = 'pipeline_outputs'; OUT = 'analysis_outputs'
OBSERVED_DIFF, OBSERVED_TEST = 0.124, 0.807                 # study split: training 0.683, test 0.807 (Table 3)

cl = pd.read_csv(f'{DATA}/data/clinical.csv')
rng = np.random.default_rng(42)
ev = cl.event.values
idx_ev = np.where(ev == 1)[0]; idx_ce = np.where(ev == 0)[0]
n_ev_tr, n_ce_tr = 55, 139
covs = ['N', 'AJCC', 'age']
res = []
for b in range(1000):
    tr_i = np.concatenate([rng.choice(idx_ev, n_ev_tr, replace=False), rng.choice(idx_ce, n_ce_tr, replace=False)])
    te_i = np.setdiff1d(np.arange(len(cl)), tr_i)
    tr = cl.iloc[tr_i]; te = cl.iloc[te_i]
    cph = CoxPHFitter(penalizer=0.0).fit(tr[covs + ['duration', 'event']], 'duration', 'event')
    res.append((cph.concordance_index_, concordance_index(te.duration, -cph.predict_partial_hazard(te[covs]).values, te.event)))
a = np.array(res); d = a[:, 1] - a[:, 0]
np.save(f'{OUT}/split_sens_clinical.npy', a)
summary = pd.DataFrame([{'splits': len(a), 'C train mean': round(a[:, 0].mean(), 3), 'C test mean': round(a[:, 1].mean(), 3),
                         'difference mean': round(d.mean(), 3), 'difference SD': round(d.std(), 3),
                         'difference 2.5th percentile': round(np.percentile(d, 2.5), 3), 'difference 97.5th percentile': round(np.percentile(d, 97.5), 3),
                         f'share of splits with difference >= +{OBSERVED_DIFF}': round((d >= OBSERVED_DIFF).mean(), 3),
                         f'share of splits with test C-index >= {OBSERVED_TEST}': round((a[:, 1] >= OBSERVED_TEST).mean(), 3)}])
summary.to_csv(f'{OUT}/Table_split_sensitivity_summary.csv', index=False); print(summary.T.to_string())
