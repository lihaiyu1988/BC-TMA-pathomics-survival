# -*- coding: utf-8 -*-
"""Deployed Clinical (age, N category, AJCC stage), Pathomics (BoW_prob(0.50), BoW_prob(0.58), BoW_pred(1)) and Combined
models of the original pipeline (their risk scores are pipeline_outputs/results/*_cox_predictions_*.csv):
  Table 3a  C-index with 2000-sample bootstrap 95% CIs            -> Table3_cindex_CI.csv
  Table 3b  paired-bootstrap ΔC-index, two-sided P, Holm-adjusted P within each cohort (three comparisons)
                                                                   -> Table3_cindex_comparison.csv
  Table 4   multivariable Cox coefficients, HRs and 95% CIs refitted on the training cohort (unpenalized)
                                                                   -> Table4_cox_coefficients.csv
  Table S1  slide-level AUC and accuracy (threshold 0.5) of each encoder after mean aggregation of patch probabilities
                                                                   -> TableS1_slide_level_encoders.csv
  Table S5  log-rank P with the training-median cut-off of each model's partial hazard (sensitivity analysis; the locked
            X-tile cut-offs are in Table_KM_locked_cutoffs.csv)   -> TableS5_median_cutoff_logrank.csv
The bootstrap uses one random-number generator (seed 42) in the order written below."""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
from sklearn.metrics import roc_auc_score
DATA = 'pipeline_outputs'; OUT = 'analysis_outputs'
RNG = np.random.default_rng(42)
MODELS, COHORTS = ['Clinical', 'Pathomics', 'Combined'], ['train', 'test']
sur = pd.read_csv(f'{DATA}/sur.csv').set_index('ID')


def cidx(dur, risk, ev):
    return concordance_index(dur, -risk, ev)                     # higher risk -> shorter survival


def load_pred(m, c):
    p = pd.read_csv(f'{DATA}/results/{m}_cox_predictions_{c}.csv').set_index('ID'); d = sur.loc[p.index]
    return pd.DataFrame({'dur': d['duration'].values, 'ev': d['event'].values, 'risk': p['HR'].values}, index=p.index)


def boot_cindex(df, nb=2000):
    vals = []; n = len(df); idx = np.arange(n)
    for _ in range(nb):
        s = df.iloc[RNG.choice(idx, n, replace=True)]
        if s['ev'].sum() < 2: continue
        vals.append(cidx(s['dur'].values, s['risk'].values, s['ev'].values))
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def compare(df_a, df_b, nb=2000):
    """Paired bootstrap on identical patients; two-sided P for H0: ΔC = 0 (floored at 1/nb)."""
    common = df_a.index.intersection(df_b.index); df_a, df_b = df_a.loc[common], df_b.loc[common]
    n = len(df_a); idx = np.arange(n); diffs = []
    for _ in range(nb):
        b = RNG.choice(idx, n, replace=True); sa, sb = df_a.iloc[b], df_b.iloc[b]
        if sa['ev'].sum() < 2: continue
        diffs.append(cidx(sa['dur'].values, sa['risk'].values, sa['ev'].values) - cidx(sb['dur'].values, sb['risk'].values, sb['ev'].values))
    diffs = np.array(diffs); p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return max(p, 1.0 / nb), diffs.mean()


def holm(p):
    """Holm step-down adjustment of a list of P values (returned in the input order)."""
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for k, i in enumerate(o):
        run = max(run, (m - k) * p[i]); adj[i] = min(1.0, run)
    return adj


# ---------------------------------------------------------------- Table 3a and 3b
preds = {m: {c: load_pred(m, c) for c in COHORTS} for m in MODELS}
rows = []
for m in MODELS:
    for c in COHORTS:
        df = preds[m][c]; lo, hi = boot_cindex(df)
        rows.append({'Model': m, 'Cohort': c, 'C-index': round(cidx(df['dur'].values, df['risk'].values, df['ev'].values), 3), '95% CI': f'{lo:.3f}-{hi:.3f}'})
t3a = pd.DataFrame(rows); t3a.to_csv(f'{OUT}/Table3_cindex_CI.csv', index=False); print(t3a.to_string(index=False))
rows = []
for c in COHORTS:
    block = []
    for a, b in [('Combined', 'Clinical'), ('Combined', 'Pathomics'), ('Pathomics', 'Clinical')]:
        p, md = compare(preds[a][c], preds[b][c]); block.append({'Cohort': c, 'Comparison': f'{a} vs {b}', 'delta_C': round(md, 3), 'p': round(p, 4)})
    for r, ph in zip(block, holm([r['p'] for r in block])): r['p_holm'] = round(ph, 4)
    rows += block
t3b = pd.DataFrame(rows); t3b.to_csv(f'{OUT}/Table3_cindex_comparison.csv', index=False); print(t3b.to_string(index=False))

# ---------------------------------------------------------------- Table 4
cl = pd.read_csv(f'{DATA}/data/clinical.csv')
spec = [('Clinical', cl[cl.group == 'train'], ['age', 'N', 'AJCC']),
        ('Pathomics', pd.read_csv(f'{DATA}/features/Pathomics_train_cox.csv'), ['prob05', 'prob058', 'pred1']),
        ('Combined', pd.read_csv(f'{DATA}/features/Combined_train_features_norm.csv'), ['Clinical', 'Pathomics'])]
rows = []
for name, df, covs in spec:
    s = CoxPHFitter().fit(df[covs + ['duration', 'event']], 'duration', 'event').summary
    for v in covs:
        r = s.loc[v]
        rows.append({'Model': name, 'Variable': v, 'n': len(df), 'events': int(df['event'].sum()), 'beta': r['coef'], 'HR': r['exp(coef)'],
                     'HR lower 95%': r['exp(coef) lower 95%'], 'HR upper 95%': r['exp(coef) upper 95%'], 'p': r['p']})
t4 = pd.DataFrame(rows); t4.to_csv(f'{OUT}/Table4_cox_coefficients.csv', index=False); print(t4.round(4).to_string(index=False))

# ---------------------------------------------------------------- Table S1
alldl = pd.read_csv(f'{DATA}/results/ALL_DL_PREDICTIONS.csv'); gtmap = alldl.groupby('ID')['gt'].first()
rows = []
for m, disp in [('resnet18', 'ResNet18'), ('resnet50', 'ResNet50'), ('densenet121', 'DenseNet121'), ('CrossFormer', 'CrossFormer (Transformer/attention)')]:
    for c in COHORTS:
        agg = pd.read_csv(f'{DATA}/results/Pathomics_Slice_{m}_{c}.csv').groupby('ID')['label-1'].mean()
        ids = agg.index.intersection(gtmap.index); y = gtmap.loc[ids].values; p = agg.loc[ids].values
        rows.append({'Backbone': disp, 'Cohort': c, 'n': len(ids), 'WSI AUC': round(roc_auc_score(y, p), 3), 'Accuracy': round(((p >= 0.5).astype(int) == y).mean(), 3)})
ts1 = pd.DataFrame(rows); ts1.to_csv(f'{OUT}/TableS1_slide_level_encoders.csv', index=False); print(ts1.to_string(index=False))

# ---------------------------------------------------------------- Table S5, training-median cut-offs
rows = []
for m in MODELS:
    med = pd.read_csv(f'{DATA}/results/{m}_cox_predictions_train.csv')['HR'].median()
    for c in COHORTS:
        d = pd.read_csv(f'{DATA}/results/{m}_cox_predictions_{c}.csv').set_index('ID').join(sur[['event', 'duration']]); hi = d.HR >= med
        r = logrank_test(d.duration[hi], d.duration[~hi], d.event[hi], d.event[~hi])
        rows.append({'Cohort': c, 'Model': m, 'median cut-off (training partial hazard)': round(med, 3), 'High-risk n (events)': f'{int(hi.sum())} ({int(d.event[hi].sum())})',
                     'Low-risk n (events)': f'{int((~hi).sum())} ({int(d.event[~hi].sum())})', 'log-rank p': r.p_value})
ts5 = pd.DataFrame(rows); ts5.to_csv(f'{OUT}/TableS5_median_cutoff_logrank.csv', index=False); print(ts5.to_string(index=False))
