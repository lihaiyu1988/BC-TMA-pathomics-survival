# -*- coding: utf-8 -*-
"""The study split and the 13 excluded hold-out patients (Section 2.1; Supplementary Table S10).
The original pipeline (original_notebooks/Step0) partitioned the 291 patients with annotated cores and follow-up into a
training block (194 patients) and a hold-out block (97 patients). Before survival modeling, 13 hold-out patients were
excluded because their H&E images were judged unusable on pathological review (clinical/clinical_excluded_holdout13.csv);
the other 84 form the test cohort (pipeline_outputs/sur.csv).
  1) Baseline balance of the original partition, 194 vs 97 patients (tests as in Table 1)
                                                      -> analysis_outputs/Table_split_balance_194_vs_97.csv
  2) The frozen Clinical, Pathomics and Combined models applied to the whole hold-out block (97 patients): C-index with
     2000-sample bootstrap 95% CIs and paired-bootstrap comparisons, Holm-adjusted (one random-number generator,
     seed 42, in the order written below)           -> analysis_outputs/TableS10_holdout_with_excluded.csv
The frozen models are the deployed Cox models refitted on the training cohort, as for Table 4; their risk scores rank
the 278 analysed patients exactly as the deployed predictions do (checked below). The 13 patients are scored with the
deployed normalization constants (results/norm_info.json, training mean and SD) and the deployed features
(features/path_features.csv covers all 291 patients)."""
import os, sys, json, warnings
import numpy as np, pandas as pd
from scipy import stats
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(encoding='utf-8')
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root
DATA, OUT = 'pipeline_outputs', 'analysis_outputs'
RNG = np.random.default_rng(42)
sur = pd.read_csv(f'{DATA}/sur.csv').set_index('ID')
ex = pd.read_csv('clinical/clinical_excluded_holdout13.csv').set_index('ID'); assert len(ex) == 13 and not ex.index.isin(sur.index).any()
dl = pd.read_csv(f'{DATA}/results/ALL_DL_PREDICTIONS.csv')
assert set(ex.index) <= set(dl['ID'].iloc[56499:]), 'the 13 patients belong to the hold-out (validation) block'

# ---------------------------------------------------------------- 1) balance of the original partition (194 vs 97)
cl = pd.read_csv('clinical/clinical_guideline_278.csv')
part = pd.concat([cl, ex.reset_index().assign(group='test')], ignore_index=True); part['HER2pos'] = part['HER2']


def table1(df):            # identical to guideline_clinical_model.table1 (Table 1)
    rows = []
    tr, te = df[df.group == 'train'], df[df.group == 'test']
    def cont(name, col):
        x, y = tr[col].dropna(), te[col].dropna()
        normal = stats.shapiro(df[col].dropna()).pvalue > 0.05
        p = stats.ttest_ind(x, y).pvalue if normal else stats.mannwhitneyu(x, y).pvalue
        rows.append([name, f'{x.mean():.2f} ± {x.std():.2f}', f'{y.mean():.2f} ± {y.std():.2f}', p, 't-test' if normal else 'Mann-Whitney U'])
    def cat(name, col):
        sub = df.dropna(subset=[col]); ct = pd.crosstab(sub[col], sub.group)
        if (stats.chi2_contingency(ct)[3] < 5).any() and ct.shape == (2, 2): p, test = stats.fisher_exact(ct.values)[1], 'Fisher exact'
        else: p, test = stats.chi2_contingency(ct)[1], 'chi-square'
        rows.append([name, '', '', p, test])
    cont('Age (years)', 'age'); cont('Lymph nodes examined', 'number_of_lymph_nodes'); cont('Positive lymph nodes', 'positive_number')
    for name, col in [('T category', 'T'), ('N category', 'N'), ('AJCC stage (6th ed.)', 'AJCC'), ('Histological grade', 'grade'), ('ER status', 'ER'),
                      ('PR status', 'PR'), ('HER2 status', 'HER2pos'), ('Surrogate subtype', 'subtype'), ('TMA set', 'TMA_set'), ('Overall-survival event', 'event')]:
        cat(name, col)
    cont('Follow-up (months)', 'duration')
    return pd.DataFrame(rows, columns=['Characteristic', f'Training (n={len(tr)})', f'Hold-out (n={len(te)})', 'p', 'Test'])
bal = table1(part); bal.to_csv(f'{OUT}/Table_split_balance_194_vs_97.csv', index=False)
print(bal.assign(p=bal.p.round(3)).to_string(index=False))

# ---------------------------------------------------------------- 2) frozen models on the whole hold-out block
pf = pd.read_csv(f'{DATA}/features/path_features.csv').set_index('ID')
ni = json.load(open(f'{DATA}/results/norm_info.json'))
path = pd.DataFrame({c: (pf[c] - ni[c]['mean']) / ni[c]['std'] for c in ['prob05', 'prob058', 'pred1']})
ptr = pd.read_csv(f'{DATA}/features/Pathomics_train_cox.csv').set_index('ID')
assert np.abs(path.loc[ptr.index, ['prob05', 'prob058', 'pred1']].values - ptr[['prob05', 'prob058', 'pred1']].values).max() < 1e-12
cdat = pd.read_csv(f'{DATA}/data/clinical.csv').set_index('ID')
clin = pd.concat([cdat[['age', 'N', 'AJCC']], ex[['age', 'N', 'AJCC']]])
cph_p = CoxPHFitter().fit(ptr[['prob05', 'prob058', 'pred1', 'duration', 'event']], 'duration', 'event')
cph_c = CoxPHFitter().fit(cdat[cdat.group == 'train'][['age', 'N', 'AJCC', 'duration', 'event']], 'duration', 'event')
cmb = pd.read_csv(f'{DATA}/features/Combined_train_features_norm.csv').set_index('ID')
cph_m = CoxPHFitter().fit(cmb[['Clinical', 'Pathomics', 'duration', 'event']], 'duration', 'event')


def scores(ids):
    ec = cph_c.predict_expectation(clin.loc[ids]).values; ep = cph_p.predict_expectation(path.loc[ids]).values
    return pd.DataFrame({'Clinical': cph_c.predict_partial_hazard(clin.loc[ids]).values,
                         'Pathomics': cph_p.predict_partial_hazard(path.loc[ids]).values,
                         'Combined': cph_m.predict_partial_hazard(pd.DataFrame({'Clinical': ec, 'Pathomics': ep}, index=ids)).values}, index=ids)


MODELS = ['Clinical', 'Pathomics', 'Combined']
sc = scores(list(sur.index) + list(ex.index))
for m in MODELS:                                   # the refitted models rank the 278 patients as the deployed predictions
    saved = pd.concat([pd.read_csv(f'{DATA}/results/{m}_cox_predictions_{c}.csv') for c in ('train', 'test')]).set_index('ID')['HR']
    assert stats.spearmanr(sc.loc[saved.index, m], saved).statistic > 0.99999, m
surv = pd.concat([sur[['duration', 'event']], ex[['duration', 'event']]])
test84 = list(sur.index[sur.group == 'test']); hold97 = test84 + list(ex.index)
cidx = lambda ids, m: concordance_index(surv.loc[ids, 'duration'], -sc.loc[ids, m], surv.loc[ids, 'event'])
ref = pd.read_csv(f'{OUT}/Table3_cindex_CI.csv') if os.path.exists(f'{OUT}/Table3_cindex_CI.csv') else None
for m in MODELS:
    if ref is not None: assert round(cidx(test84, m), 3) == ref[(ref.Model == m) & (ref.Cohort == 'test')]['C-index'].item(), m


def boot(ids, m, nb=2000):
    d = surv.loc[ids]; r = sc.loc[ids, m].values; n = len(ids); vals = []
    for _ in range(nb):
        b = RNG.choice(n, n, replace=True)
        if d.event.values[b].sum() < 2: continue
        vals.append(concordance_index(d.duration.values[b], -r[b], d.event.values[b]))
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def compare(ids, a, b, nb=2000):
    d = surv.loc[ids]; ra, rb = sc.loc[ids, a].values, sc.loc[ids, b].values; n = len(ids); diffs = []
    for _ in range(nb):
        k = RNG.choice(n, n, replace=True)
        if d.event.values[k].sum() < 2: continue
        diffs.append(concordance_index(d.duration.values[k], -ra[k], d.event.values[k]) - concordance_index(d.duration.values[k], -rb[k], d.event.values[k]))
    diffs = np.array(diffs); return max(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()), 1.0 / nb), diffs.mean()


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for k, i in enumerate(o): run = max(run, (m - k) * p[i]); adj[i] = min(1.0, run)
    return adj


rows = []
d97 = surv.loc[hold97]
for m in MODELS:
    lo, hi = boot(hold97, m)
    rows.append({'Set': 'hold-out block incl. excluded (97)', 'n': len(hold97), 'deaths': int(d97.event.sum()), 'Model': m,
                 'C-index': round(cidx(hold97, m), 3), '95% CI': f'{lo:.3f}-{hi:.3f}', 'Comparison': '', 'delta_C': np.nan, 'p': np.nan, 'p_holm': np.nan})
block = []
for a, b in [('Combined', 'Clinical'), ('Combined', 'Pathomics'), ('Pathomics', 'Clinical')]:
    p, md = compare(hold97, a, b); block.append({'Set': 'hold-out block incl. excluded (97)', 'n': len(hold97), 'deaths': int(d97.event.sum()), 'Model': '',
                                                 'C-index': np.nan, '95% CI': '', 'Comparison': f'{a} vs {b}', 'delta_C': round(md, 3), 'p': round(p, 4)})
for r, ph in zip(block, holm([r['p'] for r in block])): r['p_holm'] = round(ph, 4)
t = pd.DataFrame(rows + block); t.to_csv(f'{OUT}/TableS10_holdout_with_excluded.csv', index=False)
print(t.to_string(index=False))
print(f"\n13 excluded patients: {int(ex.event.sum())} deaths ({int(((ex.event == 1) & (ex.duration <= 60)).sum())} within 60 months); "
      f"deaths in the hold-out block {int(d97.event.sum())}/97 vs test cohort {int(surv.loc[test84].event.sum())}/84")
