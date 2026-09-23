# -*- coding: utf-8 -*-
"""Guideline-level clinical model (age, AJCC stage, ER, PR, HER2, grade) and its fusion with the pathomics signature.
Also: extended Table 1, univariable HRs, test-set nested likelihood-ratio test, time-dependent AUC, bootstrap comparisons."""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from scipy import stats
from sklearn.metrics import roc_auc_score

DATA = 'pipeline_outputs'; OUT = 'analysis_outputs'
RNG = np.random.default_rng(42)
cl = pd.read_csv(f'{OUT}/clinical_guideline_278.csv')
cl['grade3'] = (cl['grade'] == 3).astype(float); cl.loc[cl['grade'].isna(), 'grade3'] = np.nan
cl['HER2pos'] = cl['HER2_raw']
cl['TMA_set'] = cl['set']
sur = pd.read_csv(f'{DATA}/sur.csv').set_index('ID')
scores = pd.read_csv(f'{DATA}/results/joinit_info.csv').set_index('ID')   # deployed expectation scores (Clinical, Pathomics, Combined)
hr_te = {m: pd.read_csv(f'{DATA}/results/{m}_cox_predictions_test.csv').set_index('ID')['HR'] for m in ['Clinical', 'Pathomics', 'Combined']}
hr_tr = {m: pd.read_csv(f'{DATA}/results/{m}_cox_predictions_train.csv').set_index('ID')['HR'] for m in ['Clinical', 'Pathomics', 'Combined']}

def boot_ci(dur, risk, ev, nb=2000):
    n = len(dur); vals = []
    for _ in range(nb):
        b = RNG.choice(n, n, replace=True)
        if ev[b].sum() < 2: continue
        vals.append(concordance_index(dur[b], -risk[b], ev[b]))
    return np.percentile(vals, [2.5, 97.5])
def paired_p(dur, ra, rb, ev, nb=2000):
    n = len(dur); diffs = []
    for _ in range(nb):
        b = RNG.choice(n, n, replace=True)
        if ev[b].sum() < 2: continue
        diffs.append(concordance_index(dur[b], -ra[b], ev[b]) - concordance_index(dur[b], -rb[b], ev[b]))
    diffs = np.array(diffs)
    return max(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()), 1.0 / nb), diffs.mean()

# ------------------------------------------------------------------ extended Table 1
def table1(df):
    rows = []
    tr, te = df[df.group == 'train'], df[df.group == 'test']
    def cont(name, col):
        x, y = tr[col].dropna(), te[col].dropna()
        normal = stats.shapiro(df[col].dropna()).pvalue > 0.05
        p = stats.ttest_ind(x, y).pvalue if normal else stats.mannwhitneyu(x, y).pvalue
        rows.append([name, f'{df[col].mean():.2f} ± {df[col].std():.2f}', f'{x.mean():.2f} ± {x.std():.2f}', f'{y.mean():.2f} ± {y.std():.2f}', f'{p:.3f}', 't-test' if normal else 'Mann-Whitney U'])
    def cat(name, col, labels):
        sub = df.dropna(subset=[col])
        ct = pd.crosstab(sub[col], sub.group)
        if (stats.chi2_contingency(ct)[3] < 5).any() and ct.shape == (2, 2):
            p = stats.fisher_exact(ct.values)[1]; test = 'Fisher exact'
        else:
            p = stats.chi2_contingency(ct)[1]; test = 'chi-square'
        rows.append([name, '', '', '', f'{p:.3f}', test])
        for lv, lab in labels.items():
            n_all = (sub[col] == lv).sum(); n_tr = (tr[col] == lv).sum(); n_te = (te[col] == lv).sum()
            rows.append([f'  {lab}', f'{n_all} ({100*n_all/len(sub):.1f})', f'{n_tr} ({100*n_tr/tr[col].notna().sum():.1f})', f'{n_te} ({100*n_te/te[col].notna().sum():.1f})', '', ''])
        miss = df[col].isna().sum()
        if miss: rows.append([f'  Missing', f'{miss}', f'{tr[col].isna().sum()}', f'{te[col].isna().sum()}', '', ''])
    cont('Age (years)', 'age'); cont('Lymph nodes examined', 'number_of_lymph_nodes'); cont('Positive lymph nodes', 'positive_number')
    cat('T category', 'T', {1: 'T1', 2: 'T2', 3: 'T3'}); cat('N category', 'N', {0: 'N0', 1: 'N1', 2: 'N2', 3: 'N3'})
    cat('AJCC stage (6th ed.)', 'AJCC', {1: 'I', 2: 'II', 3: 'III'})
    cat('Histological grade', 'grade', {1: 'G1', 2: 'G2', 3: 'G3'})
    cat('ER status', 'ER', {1: 'Positive', 0: 'Negative'}); cat('PR status', 'PR', {1: 'Positive', 0: 'Negative'})
    cat('HER2 status', 'HER2pos', {1: 'Positive', 0: 'Negative'})
    cat('Surrogate subtype', 'subtype', {'HR+/HER2-': 'HR+/HER2-', 'HR+/HER2+': 'HR+/HER2+', 'HR-/HER2+': 'HR-/HER2+', 'TNBC': 'Triple-negative'})
    cat('TMA set', 'TMA_set', {'A': 'Set A (2001-2003 surgery)', 'B': 'Set B (2005-2007 surgery)'})
    cat('Overall-survival event', 'event', {1: 'Died', 0: 'Alive/censored'})
    cont('Follow-up (months)', 'duration')
    t = pd.DataFrame(rows, columns=['Characteristic', f'All (n={len(df)})', f'Training (n={len(tr)})', f'Test (n={len(te)})', 'p', 'Test'])
    return t
t1 = table1(cl); t1.to_csv(f'{OUT}/Table1_extended.csv', index=False, encoding='utf-8-sig'); print(t1.to_string())

# ------------------------------------------------------------------ models
G_VARS = ['age', 'AJCC', 'ER', 'PR', 'HER2pos', 'grade3']
ALT_VARS = ['age', 'T', 'N', 'ER', 'PR', 'HER2pos', 'grade3']
cc = cl.dropna(subset=G_VARS).copy()
print(f'complete cases for guideline model: {len(cc)} (train {sum(cc.group=="train")}, test {sum(cc.group=="test")}; events train {int(cc[cc.group=="train"].event.sum())}, test {int(cc[cc.group=="test"].event.sum())})')
tr = cc[cc.group == 'train'].set_index('ID'); te = cc[cc.group == 'test'].set_index('ID')

# univariable HRs (training set)
uni = []
for v, lab in [('age', 'Age (per year)'), ('AJCC', 'AJCC stage (per stage)'), ('T', 'T category (per category)'), ('N', 'N category (per category)'),
               ('ER', 'ER positive'), ('PR', 'PR positive'), ('HER2pos', 'HER2 positive'), ('grade3', 'Grade 3 vs 1-2')]:
    d = cl[cl.group == 'train'].dropna(subset=[v])
    c = CoxPHFitter().fit(d[[v, 'duration', 'event']], 'duration', 'event'); s = c.summary.iloc[0]
    uni.append({'Variable': lab, 'n': len(d), 'HR': round(s['exp(coef)'], 3), '95% CI': f"{s['exp(coef) lower 95%']:.3f}-{s['exp(coef) upper 95%']:.3f}", 'p': f"{s['p']:.3g}"})
uni = pd.DataFrame(uni); uni.to_csv(f'{OUT}/Table_guideline_univariable.csv', index=False); print(uni.to_string())

def fit_cox(df, covs, pen=0.0):
    try: return CoxPHFitter(penalizer=pen).fit(df[covs + ['duration', 'event']], 'duration', 'event')
    except Exception: return CoxPHFitter(penalizer=0.05).fit(df[covs + ['duration', 'event']], 'duration', 'event')
res_rows = []; risk_store = {}
def evaluate(name, cph, covs, dtr, dte):
    rtr = cph.predict_partial_hazard(dtr[covs]).values; rte = cph.predict_partial_hazard(dte[covs]).values
    ctr = concordance_index(dtr.duration, -rtr, dtr.event); cte = concordance_index(dte.duration, -rte, dte.event)
    lo, hi = boot_ci(dtr.duration.values, rtr, dtr.event.values); lo2, hi2 = boot_ci(dte.duration.values, rte, dte.event.values)
    res_rows.append({'Model': name, 'Covariates': ', '.join(covs), 'C train': round(ctr, 3), 'train 95% CI': f'{lo:.3f}-{hi:.3f}', 'C test': round(cte, 3), 'test 95% CI': f'{lo2:.3f}-{hi2:.3f}'})
    risk_store[name] = (pd.Series(rtr, index=dtr.index), pd.Series(rte, index=dte.index))
    return cph
# guideline clinical model (pre-specified, no selection)
cph_g = fit_cox(tr, G_VARS); evaluate('Clinical-guideline (age, AJCC, ER, PR, HER2, grade)', cph_g, G_VARS, tr, te)
multi = cph_g.summary[['coef', 'exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']].round(4).reset_index()
multi.columns = ['Variable', 'beta', 'HR', 'HR lower 95%', 'HR upper 95%', 'p']; multi.to_csv(f'{OUT}/Table_guideline_multivariable.csv', index=False); print(multi.to_string())
cph_a = fit_cox(tr, ALT_VARS); evaluate('Clinical-guideline alt (age, T, N, ER, PR, HER2, grade)', cph_a, ALT_VARS, tr, te)
# original deployed clinical model restricted to the same complete cases (age, N, AJCC as deployed)
cph_o = fit_cox(tr, ['N', 'AJCC', 'age']); evaluate('Clinical-original (age, N, AJCC) on complete cases', cph_o, ['N', 'AJCC', 'age'], tr, te)
# pathomics (deployed signature) on the same patients: use saved partial hazards
risk_store['Pathomics (deployed)'] = (hr_tr['Pathomics'].loc[tr.index], hr_te['Pathomics'].loc[te.index])
risk_store['Combined-original (deployed)'] = (hr_tr['Combined'].loc[tr.index], hr_te['Combined'].loc[te.index])
for nm in ['Pathomics (deployed)', 'Combined-original (deployed)']:
    a, b = risk_store[nm]
    lo, hi = boot_ci(tr.duration.values, a.values, tr.event.values); lo2, hi2 = boot_ci(te.duration.values, b.values, te.event.values)
    res_rows.append({'Model': nm + ' on complete cases', 'Covariates': '-', 'C train': round(concordance_index(tr.duration, -a.values, tr.event), 3), 'train 95% CI': f'{lo:.3f}-{hi:.3f}',
                     'C test': round(concordance_index(te.duration, -b.values, te.event), 3), 'test 95% CI': f'{lo2:.3f}-{hi2:.3f}'})
# Combined-guideline: late fusion of guideline clinical expectation and pathomics expectation (as deployed)
exp_g_tr = cph_g.predict_expectation(tr[G_VARS]); exp_g_te = cph_g.predict_expectation(te[G_VARS])
ftr = pd.DataFrame({'ClinicalG': exp_g_tr.values, 'Pathomics': scores.loc[tr.index, 'Pathomics'].values, 'duration': tr.duration.values, 'event': tr.event.values}, index=tr.index)
fte = pd.DataFrame({'ClinicalG': exp_g_te.values, 'Pathomics': scores.loc[te.index, 'Pathomics'].values, 'duration': te.duration.values, 'event': te.event.values}, index=te.index)
cph_cg = fit_cox(ftr, ['ClinicalG', 'Pathomics']); evaluate('Combined-guideline (guideline clinical + pathomics)', cph_cg, ['ClinicalG', 'Pathomics'], ftr, fte)
cg = cph_cg.summary[['coef', 'exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']].round(4).reset_index(); print(cg.to_string())
res = pd.DataFrame(res_rows); res.to_csv(f'{OUT}/Table_guideline_models.csv', index=False, encoding='utf-8-sig'); print(res.to_string())

# pairwise paired-bootstrap comparisons (test and train) among key models
pairs = [('Combined-guideline (guideline clinical + pathomics)', 'Clinical-guideline (age, AJCC, ER, PR, HER2, grade)'),
         ('Clinical-guideline (age, AJCC, ER, PR, HER2, grade)', 'Clinical-original (age, N, AJCC) on complete cases'),
         ('Combined-guideline (guideline clinical + pathomics)', 'Pathomics (deployed)'),
         ('Combined-guideline (guideline clinical + pathomics)', 'Combined-original (deployed)'),
         ('Pathomics (deployed)', 'Clinical-guideline (age, AJCC, ER, PR, HER2, grade)')]
cmp_rows = []
for a, b in pairs:
    for cname, d, k in [('train', tr, 0), ('test', te, 1)]:
        p, md = paired_p(d.duration.values, risk_store[a][k].loc[d.index].values, risk_store[b][k].loc[d.index].values, d.event.values)
        cmp_rows.append({'Cohort': cname, 'Comparison': f'{a} vs {b}', 'delta C': round(md, 3), 'p': round(p, 4)})
cmp = pd.DataFrame(cmp_rows); cmp.to_csv(f'{OUT}/Table_guideline_pairwise.csv', index=False, encoding='utf-8-sig'); print(cmp.to_string())

# test-set nested models: does the frozen pathomics score add to the frozen guideline clinical score in held-out patients?
lp_g = np.log(risk_store['Clinical-guideline (age, AJCC, ER, PR, HER2, grade)'][1].loc[te.index].values)
lp_p = np.log(hr_te['Pathomics'].loc[te.index].values)
lp_o = np.log(hr_te['Clinical'].loc[te.index].values)
nested = []
for base_name, lp_base in [('guideline clinical score', lp_g), ('original clinical score', lp_o)]:
    d0 = pd.DataFrame({'base': lp_base, 'duration': te.duration.values, 'event': te.event.values})
    d1 = d0.assign(path=(lp_p - lp_p.mean()) / lp_p.std())
    m0 = CoxPHFitter().fit(d0, 'duration', 'event'); m1 = CoxPHFitter().fit(d1, 'duration', 'event')
    lrt = 2 * (m1.log_likelihood_ - m0.log_likelihood_); p = stats.chi2.sf(lrt, 1); s = m1.summary.loc['path']
    nested.append({'Base model (test set, n=%d, events=%d)' % (len(te), int(te.event.sum())): base_name, 'HR per SD of pathomics score (adjusted)': round(s['exp(coef)'], 2),
                   '95% CI': f"{s['exp(coef) lower 95%']:.2f}-{s['exp(coef) upper 95%']:.2f}", 'Wald p': f"{s['p']:.3f}", 'LRT chi2': round(lrt, 2), 'LRT p': f'{p:.3f}'})
nested = pd.DataFrame(nested); nested.to_csv(f'{OUT}/Table_test_nested_LRT.csv', index=False); print(nested.to_string())

# time-dependent AUC (definition as in the original analysis: event by t = positive; censored before/at t excluded)
def tdauc(d, risk, t):
    keep = ~((d.event == 0) & (d.duration <= t)); dd = d[keep]; y = ((dd.event == 1) & (dd.duration <= t)).astype(int)
    return roc_auc_score(y, risk.loc[dd.index]), int(y.sum()), len(dd)
td = []
for nm in ['Clinical-original (age, N, AJCC) on complete cases', 'Clinical-guideline (age, AJCC, ER, PR, HER2, grade)', 'Pathomics (deployed)', 'Combined-original (deployed)', 'Combined-guideline (guideline clinical + pathomics)']:
    for cname, d, k in [('train', tr, 0), ('test', te, 1)]:
        for t, tl in [(60, '5 y'), (84, '7 y'), (108, '9 y')]:
            auc, ne, n = tdauc(d, risk_store[nm][k], t)
            td.append({'Model': nm, 'Cohort': cname, 'Horizon': tl, 'AUC': round(auc, 3), 'events': ne, 'n': n})
td = pd.DataFrame(td); td.to_csv(f'{OUT}/Table_guideline_tdAUC.csv', index=False); print(td.pivot_table(index=['Model', 'Cohort'], columns='Horizon', values='AUC').to_string())
# save risk scores for figures
pd.DataFrame({k + '|train': v[0] for k, v in risk_store.items()}).to_csv(f'{OUT}/guideline_risks_train.csv')
pd.DataFrame({k + '|test': v[1] for k, v in risk_store.items()}).to_csv(f'{OUT}/guideline_risks_test.csv')
cc[['ID', 'group', 'event', 'duration'] + G_VARS + ['T', 'N', 'subtype']].to_csv(f'{OUT}/guideline_complete_cases.csv', index=False)
print('DONE')
