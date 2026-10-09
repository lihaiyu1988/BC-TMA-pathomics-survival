# -*- coding: utf-8 -*-
"""External validation of the frozen clinical models in TCGA-BRCA (public clinical data).

Pre-specified protocol (fixed before any outcome was examined in TCGA-BRCA):
  * Data: GDC BCR Biotab patient file (nationwidechildrens.org_clinical_patient_brca.txt; receptor status, pathologic
    TNM, AJCC stage, neoadjuvant therapy, sex) and the TCGA Pan-Cancer Clinical Data Resource (TCGA-CDR, Liu et al.,
    Cell 2018; overall survival OS / OS.time and age at diagnosis). Histological grade is not part of either resource;
    the source used for grade is documented in GRADE_SOURCE below.
  * Eligibility, mirroring the development cohort: women; no neoadjuvant therapy; no distant metastasis (pathologic M1
    or AJCC stage IV excluded); AJCC pathologic stage I-III; overall-survival time > 0.
  * Encodings, identical to the development cohort: age in years; AJCC stage I/II/III = 1/2/3 (sub-stages pooled);
    pathologic N category N0/N1/N2/N3 = 0/1/2/3 (N0(i+), N0(i-), N0(mol+) = N0; N1mi = N1; NX = missing);
    ER and PR = status reported by the contributing sites; HER2 = FISH when reported, otherwise the IHC score
    (3+ positive, 0/1+ negative, 2+ missing) or, when no score was recorded, the site-reported IHC status.
  * Models are applied frozen: coefficients, centring means and baseline hazard come from the development (training)
    cohort; nothing is refitted in TCGA-BRCA. Deployed Clinical model: age + N + AJCC stage (194 training patients);
    Clinical-guideline model: age + AJCC stage + ER + PR + HER2 + grade 3 (183 complete-case training patients).
  * Risk groups: locked development cut-offs on the partial hazard (Clinical: X-tile 1.03; Clinical-guideline: the
    training-cohort median, the cut-off rule of the sensitivity analysis of the development study).
  * Histological grade (not in the TCGA clinical data): primary analysis = Nottingham grade reconstructed from the
    expert-panel morphology scores of the TCGA breast cancer pathology review (Thennavan et al., Cell Genomics 2021,
    Data S2; tubule formation + nuclear pleomorphism + mitotic count, each 1-3; total 3-5 = G1, 6-7 = G2, 8-9 = G3) when
    the scores are obtainable; sensitivity analysis (and the only option otherwise) = grade 3 replaced by its expected
    value given age, AJCC stage, ER, PR and HER2 from a logistic model fitted in the development training cohort.
  * Metrics: Harrell's C-index (2000 bootstrap CI); paired-bootstrap difference between the two models on the same
    patients; time-dependent AUC at 3 and 5 years (deaths by the horizon vs survivors beyond it; patients censored
    before the horizon excluded); calibration at 5 years (predicted vs Kaplan-Meier observed risk in quintiles of
    predicted risk, observed/expected ratio and calibration slope); Kaplan-Meier curves with log-rank tests.
Outputs -> analysis_outputs/external_tcga/ (CSV tables, per-patient analysis file); Figure 11 by make_figure11_external.py.
Raw inputs (open access; see README): GDC file 8162d394-8b64-4da2-9f5b-d164c54b9608 (nationwidechildrens.org_clinical_patient_brca.txt),
GDC file 1b5f413e-a8d1-4d10-92eb-7c4ae739ed81 (TCGA-CDR-SupplementalTableS1.xlsx), Data S2 of Thennavan et al. 2021 (mmc3.xlsx; set TCGA_GRADE_FILE).
Usage: python external_validation_tcga.py <dir with raw downloads>"""
import os, sys, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)   # repository root
A = 'clinical'; D = 'pipeline_outputs'; OUT = 'analysis_outputs/external_tcga'
RAW = sys.argv[1] if len(sys.argv) > 1 else f'{OUT}/raw'
DAYS_PER_MONTH = 30.4375
RNG = np.random.default_rng(20261008)
NB = 2000

# ------------------------------------------------------------------ frozen development models
cl = pd.read_csv(f'{A}/clinical_guideline_278.csv')
cl['grade3'] = (cl['grade'] == 3).astype(float); cl.loc[cl['grade'].isna(), 'grade3'] = np.nan
cl['HER2pos'] = cl['HER2']          # distributed table: HER2 = positivity (FISH first, otherwise IHC 3+)
dev_tr = pd.read_csv(f'{D}/features/Clinical_train_features_norm.csv').set_index('ID')
CLIN = ['N', 'AJCC', 'age']
cph_clin = CoxPHFitter().fit(dev_tr[CLIN + ['duration', 'event']], 'duration', 'event')
G = ['age', 'AJCC', 'ER', 'PR', 'HER2pos', 'grade3']
g_tr = cl[cl.group == 'train'].dropna(subset=G)
cph_g = CoxPHFitter().fit(g_tr[G + ['duration', 'event']], 'duration', 'event')
# the deployed Clinical model must be reproduced exactly (Table 4: 0.034 / 0.033 / 0.828; deployed partial hazards)
dep = pd.read_csv(f'{D}/results/Clinical_cox_predictions_train.csv').set_index('ID')
assert float((cph_clin.predict_partial_hazard(dev_tr[CLIN]) - dep.loc[dev_tr.index, 'HR']).abs().max()) < 1e-3
assert len(g_tr) == 183
CUT_CLIN = 1.03                                                     # locked X-tile cut-off of the deployed Clinical model
CUT_G = float(np.median(cph_g.predict_partial_hazard(g_tr[G])))     # locked training-median cut-off (guideline model)

# ------------------------------------------------------------------ TCGA-BRCA cohort
pat = pd.read_csv(f'{RAW}/nationwidechildrens.org_clinical_patient_brca.txt', sep='\t', dtype=str).iloc[2:].set_index('bcr_patient_barcode')
cdr = pd.read_excel(f'{RAW}/TCGA-CDR-SupplementalTableS1.xlsx', sheet_name=0)
cdr = cdr[cdr['type'] == 'BRCA'].set_index('bcr_patient_barcode')
df = pat.join(cdr[['age_at_initial_pathologic_diagnosis', 'OS', 'OS.time']], how='inner')
flow = [('TCGA-BRCA patients in the GDC clinical file and TCGA-CDR', len(df))]
def step(mask, label):
    global df
    df = df[mask]; flow.append((label, len(df)))
step(df.gender == 'FEMALE', 'women')
step(df.history_neoadjuvant_treatment == 'No', 'no neoadjuvant therapy')
step(~df.ajcc_metastasis_pathologic_pm.isin(['M1']) & (df.ajcc_pathologic_tumor_stage != 'Stage IV'), 'no distant metastasis (M1 / stage IV excluded)')
STAGE = {'Stage I': 1, 'Stage IA': 1, 'Stage IB': 1, 'Stage II': 2, 'Stage IIA': 2, 'Stage IIB': 2,
         'Stage III': 3, 'Stage IIIA': 3, 'Stage IIIB': 3, 'Stage IIIC': 3}
df['AJCC'] = df.ajcc_pathologic_tumor_stage.map(STAGE)
step(df.AJCC.notna(), 'AJCC pathologic stage I-III')
df['duration'] = pd.to_numeric(df['OS.time'], errors='coerce') / DAYS_PER_MONTH
df['event'] = pd.to_numeric(df['OS'], errors='coerce')
step(df.duration.notna() & (df.duration > 0) & df.event.notna(), 'overall-survival time > 0')
df['age'] = pd.to_numeric(df.age_at_initial_pathologic_diagnosis, errors='coerce')

def n_cat(v):
    if not isinstance(v, str): return np.nan
    v = v.strip()
    if v.startswith('N0'): return 0
    if v.startswith('N1'): return 1
    if v.startswith('N2'): return 2
    if v.startswith('N3'): return 3
    return np.nan                                                   # NX or missing
df['N'] = df.ajcc_nodes_pathologic_pn.map(n_cat)
rec = lambda v: 1.0 if v == 'Positive' else (0.0 if v == 'Negative' else np.nan)
df['ER'] = df.er_status_by_ihc.map(rec); df['PR'] = df.pr_status_by_ihc.map(rec)
def her2(r):
    if r.her2_fish_status in ('Positive', 'Negative'): return rec(r.her2_fish_status)
    if r.her2_ihc_score == '3+': return 1.0                         # IHC score, as in the development cohort
    if r.her2_ihc_score in ('0', '1+'): return 0.0
    if r.her2_ihc_score == '2+': return np.nan                      # equivocal without FISH
    return rec(r.her2_status_by_ihc)                                # no score recorded: site-reported IHC status
df['HER2pos'] = df.apply(her2, axis=1)

if __name__ == '__main__':
    print('frozen deployed Clinical model:', cph_clin.params_.round(4).to_dict())
    print('frozen Clinical-guideline model:', cph_g.params_.round(4).to_dict(), '; locked median cut-off', round(CUT_G, 3))
    for lab, n in flow: print(f'{n:5d}  {lab}')
    print('missing: N', int(df.N.isna().sum()), '| ER', int(df.ER.isna().sum()), '| PR', int(df.PR.isna().sum()), '| HER2', int(df.HER2pos.isna().sum()))
    print('deaths', int(df.event.sum()), '| median follow-up (reverse KM, months)',
          round(KaplanMeierFitter().fit(df.duration, 1 - df.event).median_survival_time_, 1))


# ------------------------------------------------------------------ metrics (frozen models, no refitting)
def lp(cph, X):                     # uncentred linear predictor x'beta (ranking and calibration slope)
    return X[cph.params_.index].values @ cph.params_.values
def boot_c(t, r, e, nb=NB):
    n = len(t); v = []
    for _ in range(nb):
        b = RNG.choice(n, n, replace=True)
        if e[b].sum() >= 2: v.append(concordance_index(t[b], -r[b], e[b]))
    return np.percentile(v, [2.5, 97.5])
def td_auc(t, r, e, h):
    case = (t <= h) & (e == 1); ctrl = t > h; keep = case | ctrl
    return roc_auc_score(case[keep].astype(int), r[keep]), int(case.sum()), int(keep.sum())
def boot_auc(t, r, e, h, nb=NB):
    n = len(t); v = []
    for _ in range(nb):
        b = RNG.choice(n, n, replace=True); tb, rb, eb = t[b], r[b], e[b]
        case = (tb <= h) & (eb == 1); ctrl = tb > h
        if case.sum() >= 2 and ctrl.sum() >= 2: v.append(td_auc(tb, rb, eb, h)[0])
    return np.percentile(v, [2.5, 97.5])
def km_risk(t, e, h):
    k = KaplanMeierFitter().fit(t, e); s = float(k.survival_function_at_times(h).iloc[0])
    ci = k.confidence_interval_survival_function_
    lo = float(np.interp(h, ci.index.values, ci.iloc[:, 0].values)); hi = float(np.interp(h, ci.index.values, ci.iloc[:, 1].values))
    return 1 - s, 1 - hi, 1 - lo
def calibration(cph, X, t, e, h=60, q=5):
    pred = 1 - cph.predict_survival_function(X[cph.params_.index], times=[h]).iloc[0].values
    grp = pd.qcut(pd.Series(pred).rank(method='first'), q, labels=False).values
    rows = []
    for g in range(q):
        m = grp == g; o, olo, ohi = km_risk(t[m], e[m], h)
        rows.append({'group': g + 1, 'n': int(m.sum()), 'deaths': int(e[m].sum()), 'predicted': pred[m].mean(), 'observed': o, 'obs_lo': olo, 'obs_hi': ohi})
    o_all = km_risk(t, e, h)[0]
    slope_fit = CoxPHFitter().fit(pd.DataFrame({'lp': lp(cph, X), 'T': t, 'E': e}), 'T', 'E').summary.loc['lp']
    o_all, o_lo, o_hi = km_risk(t, e, h)
    return pd.DataFrame(rows), {'O/E': o_all / pred.mean(), 'O/E lo': o_lo / pred.mean(), 'O/E hi': o_hi / pred.mean(), 'observed 5-y risk': o_all, 'mean predicted 5-y risk': pred.mean(),
                                'slope': slope_fit['coef'], 'slope_lo': slope_fit['coef lower 95%'], 'slope_hi': slope_fit['coef upper 95%']}, pred
def evaluate(name, cph, data, cut):
    t, e = data.duration.values, data.event.values.astype(int)
    r = lp(cph, data); c = concordance_index(t, -r, e); clo, chi = boot_c(t, r, e)
    out = {'Model': name, 'n': len(data), 'deaths': int(e.sum()), 'C-index': c, 'C lo': clo, 'C hi': chi}
    for h in (36, 60):
        a, ncase, nev = td_auc(t, r, e, h); alo, ahi = boot_auc(t, r, e, h)
        out.update({f'AUC {h // 12}y': a, f'AUC {h // 12}y lo': alo, f'AUC {h // 12}y hi': ahi, f'cases {h // 12}y': ncase, f'evaluable {h // 12}y': nev})
    cal, cs, pred = calibration(cph, data, t, e)
    out.update(cs)
    ph = cph.predict_partial_hazard(data[cph.params_.index]).values; hi_risk = ph > cut
    lr = logrank_test(t[hi_risk], t[~hi_risk], e[hi_risk], e[~hi_risk])
    out.update({'cut-off': cut, 'high n': int(hi_risk.sum()), 'high deaths': int(e[hi_risk].sum()), 'low n': int((~hi_risk).sum()),
                'low deaths': int(e[~hi_risk].sum()), 'log-rank P': lr.p_value})
    return out, cal, r, hi_risk, pred
def paired(data, ra, rb, nb=NB):
    t, e = data.duration.values, data.event.values.astype(int); n = len(t); d = []
    for _ in range(nb):
        b = RNG.choice(n, n, replace=True)
        if e[b].sum() >= 2: d.append(concordance_index(t[b], -ra[b], e[b]) - concordance_index(t[b], -rb[b], e[b]))
    d = np.array(d); return d.mean(), min(1.0, max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1.0 / nb))


# ------------------------------------------------------------------ histological grade
def reviewed_grade(path):
    """Nottingham grade from the expert-panel component scores (tubules, nuclear pleomorphism, mitoses; 1-3 each)."""
    if not path or not os.path.exists(path): return None
    x = pd.read_excel(path, sheet_name=0) if path.endswith('xlsx') else pd.read_csv(path)
    cols = {c.lower(): c for c in x.columns}
    pick = lambda *keys: next(cols[c] for c in cols if all(k in c for k in keys))
    idc, tub, nuc, mit = pick('patient') if any('patient' in c for c in cols) else x.columns[0], pick('tubul'), pick('pleomorph'), pick('mitos')
    g = x[[idc, tub, nuc, mit]].copy(); g.columns = ['id', 'tub', 'nuc', 'mit']
    for c in ['tub', 'nuc', 'mit']:                  # cells read e.g. '(score = 3) <10%'
        g[c] = pd.to_numeric(g[c].astype(str).str.extract(r'score\s*=\s*(\d)', expand=False), errors='coerce')
    g = g.dropna(); g = g[g[['tub', 'nuc', 'mit']].isin([1, 2, 3]).all(axis=1)]
    g['id'] = g['id'].astype(str).str[:12]
    total = g[['tub', 'nuc', 'mit']].sum(axis=1)
    g['grade'] = np.where(total <= 5, 1, np.where(total <= 7, 2, 3))
    return g.groupby('id')['grade'].max()            # several slides per patient: highest grade, as in the development cohort
GRADE_FILE = os.environ.get('TCGA_GRADE_FILE', '')
rg = reviewed_grade(GRADE_FILE)
if rg is not None:
    df['grade3_reviewed'] = df.index.map(lambda b: np.nan if b not in rg.index else float(rg[b] == 3))
# development-data imputation of grade 3 (sensitivity analysis / fallback)
from sklearn.linear_model import LogisticRegression
IMP = ['age', 'AJCC', 'ER', 'PR', 'HER2pos']
imp = LogisticRegression(penalty=None, max_iter=5000).fit(g_tr[IMP].values, g_tr['grade3'].values)
ok = df[IMP].notna().all(axis=1)
df['grade3_imputed'] = np.nan
df.loc[ok, 'grade3_imputed'] = imp.predict_proba(df.loc[ok, IMP].values)[:, 1]

# ------------------------------------------------------------------ run
if __name__ == '__main__' and os.environ.get('RUN_EXTERNAL', '1') == '1':
    os.makedirs(OUT, exist_ok=True)
    rows, cals, keep = [], {}, {}
    dc = df.dropna(subset=CLIN)
    r, cal, lp_c, hi_c, pr_c = evaluate('Clinical (deployed: age, N, AJCC stage)', cph_clin, dc, CUT_CLIN)
    rows.append(r); cals['Clinical'] = cal; keep['Clinical'] = (dc, lp_c, hi_c, pr_c)
    variants = [('reviewed', 'grade3_reviewed', 'Clinical-guideline (expert-panel grade)')] if 'grade3_reviewed' in df else []
    variants.append(('imputed', 'grade3_imputed', 'Clinical-guideline (grade imputed from development data)'))
    for tag, col, name in variants:
        dg = df.dropna(subset=['age', 'AJCC', 'ER', 'PR', 'HER2pos', col]).copy(); dg['grade3'] = dg[col]
        r, cal, lp_g, hi_g, pr_g = evaluate(name, cph_g, dg, CUT_G)
        common = dg.dropna(subset=CLIN)
        dlt, p = paired(common, lp(cph_g, common), lp(cph_clin, common))
        r.update({'paired n': len(common), 'delta C vs Clinical': dlt, 'paired P': p,
                  'paired C Clinical': concordance_index(common.duration, -lp(cph_clin, common), common.event),
                  'paired C guideline': concordance_index(common.duration, -lp(cph_g, common), common.event),
                  'excluded n': len(df) - len(dg), 'excluded deaths': int(df.drop(dg.index).event.sum())})
        rows.append(r); cals[tag] = cal; keep[tag] = (dg, lp_g, hi_g, pr_g)
    res = pd.DataFrame(rows); res.to_csv(f'{OUT}/Table_external_TCGA.csv', index=False, encoding='utf-8-sig')
    # Supplementary Table S9: case-mix of the development cohort and the TCGA-BRCA validation cohort (descriptive only)
    pct = lambda x: f'{int(x.sum())} ({100 * x.mean():.1f})'
    def column(d, grade_col, tcol):
        out = {'Patients': f'{len(d)}', 'Age, years (mean ± SD)': f'{d.age.mean():.1f} ± {d.age.std():.1f}'}
        for k, lab in [(1, 'T1'), (2, 'T2'), (3, 'T3'), (4, 'T4')]: out[f'  {lab}'] = pct(d[tcol].dropna() == k)
        for k in range(4): out[f'  N{k}'] = pct(d.N.dropna() == k)
        for k, lab in [(1, 'I'), (2, 'II'), (3, 'III')]: out[f'  Stage {lab}'] = pct(d.AJCC.dropna() == k)
        for v, lab in [('ER', 'ER positive'), ('PR', 'PR positive'), ('HER2pos', 'HER2 positive')]: out[lab] = pct(d[v].dropna() == 1)
        out['Grade 3'] = pct(d[grade_col].dropna() == 1)
        out['Deaths'] = pct(d.event == 1)
        out['Median follow-up, months (reverse Kaplan–Meier)'] = f'{KaplanMeierFitter().fit(d.duration, 1 - d.event).median_survival_time_:.1f}'
        for v in ['N', 'ER', 'PR', 'HER2pos', grade_col]: out[f'Missing {v.replace("HER2pos", "HER2").replace(grade_col, "grade")}'] = f'{int(d[v].isna().sum())}'
        return out
    dev = cl.copy(); dev['T_cat'] = dev['T']
    pt = pat.loc[df.index, 'ajcc_tumor_pathologic_pt'].fillna('').str.extract(r'^T([0-4])', expand=False)
    tc = df.copy(); tc['T_cat'] = pd.to_numeric(pt, errors='coerce')
    s9 = pd.DataFrame({'Development cohort (TMA)': column(dev, 'grade3', 'T_cat'),
                       'TCGA-BRCA': column(tc, 'grade3_reviewed' if 'grade3_reviewed' in tc else 'grade3_imputed', 'T_cat')})
    s9.loc['Period of surgery / diagnosis'] = ['2001–2003 and 2005–2007 (surgery)',
        f"{int(pd.to_numeric(pat.loc[df.index, 'initial_pathologic_dx_year'], errors='coerce').min())}–{int(pd.to_numeric(pat.loc[df.index, 'initial_pathologic_dx_year'], errors='coerce').max())} (diagnosis)"]
    ed = pat.loc[df.index, 'ajcc_staging_edition'].value_counts()
    s9.loc['AJCC edition'] = ['6th', '; '.join(f"{k.replace('[Not Available]', 'not recorded')} {v}" for k, v in ed.items())]
    s9.loc['Grade source'] = ['Pathology report (Nottingham)', 'Central expert review (Nottingham components)']
    s9.loc['Receptor status source'] = ['TMA IHC/FISH records and pathology reports', 'Reported by contributing sites']
    s9.index.name = 'Characteristic'; s9.to_csv(f'{OUT}/Table_S9_casemix.csv', encoding='utf-8-sig')

    for k, c in cals.items(): c.to_csv(f'{OUT}/calibration_5y_{k}.csv', index=False)
    pd.DataFrame(flow, columns=['step', 'n']).to_csv(f'{OUT}/cohort_flow.csv', index=False)
    per = df[['age', 'AJCC', 'N', 'ER', 'PR', 'HER2pos'] + [c for c in ['grade3_reviewed', 'grade3_imputed'] if c in df] + ['duration', 'event']].copy()
    per['lp_clinical'] = np.nan; per.loc[dc.index, 'lp_clinical'] = lp_c
    for tag in [t for t, _, _ in variants]:
        dg, lp_g = keep[tag][0], keep[tag][1]
        per[f'lp_guideline_{tag}'] = np.nan; per.loc[dg.index, f'lp_guideline_{tag}'] = lp_g
    per.index.name = 'bcr_patient_barcode'
    per.to_csv(f'{OUT}/tcga_brca_external_cohort.csv', encoding='utf-8-sig')
    json.dump({'clinical_coef': cph_clin.params_.to_dict(), 'guideline_coef': cph_g.params_.to_dict(), 'cut_clinical': CUT_CLIN,
               'cut_guideline_training_median': CUT_G, 'grade_file': GRADE_FILE or None,
               'imputation_model': dict(zip(['intercept'] + IMP, [float(imp.intercept_[0])] + [float(v) for v in imp.coef_[0]]))},
              open(f'{OUT}/frozen_models.json', 'w'), indent=1)
    with pd.option_context('display.width', 250, 'display.max_columns', 40):
        print(res.round(3).T.to_string())
    for lab, n in flow: print(f'{n:5d}  {lab}')
    import pickle; pickle.dump({'keep': keep, 'cals': cals, 'res': res, 'variants': variants}, open(f'{OUT}/_results.pkl', 'wb'))
