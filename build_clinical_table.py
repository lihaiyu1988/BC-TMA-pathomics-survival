# -*- coding: utf-8 -*-
"""Harmonize the raw TMA pathology sheets into guideline-level clinical variables for the 278 study patients."""
import re, sys, pandas as pd, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
OUT = "analysis_outputs"
raw = pd.read_csv(f"{OUT}/raw_clinical_long.csv", dtype=str)
raw = raw.replace({'nan': np.nan, '——': np.nan, '': np.nan})
S = lambda x: None if pd.isna(x) else str(x).strip()

# ---------- element-level decoders ----------
def dec_grade(v):
    v = S(v)
    if v is None: return np.nan
    v = v.replace('级', '').replace('和Ⅱ', '').strip()
    m = {'Ⅰ': 1, 'Ⅰ-Ⅱ': 2, 'Ⅱ': 2, 'Ⅱ-Ⅲ': 3, 'Ⅲ': 3, 'Ⅰ-Ⅲ': 3, 'G2': 2, 'G3': 3}
    return m.get(v, np.nan)          # mixed grades -> highest component
def dec_TN(v):
    v = S(v)
    if v is None: return np.nan
    m = re.search(r'(\d)', v); return int(m.group(1)) if m else np.nan
def dec_ajcc(v):
    v = S(v)
    if v is None: return np.nan
    m = re.match(r'(\d)', v); return int(m.group(1)) if m else np.nan
def dec_size(v):
    v = S(v)
    if v is None: return np.nan
    nums = [float(x) for x in re.findall(r'\d+\.?\d*', v.replace('×', 'x').replace('X', 'x'))]
    return max(nums) if nums else np.nan
def dec_posneg_cn(v):        # 阳性 / 阴性 / 个别阳性 (set A categorical IHC)
    v = S(v)
    if v is None: return np.nan
    if v.startswith('阳') or v == '个别阳性': return 1
    if v.startswith('阴'): return 0
    return np.nan            # 脱片 / 组织折叠 / 肿瘤较少 / 未读片 -> missing
def dec_ihc_text(v):         # set B free text like （90%++）, （-）, 弱（+）, （+/-）
    v = S(v)
    if v is None: return np.nan
    if '脱片' in v or '肿瘤较少' in v: return np.nan
    if '+' in v: return 1    # any positive staining (>=1% convention); '+/-' treated as positive-low
    if '-' in v or '−' in v: return 0
    return np.nan
def dec_ihc_text_her2(v):    # HER2 IHC text -> 3 = +++, 2 = ++, 1 = +, 0 = -
    v = S(v)
    if v is None: return np.nan
    if '脱片' in v: return np.nan
    if '+++' in v: return 3
    if '++' in v: return 2
    if '+' in v: return 1
    if '-' in v: return 0
    return np.nan
def dec_fish(v):
    v = S(v)
    if v is None: return np.nan
    if v.startswith('阳'): return 1
    if v.startswith('阴'): return 0
    return np.nan
def dec_score(v):            # '1-2' -> 2 (higher), '0' -> 0
    v = S(v)
    if v is None: return np.nan
    nums = re.findall(r'\d', v); return max(int(x) for x in nums) if nums else np.nan
def dec_frac(v):
    v = S(v)
    if v is None: return np.nan
    if v == '个别': return 0.005
    v = v.replace('%', '')
    m = re.search(r'<\s*(\d+\.?\d*)', v)
    if m: return float(m.group(1)) / 100 / 2      # '<5%' -> 2.5%
    try:
        x = float(v); return x / 100 if x > 1 else x
    except Exception:
        return np.nan

# ---------- per-row decoded variables ----------
d = pd.DataFrame({'file': raw.file, 'ID': raw.ID})
d['age'] = pd.to_numeric(raw.age, errors='coerce')
d['grade'] = raw.grade.map(dec_grade)
d['T'] = raw['T'].map(dec_TN); d['N'] = raw['N'].map(dec_TN); d['AJCC'] = raw.AJCC.map(dec_ajcc)
d['size_cm'] = raw['size'].map(dec_size)
d['ln_total'] = pd.to_numeric(raw.ln_total, errors='coerce'); d['ln_pos'] = pd.to_numeric(raw.ln_pos, errors='coerce')
setA = raw.file.str.startswith('A_')
d['ER_a1'] = np.where(setA, raw.ER_ihc.map(dec_posneg_cn), np.nan)
d['ER_a2'] = raw.ER_ihc2.map(dec_posneg_cn)
d['ER_b'] = np.where(~setA, raw.ER_ihc.map(dec_ihc_text), np.nan)
d['ER_frac'] = raw.ER_frac.map(dec_frac); d['ER_score'] = raw.ER_score.map(dec_score)
d['PR_a1'] = np.where(setA, raw.PR_ihc.map(dec_posneg_cn), np.nan)
d['PR_a2'] = raw.PR_ihc2.map(dec_posneg_cn)
d['PR_b'] = np.where(~setA, raw.PR_ihc.map(dec_ihc_text), np.nan)
d['PR_frac'] = raw.PR_frac.map(dec_frac); d['PR_score'] = raw.PR_score.map(dec_score)
d['HER2_fish1'] = raw.HER2_fish.map(dec_fish); d['HER2_fish2'] = raw.HER2_fish2.map(dec_fish)
d['HER2_ihc_a1'] = np.where(setA, raw.HER2_ihc.map(dec_posneg_cn), np.nan)
d['HER2_ihc_a2'] = raw.HER2_ihc2.map(dec_posneg_cn)
d['HER2_ihc_b'] = np.where(~setA, raw.HER2_ihc.map(dec_ihc_text_her2), np.nan)
d['HER2_score'] = raw.HER2_score.map(dec_score)
d['ki67_frac'] = raw.ki67_frac.map(dec_frac)
d['recur'] = raw.recur.map(lambda v: np.nan if S(v) is None else (1 if S(v) == '复发' else 0))
d['status'] = raw.status.map(lambda v: np.nan if S(v) is None else (1 if S(v) == '死亡' else 0))
d['os_months'] = pd.to_numeric(raw.os_months, errors='coerce')
d['histology'] = raw.histology

# ---------- collapse to one row per patient ----------
def agg_first(s):
    s = s.dropna(); return s.iloc[0] if len(s) else np.nan
def agg_any(s):
    s = s.dropna()
    if not len(s): return np.nan
    return 1 if (s == 1).any() else 0
def agg_max(s):
    s = s.dropna(); return s.max() if len(s) else np.nan
g = d.groupby('ID')
pt = pd.DataFrame({
    'age': g.age.agg(agg_first), 'grade': g.grade.agg(agg_max), 'T': g['T'].agg(agg_first), 'N': g['N'].agg(agg_first),
    'AJCC': g.AJCC.agg(agg_first), 'size_cm': g.size_cm.agg(agg_first), 'ln_total': g.ln_total.agg(agg_first), 'ln_pos': g.ln_pos.agg(agg_first),
    'ER_a1': g.ER_a1.agg(agg_first), 'ER_a2': g.ER_a2.agg(agg_first), 'ER_b': g.ER_b.agg(agg_first), 'ER_frac': g.ER_frac.agg(agg_first), 'ER_score': g.ER_score.agg(agg_first),
    'PR_a1': g.PR_a1.agg(agg_first), 'PR_a2': g.PR_a2.agg(agg_first), 'PR_b': g.PR_b.agg(agg_first), 'PR_frac': g.PR_frac.agg(agg_first), 'PR_score': g.PR_score.agg(agg_first),
    'HER2_fish': g[['HER2_fish1', 'HER2_fish2']].apply(lambda x: agg_any(pd.concat([x.HER2_fish1, x.HER2_fish2]))),
    'HER2_ihc_a': g[['HER2_ihc_a1', 'HER2_ihc_a2']].apply(lambda x: agg_any(pd.concat([x.HER2_ihc_a1, x.HER2_ihc_a2]))),
    'HER2_ihc_b': g.HER2_ihc_b.agg(agg_max), 'HER2_score': g.HER2_score.agg(agg_max),
    'ki67_frac': g.ki67_frac.agg(agg_first), 'recur': g.recur.agg(agg_first), 'status_raw': g.status.agg(agg_first), 'os_raw': g.os_months.agg(agg_first),
    'histology': g.histology.agg(agg_first), 'n_files': g.size(),
})
pt['set'] = np.where(pt.index.str.startswith('J07A3'), 'B', 'A')
def hr_final(r, p):
    if r['set'] == 'A':
        for k in [f'{p}_a1', f'{p}_a2']:
            if not pd.isna(r[k]): return r[k]
        if not pd.isna(r[f'{p}_frac']): return 1 if r[f'{p}_frac'] >= 0.01 else 0
        return np.nan
    return r[f'{p}_b']
pt['ER'] = pt.apply(lambda r: hr_final(r, 'ER'), axis=1)
pt['PR'] = pt.apply(lambda r: hr_final(r, 'PR'), axis=1)
def her2_final(r):
    if not pd.isna(r.HER2_fish): return r.HER2_fish
    if r['set'] == 'B':
        if not pd.isna(r.HER2_ihc_b): return 1 if r.HER2_ihc_b == 3 else 0
        return np.nan
    if not pd.isna(r.HER2_score): return 1 if r.HER2_score == 3 else 0
    if not pd.isna(r.HER2_ihc_a): return r.HER2_ihc_a
    return np.nan
pt['HER2'] = pt.apply(her2_final, axis=1)
pt['HR_pos'] = np.where(pt.ER.isna() & pt.PR.isna(), np.nan, ((pt.ER == 1) | (pt.PR == 1)).astype(float))
def subtype(r):
    if pd.isna(r.HR_pos) or pd.isna(r.HER2): return np.nan
    if r.HR_pos == 1 and r.HER2 == 0: return 'HR+/HER2-'
    if r.HR_pos == 1 and r.HER2 == 1: return 'HR+/HER2+'
    if r.HR_pos == 0 and r.HER2 == 1: return 'HR-/HER2+'
    return 'TNBC'
pt['subtype'] = pt.apply(subtype, axis=1)
pt = pt.reset_index()
pt.to_csv(f"{OUT}/clinical_guideline_all.csv", index=False, encoding='utf-8-sig')

# ---------- cross-check against the study file ----------
cl = pd.read_csv('pipeline_outputs/data/clinical.csv')
m = cl.merge(pt, on='ID', how='left', suffixes=('', '_raw'))
print('study patients', len(m), 'matched', m.age_raw.notna().sum())
for a, b in [('age', 'age_raw'), ('T', 'T_raw'), ('N', 'N_raw'), ('AJCC', 'AJCC_raw'), ('number_of_lymph_nodes', 'ln_total'), ('positive_number', 'ln_pos')]:
    ok = (m[a] == m[b]); print(f"  {a} vs raw: agree {ok.sum()}/{m[b].notna().sum()} (raw missing {m[b].isna().sum()})")
print('  study HER2 (0=positive) vs raw HER2 (1=positive):'); print(pd.crosstab(m.HER2, m.HER2_raw, dropna=False))
print('  study M vs set:'); print(pd.crosstab(m.M, m['set']))
print('Missingness among 278:', m[['grade', 'size_cm', 'ER', 'PR', 'HER2_raw', 'ki67_frac', 'subtype']].isna().sum().to_dict())
print('Missingness by set:'); print(m.groupby('set')[['grade', 'size_cm', 'ER', 'PR', 'HER2_raw', 'ki67_frac']].apply(lambda x: x.isna().sum()))
print('Distributions (278):')
for c in ['grade', 'ER', 'PR', 'HER2_raw', 'subtype', 'T_raw', 'N_raw', 'AJCC_raw']:
    print(' ', c, m[c].value_counts(dropna=False).to_dict())
print('size_cm:', m.size_cm.describe().round(2).to_dict())
print('ki67 (set A):', m.ki67_frac.describe().round(3).to_dict())
m['status_ok'] = (m.event == m.status_raw); print('status agree', m.status_ok.sum(), '/', m.status_raw.notna().sum())
print('duration vs raw os (abs diff <=1 mo):', ((m.duration - m.os_raw).abs() <= 1).sum(), '/', m.os_raw.notna().sum())
ex = ['J07A0946', 'J07A0980', 'J07A0987', 'J07A0999', 'J07A1023', 'J07A1076', 'J07A1098', 'J07A3073', 'J07A3081', 'J07A3093', 'J07A3112', 'J07A3159', 'J07A3171']
print('Excluded 13:'); print(pt[pt.ID.isin(ex)][['ID', 'set', 'age', 'T', 'N', 'AJCC', 'grade', 'ER', 'PR', 'HER2', 'status_raw', 'os_raw']].to_string())
m.to_csv(f"{OUT}/clinical_guideline_278.csv", index=False, encoding='utf-8-sig')
