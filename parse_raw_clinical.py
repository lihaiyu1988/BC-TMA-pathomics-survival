# -*- coding: utf-8 -*-
"""Parse the 7 TMA pathology Excel sheets into one raw per-patient table (no recoding yet)."""
import os, re, sys, pandas as pd, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
ROOT = "乳腺癌HE_生存期"
OUT = "analysis_outputs"
FILES = [
 ("A_140Sur01", "HBre-Duc140Sur-01 生存期/生存期乳腺癌HBre-Duc140Sur-01添加数据ER和PR和Her2和EGFR和Ki67和p53和AR和CK56.xls"),
 ("A_170Sur01", "HBre-Duc170Sur-01 生存期/生存期乳腺癌HBre-Duc170Sur-01详细病理资料 添加数据ER和PR和Her2和EGFR和Ki67和p53和AR和CK56.xls"),
 ("A_D145Su01", "HBreD145Su01  生存期/生存期乳腺癌HBreD145Su01详细病理资料  添加数据ER和PR和Her2和EGFR和Ki67和p53和AR和CK56.xls"),
 ("A_D129Su01", "生存期乳腺癌HBreD129Su01/生存期乳腺癌HBreD129Su01详细病理资料及阵列表--2023.07.25.xlsx"),
 ("B_D130Su09", "生存期乳腺癌HBreD130Su09详细病理资料及阵列排布表 添加了FISH数据HER2和PDL1和CD8数据-官网【2023.03.15】(1).xls"),
 ("B_D140Su03", "HBreD140Su03 生存期/HBreD140Su03/生存期乳腺癌HBreD140Su03详细病理资料 添加了FISH数据HER2和PDL1和CD8数据.xls"),
 ("B_D132Su07", "生存期乳腺癌HBreD132Su07 添加了FISH数据HER2 - 及阵列排布/生存期乳腺癌HBreD132Su07 添加了FISH数据HER2 - 及阵列排布.xls"),
]
def dedupe(cols):
    seen, out = {}, []
    for c in cols:
        c = str(c).strip()
        if c in seen: seen[c] += 1; out.append(f"{c}__{seen[c]}")
        else: seen[c] = 0; out.append(c)
    return out
def first_col(df, *cands, contains=None):
    for c in cands:
        if c in df.columns: return c
    if contains:
        for c in df.columns:
            if all(k in c for k in contains): return c
    return None
rows = []
for tag, rel in FILES:
    fp = os.path.join(ROOT, rel)
    xl = pd.ExcelFile(fp)
    sh = xl.sheet_names[0]
    df = xl.parse(sh, header=None)
    df.columns = dedupe(df.iloc[0].tolist()); df = df.iloc[1:].reset_index(drop=True)
    idc = first_col(df, '组织编码', '病例编码')
    df['ID'] = df[idc].astype(str).str.strip()
    df = df[df['ID'].str.match(r'^J07A\d{4}$')]
    if '组织类型' in df.columns:
        df = df[df['组织类型'].astype(str).str.strip() != '癌旁']
    print(tag, sh, 'rows', len(df), 'unique IDs', df.ID.nunique())
    for _, r in df.iterrows():
        d = {'file': tag, 'ID': r['ID']}
        def g(*cands, contains=None):
            c = first_col(df, *cands, contains=contains)
            return r[c] if c is not None else np.nan
        d['age'] = g('年龄'); d['status'] = g('生存状态')
        d['os_months'] = g('2013生存期', '生存期201601', '总生存期201601', '201307生存期')
        d['os_months_2014'] = g('2014生存期', '201407生存期')
        d['grade'] = g('病理分级'); d['histology'] = g('病理分型', '病理类型')
        d['size'] = g('肿瘤大小CM', '肿瘤大小cm', '肿瘤大小')
        d['T'] = g('T'); d['N'] = g('N'); d['M'] = g('M'); d['AJCC'] = g('AJCC第六版临床分期')
        d['ln_total'] = g('淋巴结总数'); d['ln_pos'] = g('阳性数')
        d['LVI'] = g('脉管侵犯')
        # ER / PR / HER2
        d['ER_ihc'] = g('癌组织ER（IHC数据）', 'ER'); d['ER_ihc2'] = g('癌组织ER（IHC数据）__1')
        d['PR_ihc'] = g('癌组织PR（IHC数据）', 'PR'); d['PR_ihc2'] = g('癌组织PR（IHC数据）__1')
        d['HER2_ihc'] = g('癌组织HER2（IHC数据）', 'Her-2'); d['HER2_ihc2'] = g('癌组织HER2（IHC数据）__1')
        d['HER2_fish'] = g('癌组织HER2（FISH数据）', contains=['FISH'])
        d['HER2_fish2'] = g('癌组织HER2（FISH数据）__1', 'HBreD140Su03的FISH数据HER2')
        d['ER_score'] = g(contains=['ER核数据']); d['ER_frac'] = g(contains=['ER核数据__1'])
        d['PR_score'] = g(contains=['PR核数据']); d['PR_frac'] = g(contains=['PR核数据__1'])
        d['HER2_score'] = g(contains=['HER2膜']); d['HER2_frac'] = g(contains=['HER2膜数据__1']) if first_col(df, contains=['HER2膜数据__1']) else g(contains=['HER2膜__1'])
        d['ki67_score'] = g(contains=['ki67核数据']); d['ki67_frac'] = g(contains=['ki67核数据__1'])
        d['p53_score'] = g(contains=['p53核数据']); d['AR_score'] = g(contains=['AR核数据'])
        d['recur'] = g('复发情况'); d['recur_time'] = g('复发时间'); d['dfs_months'] = g('复发生存期201601', '无病生存期201601')
        d['dfs_status'] = g('无病生存状态')
        d['qc1'] = g(contains=['质检']); 
        rows.append(d)
raw = pd.DataFrame(rows)
raw.to_csv(f"{OUT}/raw_clinical_long.csv", index=False, encoding='utf-8-sig')
print('long table', raw.shape, 'unique IDs', raw.ID.nunique())
sur = pd.read_csv('pipeline_outputs/sur.csv')
print('IDs in sur.csv covered:', sur.ID.isin(raw.ID).sum(), '/', len(sur))
ex = ['J07A0946','J07A0980','J07A0987','J07A0999','J07A1023','J07A1076','J07A1098','J07A3073','J07A3081','J07A3093','J07A3112','J07A3159','J07A3171']
print('extra13 covered:', [i for i in ex if i in set(raw.ID)])
# value sets for mapping design
for c in ['grade','T','N','M','AJCC','ER_ihc','ER_ihc2','PR_ihc','PR_ihc2','HER2_ihc','HER2_ihc2','HER2_fish','HER2_fish2','ER_score','PR_score','HER2_score','ki67_score','recur','status','LVI','qc1']:
    vc = raw[c].astype(str).str.strip().value_counts()
    print(f"--- {c}: {len(vc)} values:", dict(list(vc.items())[:60]))
