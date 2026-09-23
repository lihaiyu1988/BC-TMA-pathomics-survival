# -*- coding: utf-8 -*-
"""Re-implementation of the patch-to-slide pathomics pipeline (PLH + BoW/TF-IDF -> z-score on train ->
correlation filter |r|>0.8 -> univariable Cox p<0.05 -> elastic-net Cox (l1_ratio 0.1, 10-fold CV) -> Cox PH).
Used for: backbone sensitivity (ResNet18/50, DenseNet121, CrossFormer), PLH/BoW ablation, and IDF-on-train check."""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import KFold
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

DATA = "pipeline_outputs"
SUR = pd.read_csv(f"{DATA}/sur.csv").set_index('ID')
BINS = [round(i/100, 2) for i in range(101)]

def aggregate(patch_df, idf_fit_ids=None):
    """patch_df: columns ID, prob (rounded 2 dp), pred (0/1). Returns 206-d slide features (PLH 103 + BoW 103)."""
    p = patch_df.copy(); p['prob'] = p['prob'].round(2)
    Cp = p.groupby(['ID', 'prob']).size().unstack(fill_value=0).reindex(columns=BINS, fill_value=0)
    Cl = p.groupby(['ID', 'pred']).size().unstack(fill_value=0).reindex(columns=[0, 1], fill_value=0)
    Hp = Cp.div(Cp.sum(1), axis=0); Hp.columns = [f'PLH_prob_{b:.2f}' for b in BINS]
    Hl = Cl.div(Cl.sum(1), axis=0); Hl.columns = ['PLH_pred_0', 'PLH_pred_1']
    fit_ids = Cp.index if idf_fit_ids is None else Cp.index.intersection(idf_fit_ids)
    tp = TfidfTransformer().fit(Cp.loc[fit_ids].values); tl = TfidfTransformer().fit(Cl.loc[fit_ids].values)
    Bp = pd.DataFrame(tp.transform(Cp.values).toarray(), index=Cp.index, columns=[f'BoW_prob_{b:.2f}' for b in BINS])
    Bl = pd.DataFrame(tl.transform(Cl.values).toarray(), index=Cl.index, columns=['BoW_pred_0', 'BoW_pred_1'])
    return pd.concat([Hp, Hl, Bp, Bl], axis=1)

def load_backbone(model):
    frames = []
    for c in ['train', 'test']:
        d = pd.read_csv(f"{DATA}/results/Pathomics_Slice_{model}_{c}.csv")
        frames.append(pd.DataFrame({'ID': d['ID'], 'prob': d['label-1'].round(2), 'pred': (d['label-1'] >= 0.5).astype(int)}))
    return pd.concat(frames, ignore_index=True)

def corr_filter(X, thr=0.8):
    corr = X.corr('pearson').abs(); keep = []
    for f in list(X.columns)[::-1]:        # same greedy order as the original implementation (later columns retained)
        if all(corr.loc[f, k] <= thr for k in keep): keep.append(f)
    return keep[::-1]

def uni_screen(df, feats, p=0.05):
    out = []
    for f in feats:
        try:
            c = CoxPHFitter(penalizer=0.0).fit(df[[f, 'duration', 'event']], 'duration', 'event')
            if c.summary['p'].iloc[0] < p: out.append(f)
        except Exception:
            pass
    return out

def coxnet_select(df, feats, l1_ratio=0.1, n_alphas=50, cv=10, seed=42, force_k=None):
    X = df[feats].values; y = Surv.from_arrays(df['event'].astype(bool).values, df['duration'].values)
    path = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alpha_min_ratio=0.01, n_alphas=n_alphas, max_iter=100000).fit(X, y)
    alphas = path.alphas_
    if force_k is not None:   # matched complexity: largest alpha giving exactly k non-zero coefficients
        for a, coef in zip(alphas, path.coef_.T):
            nz = [f for f, c in zip(feats, coef) if abs(c) > 1e-6]
            if len(nz) == force_k: return nz, a, None
        # fallback: closest to k
        best = min(zip(alphas, path.coef_.T), key=lambda t: abs(sum(abs(t[1]) > 1e-6) - force_k))
        return [f for f, c in zip(feats, best[1]) if abs(c) > 1e-6], best[0], None
    kf = KFold(cv, shuffle=True, random_state=seed); scores = np.zeros((cv, len(alphas)))
    for i, (tr, va) in enumerate(kf.split(X)):
        m = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alphas=alphas, max_iter=100000).fit(X[tr], y[tr])
        for j, a in enumerate(alphas):
            r = m.predict(X[va], alpha=a)
            scores[i, j] = concordance_index(df['duration'].values[va], -r, df['event'].values[va]) if np.ptp(r) > 0 else 0.5
    mean = scores.mean(0); j = int(np.argmax(mean))
    coef = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alphas=[alphas[j]], max_iter=100000).fit(X, y).coef_[:, 0]
    return [f for f, c in zip(feats, coef) if abs(c) > 1e-6], alphas[j], mean[j]

def run_pipeline(feat, tag, branch=None, force_k=None, seed=42, verbose=True):
    """feat: slide features indexed by ID (train+test). Returns dict with selected features and C-indices."""
    cols = [c for c in feat.columns if branch is None or c.startswith(branch)]
    df = feat[cols].join(SUR, how='inner')
    tr = df[df.group == 'train'].copy(); te = df[df.group == 'test'].copy()
    mu, sd = tr[cols].mean(), tr[cols].std().replace(0, np.nan)
    tr[cols] = (tr[cols] - mu) / sd; te[cols] = (te[cols] - mu) / sd
    cols = [c for c in cols if not tr[c].isna().any()]
    tr = tr.dropna(axis=1); te = te[tr.columns]
    k1 = corr_filter(tr[cols]); k2 = uni_screen(tr, k1)
    if len(k2) == 0:
        return {'tag': tag, 'n_corr': len(k1), 'n_uni': 0, 'selected': [], 'C_train': np.nan, 'C_test': np.nan}
    sel, alpha, cvscore = coxnet_select(tr, k2, force_k=force_k, seed=seed)
    if len(sel) == 0: sel = k2[:1]
    try:
        cph = CoxPHFitter(penalizer=0.0).fit(tr[sel + ['duration', 'event']], 'duration', 'event')
    except Exception:
        # perfectly collinear selections (e.g. the two predicted-label bins) -> light ridge penalty
        cph = CoxPHFitter(penalizer=0.01).fit(tr[sel + ['duration', 'event']], 'duration', 'event')
    r_tr = cph.predict_partial_hazard(tr[sel]); r_te = cph.predict_partial_hazard(te[sel])
    res = {'tag': tag, 'n_corr': len(k1), 'n_uni': len(k2), 'alpha': alpha, 'cv_C': cvscore, 'selected': sel,
           'C_train': concordance_index(tr.duration, -r_tr.values, tr.event), 'C_test': concordance_index(te.duration, -r_te.values, te.event),
           'risk_train': pd.Series(r_tr.values, index=tr.index), 'risk_test': pd.Series(r_te.values, index=te.index), 'cph': cph}
    if verbose:
        print(f"[{tag}] corr->{len(k1)} uni->{len(k2)} alpha={alpha:.4f} selected({len(sel)})={sel} C_train={res['C_train']:.3f} C_test={res['C_test']:.3f}")
    return res

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    # validation: reproduce saved ResNet50 features
    f50 = aggregate(load_backbone('resnet50'))
    saved = pd.read_csv(f"{DATA}/features/prob_tfidf.csv").set_index('ID')
    common = f50.index.intersection(saved.index); print('BoW reproduction (274 pts, IDF differs slightly from 291-pt run) max diff:', np.abs(f50.loc[common, 'BoW_prob_0.50'] - saved.loc[common, 'prob05']).max().round(4))
    print('deployed signature C-index (from saved files):')
    ptr = pd.read_csv(f"{DATA}/features/Pathomics_train_cox.csv"); pte = pd.read_csv(f"{DATA}/features/Pathomics_test_cox.csv")
    cph = CoxPHFitter().fit(ptr[['prob058','prob05','pred1','duration','event']], 'duration','event')
    print('  train', round(cph.concordance_index_,3), 'test', round(concordance_index(pte.duration, -cph.predict_partial_hazard(pte).values, pte.event),3))
    print('--- CV-selected alpha, ResNet50 (all patients IDF) ---')
    run_pipeline(f50, 'ResNet50 CV')
    print('--- matched complexity k=3 ---')
    run_pipeline(f50, 'ResNet50 k=3', force_k=3)
