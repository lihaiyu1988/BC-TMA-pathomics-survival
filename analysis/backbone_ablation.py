# -*- coding: utf-8 -*-
"""Backbone sensitivity (ResNet18/ResNet50/DenseNet121/CrossFormer), PLH/BoW ablation at matched complexity,
IDF-on-train-only sensitivity, leave-one-out, and Combined (late-fusion) models for every backbone.
All analyses use the 274 patients with saved patch predictions for all four backbones (193 train / 81 test)."""
import sys, warnings, json, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'analysis_outputs')
from pathomics_pipeline import aggregate, load_backbone, run_pipeline, SUR, DATA
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

OUT = 'analysis_outputs'
RNG = np.random.default_rng(42)

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

clin_tr = pd.read_csv(f'{DATA}/results/Clinical_cox_predictions_train.csv').set_index('ID')['expectation']
clin_te = pd.read_csv(f'{DATA}/results/Clinical_cox_predictions_test.csv').set_index('ID')['expectation']

def combined_model(res):
    """Late fusion exactly as deployed: Cox on (Clinical expectation, Pathomics expectation)."""
    cph = res['cph']; sel = list(cph.params_.index)
    feat = res['feat']
    tr_ids = res['risk_train'].index; te_ids = res['risk_test'].index
    exp_tr = cph.predict_expectation(feat.loc[tr_ids, sel]); exp_te = cph.predict_expectation(feat.loc[te_ids, sel])
    dtr = pd.DataFrame({'Clinical': clin_tr.loc[tr_ids], 'Pathomics': exp_tr.values, 'duration': SUR.loc[tr_ids, 'duration'], 'event': SUR.loc[tr_ids, 'event']})
    dte = pd.DataFrame({'Clinical': clin_te.loc[te_ids], 'Pathomics': exp_te.values, 'duration': SUR.loc[te_ids, 'duration'], 'event': SUR.loc[te_ids, 'event']})
    c2 = CoxPHFitter(penalizer=0.0).fit(dtr, 'duration', 'event')
    rtr = c2.predict_partial_hazard(dtr); rte = c2.predict_partial_hazard(dte)
    return (concordance_index(dtr.duration, -rtr.values, dtr.event), concordance_index(dte.duration, -rte.values, dte.event),
            pd.Series(rte.values, index=te_ids), pd.Series(rtr.values, index=tr_ids))

rows, risks_test = [], {}
feats = {}
for model, disp in [('resnet50', 'ResNet50'), ('resnet18', 'ResNet18'), ('densenet121', 'DenseNet121'), ('CrossFormer', 'CrossFormer')]:
    patch = load_backbone(model)
    train_ids = SUR.index[SUR.group == 'train']
    feats[disp] = {'allIDF': aggregate(patch), 'trainIDF': aggregate(patch, idf_fit_ids=train_ids)}
    for idf in ['allIDF', 'trainIDF']:
        if idf == 'trainIDF' and disp != 'ResNet50':
            continue
        f = feats[disp][idf]
        for mode, fk in [('k=3', 3), ('CV', None)]:
            for branch, bname in [(None, 'PLH+BoW'), ('PLH_', 'PLH only'), ('BoW_', 'BoW only')]:
                if mode == 'CV' and branch is not None and disp != 'ResNet50':
                    continue
                tag = f'{disp} | {bname} | {mode} | {idf}'
                res = run_pipeline(f, tag, branch=branch, force_k=fk)
                if not res['selected']:
                    continue
                res['feat'] = f
                # normalise features stored in cph are z-scored; recompute z-scoring for expectation
                sel = res['selected']
                tr = f.loc[res['risk_train'].index, sel]; mu, sd = tr.mean(), tr.std()
                res['feat'] = (f[sel] - mu) / sd
                te = SUR.loc[res['risk_test'].index]
                lo, hi = boot_ci(te.duration.values, res['risk_test'].values, te.event.values)
                ctr_c, cte_c, rte_c, rtr_c = combined_model(res)
                lo2, hi2 = boot_ci(te.duration.values, rte_c.values, te.event.values)
                rows.append({'Backbone': disp, 'Branch': bname, 'Selection': mode, 'IDF': idf, 'n_after_corr': res['n_corr'], 'n_after_uni': res['n_uni'],
                             'n_selected': len(sel), 'Selected features': ', '.join(sel),
                             'Pathomics C train': round(res['C_train'], 3), 'Pathomics C test': round(res['C_test'], 3), 'Pathomics test 95% CI': f'{lo:.3f}-{hi:.3f}',
                             'Combined C train': round(ctr_c, 3), 'Combined C test': round(cte_c, 3), 'Combined test 95% CI': f'{lo2:.3f}-{hi2:.3f}'})
                risks_test[tag] = (res['risk_test'], rte_c)
tab = pd.DataFrame(rows)
tab.to_csv(f'{OUT}/Table_backbone_ablation.csv', index=False, encoding='utf-8-sig')
print(tab.drop(columns=['Selected features']).to_string())
print(tab[['Backbone', 'Branch', 'Selection', 'IDF', 'Selected features']].to_string())

# pairwise paired-bootstrap comparisons of test C-index vs the deployed ResNet50 k=3 signature (274-patient version)
ref = 'ResNet50 | PLH+BoW | k=3 | allIDF'
te = SUR.loc[risks_test[ref][0].index]
cmp_rows = []
for tag, (rp, rc) in risks_test.items():
    if tag == ref: continue
    ids = rp.index.intersection(te.index)
    p, md = paired_p(te.loc[ids].duration.values, rp.loc[ids].values, risks_test[ref][0].loc[ids].values, te.loc[ids].event.values)
    cmp_rows.append({'Comparison': f'{tag} vs {ref}', 'delta C (pathomics, test)': round(md, 3), 'p': round(p, 3)})
cmp = pd.DataFrame(cmp_rows); cmp.to_csv(f'{OUT}/Table_backbone_pairwise.csv', index=False)
print(cmp.to_string())

# leave-one-out on the deployed 3-feature signature (278-patient saved features) and IDF-on-train refit
ptr = pd.read_csv(f'{DATA}/features/Pathomics_train_cox.csv'); pte = pd.read_csv(f'{DATA}/features/Pathomics_test_cox.csv')
final = ['prob058', 'prob05', 'pred1']; loo = []
for drop in [None] + final:
    use = [f for f in final if f != drop]
    cph = CoxPHFitter(penalizer=0.0).fit(ptr[use + ['duration', 'event']], 'duration', 'event')
    rte = cph.predict_partial_hazard(pte[use]); lo, hi = boot_ci(pte.duration.values, rte.values, pte.event.values)
    loo.append({'Configuration': 'Full signature (BoW_prob_0.50 + BoW_prob_0.58 + BoW_pred_1)' if drop is None else f'Remove {drop}',
                'C train': round(cph.concordance_index_, 3), 'C test': round(concordance_index(pte.duration, -rte.values, pte.event), 3), 'test 95% CI': f'{lo:.3f}-{hi:.3f}'})
loo = pd.DataFrame(loo); loo.to_csv(f'{OUT}/Table_LOO.csv', index=False); print(loo.to_string())
json.dump({k: {'path': v[0].round(6).to_dict(), 'comb': v[1].round(6).to_dict()} for k, v in risks_test.items()}, open(f'{OUT}/backbone_test_risks.json', 'w'))
print('DONE')
