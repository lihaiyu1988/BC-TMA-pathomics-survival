# -*- coding: utf-8 -*-
"""Regenerate all analysis figures in one consistent style. Every figure is drawn at its final print size by
fig_layout.Grid (width 15.0 cm, identical 0.3-cm gaps between panels horizontally and vertically, 0.5-cm outer
margin, aligned axes, one type scale and one set of line widths, bold letters at the top-left of every panel) and is
inserted into the manuscript at 100 %. Numbers are computed exactly as before (same random-number sequence).
Outputs -> analysis_outputs/figures/ (+ figure_layout_metrics.csv)"""
import os, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
import fig_layout as fl                      # sets the final-size type scale and line widths (rcParams)
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

DATA = 'pipeline_outputs'; OUT = 'analysis_outputs/figures'; os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(42)
COL = {'Clinical': '#1f77b4', 'Pathomics': '#ff7f0e', 'Combined': '#d62728',
       'ResNet18': '#2ca02c', 'ResNet50': '#d62728', 'DenseNet121': '#9467bd', 'CrossFormer': '#17becf'}
sur = pd.read_csv(f'{DATA}/sur.csv').set_index('ID')
join = pd.read_csv(f'{DATA}/results/joinit_info.csv').set_index('ID')
METRICS = []
TICKS01 = np.linspace(0, 1, 6)
pval = lambda p: r'$\it{P}$ < 0.001' if p < 0.001 else rf'$\it{{P}}$ = {p:.3f}'


def save(grid, name):
    path = f'{OUT}/{name}.png'; lay = grid.finalize(path)
    m = fl.measure_png(path, lay); m['figure'] = name; METRICS.append(m)
    print(f'  {name}: {lay["W"]:.2f} x {lay["H"]:.2f} cm, axes {lay["aw"]:.2f} cm wide')
    return path


def write_metrics():
    p = f'{OUT}/figure_layout_metrics.csv'; new = pd.DataFrame(METRICS)
    old = pd.read_csv(p) if os.path.exists(p) else new.iloc[:0]
    pd.concat([old[~old.figure.isin(new.figure)], new]).to_csv(p, index=False)      # one row per figure


def roc_axes(ax):
    ax.plot([0, 1], [0, 1], 'k--', lw=fl.LW_REF, alpha=0.6)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks(TICKS01); ax.set_yticks(TICKS01)
    ax.set_xlabel('1 − Specificity'); ax.set_ylabel('Sensitivity')


def km_panel(ax, t, e, hi, title, xticks):
    """Kaplan-Meier curves of the high- and low-risk groups with the log-rank P value and the number-at-risk table."""
    t = np.asarray(t); e = np.asarray(e).astype(int); hi = np.asarray(hi)
    k1 = KaplanMeierFitter().fit(t[hi], e[hi], label='High risk')      # group sizes: first column of the at-risk table
    k0 = KaplanMeierFitter().fit(t[~hi], e[~hi], label='Low risk')
    k1.plot_survival_function(ax=ax, color='#d62728', ci_alpha=0.15, lw=fl.LW_DATA)
    k0.plot_survival_function(ax=ax, color='#2ca02c', ci_alpha=0.15, lw=fl.LW_DATA)
    r = logrank_test(t[hi], t[~hi], e[hi], e[~hi])
    ax.text(0.97, 0.04, pval(r.p_value), transform=ax.transAxes, fontsize=fl.ANNOT_PT, ha='right', va='bottom')   # log-rank (caption)
    ax.set_title(title); ax.set_xlabel('Time (months)'); ax.set_ylabel('Overall survival\nprobability')
    ax.set_xlim(xticks[0], xticks[-1]); ax.set_xticks(xticks); ax.set_ylim(0, 1.02); ax.set_yticks(TICKS01)
    ax.legend(loc='lower left')
    cnt = fl.at_risk_table(ax, [k1, k0], ['High', 'Low'], xticks, header='At risk')
    assert cnt == fl.lifelines_counts([k1, k0], xticks), 'numbers at risk differ from lifelines'
    return r


def boot_auc(y, s, nb=2000):
    y = np.asarray(y); s = np.asarray(s); n = len(y); v = []
    for _ in range(nb):
        b = RNG.choice(n, n, replace=True)
        if y[b].sum() == 0 or y[b].sum() == n: continue
        v.append(roc_auc_score(y[b], s[b]))
    return np.percentile(v, [2.5, 97.5])

# ------------------------------------------------------------------ Figure 2: patch-level ROC, 4 backbones
def fig_patch_roc():
    alldl = pd.read_csv(f'{DATA}/results/ALL_DL_PREDICTIONS.csv'); gt = alldl.groupby('ID')['gt'].first()
    G = fl.Grid(1, 2, aspect=1.0)
    for i, (c, title) in enumerate([('train', 'Training cohort (patch level)'), ('test', 'Test cohort (patch level)')]):
        ax = G.ax[0][i]
        for m, disp in [('resnet18', 'ResNet18'), ('resnet50', 'ResNet50'), ('densenet121', 'DenseNet121'), ('CrossFormer', 'CrossFormer')]:
            d = pd.read_csv(f'{DATA}/results/Pathomics_Slice_{m}_{c}.csv'); y = d['ID'].map(gt).values; s = d['label-1'].values
            fpr, tpr, _ = roc_curve(y, s); auc = roc_auc_score(y, s); lo, hi = boot_auc(y, s, nb=300)
            ax.plot(fpr, tpr, color=COL[disp], label=f'{disp} {auc:.3f} ({lo:.3f}–{hi:.3f})')
        roc_axes(ax); ax.set_title(title); fl.legend_below(ax, title='AUC (95% CI)')
    save(G, 'Figure2_patch_ROC')

# ------------------------------------------------------------------ Figure 5: KM with locked cut-offs
def fig_km():
    thr = {'Clinical': 1.03, 'Pathomics': 0.80, 'Combined': 1.03}
    G = fl.Grid(2, 3, aspect=0.8); stats_rows = []
    for i, c in enumerate(['train', 'test']):
        for j, m in enumerate(['Clinical', 'Pathomics', 'Combined']):
            d = pd.read_csv(f'{DATA}/results/{m}_cox_predictions_{c}.csv').set_index('ID').join(sur[['event', 'duration']])
            hi = (d.HR >= thr[m]).values
            r = km_panel(G.ax[i][j], d.duration.values, d.event.values, hi, f'{m}, {"training" if c == "train" else "test"} cohort',
                         [0, 25, 50, 75, 100, 125, 150])
            stats_rows.append({'Cohort': c, 'Model': m, 'cut-off (partial hazard)': thr[m], 'High-risk n (events)': f'{int(hi.sum())} ({int(d.event[hi].sum())})',
                               'Low-risk n (events)': f'{int((~hi).sum())} ({int(d.event[~hi].sum())})', 'log-rank p': f'{r.p_value:.2e}'})
    pd.DataFrame(stats_rows).to_csv('analysis_outputs/Table_KM_locked_cutoffs.csv', index=False)
    save(G, 'Figure5_KM')

# ------------------------------------------------------------------ Figures 7/8: time-dependent ROC
def td_labels(d, t):
    keep = ~((d.event == 0) & (d.duration <= t)); dd = d[keep]; return dd, ((dd.event == 1) & (dd.duration <= t)).astype(int)
def fig_tdroc():
    rows = []
    for c, fname in [('train', 'Figure7_tdROC_train'), ('test', 'Figure8_tdROC_test')]:
        d = join[join.group == c]; G = fl.Grid(1, 3, aspect=1.0)
        for j, (t, tl) in enumerate([(60, '5-year'), (84, '7-year'), (108, '9-year')]):
            dd, y = td_labels(d, t); ax = G.ax[0][j]
            for m in ['Clinical', 'Pathomics', 'Combined']:
                s = -dd[m].values; fpr, tpr, _ = roc_curve(y, s); auc = roc_auc_score(y, s); lo, hi = boot_auc(y.values, s)
                ax.plot(fpr, tpr, color=COL[m], label=f'{m} {auc:.3f} ({lo:.3f}–{hi:.3f})')
                rows.append({'Cohort': c, 'Horizon': tl, 'Model': m, 'AUC': round(auc, 3), '95% CI': f'{lo:.3f}-{hi:.3f}', 'events': int(y.sum()), 'n': len(dd)})
            roc_axes(ax); ax.set_title(f'{tl} OS (events = {int(y.sum())})')
            fl.legend_below(ax, title='AUC (95% CI)')
        save(G, fname)
    pd.DataFrame(rows).to_csv('analysis_outputs/Table5_tdAUC_CI.csv', index=False)

# ------------------------------------------------------------------ Figures 9/10: calibration & DCA at 9 years
HOR = 108
def model_event_prob():
    cl = pd.read_csv(f'{DATA}/data/clinical.csv'); cl_tr = cl[cl.group == 'train'].set_index('ID'); cl_te = cl[cl.group == 'test'].set_index('ID')
    ptr = pd.read_csv(f'{DATA}/features/Pathomics_train_cox.csv').set_index('ID'); pte = pd.read_csv(f'{DATA}/features/Pathomics_test_cox.csv').set_index('ID')
    ctr = pd.read_csv(f'{DATA}/features/Combined_train_features_norm.csv').set_index('ID'); cte = pd.read_csv(f'{DATA}/features/Combined_test_features_norm.csv').set_index('ID')
    spec = {'Clinical': (['N', 'AJCC', 'age'], cl_tr, cl_te), 'Pathomics': (['prob058', 'prob05', 'pred1'], ptr, pte), 'Combined': (['Clinical', 'Pathomics'], ctr, cte)}
    out = {}
    for m, (covs, tr, te) in spec.items():
        cph = CoxPHFitter(penalizer=0.0).fit(tr[covs + ['duration', 'event']], 'duration', 'event')
        for c, d in [('train', tr), ('test', te)]:
            sf = cph.predict_survival_function(d[covs], times=[HOR]); out[(m, c)] = (1 - sf.T.iloc[:, 0].values, d)
    return out
def fig_cal_dca():
    ep = model_event_prob(); G = fl.Grid(1, 2, aspect=1.0)
    for i, c in enumerate(['train', 'test']):
        ax = G.ax[0][i]
        for m in ['Clinical', 'Pathomics', 'Combined']:
            pred, d = ep[(m, c)]; dd = d.reset_index(drop=True).copy(); dd['pred'] = pred
            nb = 4 if c == 'train' else 3; dd['q'] = pd.qcut(dd['pred'].rank(method='first'), nb, labels=False)
            xs, ys, lo, hi = [], [], [], []
            for q in sorted(dd['q'].unique()):
                g = dd[dd['q'] == q]; kmf = KaplanMeierFitter().fit(g['duration'], g['event'])
                xs.append(g['pred'].mean()); ys.append(1 - float(kmf.predict(HOR)))
                ci = kmf.confidence_interval_survival_function_; idx = ci.index[ci.index <= HOR][-1]
                lo.append(1 - ci.loc[idx].iloc[1]); hi.append(1 - ci.loc[idx].iloc[0])
            ax.errorbar(xs, ys, yerr=[np.array(ys) - np.array(lo), np.array(hi) - np.array(ys)], fmt='o-', color=COL[m], label=m)
        ax.plot([0, 1], [0, 1], 'k--', lw=fl.LW_REF, alpha=0.6, label='Ideal')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks(TICKS01); ax.set_yticks(TICKS01)
        ax.set_xlabel('Predicted 9-year event probability'); ax.set_ylabel('Observed 9-year event probability (KM)')
        ax.set_title(f'Calibration, {"training" if c == "train" else "test"} cohort'); ax.legend(loc='upper left')
    save(G, 'Figure9_calibration')
    thr = np.linspace(0.02, 0.7, 60); G = fl.Grid(1, 2, aspect=0.85)
    for i, c in enumerate(['train', 'test']):
        ax = G.ax[0][i]; prev = None
        for m in ['Clinical', 'Pathomics', 'Combined']:
            pred, d = ep[(m, c)]; y = ((d['duration'].values <= HOR) & (d['event'].values == 1)).astype(int); n = len(y); prev = y.mean()
            nbv = [(((pred >= pt) & (y == 1)).sum() / n - ((pred >= pt) & (y == 0)).sum() / n * (pt / (1 - pt))) for pt in thr]
            ax.plot(thr, nbv, color=COL[m], label=m)
        ax.plot(thr, prev - (1 - prev) * (thr / (1 - thr)), 'k--', lw=fl.LW_REF, alpha=0.7, label='Treat all')
        ax.axhline(0, color='gray', ls=':', lw=fl.LW_REF, label='Treat none')
        ax.set_ylim(-0.08, prev + 0.06); ax.set_xlim(0, 0.7); ax.set_xticks(np.arange(0, 0.71, 0.1))
        ax.set_xlabel('Threshold probability'); ax.set_ylabel('Net benefit')
        ax.set_title(f'Decision curve, {"training" if c == "train" else "test"} cohort'); ax.legend(loc='upper right')
    save(G, 'Figure10_DCA')

# ------------------------------------------------------------------ Supplementary: split sensitivity + backbone forest + guideline forest
def fig_supp():
    a = np.load('analysis_outputs/split_sens_clinical.npy'); d = a[:, 1] - a[:, 0]
    G = fl.Grid(1, 2, aspect=0.85)
    ax = G.ax[0][0]
    ax.hist(d, bins=40, color='#9ecae1', edgecolor='white', linewidth=0.3); ax.axvline(0.124, color='#d62728', lw=fl.LW_DATA, label='Observed split (+0.124)')
    ax.set_ylim(0, np.histogram(d, bins=40)[0].max() * 1.25)
    ax.set_xlabel('Test C-index minus training C-index\n(clinical model)'); ax.set_ylabel('Number of random splits'); ax.legend(loc='upper left')
    ax.set_title('1000 random stratified 7:3 splits')
    ax = G.ax[0][1]
    ax.scatter(a[:, 0], a[:, 1], s=5, alpha=0.4, color='#6baed6', linewidths=0); ax.scatter([0.683], [0.807], color='#d62728', s=20, zorder=5, label='Observed split')
    ax.plot([0.5, 0.95], [0.5, 0.95], 'k--', lw=fl.LW_REF); ax.set_xlabel('Training C-index'); ax.set_ylabel('Test C-index'); ax.legend(loc='upper left')
    ax.set_title('Clinical model (age, N, AJCC)')
    save(G, 'FigureS1_split_sensitivity')
    # backbone forest (k=3 matched complexity, PLH+BoW)
    b = pd.read_csv('analysis_outputs/Table_backbone_ablation.csv')
    b = b[(b.Selection == 'k=3') & (b.Branch == 'PLH+BoW') & (b.IDF == 'allIDF')]
    G = fl.Grid(1, 1, aspect=0.55, width_cm=12.0, letters=False); ax = G.ax[0][0]
    yy = np.arange(len(b))[::-1]
    for i, (_, r) in enumerate(b.iterrows()):
        lo, hi = map(float, r['Pathomics test 95% CI'].split('-')); lo2, hi2 = map(float, r['Combined test 95% CI'].split('-'))
        ax.errorbar(r['Pathomics C test'], yy[i] + 0.15, xerr=[[r['Pathomics C test'] - lo], [hi - r['Pathomics C test']]], fmt='o', color=COL['Pathomics'], label='Pathomics signature' if i == 0 else None)
        ax.errorbar(r['Combined C test'], yy[i] - 0.15, xerr=[[r['Combined C test'] - lo2], [hi2 - r['Combined C test']]], fmt='s', color=COL['Combined'], label='Combined model' if i == 0 else None)
    ax.set_yticks(yy); ax.set_yticklabels(b.Backbone); ax.set_xlabel('Test-cohort C-index (95% bootstrap CI)'); ax.axvline(0.5, color='gray', ls=':', lw=fl.LW_REF)
    ax.set_xlim(0.4, 1.0); ax.set_ylim(-0.6, len(b) - 0.4)
    ax.set_title('Backbone sensitivity (identical downstream pipeline)'); fl.legend_below(ax, ncol=2)
    save(G, 'FigureS2_backbone_forest')
    # guideline model forest
    g = pd.read_csv('analysis_outputs/Table_guideline_models.csv')
    G = fl.Grid(1, 1, aspect=0.6, letters=False); ax = G.ax[0][0]; yy = np.arange(len(g))[::-1]
    for i, (_, r) in enumerate(g.iterrows()):
        lo, hi = map(float, r['train 95% CI'].split('-')); lo2, hi2 = map(float, r['test 95% CI'].split('-'))
        ax.errorbar(r['C train'], yy[i] + 0.15, xerr=[[r['C train'] - lo], [hi - r['C train']]], fmt='o', color='#7f7f7f', label='Training (optimistic)' if i == 0 else None)
        ax.errorbar(r['C test'], yy[i] - 0.15, xerr=[[r['C test'] - lo2], [hi2 - r['C test']]], fmt='s', color='#d62728', label='Test' if i == 0 else None)
    _nm = {'Clinical-guideline (age, AJCC, ER, PR, HER2, grade)': 'Clinical-guideline (age, AJCC stage, ER, PR, HER2, grade)',
           'Clinical-guideline alt (age, T, N, ER, PR, HER2, grade)': 'Clinical-guideline, T/N variant (age, T, N, ER, PR, HER2, grade)',
           'Clinical-original (age, N, AJCC) on complete cases': 'Clinical-original (age, N, AJCC stage)',
           'Pathomics (deployed) on complete cases': 'Pathomics signature',
           'Combined-original (deployed) on complete cases': 'Combined-original (Clinical-original + Pathomics)',
           'Combined-guideline (guideline clinical + pathomics)': 'Combined-guideline (Clinical-guideline + Pathomics)'}
    names = [_nm.get(n, n).replace(' (', '\n(', 1) for n in g.Model]   # names identical to Table 5, broken before the parenthesis
    ax.set_yticks(yy); ax.set_yticklabels(names); ax.set_xlabel('C-index (95% bootstrap CI), complete cases n = 259'); ax.set_xlim(0.5, 1.0)
    ax.set_ylim(-0.6, len(g) - 0.4); ax.legend(loc='upper left'); ax.set_title('Guideline-level clinical model versus fusion models')
    save(G, 'FigureS3_guideline_forest')

if __name__ == '__main__':
    if os.path.exists(f'{OUT}/figure_layout_metrics.csv'): os.remove(f'{OUT}/figure_layout_metrics.csv')
    print('Figure 2 ...'); fig_patch_roc()
    print('Figure 5 ...'); fig_km()
    print('Figures 7/8 ...'); fig_tdroc()
    print('Figures 9/10 ...'); fig_cal_dca()
    print('Supplementary ...'); fig_supp()
    write_metrics(); print('DONE')
