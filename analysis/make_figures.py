# -*- coding: utf-8 -*-
"""Regenerate all analysis figures in one consistent style and compose multi-panel figures following the
layout rules (equal gutters 0.3 cm, outer margin 0.5 cm, bold panel letters at the top-left, aligned axes).
Outputs -> analysis_outputs/figures/"""
import os, sys, io, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import roc_auc_score, roc_curve
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts

DATA = 'pipeline_outputs'; OUT = 'analysis_outputs/figures'; os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(42)
rcParams.update({'font.family': 'Arial', 'font.size': 9, 'axes.linewidth': 0.8, 'lines.linewidth': 1.4,
                 'axes.titlesize': 10, 'axes.labelsize': 9, 'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                 'savefig.dpi': 300, 'figure.dpi': 100})
COL = {'Clinical': '#1f77b4', 'Pathomics': '#ff7f0e', 'Combined': '#d62728',
       'ResNet18': '#2ca02c', 'ResNet50': '#d62728', 'DenseNet121': '#9467bd', 'CrossFormer': '#17becf'}
DPI = 300; CM = DPI / 2.54
sur = pd.read_csv(f'{DATA}/sur.csv').set_index('ID')
join = pd.read_csv(f'{DATA}/results/joinit_info.csv').set_index('ID')

def panel(fig, path):
    fig.savefig(path, dpi=DPI, bbox_inches='tight', pad_inches=0.04, facecolor='white'); plt.close(fig)
    return path

def compose(paths, ncols, out, labels=None, gutter_cm=0.3, margin_cm=0.5, panel_w_cm=None, label_pt=12):  # gutters 0.3 cm, margin 0.5 cm
    """Equal panel widths, equal gutters, uniform outer margin; panel letters in the top-left corner inside the margin."""
    ims = [Image.open(p).convert('RGB') for p in paths]
    n = len(ims); nrows = int(np.ceil(n / ncols))
    if panel_w_cm is None: panel_w_cm = (17.0 - 2 * margin_cm - (ncols - 1) * gutter_cm) / ncols
    pw = int(round(panel_w_cm * CM)); gut = int(round(gutter_cm * CM)); mar = int(round(margin_cm * CM))
    scaled = [im.resize((pw, int(round(im.size[1] * pw / im.size[0]))), Image.LANCZOS) for im in ims]
    row_h = [max(scaled[i].size[1] for i in range(r * ncols, min(n, (r + 1) * ncols))) for r in range(nrows)]
    W = 2 * mar + ncols * pw + (ncols - 1) * gut; H = 2 * mar + sum(row_h) + (nrows - 1) * gut
    canvas = Image.new('RGB', (W, H), 'white'); draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(r'C:\Windows\Fonts\arialbd.ttf', int(label_pt * DPI / 72))
    labels = labels or [chr(65 + i) for i in range(n)]
    y = mar
    for r in range(nrows):
        for c in range(ncols):
            i = r * ncols + c
            if i >= n: break
            x = mar + c * (pw + gut)
            # top-align panels within the row; keep axes aligned by identical panel sizes where possible
            canvas.paste(scaled[i], (x, y))
            draw.text((x + int(0.03 * CM), y), labels[i], fill='black', font=font)   # top-left corner of every panel
        y += row_h[r] + gut
    canvas.save(out, dpi=(DPI, DPI))
    content = (ncols * pw * sum(row_h)) / (W * H)
    print(f'  composed {os.path.basename(out)} {W}x{H}px  ({W/CM:.1f}x{H/CM:.1f} cm)  panel area {content:.0%}')
    return out

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
    paths = []
    for c, title in [('train', 'Training cohort (patch level)'), ('test', 'Test cohort (patch level)')]:
        fig, ax = plt.subplots(figsize=(3.4, 3.4))
        for m, disp in [('resnet18', 'ResNet18'), ('resnet50', 'ResNet50'), ('densenet121', 'DenseNet121'), ('CrossFormer', 'CrossFormer')]:
            d = pd.read_csv(f'{DATA}/results/Pathomics_Slice_{m}_{c}.csv'); y = d['ID'].map(gt).values; s = d['label-1'].values
            fpr, tpr, _ = roc_curve(y, s); auc = roc_auc_score(y, s); lo, hi = boot_auc(y, s, nb=300)
            ax.plot(fpr, tpr, color=COL[disp], label=f'{disp}: AUC {auc:.3f} ({lo:.3f}-{hi:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.6)
        ax.set_xlabel('1 - Specificity'); ax.set_ylabel('Sensitivity'); ax.set_title(title); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(loc='lower right', frameon=False); ax.set_aspect('equal')
        paths.append(panel(fig, f'{OUT}/_fig2_{c}.png'))
    compose(paths, 2, f'{OUT}/Figure2_patch_ROC.png', panel_w_cm=7.5)

# ------------------------------------------------------------------ Figure 5: KM with locked cut-offs
def fig_km():
    thr = {'Clinical': 1.03, 'Pathomics': 0.80, 'Combined': 1.03}
    paths = []; stats_rows = []
    for c in ['train', 'test']:
        for m in ['Clinical', 'Pathomics', 'Combined']:
            d = pd.read_csv(f'{DATA}/results/{m}_cox_predictions_{c}.csv').set_index('ID').join(sur[['event', 'duration']])
            hi = d.HR >= thr[m]
            r = logrank_test(d.duration[hi], d.duration[~hi], d.event[hi], d.event[~hi])
            fig, ax = plt.subplots(figsize=(3.6, 3.6))
            k1 = KaplanMeierFitter().fit(d.duration[hi], d.event[hi], label=f'High risk (n={int(hi.sum())})')
            k0 = KaplanMeierFitter().fit(d.duration[~hi], d.event[~hi], label=f'Low risk (n={int((~hi).sum())})')
            k1.plot_survival_function(ax=ax, color='#d62728', ci_alpha=0.15); k0.plot_survival_function(ax=ax, color='#2ca02c', ci_alpha=0.15)
            ptxt = 'log-rank p < 0.001' if r.p_value < 0.001 else f'log-rank p = {r.p_value:.3f}'
            ax.text(0.03, 0.06, ptxt, transform=ax.transAxes, fontsize=8)
            ax.set_title(f'{m} model, {"training" if c == "train" else "test"} cohort'); ax.set_xlabel('Time (months)'); ax.set_ylabel('Overall survival probability')
            ax.set_ylim(0, 1.02); ax.set_xlim(0, 150); ax.legend(loc='lower left', frameon=False, bbox_to_anchor=(0.0, 0.12))
            add_at_risk_counts(k1, k0, ax=ax, rows_to_show=['At risk'], fontsize=7)
            plt.tight_layout()
            paths.append(panel(fig, f'{OUT}/_fig5_{c}_{m}.png'))
            stats_rows.append({'Cohort': c, 'Model': m, 'cut-off (partial hazard)': thr[m], 'High-risk n (events)': f'{int(hi.sum())} ({int(d.event[hi].sum())})',
                               'Low-risk n (events)': f'{int((~hi).sum())} ({int(d.event[~hi].sum())})', 'log-rank p': f'{r.p_value:.2e}'})
    pd.DataFrame(stats_rows).to_csv('analysis_outputs/Table_KM_locked_cutoffs.csv', index=False)
    compose(paths, 3, f'{OUT}/Figure5_KM.png')

# ------------------------------------------------------------------ Figures 7/8: time-dependent ROC
def td_labels(d, t):
    keep = ~((d.event == 0) & (d.duration <= t)); dd = d[keep]; return dd, ((dd.event == 1) & (dd.duration <= t)).astype(int)
def fig_tdroc():
    rows = []
    for c, fname in [('train', 'Figure7_tdROC_train'), ('test', 'Figure8_tdROC_test')]:
        d = join[join.group == c]; paths = []
        for t, tl in [(60, '5-year'), (84, '7-year'), (108, '9-year')]:
            dd, y = td_labels(d, t)
            fig, ax = plt.subplots(figsize=(3.4, 3.4))
            for m in ['Clinical', 'Pathomics', 'Combined']:
                s = -dd[m].values; fpr, tpr, _ = roc_curve(y, s); auc = roc_auc_score(y, s); lo, hi = boot_auc(y.values, s)
                ax.plot(fpr, tpr, color=COL[m], label=f'{m}: AUC {auc:.3f} ({lo:.3f}-{hi:.3f})')
                rows.append({'Cohort': c, 'Horizon': tl, 'Model': m, 'AUC': round(auc, 3), '95% CI': f'{lo:.3f}-{hi:.3f}', 'events': int(y.sum()), 'n': len(dd)})
            ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.6); ax.set_aspect('equal')
            ax.set_xlabel('1 - Specificity'); ax.set_ylabel('Sensitivity'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_title(f'{tl} OS, {"training" if c == "train" else "test"} cohort (events = {int(y.sum())})'); ax.legend(loc='lower right', frameon=False)
            paths.append(panel(fig, f'{OUT}/_{fname}_{t}.png'))
        compose(paths, 3, f'{OUT}/{fname}.png')
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
    ep = model_event_prob(); paths = []
    for c in ['train', 'test']:
        fig, ax = plt.subplots(figsize=(3.6, 3.6))
        for m in ['Clinical', 'Pathomics', 'Combined']:
            pred, d = ep[(m, c)]; dd = d.reset_index(drop=True).copy(); dd['pred'] = pred
            nb = 4 if c == 'train' else 3; dd['q'] = pd.qcut(dd['pred'].rank(method='first'), nb, labels=False)
            xs, ys, lo, hi = [], [], [], []
            for q in sorted(dd['q'].unique()):
                g = dd[dd['q'] == q]; kmf = KaplanMeierFitter().fit(g['duration'], g['event'])
                xs.append(g['pred'].mean()); ys.append(1 - float(kmf.predict(HOR)))
                ci = kmf.confidence_interval_survival_function_; idx = ci.index[ci.index <= HOR][-1]
                lo.append(1 - ci.loc[idx].iloc[1]); hi.append(1 - ci.loc[idx].iloc[0])
            ax.errorbar(xs, ys, yerr=[np.array(ys) - np.array(lo), np.array(hi) - np.array(ys)], fmt='o-', color=COL[m], label=m, capsize=2, ms=4)
        ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.6, label='Ideal'); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
        ax.set_xlabel('Predicted 9-year event probability'); ax.set_ylabel('Observed 9-year event probability (KM)')
        ax.set_title(f'Calibration, {"training" if c == "train" else "test"} cohort'); ax.legend(loc='upper left', frameon=False)
        paths.append(panel(fig, f'{OUT}/_fig9_{c}.png'))
    compose(paths, 2, f'{OUT}/Figure9_calibration.png', panel_w_cm=7.5)
    thr = np.linspace(0.02, 0.7, 60); paths = []
    for c in ['train', 'test']:
        fig, ax = plt.subplots(figsize=(3.9, 3.4)); prev = None
        for m in ['Clinical', 'Pathomics', 'Combined']:
            pred, d = ep[(m, c)]; y = ((d['duration'].values <= HOR) & (d['event'].values == 1)).astype(int); n = len(y); prev = y.mean()
            nbv = [(((pred >= pt) & (y == 1)).sum() / n - ((pred >= pt) & (y == 0)).sum() / n * (pt / (1 - pt))) for pt in thr]
            ax.plot(thr, nbv, color=COL[m], label=m)
        ax.plot(thr, prev - (1 - prev) * (thr / (1 - thr)), 'k--', lw=0.9, alpha=0.7, label='Treat all'); ax.axhline(0, color='gray', ls=':', label='Treat none')
        ax.set_ylim(-0.08, prev + 0.06); ax.set_xlim(0, 0.7); ax.set_xlabel('Threshold probability'); ax.set_ylabel('Net benefit')
        ax.set_title(f'Decision curve (9-year OS), {"training" if c == "train" else "test"} cohort'); ax.legend(frameon=False)
        paths.append(panel(fig, f'{OUT}/_fig10_{c}.png'))
    compose(paths, 2, f'{OUT}/Figure10_DCA.png', panel_w_cm=7.5)

# ------------------------------------------------------------------ Supplementary: split sensitivity + backbone forest + guideline forest
def fig_supp():
    a = np.load('analysis_outputs/split_sens_clinical.npy'); d = a[:, 1] - a[:, 0]
    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    ax.hist(d, bins=40, color='#9ecae1', edgecolor='white'); ax.axvline(0.124, color='#d62728', lw=1.5, label='Observed split (+0.124)')
    ax.set_xlabel('Test C-index minus training C-index (clinical model)'); ax.set_ylabel('Number of random splits'); ax.legend(frameon=False)
    ax.set_title('1000 random stratified 7:3 splits')
    p1 = panel(fig, f'{OUT}/_supp_split_a.png')
    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    ax.scatter(a[:, 0], a[:, 1], s=6, alpha=0.4, color='#6baed6'); ax.scatter([0.683], [0.807], color='#d62728', s=30, zorder=5, label='Observed split')
    ax.plot([0.5, 0.95], [0.5, 0.95], 'k--', lw=0.8); ax.set_xlabel('Training C-index'); ax.set_ylabel('Test C-index'); ax.legend(frameon=False)
    ax.set_title('Clinical model (age, N, AJCC)')
    p2 = panel(fig, f'{OUT}/_supp_split_b.png')
    compose([p1, p2], 2, f'{OUT}/FigureS1_split_sensitivity.png', panel_w_cm=7.5)
    # backbone forest (k=3 matched complexity, PLH+BoW)
    b = pd.read_csv('analysis_outputs/Table_backbone_ablation.csv')
    b = b[(b.Selection == 'k=3') & (b.Branch == 'PLH+BoW') & (b.IDF == 'allIDF')]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    yy = np.arange(len(b))[::-1]
    for i, (_, r) in enumerate(b.iterrows()):
        lo, hi = map(float, r['Pathomics test 95% CI'].split('-')); lo2, hi2 = map(float, r['Combined test 95% CI'].split('-'))
        ax.errorbar(r['Pathomics C test'], yy[i] + 0.15, xerr=[[r['Pathomics C test'] - lo], [hi - r['Pathomics C test']]], fmt='o', color=COL['Pathomics'], capsize=3, label='Pathomics signature' if i == 0 else None)
        ax.errorbar(r['Combined C test'], yy[i] - 0.15, xerr=[[r['Combined C test'] - lo2], [hi2 - r['Combined C test']]], fmt='s', color=COL['Combined'], capsize=3, label='Combined model' if i == 0 else None)
    ax.set_yticks(yy); ax.set_yticklabels(b.Backbone); ax.set_xlabel('Test-cohort C-index (95% bootstrap CI)'); ax.axvline(0.5, color='gray', ls=':')
    ax.set_xlim(0.4, 1.0); ax.legend(frameon=False, loc='lower left'); ax.set_title('Backbone sensitivity (identical downstream pipeline)')
    p3 = panel(fig, f'{OUT}/FigureS2_backbone_forest.png')
    # guideline model forest
    g = pd.read_csv('analysis_outputs/Table_guideline_models.csv')
    fig, ax = plt.subplots(figsize=(6.0, 3.2)); yy = np.arange(len(g))[::-1]
    for i, (_, r) in enumerate(g.iterrows()):
        lo, hi = map(float, r['train 95% CI'].split('-')); lo2, hi2 = map(float, r['test 95% CI'].split('-'))
        ax.errorbar(r['C train'], yy[i] + 0.15, xerr=[[r['C train'] - lo], [hi - r['C train']]], fmt='o', color='#7f7f7f', capsize=3, label='Training (optimistic)' if i == 0 else None)
        ax.errorbar(r['C test'], yy[i] - 0.15, xerr=[[r['C test'] - lo2], [hi2 - r['C test']]], fmt='s', color='#d62728', capsize=3, label='Test' if i == 0 else None)
    names = [n.replace(' on complete cases', '').replace(' (deployed)', '') for n in g.Model]
    ax.set_yticks(yy); ax.set_yticklabels(names, fontsize=7); ax.set_xlabel('C-index (95% bootstrap CI), complete cases n = 259'); ax.set_xlim(0.5, 1.0)
    ax.legend(frameon=False, loc='upper left'); ax.set_title('Guideline-level clinical model versus fusion models')
    p4 = panel(fig, f'{OUT}/FigureS3_guideline_forest.png')

if __name__ == '__main__':
    print('Figure 2 ...'); fig_patch_roc()
    print('Figure 5 ...'); fig_km()
    print('Figures 7/8 ...'); fig_tdroc()
    print('Figures 9/10 ...'); fig_cal_dca()
    print('Supplementary ...'); fig_supp()
    print('DONE')
