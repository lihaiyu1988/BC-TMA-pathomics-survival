# -*- coding: utf-8 -*-
"""Figure 11: external validation of the frozen clinical models in TCGA-BRCA (run external_validation_tcga.py first).
Drawn at its final print size with fig_layout.Grid, like every other figure (width 15.0 cm, identical 0.3-cm gaps,
0.5-cm outer margin, aligned axes, one type scale and one set of line widths, bold letters at the top-left)."""
import os, sys, pickle, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); sys.path.insert(0, f'{ROOT}/analysis')
import make_figures as mf                                   # km_panel(), roc_axes(), save(), style
import fig_layout as fl
from sklearn.metrics import roc_curve

A = 'analysis_outputs'; OUT = f'{A}/external_tcga'; FIG = f'{A}/figures'
R = pickle.load(open(f'{OUT}/_results.pkl', 'rb')); keep, cals, res = R['keep'], R['cals'], R['res']
PRIMARY = 'reviewed' if 'reviewed' in keep else 'imputed'
GLAB = 'Clinical-guideline'
C_CLIN, C_GUIDE = '#1f77b4', '#9467bd'
r_clin = res.iloc[0]; r_g = res[res.Model.str.contains('expert' if PRIMARY == 'reviewed' else 'imputed')].iloc[0]
mf.OUT = FIG

G = fl.Grid(2, 3, aspect=[0.8, 1.0])
# A: C-index in the development training cohort, the internal test cohort and TCGA-BRCA
ci = pd.read_csv('pipeline_outputs/revision_outputs/Table3_cindex_CI.csv')
g = pd.read_csv(f'{A}/Table_guideline_models.csv').set_index('Model').loc['Clinical-guideline (age, AJCC, ER, PR, HER2, grade)']
cl_tr = ci[(ci.Model == 'Clinical') & (ci.Cohort == 'train')].iloc[0]; cl_te = ci[(ci.Model == 'Clinical') & (ci.Cohort == 'test')].iloc[0]
rng = lambda s: tuple(float(v) for v in str(s).replace('–', '-').split('-'))
rows = [('Clinical', 'Training', float(cl_tr['C-index']), *rng(cl_tr['95% CI'])), ('Clinical', 'Internal test', float(cl_te['C-index']), *rng(cl_te['95% CI'])),
        ('Clinical', 'TCGA-BRCA', r_clin['C-index'], r_clin['C lo'], r_clin['C hi']),
        (GLAB, 'Training', g['C train'], *rng(g['train 95% CI'])), (GLAB, 'Internal test', g['C test'], *rng(g['test 95% CI'])),
        (GLAB, 'TCGA-BRCA', r_g['C-index'], r_g['C lo'], r_g['C hi'])]
COH = {'Training': ('#7f7f7f', 'o'), 'Internal test': ('#d62728', 's'), 'TCGA-BRCA': ('#1f4e9a', 'D')}
ax = G.ax[0][0]
ypos = {('Clinical', 'Training'): 5.5, ('Clinical', 'Internal test'): 5.0, ('Clinical', 'TCGA-BRCA'): 4.5,
        (GLAB, 'Training'): 3.5, (GLAB, 'Internal test'): 3.0, (GLAB, 'TCGA-BRCA'): 2.5}
seen = set()
for m, c, v, lo, hi in rows:
    col, mk = COH[c]; y = ypos[(m, c)]
    ax.errorbar(v, y, xerr=[[v - lo], [hi - v]], fmt=mk, color=col, label=c if c not in seen else None); seen.add(c)
    ax.text(1.115, y, f'{v:.3f}', va='center', ha='right', fontsize=fl.ANNOT_PT)   # value column inside the frame (CIs end <= 0.92)
ax.set_yticks([5.0, 3.0]); ax.set_yticklabels(['Clinical', 'Clinical-\nguideline']); ax.set_ylim(2.0, 6.0)
ax.axvline(0.5, color='gray', ls=':', lw=fl.LW_REF); ax.set_xlim(0.5, 1.12); ax.set_xticks(np.arange(0.5, 1.01, 0.1))
assert max(r[4] for r in rows) < 0.995, 'a confidence limit would run into the value column'
ax.set_xlabel('C-index (95% bootstrap CI)'); ax.set_title('Discrimination by cohort')
fl.legend_below(ax)

# B, C: Kaplan-Meier curves with the locked development cut-offs
for j, (key, lab) in enumerate([('Clinical', 'Clinical model'), (PRIMARY, f'{GLAB} model')], start=1):
    d, lp_, hi, _ = keep[key]
    mf.km_panel(G.ax[0][j], d.duration.values, d.event.values.astype(int), np.asarray(hi), lab, [0, 24, 48, 72, 96, 120])

# D: 5-year time-dependent ROC (deaths by 5 years vs survivors beyond 5 years)
ax = G.ax[1][0]
for key, lab, col, rr in [('Clinical', 'Clinical', C_CLIN, r_clin), (PRIMARY, GLAB, C_GUIDE, r_g)]:
    d, lp_, _, _ = keep[key]; t, e = d.duration.values, d.event.values.astype(int)
    case = (t <= 60) & (e == 1); ctrl = t > 60; k = case | ctrl
    fpr, tpr, _ = roc_curve(case[k].astype(int), lp_[k])
    ax.plot(fpr, tpr, color=col, label=f"{lab} ({int(rr['cases 5y'])} deaths)" + '\n' + f"{rr['AUC 5y']:.3f} ({rr['AUC 5y lo']:.3f}\u2013{rr['AUC 5y hi']:.3f})")
mf.roc_axes(ax); ax.set_title('5-year OS'); fl.legend_below(ax, title='AUC (95% CI)')

# E, F: calibration of the predicted 5-year risk (frozen baseline hazard of the development cohort)
for j, (key, lab, col, rr) in enumerate([('Clinical', 'Clinical model', C_CLIN, r_clin), (PRIMARY, f'{GLAB} model', C_GUIDE, r_g)], start=1):
    c = cals[key]; ax = G.ax[1][j]
    ax.errorbar(c.predicted, c.observed, yerr=[c.observed - c.obs_lo, c.obs_hi - c.observed], fmt='o-', color=col, label='Quintiles of predicted risk')
    ax.plot([0, 0.6], [0, 0.6], 'k--', lw=fl.LW_REF, alpha=0.6, label='Ideal')
    ax.set_xlim(0, 0.6); ax.set_ylim(0, 0.6); ax.set_xticks(np.arange(0, 0.61, 0.1)); ax.set_yticks(np.arange(0, 0.61, 0.1))
    ax.set_xlabel('Predicted 5-year\nevent probability'); ax.set_ylabel('Observed 5-year event\nprobability (KM)')
    ax.text(0.97, 0.04, f"O/E = {rr['O/E']:.2f}\nCalibration slope\n{rr['slope']:.2f} ({rr['slope_lo']:.2f}\u2013{rr['slope_hi']:.2f})",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=fl.ANNOT_PT)
    ax.set_title(lab); fl.legend_below(ax)

mf.save(G, 'Figure11_TCGA_external'); mf.write_metrics()
print('Figure 11 saved; primary grade source:', PRIMARY)
