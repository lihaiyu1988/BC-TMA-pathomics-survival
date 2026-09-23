# -*- coding: utf-8 -*-
"""Accurate end-to-end flow chart (Figure 1) matching the Methods exactly."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams['font.family'] = 'Arial'
W, H = 124, 62
fig, ax = plt.subplots(figsize=(W / 8.2, H / 8.2)); ax.axis('off'); ax.set_xlim(0, W); ax.set_ylim(0, H)
def box(x, y, w, h, title, lines, fc, ec, fs=7.6, ts=8.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.25,rounding_size=1.2', fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h - 1.5, title, ha='center', va='top', fontsize=ts, fontweight='bold', color=ec)
    ax.text(x + 1.2, y + h - 4.8, '\n'.join(lines), ha='left', va='top', fontsize=fs, linespacing=1.4)
def arrow(x1, y1, x2, y2, c='#444444', lw=1.3):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=13, lw=lw, color=c, shrinkA=0, shrinkB=0))
def stage(x, w, num, label, c):
    ax.add_patch(FancyBboxPatch((x, 57.5), w, 3.8, boxstyle='round,pad=0.1,rounding_size=0.9', fc=c, ec=c, lw=0))
    ax.text(x + w / 2, 59.4, f'{num}   {label}', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
C1, C2, C3, C4, C5 = '#1f5f8b', '#2e7d32', '#6a1b9a', '#c1550a', '#8b0000'
F1, F2, F3, F4, F5 = '#e8f1f8', '#eaf5ea', '#f3eaf8', '#fdf0e6', '#fbeaea'
X = [1, 31, 61, 91]; BW = 28
stage(X[0], BW, 1, 'Data and preprocessing', C1); stage(X[1], BW, 2, 'Weakly supervised patch encoder', C2)
stage(X[2], BW, 3, 'Patch-to-slide aggregation', C3); stage(X[3], BW, 4, 'Survival modelling and evaluation', C4)
# column 1
box(X[0], 33.5, BW, 22, 'H&E tissue microarray (TMA) cores', [
    '278 women with invasive breast carcinoma',
    '7 TMA slides, 1.5-mm cores, 1-4 cores per patient',
    'Scanned at 20x (0.5 um per pixel)',
    'Tiling into 256 x 256-pixel patches',
    'Removal of background / non-tissue patches',
    'Macenko stain normalization (fixed reference)',
    'Per-channel z-score standardization',
    '81,402 patches (median 262 per patient)'], F1, C1)
box(X[0], 12, BW, 19.5, 'Clinical variables and outcome', [
    'Original clinical model: age, N category, AJCC stage',
    'Guideline model: age, AJCC stage, ER, PR, HER2, grade',
    'Endpoint: overall survival (65 deaths / 278)',
    'Patient-level random split 7:3 (fixed seed):',
    '   training n = 194, test n = 84',
    'Test set untouched until final evaluation'], F1, C1)
# column 2
box(X[1], 27, BW, 28.5, 'Patch-level CNN classifier', [
    'Weak label = patient 5-year OS status:',
    '   died within 60 months (1) vs alive beyond 60 (0);',
    '   every patch inherits its patient label',
    'Trained on training-cohort patients only',
    '   (193 patients, 56,499 patches)',
    'ImageNet-initialized ResNet50 (primary);',
    '   ResNet18, DenseNet121, CrossFormer compared',
    'SGD, cosine learning-rate schedule, 12 epochs',
    'Output per patch: risk probability p in [0, 1]',
    '   and predicted label 1(p >= 0.5)'], F2, C2)
box(X[1], 12, BW, 13, 'Interpretability (qualitative)', [
    'Grad-CAM on the patch classifier',
    'Spatial risk-probability maps of TMA slides',
    '(not validated by pathologist review)'], F2, C2)
# column 3
box(X[2], 40, BW, 15.5, 'A. Patch-level histogram (PLH)', [
    'Fraction of a patient\'s patches in each of the',
    '101 probability bins (0.00, 0.01, ..., 1.00): 101-d',
    'Fraction predicted label 0 / label 1: 2-d',
    '=> 103 PLH features'], F3, C3)
box(X[2], 22, BW, 16, 'B. Bag-of-words with TF-IDF (BoW)', [
    'Bins are "words", a patient is a "document"',
    'Weight = term count x log(N / document frequency),',
    'L2-normalized per patient (101 + 2 words)',
    '=> 103 BoW features'], F3, C3)
box(X[2], 12, BW, 8, 'Concatenation', ['206-dimensional pathomics feature vector'], F3, C3)
# column 4
box(X[3], 38.5, BW, 17, 'Pathomics signature (training set only)', [
    'Z-score with training mean / SD',
    'Correlation filter |r| > 0.8:   206 -> 95 features',
    'Univariable Cox p < 0.05:        95 -> 79 features',
    'Elastic-net Cox (l1 ratio 0.1, 10-fold CV): 3 features',
    '   BoW_prob(0.50), BoW_prob(0.58), BoW_pred(1)',
    'Cox PH -> pathomics prognostic score'], F4, C4)
box(X[3], 26, BW, 10.5, 'Cox proportional-hazards models', [
    'Clinical  |  Pathomics  |  Combined (late fusion of',
    'the clinical and pathomics scores)',
    'Risk cut-offs fixed on the training set (X-tile),',
    'then applied unchanged to the test set'], F4, C4)
box(X[3], 6, BW, 18, 'Evaluation (test cohort is primary)', [
    'Harrell C-index, 2000-sample bootstrap 95% CI,',
    '   paired-bootstrap model comparison (Holm-adjusted)',
    'Time-dependent AUC (5, 7, 9 y), Kaplan-Meier',
    'Calibration and decision-curve analysis',
    'Sensitivity analyses: four backbones, PLH vs BoW,',
    '   1000 random splits, guideline clinical baseline'], F5, C5)
# arrows
arrow(X[0] + BW, 44.5, X[1], 44.5)                                        # patches -> encoder
arrow(X[1] + BW, 41, X[2], 47.5); arrow(X[1] + BW, 41, X[2], 30)          # encoder -> PLH / BoW
arrow(X[2] + BW / 2, 40, X[2] + BW / 2, 38.2); arrow(X[2] + BW / 2, 22, X[2] + BW / 2, 20.2)   # PLH -> BoW -> concat
# concatenated vector -> pathomics signature (routed through the inter-column gap)
ax.plot([X[2] + BW, 90, 90], [16, 16, 47], color='#444444', lw=1.3); arrow(90, 47, X[3], 47)
arrow(X[3] + BW / 2, 38.5, X[3] + BW / 2, 36.7); arrow(X[3] + BW / 2, 26, X[3] + BW / 2, 24.2)
# clinical variables -> Cox models (routed below the boxes and up the right margin)
ax.plot([15, 15, 122, 122], [12, 2.2, 2.2, 31], color=C1, lw=1.3); arrow(122, 31, X[3] + BW, 31, c=C1)
ax.text(66, 3.4, 'clinical variables -> Clinical model (and clinical component of the Combined model)', fontsize=7, color=C1, ha='center', style='italic')
arrow(X[1] + 4, 27, X[1] + 4, 25.2, c=C2)
fig.savefig('analysis_outputs/figures/Figure1_workflow.png', dpi=300, bbox_inches='tight', pad_inches=0.06, facecolor='white')
print('Figure 1 saved')
