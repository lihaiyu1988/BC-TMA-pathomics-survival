# Weakly supervised patch-to-slide deep pathomics for overall-survival prediction in breast cancer (H&E TMA cores)

Code, de-identified per-patient tables, locked cut-offs and analysis outputs accompanying the manuscript
*A weakly supervised patch-to-slide deep pathomics framework for overall-survival prediction from H&E tissue-microarray histopathology in breast cancer* (Biomedical Signal Processing and Control, revised version).

## Repository layout

| Folder / file | Content |
|---|---|
| `original_notebooks/` | The original OnekeyAI-based pipeline (Jupyter notebooks Step0–Step5): patch preprocessing, encoder training (ResNet18/50, DenseNet121, CrossFormer), patch-to-slide aggregation (PLH histogram and BoW/TF-IDF), feature selection, Cox models, Kaplan–Meier, time-dependent AUC, nomogram. These notebooks depend on the `onekey_algo` package and on the patch images, which are not distributed. |
| `pipeline_outputs/` | Outputs of that pipeline needed to reproduce every number in the paper without the images: patch-level predictions of the four encoders (`results/Pathomics_Slice_*`, `results/ALL_DL_PREDICTIONS.csv`), the 206-dimensional PLH/BoW features (`features/`), the deployed Cox inputs and predictions (`features/*_cox.csv`, `results/*_cox_predictions_*.csv`), the survival file (`sur.csv`) and the clinical file used by the original Clinical model (`data/clinical.csv`; note that column `M` is the TMA-set indicator and column `HER2` is the original inverse coding, see the paper). |
| `clinical/clinical_guideline_278.csv` | Curated guideline-level clinical variables (ER, PR, HER2 [FISH-first], grade, surrogate subtype, T, N, AJCC stage, age) for the 278 patients, with the encodings described in Supplementary Table S4 of the paper. |
| `analysis/` | Open-source re-implementation of the downstream pipeline and of every analysis added in the revision (Python; no proprietary dependency). |
| `analysis_outputs/` | All tables (CSV) and figures (PNG, 300 dpi) of the revised paper, plus `locked_cutoffs.json` (X-tile cut-offs, patch-level thresholds, feature-selection settings, seed). |
| `requirements.txt` | Python dependencies (tested with Python 3.14, lifelines 0.30.3, scikit-survival 0.28.0). |

Trained encoder weights (`resnet50_CV1.pth`, `CrossFormer_CV1.pth`) and the inference script are exported from the training environment and attached to the Zenodo record of this repository (they exceed the GitHub file-size limit).

## Reproducing the analyses

```bash
pip install -r requirements.txt
cd <repository root>
python analysis/pathomics_pipeline.py          # validates that the re-implemented pipeline reproduces the deployed 3-feature signature
python analysis/backbone_ablation.py           # Supplementary Table S2, S3 (backbones, PLH/BoW ablation, IDF check, leave-one-out)
python analysis/guideline_clinical_model.py    # Table 1 (extended), Table 5, Supplementary Tables S4 and S6
python analysis/make_figures.py                # Figures 2, 5, 7, 8, 9, 10, S1–S3 and Tables 5/6 CSVs
python analysis/make_figure1.py                # Figure 1 (flow chart)
```

`analysis/parse_raw_clinical.py` and `analysis/build_clinical_table.py` document how the guideline variables were decoded from the original pathology sheets; the raw sheets are not distributed (they contain free-text pathology reports), and `clinical/clinical_guideline_278.csv` is their output.

## Pipeline summary (Algorithm 1 of the paper)

1. Patient-level random 7:3 split with a fixed seed (training n = 194, test n = 84); the test cohort is untouched until the final evaluation.
2. TMA cores tiled into 256 × 256 patches at 20×; background removal; Macenko normalisation with one fixed reference; z-score.
3. Weak label = patient 5-year overall-survival status (death ≤ 60 months vs survival > 60 months) inherited by every patch; encoder trained on training patients only.
4. Per-patient aggregation: PLH (fraction of patches per 0.01 probability bin + predicted-label bins, 103-d) and BoW (L2-normalised TF-IDF of the same bins, 103-d) → 206-d vector.
5. Training-only feature selection: z-score → correlation filter |r| > 0.8 → univariable Cox p < 0.05 → elastic-net Cox (l1 ratio 0.1), matched complexity 3 features: `BoW_prob(0.50)`, `BoW_prob(0.58)`, `BoW_pred(1)`.
6. Cox models: Clinical (age, N, AJCC), Pathomics, Combined (late fusion); guideline-level Clinical (age, AJCC, ER, PR, HER2, grade) and its fusion.
7. Evaluation on the test cohort: C-index with 2000-sample bootstrap CI, paired-bootstrap comparisons (Holm), time-dependent AUC, Kaplan–Meier with cut-offs locked on the training cohort, calibration, decision curves, nested likelihood-ratio test, sensitivity analyses.

## Data statement

All tables are de-identified (study codes only). Patient-level clinical and survival data are shared under the institutional approval described in the paper; the TMA images are available from the corresponding authors on reasonable request.

## Citation

Please cite the paper (reference to be added after acceptance) and this repository (Zenodo DOI to be added).

## License

MIT (see `LICENSE`).
