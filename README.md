# Weakly supervised patch-to-slide deep pathomics for overall-survival prediction in breast cancer (H&E TMA cores)

Code, de-identified per-patient tables, locked cut-offs and analysis outputs accompanying the manuscript
*A weakly supervised patch-to-slide deep pathomics framework for overall-survival prediction from H&E tissue-microarray histopathology in breast cancer* (Biomedical Signal Processing and Control, revised version).
The version used for the revised manuscript is tagged `v1.2` (v1.0: pipeline outputs and analysis scripts; v1.1: TCGA-BRCA external validation; v1.2: the scripts of every remaining table, exact reproduction of the deployed features, the data and analyses of the 13 excluded hold-out patients and of every saved ResNet50 checkpoint, and corrections listed in the release notes).

## Repository layout

| Folder / file | Content |
|---|---|
| `original_notebooks/` | The original pipeline as run for the study (Jupyter notebooks Step0–Step5): patient split and label files (Step0), patch tumor prediction (Step0.1), encoder training (Step1.1), Grad-CAM visualization (Step1.1.2), patch-level metrics and export of the deployed predictions (Step1.2), patch-to-slide aggregation (Step1.3), feature selection and Cox models (Step2.2, Step3.2, Step4.2), C-index and time-dependent AUC (Step5.1, Step5.2). They depend on the `onekey_algo` package and on the patch images, which are not distributed; their saved outputs document the original run. |
| `pipeline_outputs/` | Outputs of that pipeline, from which every number of the paper is reproduced: patch-level predictions of the four encoders (`results/Pathomics_Slice_*`; `results/ALL_DL_PREDICTIONS.csv` holds the deployed ResNet50 probabilities and predicted labels of all 291 patients with cores and follow-up; `results/resnet50_checkpoints.csv.gz` holds those of all 12 saved ResNet50 checkpoints), the 206 slide-level features (`features/prob_histogram.csv`, `pred_histogram.csv`, `prob_tfidf.csv`, `pred_tfidf.csv`), the Cox inputs and predictions of the deployed models (`features/*_cox.csv`, `features/*_features_norm.csv`, `results/*_cox_predictions_*.csv`), the survival file (`sur.csv`: 278 patients, `group` = train / test), the variables of the original Clinical model (`data/clinical.csv`; column `M` is the TMA-set indicator and column `HER2` the original inverse coding) and the nomogram exported by the pipeline (`img/nomogram.png`). `config.yaml` is the original configuration file (paths of the original workstation). |
| `clinical/clinical_guideline_278.csv` | Curated guideline-level clinical variables of the 278 patients (ER, PR, HER2 [FISH first], grade, surrogate subtype, T, N, AJCC stage, age, TMA set); encodings in Supplementary Table S4 of the paper. |
| `clinical/clinical_excluded_holdout13.csv` | The same variables, with overall survival, for the 13 hold-out patients excluded before survival modeling because their H&E images were judged unusable on pathological review (`group` = excluded). |
| `analysis/` | Open-source re-implementation of the downstream pipeline and every analysis of the paper (Python). |
| `analysis_outputs/` | All tables (CSV) and data figures (PNG) produced by `analysis/` (see the mapping below), `document_frequencies_291.csv` (the fixed document-frequency vector of the BoW features) and `locked_cutoffs.json` (locked cut-offs, thresholds, selection settings, analysis seed and study split). |

The trained encoder weights (`resnet50_CV1.pth`, `CrossFormer_CV1.pth`) and the inference script are available from the corresponding authors on request.

## Study split and encoder data, as run by the original pipeline

- `original_notebooks/Step0` generated ten random patient-level partitions of the 291 patients with annotated cores and survival follow-up (test fraction 0.33); the pipeline used the second, RND-1, with 194 training and 97 hold-out patients. As specified in the original protocol, the partition was required to be balanced in the clinical variables then available (age, examined and positive lymph nodes, T and N categories, AJCC stage, HER2 status, TMA set); `analysis_outputs/Table_split_balance_194_vs_97.csv` gives the tests for this partition (`analysis/holdout_sensitivity.py`).
- 13 of the 97 hold-out patients were excluded before survival modeling because their H&E images were judged unusable on pathological review; their clinical variables and survival are in `clinical/clinical_excluded_holdout13.csv` and their slide features in `pipeline_outputs/features/path_features.csv`. The other 84 form the test cohort (`pipeline_outputs/sur.csv`). `analysis/holdout_sensitivity.py` applies the frozen models to all 97 hold-out patients (Supplementary Table S10).
- The encoders were trained on the 193 training patients with a defined 5-year label; patients alive with less than 60 months of follow-up have no label. The validation set of encoder training was the hold-out set: in `pipeline_outputs/results/ALL_DL_PREDICTIONS.csv`, rows 1–56,499 are the training patches and the remaining 28,955 rows the validation patches of 98 patients (the 97 hold-out patients and the one training patient without a defined label). Patients without a defined label carry the placeholder `gt` = 1 in that file.
- The framework computed the patch-level AUC of this validation set after every epoch. The deployed ResNet50 checkpoint (`Epoch-1`, 0-based, i.e. the second of 12 epochs) is the epoch with the highest validation AUC; ResNet18 and DenseNet121 use the epochs set in `original_notebooks/Step1.2` (`epoch_mapping`: 5 and 2, 0-based), and CrossFormer the last completed epoch (9, 0-based; training stopped after 10 of 16 planned epochs). `pipeline_outputs/results/resnet50_checkpoints.csv.gz` holds the patch predictions of all 12 ResNet50 checkpoints as written by the training framework (file paths replaced by patient IDs; epoch 1 equals `ALL_DL_PREDICTIONS.csv`), and `analysis/checkpoint_sensitivity.py` reruns the signature on each of them (Supplementary Table S11).

## Reproducing the analyses

```bash
pip install -r requirements.txt
cd <repository root>
python analysis/pathomics_pipeline.py          # reproduces the 206 deployed features of the 291 patients exactly; writes document_frequencies_291.csv
python analysis/patch_level_metrics.py         # Table 2
python analysis/deployed_models.py             # Tables 3a, 3b (with Holm adjustment) and 4; Supplementary Tables S1 and S5 (median cut-offs)
python analysis/split_sensitivity.py           # Section 3.3: 1000 random splits; data of Supplementary Figure S1
python analysis/holdout_sensitivity.py         # Supplementary Table S10 (whole hold-out block) and balance of the original 194/97 partition
python analysis/checkpoint_sensitivity.py      # Supplementary Table S11 (all 12 ResNet50 checkpoints)
python analysis/backbone_ablation.py           # Supplementary Tables S2, S3 and S8; data of Supplementary Figure S2
python analysis/guideline_clinical_model.py    # Table 1, Table 5, Supplementary Tables S4 and S6
python analysis/make_figures.py                # Figures 2, 5, 7, 8, 9, 10 and Supplementary Figures S1–S3; Table 6 and Supplementary Table S5 (locked cut-offs)
python analysis/make_figure6_labels.py         # Figure 6 (nomogram exported by the original pipeline, axis labels unified)
python analysis/external_validation_tcga.py <dir with the TCGA downloads>   # Table 7 and Supplementary Table S9 (set TCGA_GRADE_FILE, see below)
python analysis/make_figure11_external.py      # Figure 11
```

Run the scripts in this order: later scripts read tables written by earlier ones. Every script changes to the repository root itself. Tested with Python 3.14.3, numpy 2.4.4, pandas 2.3.3, scipy 1.17.1, scikit-learn 1.9.1, scikit-survival 0.28.0, lifelines 0.30.3, matplotlib 3.10.8, pillow 12.2.0, openpyxl 3.1.5 and xlrd 2.0.2. Run in a copy of the repository from which all outputs had been deleted, the scripts regenerate every table and array of `analysis_outputs/` exactly and every figure pixel for pixel.

`analysis/parse_raw_clinical.py` and `analysis/build_clinical_table.py` document how the guideline variables were decoded from the original pathology sheets. The raw sheets are not distributed (they contain free-text pathology reports); `clinical/clinical_guideline_278.csv` is the decoded table without the free-text columns.

## Where each result of the paper comes from

| Paper item | File in `analysis_outputs/` | Script |
|---|---|---|
| Table 1 | `Table1_extended.csv` | `guideline_clinical_model.py` |
| Table 2 | `Table2_patch_level_4backbones.csv` | `patch_level_metrics.py` |
| Tables 3a, 3b | `Table3_cindex_CI.csv`, `Table3_cindex_comparison.csv` | `deployed_models.py` |
| Table 4 | `Table4_cox_coefficients.csv` | `deployed_models.py` |
| Table 5 | `Table_guideline_models.csv`, `Table_guideline_pairwise.csv` | `guideline_clinical_model.py` |
| Table 6 | `Table5_tdAUC_CI.csv` (file name from an earlier table numbering) | `make_figures.py` |
| Table 7 | `external_tcga/Table_external_TCGA.csv` | `external_validation_tcga.py` |
| Supplementary Table S1 | `TableS1_slide_level_encoders.csv` | `deployed_models.py` |
| Supplementary Tables S2 and S8 | `Table_backbone_ablation.csv` (paired comparisons: `Table_backbone_pairwise.csv`) | `backbone_ablation.py` |
| Supplementary Table S3 | `Table_LOO.csv` | `backbone_ablation.py` |
| Supplementary Table S4 | `Table_guideline_univariable.csv`, `Table_guideline_multivariable.csv` | `guideline_clinical_model.py` |
| Supplementary Table S5 | `Table_KM_locked_cutoffs.csv`, `TableS5_median_cutoff_logrank.csv` | `make_figures.py`, `deployed_models.py` |
| Supplementary Table S6 | `Table_test_nested_LRT.csv` | `guideline_clinical_model.py` |
| Supplementary Table S9 | `external_tcga/Table_S9_casemix.csv` | `external_validation_tcga.py` |
| Supplementary Table S10 | `TableS10_holdout_with_excluded.csv` | `holdout_sensitivity.py` |
| Supplementary Table S11 | `TableS11_checkpoint_sensitivity.csv` | `checkpoint_sensitivity.py` |
| Section 2.1, balance of the original 194/97 partition | `Table_split_balance_194_vs_97.csv` | `holdout_sensitivity.py` |
| Section 3.3, 1000 random splits | `split_sens_clinical.npy`, `Table_split_sensitivity_summary.csv` | `split_sensitivity.py` |
| Figures 2, 5, 7–10; Supplementary Figures S1–S3 | `figures/*.png` | `make_figures.py` |
| Figure 6 | `figures/Figure6_nomogram.png` | `make_figure6_labels.py` |
| Figure 11 | `figures/Figure11_TCGA_external.png` | `make_figure11_external.py` |

Supplementary Table S7 (feature nomenclature) contains no data. Figure 1 is a schematic, and Figures 3 and 4 are illustrations that require the TMA images (Figure 3 is produced by `original_notebooks/Step1.1.2`); they are not part of this repository.

## Figures

Figures are drawn at their final print size by `analysis/fig_layout.py`: 15.0 cm wide (12.0 cm for Supplementary Figure S2), identical 0.3-cm gaps between panels, a 0.5-cm outer margin, 7–8-pt Arial with 10-pt bold panel letters, 600 dpi. `analysis_outputs/figures/figure_layout_metrics.csv` records the measured margins and gaps. With Arial installed (Windows, macOS) the figures are pixel-identical to those of the paper; without it (e.g., Linux without the Microsoft core fonts) a fallback font is used, layout checks become warnings, and all numbers are unaffected.

## External validation in TCGA-BRCA (Section 3.10, Table 7, Figure 11)

The frozen clinical models (deployed Clinical model: age, N category, AJCC stage; Clinical-guideline model: age, AJCC stage, ER, PR, HER2, grade 3) are applied without refitting to the public TCGA-BRCA cohort. The pathomics signature cannot be applied (TMA-core encoder; TCGA-BRCA provides whole-slide sections). Open-access inputs, placed in one directory that is passed as the first argument:

| File | Source | MD5 |
|---|---|---|
| `nationwidechildrens.org_clinical_patient_brca.txt` | GDC, https://api.gdc.cancer.gov/data/8162d394-8b64-4da2-9f5b-d164c54b9608 | `f271f56d44b62d41bd6ddb0d26e525b4` |
| `TCGA-CDR-SupplementalTableS1.xlsx` | TCGA Pan-Cancer Clinical Data Resource (Liu et al., Cell 2018), https://api.gdc.cancer.gov/data/1b5f413e-a8d1-4d10-92eb-7c4ae739ed81 | `a4591b2dcee39591f59e5e25a6ce75fa` |
| `mmc3.xlsx` (Data S2) | Thennavan et al., Molecular analysis of TCGA breast cancer histologic types, Cell Genomics 2021;1:100067, doi:10.1016/j.xgen.2021.100067 (PMC9028992): expert-panel tubule, pleomorphism and mitosis scores | – |

Download Data S2 from the supplementary material of the article in a browser (scripted downloads may receive an HTML page instead of the workbook) and point the environment variable `TCGA_GRADE_FILE` to it, e.g. `export TCGA_GRADE_FILE=/path/to/mmc3.xlsx` (bash) or `$env:TCGA_GRADE_FILE = "C:\path\to\mmc3.xlsx"` (PowerShell). The script stops if the file is missing or cannot be read; without the variable, grade is imputed from the development data (a sensitivity analysis only). A correct run reports 1014 eligible patients (`cohort_flow.csv`: 1097 → 1014) and an expert-panel Nottingham grade for 956 of them.

`analysis_outputs/external_tcga/` holds the results (`Table_external_TCGA.csv`, `Table_S9_casemix.csv`, calibration tables, cohort flow, frozen coefficients and cut-offs in `frozen_models.json`) and the derived per-patient table `tcga_brca_external_cohort.csv` (TCGA barcodes, coded variables, survival, frozen linear predictors).

## Pipeline summary (Algorithm 1 of the paper)

1. Patient-level 2:1 split of the original pipeline (see above): training cohort n = 194; hold-out block n = 97, of whom 13 were excluded for unusable images (test cohort n = 84).
2. TMA cores tiled into 256 × 256 patches at 20×; background removal; Macenko normalization with one fixed reference; z-score.
3. Weak label = patient 5-year overall-survival status (death ≤ 60 months vs survival > 60 months) inherited by every patch; encoder trained on training patients only, with the hold-out block as validation set (ResNet50 checkpoint = epoch with the highest validation AUC).
4. Per-patient aggregation: PLH (fraction of patches per 0.01 probability bin + predicted-label bins, 103-d) and BoW (L2-normalized TF-IDF of the same bins, 103-d) → 206-d vector.
5. Training-only feature selection: z-score → correlation filter |r| > 0.8 → univariable Cox p < 0.05 → elastic-net Cox (L1 ratio 0.1), matched complexity 3 features: `BoW_prob(0.50)`, `BoW_prob(0.58)`, `BoW_pred(1)`.
6. Cox models: Clinical (age, N, AJCC), Pathomics, Combined (late fusion); guideline-level Clinical (age, AJCC, ER, PR, HER2, grade) and its fusion.
7. Evaluation in the test cohort: C-index with 2000-sample bootstrap CI, paired-bootstrap comparisons (Holm), time-dependent AUC, Kaplan–Meier with cut-offs locked on the training cohort, calibration, decision curves, nested likelihood-ratio test, sensitivity analyses.

## Data statement

All tables are de-identified (study codes only). Patient-level clinical and survival data are shared under the institutional approval described in the paper; the TMA images are available from the corresponding authors on reasonable request.

## Citation

Please cite the paper (reference to be added after acceptance) and this repository (https://github.com/lihaiyu1988/BC-TMA-pathomics-survival, release v1.2).

## License

MIT (see `LICENSE`).
