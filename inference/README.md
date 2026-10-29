# Trained encoder weights and patch-level inference (release v1.3)

The checkpoints are attached to release v1.3 of this repository as release assets (they are not part of the source tree):

| Asset | Encoder | Role in the paper | Size | SHA-256 |
|---|---|---|---|---|
| `resnet50_CV1_epoch2.pth` | resnet50 | deployed encoder (epoch 2 of 12, highest validation AUC; Epoch-1 in the 0-based file names of the training framework) | 94.4 MB | `7d7d6db7007431cf171110a9ff54e82fe29f2ae9813ca6758c4386c3e1c176af` |
| `densenet121_CV1_epoch3.pth` | densenet121 | encoder comparison (epoch 3 of 12, as set in original_notebooks/Step1.2) | 28.4 MB | `96087dc8d6470af0e8e5953f2e25af048b34d71e03e31b4ee758a35da1878c3f` |
| `resnet18_CV1_epoch6.pth` | resnet18 | encoder comparison (epoch 6 of 12, as set in original_notebooks/Step1.2) | 44.8 MB | `064673c1f770411251f4f42c67c5a4bf05dcb49dc540cb0625bb18089af1c7ff` |

Each file is the checkpoint written by the training framework at the epoch given, renamed but otherwise unchanged
(the SHA-256 values are those of the original files): a Python dict with the keys `global_step`, `model_state_dict`,
`optimizer_state_dict` and `lr_scheduler_state_dict`. The entry `model_state_dict` is the torchvision `state_dict` of
the architecture named in the table with a two-class output layer (class 1 = poor prognosis, i.e. death within
60 months of surgery); `infer_patches.py` reads this entry and ignores the optimizer and scheduler state, which is
not needed for inference. The CrossFormer checkpoint of the comparison is not included in the release.

## Input and preprocessing

The encoders were trained and applied on the 256 × 256 patches as tiled from the scanned TMA slides: no stain
normalization was applied in the original pipeline (the normalized patches displayed by `original_notebooks/Step0`
were an illustration that the pipeline did not use), and the patch filtering that precedes inference (background removal and the automated tissue classifier)
(`original_notebooks/Step0.1`) is not re-implemented here. `infer_patches.py` applies a checkpoint to such patches with
the evaluation transform of the original run (`viz/task.json`: input size 224 × 224, ImageNet normalization): the image
is scaled to [0, 1], resized to 224 × 224 by bilinear interpolation without antialiasing
(`torch.nn.functional.interpolate`, the resize of the training framework) and standardized with the ImageNet mean and
standard deviation. The antialiased resize that torchvision applies to PIL images changes the probabilities by about
0.01 on average and does not reproduce the released predictions. The script writes `fpath, label-1, pred_label` per
patch, the two quantities from which `analysis/pathomics_pipeline.py` builds the PLH/BoW features
(`pipeline_outputs/results/ALL_DL_PREDICTIONS.csv` holds them for all patches of the study, with the patch path
replaced by the patient ID).

```bash
pip install torch torchvision          # CPU builds are sufficient
python inference/infer_patches.py --weights resnet50_CV1_epoch2.pth --arch resnet50 --input <patch directory> --output predictions.csv
```

## Example patches and the two reference files

`example_patches/` holds nine patches of the validation set of the original run (nine test-cohort patients, five of
the six scanned slides, as tiled), chosen to span the range of predicted probabilities.

* `example_predictions.tsv` holds the rows of these patches in the prediction file of the original run
  (`valid/Epoch-1.txt` of the training framework: path, probability of the predicted class, predicted label, label;
  `label-1` = score if the predicted label is 1, else 1 − score), i.e. the values that entered
  `pipeline_outputs/results/ALL_DL_PREDICTIONS.csv`. Running

  ```bash
  python inference/infer_patches.py --weights resnet50_CV1_epoch2.pth --arch resnet50 --input inference/example_patches --verify-log inference/example_predictions.tsv
  ```

  reproduces them: the predicted class is identical and the probabilities match the three decimals of the file up to
  the numerical precision of the computation (differences of up to about 0.001 come from the GPU arithmetic of the
  original run).

* `example_predictions_checkpoint.csv` is the output of `infer_patches.py` with `resnet50_CV1_epoch2.pth` on these
  patches (`fpath, label-1, pred_label`; paths written relative to `inference/`); `--verify-log inference/example_predictions_checkpoint.csv` should reproduce
  it to about 1e-4 (BLAS library, GPU, JPEG decoder) and checks the installation and the integrity of the weight
  file. `example_predictions_checkpoint.csv` was written on a machine other than the training workstation (torch 2.14.1, CPU); in that run the predicted class of all nine patches agreed with the prediction file and the probabilities differed from its three-decimal values by at most 0.0003.

## The two parts of the released prediction file

The validation rows of `ALL_DL_PREDICTIONS.csv` (rows 56,500–85,454: the 97 hold-out patients, including the test
cohort, and one training patient) were computed after the epoch (`valid/Epoch-1.txt`; the patch-level AUC of this file
is the validation AUC stored by the framework). Applied to the patches of the original run (retrieved from the training workstation for the second revision; not distributed), the released checkpoint with `infer_patches.py` reproduced all 28,955 validation rows to within 0.0026 of their three-decimal values (mean absolute difference 0.00028; 89% of the patches identical to three decimals; predicted class identical for all but two patches, whose probabilities lie at the 0.5 threshold), patch-level AUC 0.5439 recomputed against 0.5441 in the file. The residual differences are the numerical noise of the GPU arithmetic of the original run.

For the training split (rows 1–56,499) the framework recorded these values during the training epoch itself: the
model was in training mode, the patches were augmented online (random cropping, horizontal and vertical flipping), and
the weights were still being updated over the 1,766 steps of the epoch. The saved checkpoint applied in evaluation
mode therefore does not reproduce these values exactly. On the training split the same run differed from the training-time values by up to 0.99 (mean 0.112; predicted class identical for 52,879 of 56,499 patches), as expected from the way those values were recorded.
