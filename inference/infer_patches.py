#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Patch-level inference with the released encoder weights (release v1.3).

Applies a trained encoder (ResNet50 = the deployed encoder; ResNet18 or DenseNet121 = the encoders of the
backbone comparison) to 256 x 256 H&E patch images as tiled (no stain normalization) and writes one row per patch:

    fpath, label-1, pred_label

`label-1` is the softmax probability of class 1 (poor prognosis: death within 60 months of surgery) and
`pred_label` the predicted class (1 if label-1 >= 0.5, else 0). These are the two quantities from which the
PLH/BoW features of the paper are built (analysis/pathomics_pipeline.py reads them from
pipeline_outputs/results/ALL_DL_PREDICTIONS.csv, where the patch path has been replaced by the patient ID).

Preprocessing is the evaluation transform of the original training run (viz/task.json: input_size 224 x 224,
normalize_method "imagenet"): the patch is scaled to [0, 1], resized to 224 x 224 pixels by bilinear interpolation
without antialiasing (torch.nn.functional.interpolate with align_corners=False, the resize of the training framework;
the antialiased resize that torchvision applies to PIL images changes the probabilities by about 0.01 on average and
does not reproduce the released predictions) and standardized with the ImageNet mean and standard deviation. No stain
normalization is applied, because none was applied in the original pipeline: the patches enter the encoder as tiled.
The patch filtering of the original pipeline (background removal and the automated tissue classifier) is not re-implemented here. With this transform the deployed
checkpoint reproduces the validation rows of pipeline_outputs/results/ALL_DL_PREDICTIONS.csv (inference/README.md).

Usage
    python inference/infer_patches.py --weights resnet50_CV1_epoch2.pth --arch resnet50 \
        --input <directory with patch .jpg/.png files (searched recursively)> --output predictions.csv

    # check of the installation against the reference output distributed with the script (nine example patches):
    python inference/infer_patches.py --weights resnet50_CV1_epoch2.pth --arch resnet50 \
        --input inference/example_patches --verify-log inference/example_predictions_checkpoint.csv

    # comparison with the rows of the same patches in the prediction file of the original run (tab-separated:
    # fpath, probability of the predicted class, predicted label, label); the nine example patches are validation-set
    # patches, whose rows the checkpoint reproduces (inference/README.md):
    python inference/infer_patches.py --weights resnet50_CV1_epoch2.pth --arch resnet50 \
        --input inference/example_patches --verify-log inference/example_predictions.tsv

Requires torch >= 2.0 and torchvision (a CPU build is sufficient).
"""
import argparse, csv, hashlib, os, sys, time

import torch
import torchvision
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
EXTS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')


def build_model(arch, weights):
    if arch == 'resnet50':
        model = torchvision.models.resnet50(weights=None, num_classes=2)
    elif arch == 'resnet18':
        model = torchvision.models.resnet18(weights=None, num_classes=2)
    elif arch == 'densenet121':
        model = torchvision.models.densenet121(weights=None, num_classes=2)
    else:
        raise SystemExit(f'unknown architecture {arch}')
    state = torch.load(weights, map_location='cpu', weights_only=True)
    # the released files are the checkpoints written by the training framework: a dict with the keys global_step,
    # model_state_dict, optimizer_state_dict and lr_scheduler_state_dict; only the model weights are needed here
    for key in ('model_state_dict', 'state_dict'):
        if isinstance(state, dict) and key in state:
            state = state[key]
            break
    state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def transform(size=224):
    """Evaluation transform of the original run: [0, 1] tensor -> bilinear resize to size x size without antialiasing
    (identical to torchvision.transforms.Resize(..., antialias=False) applied to a tensor) -> ImageNet standardization."""
    import torch.nn.functional as F
    from torchvision import transforms
    to_tensor, normalize = transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def apply(img):
        x = to_tensor(img)
        if tuple(x.shape[-2:]) != (size, size):
            x = F.interpolate(x[None], size=(size, size), mode='bilinear', align_corners=False)[0]
        return normalize(x)
    return apply


def list_patches(root):
    if os.path.isfile(root):
        return [l.strip() for l in open(root, encoding='utf-8') if l.strip()]
    out = []
    for dp, _, fn in os.walk(root):
        for f in sorted(fn):
            if f.lower().endswith(EXTS):
                out.append(os.path.join(dp, f))
    return sorted(out)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--weights', required=True, help='checkpoint file (.pth): a training checkpoint of the original run (its model_state_dict entry is read) or a plain torchvision state_dict')
    ap.add_argument('--arch', default='resnet50', choices=['resnet50', 'resnet18', 'densenet121'])
    ap.add_argument('--input', required=True, help='directory of patch images (recursive) or a text file listing patch paths')
    ap.add_argument('--output', default='predictions.csv')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--verify-log', help='reference file to compare with: a prediction log of the training framework (tab-separated: fpath, pred_score, pred_label, gt) or an output file of this script (fpath, label-1, pred_label)')
    args = ap.parse_args()

    print(f'weights {args.weights}  sha256 {sha256(args.weights)}')
    model = build_model(args.arch, args.weights).to(args.device)
    tf = transform()
    paths = list_patches(args.input)
    if not paths:
        raise SystemExit(f'no patch images found under {args.input}')
    print(f'{len(paths)} patches, device {args.device}')
    rows, t0 = [], time.time()
    with torch.no_grad():
        for i in range(0, len(paths), args.batch_size):
            batch = paths[i:i + args.batch_size]
            x = torch.stack([tf(Image.open(p).convert('RGB')) for p in batch]).to(args.device)
            prob = torch.softmax(model(x), dim=1)[:, 1].cpu().numpy()
            for p, q in zip(batch, prob):
                rows.append((p, float(q), int(q >= 0.5)))
            if (i // args.batch_size) % 20 == 0:
                print(f'  {min(i + args.batch_size, len(paths))}/{len(paths)}  {time.time() - t0:.0f} s', flush=True)
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['fpath', 'label-1', 'pred_label'])
        for p, q, l in rows:
            w.writerow([p, f'{q:.6f}', l])
    print(f'written {args.output} ({len(rows)} rows)')

    if args.verify_log:
        # the original log stores the probability of the PREDICTED class (pred_score) and the predicted label;
        # label-1 = pred_score if pred_label == 1 else 1 - pred_score (original_notebooks/Step1.2)
        ref = {}
        for line in open(args.verify_log, encoding='utf-8'):
            line = line.rstrip('\n')
            if not line or line.startswith('fpath'):
                continue
            if '\t' in line:      # log of the training framework
                parts = line.split('\t')
                score, label = float(parts[1]), int(parts[2])
                value, name = (score if label == 1 else 1.0 - score), parts[0]
            else:                 # output file of this script
                name, value, label = line.rsplit(',', 2)
                value, label = float(value), int(label)
            ref[os.path.basename(name.replace('\\', '/'))] = (value, label)
        diffs, agree, n = [], 0, 0
        for p, q, l in rows:
            r = ref.get(os.path.basename(p))
            if r is None:
                continue
            n += 1; diffs.append(abs(q - r[0])); agree += int(l == r[1])
        if n == 0:
            print('verify: no patch of the input appears in the log')
        else:
            print(f'verify: {n} patches found in the log; max |label-1 difference| = {max(diffs):.4f}, '
                  f'mean = {sum(diffs) / n:.4f}; predicted labels agree for {agree}/{n} '
                  f'(a log of the training framework stores 3 decimals; inference/README.md explains what each reference file contains)')


if __name__ == '__main__':
    main()
