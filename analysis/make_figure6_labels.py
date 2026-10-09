# -*- coding: utf-8 -*-
"""Figure 6: relabel the nomogram exported by the original pipeline (pipeline_outputs/img/nomogram.png) with uniform,
grammatical axis names ('5-year survival' instead of '5 years survival' / '9 year survival'); scales are untouched."""
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root
im = Image.open('pipeline_outputs/img/nomogram.png').convert('RGB'); a = np.array(im.convert('L')) < 128
labels = ['Points', 'Clinical score', 'Pathomics score', 'Total points', '5-year survival', '7-year survival', '9-year survival']
col = a[:, :1400]; rows = np.where(col.any(1))[0]
groups, start, prev = [], rows[0], rows[0]
for r in rows[1:]:
    if r - prev > 5: groups.append((start, prev)); start = r
    prev = r
groups.append((start, prev)); assert len(groups) == len(labels), groups
d = ImageDraw.Draw(im); from matplotlib import font_manager
font = ImageFont.truetype(font_manager.findfont(font_manager.FontProperties(family=['Arial', 'Liberation Sans', 'DejaVu Sans'])), 102)
for (y0, y1), lab in zip(groups, labels):
    d.rectangle([380, y0 - 10, 1440, y1 + 10], fill='white')
    top = font.getbbox('P')[1]                       # align the cap height with the original label
    d.text((400, y0 - top), lab, fill='black', font=font)
im.save('analysis_outputs/figures/Figure6_nomogram.png', dpi=(300, 300)); print('Figure 6 relabeled')
