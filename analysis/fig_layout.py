# -*- coding: utf-8 -*-
"""Layout engine for the multi-panel figures. Every figure is drawn at its final print size and inserted at 100 %,
so nothing is rescaled after drawing.

Layout rules (author's figure standard, 2026-10-09):
  * width 15.0 cm (the text width of the manuscript is 15.24 cm);
  * identical gap between panel boxes horizontally and vertically (GAP_CM = 0.3 cm); outer margin 0.5 cm on all sides;
  * aligned axes: panels in one row share the top and bottom of their axes, panels in one column share the left and
    right edges, and all axes in a figure have the same width;
  * one type scale and one set of line widths for every panel (STYLE);
  * bold panel letters at the top-left corner of every panel box.
A panel box is the tight box around everything a panel draws (title, axis and tick labels, legend, at-risk table,
letter); the boxes of a row share the row's top and bottom bands and the boxes of a column share its left and right
bands, so the gap between neighboring boxes is exactly GAP_CM everywhere."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.transforms import blended_transform_factory, offset_copy

CM = 1 / 2.54                                   # inches per cm
DPI = 600
WIDTH_CM, GAP_CM, MARGIN_CM = 15.0, 0.3, 0.5
LETTER_PT, TITLE_PT, LABEL_PT, TICK_PT, LEGEND_PT, ANNOT_PT = 10, 8, 8, 7, 7, 7
LW_DATA, LW_REF, LW_AXES = 1.0, 0.6, 0.6
STYLE = {'font.family': 'Arial', 'font.size': LABEL_PT, 'axes.titlesize': TITLE_PT, 'axes.labelsize': LABEL_PT,
         'xtick.labelsize': TICK_PT, 'ytick.labelsize': TICK_PT, 'legend.fontsize': LEGEND_PT, 'legend.title_fontsize': LEGEND_PT,
         'axes.linewidth': LW_AXES, 'xtick.major.width': LW_AXES, 'ytick.major.width': LW_AXES, 'xtick.major.size': 2.5,
         'ytick.major.size': 2.5, 'xtick.major.pad': 1.5, 'ytick.major.pad': 1.5, 'axes.titlepad': 3.0, 'axes.labelpad': 2.0,
         'lines.linewidth': LW_DATA, 'lines.markersize': 3.5, 'errorbar.capsize': 2, 'patch.linewidth': LW_AXES,
         'legend.frameon': True, 'legend.framealpha': 0.85, 'legend.facecolor': 'white', 'legend.edgecolor': 'none',
         'legend.fancybox': False, 'legend.handlelength': 1.6, 'legend.handletextpad': 0.5, 'legend.borderaxespad': 0.3,
         'legend.borderpad': 0.2, 'legend.labelspacing': 0.3, 'savefig.dpi': DPI, 'figure.dpi': 100,
         'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold'}
rcParams.update(STYLE)


def _texts(fig):
    """Every text item that is actually drawn: titles, axis labels, tick labels inside the view interval, legend texts,
    free texts of the axes (annotations, at-risk table) and figure texts (panel letters)."""
    out = list(fig.texts)
    for ax in fig.axes:
        out += [ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label] + list(ax.texts)
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval()); eps = 1e-9 * max(1.0, abs(hi - lo))
            for tick, loc in zip(axis.get_major_ticks(), axis.get_majorticklocs()):
                if lo - eps <= loc <= hi + eps:
                    out += [lab for lab in (tick.label1, tick.label2) if lab.get_visible()]
        leg = ax.get_legend()
        if leg is not None: out += list(leg.get_texts()) + [leg.get_title()]
    return [t for t in out if t.get_visible() and t.get_text().strip()]


class Grid:
    """nrows x ncols panels; aspect = axes height / axes width (one value, or one per row)."""

    def __init__(self, nrows, ncols, aspect=1.0, width_cm=WIDTH_CM, gap_cm=GAP_CM, margin_cm=MARGIN_CM, letters=True):
        self.nr, self.nc, self.W, self.g, self.m = nrows, ncols, width_cm, gap_cm, margin_cm
        self.aspect = list(aspect) if isinstance(aspect, (list, tuple)) else [aspect] * nrows
        self.fig = plt.figure(figsize=(width_cm * CM, width_cm * CM))
        self.ax = [[self.fig.add_axes([0.1 + 0.8 * c / ncols, 0.9 - 0.8 * (r + 1) / nrows, 0.5 / ncols, 0.5 / nrows])
                    for c in range(ncols)] for r in range(nrows)]
        n = nrows * ncols
        self.letters = [chr(65 + i) for i in range(n)] if letters is True else (list(letters) if letters else [None] * n)
        self.layout = None

    # ------------------------------------------------------------------ measuring
    def _renderer(self):
        self.fig.canvas.draw(); return self.fig.canvas.get_renderer()

    def _extents(self):
        rd = self._renderer(); px = 1 / self.fig.dpi / CM
        ext = {}
        for r in range(self.nr):
            for c in range(self.nc):
                ax = self.ax[r][c]; bb = ax.get_tightbbox(rd); p = ax.get_window_extent(rd)
                ext[r, c] = ((p.x0 - bb.x0) * px, (bb.x1 - p.x1) * px, (bb.y1 - p.y1) * px, (p.y0 - bb.y0) * px)
        return ext

    def _letter_size(self):
        t = self.fig.text(0, 0, 'W', fontsize=LETTER_PT, fontweight='bold'); e = t.get_window_extent(self._renderer()); t.remove()
        return e.width / self.fig.dpi / CM, e.height / self.fig.dpi / CM

    def _ink(self, dpi=DPI):
        """Rendered image as a boolean ink mask (pixels darker than near-white) and pixels per cm."""
        fig = self.fig; old = fig.dpi; fig.set_dpi(dpi); fig.canvas.draw()
        a = np.asarray(fig.canvas.buffer_rgba())[..., :3].min(axis=2) < 245
        axes_px = {k: self.ax[k[0]][k[1]].get_window_extent(fig.canvas.get_renderer()).frozen() for k in self.layout['boxes']}   # frozen: the lazy bbox would follow the dpi reset
        fig.set_dpi(old)
        return a, dpi * CM, axes_px

    def _ink_extents(self):
        """Visible extents (ink) of every panel beyond its axes rectangle: left, right, top, bottom in cm. The ink of a
        panel is searched in its box widened by half a gap, so neighboring panels never mix."""
        a, pc, axes_px = self._ink(); g = self.g; ext = {}
        for k, (x, y, w, h) in self.layout['boxes'].items():
            x0 = max(0, int((x - g / 2) * pc)); x1 = min(a.shape[1], int(np.ceil((x + w + g / 2) * pc)))
            y0 = max(0, int((y - g / 2) * pc)); y1 = min(a.shape[0], int(np.ceil((y + h + g / 2) * pc)))
            ys, xs = np.where(a[y0:y1, x0:x1])
            il, it, ir, ib = (x0 + xs.min()) / pc, (y0 + ys.min()) / pc, (x0 + xs.max() + 1) / pc, (y0 + ys.max() + 1) / pc
            p = axes_px[k]; al, ar = p.x0 / pc, p.x1 / pc; at, ab = (a.shape[0] - p.y1) / pc, (a.shape[0] - p.y0) / pc
            ext[k] = (al - il, ir - ar, at - it, ib - ab)
        return ext

    @staticmethod
    def _letter_ink(letter):
        """Offsets (cm) from the top-left corner of a letter's text box to the top-left corner of its ink, and the ink
        height; measured on a blank figure so that nothing else is inked."""
        f = plt.figure(figsize=(1, 1), dpi=300); t = f.text(0.2, 0.8, letter, ha='left', va='top', fontsize=LETTER_PT, fontweight='bold')
        f.canvas.draw(); e = t.get_window_extent(f.canvas.get_renderer())
        buf = np.asarray(f.canvas.buffer_rgba())[..., :3].min(axis=2) < 245; pc = 300 * CM; H = buf.shape[0]
        ys, xs = np.where(buf); plt.close(f)
        return (xs.min() - e.x0) / pc, (ys.min() - (H - e.y1)) / pc, (ys.max() - ys.min() + 1) / pc

    def _apply(self, ext, need_top):
        L = [max(ext[r, c][0] for r in range(self.nr)) for c in range(self.nc)]
        R = [max(ext[r, c][1] for r in range(self.nr)) for c in range(self.nc)]
        T = [max(max(ext[r, c][2] for c in range(self.nc)), need_top) for r in range(self.nr)]
        B = [max(ext[r, c][3] for c in range(self.nc)) for r in range(self.nr)]
        aw = (self.W - 2 * self.m - (self.nc - 1) * self.g - sum(L) - sum(R)) / self.nc
        assert aw > 1.5, f'axes too narrow ({aw:.2f} cm)'
        ah = [a * aw for a in self.aspect]
        H = 2 * self.m + sum(T) + sum(ah) + sum(B) + (self.nr - 1) * self.g
        self.fig.set_size_inches(self.W * CM, H * CM)
        boxes = {}
        for r in range(self.nr):
            top = self.m + sum(T[:r]) + sum(ah[:r]) + sum(B[:r]) + r * self.g
            for c in range(self.nc):
                left = self.m + sum(L[:c]) + sum(R[:c]) + c * (aw + self.g)
                self.ax[r][c].set_position([(left + L[c]) / self.W, 1 - (top + T[r] + ah[r]) / H, aw / self.W, ah[r] / H])
                boxes[r, c] = (left, top, L[c] + aw + R[c], T[r] + ah[r] + B[r])      # cm from the top-left corner: x, y, w, h
        self.layout = dict(W=self.W, H=H, gap=self.g, margin=self.m, L=L, R=R, T=T, B=B, aw=aw, ah=ah, boxes=boxes)

    def finalize(self, path, max_iter=12):
        used = [lt for lt in self.letters if lt]
        # 1) layout on text boxes (fast, converges to within a few hundredths of a centimeter)
        lw, lh = self._letter_size(); need = lh + 1.2 * TICK_PT / 72 * 2.54 / 2 + 0.05 if used else 0.0
        prev = None
        for _ in range(max_iter):
            ext = self._extents(); self._apply(ext, need)
            cur = np.array([ext[k] for k in sorted(ext)])
            if prev is not None and np.abs(cur - prev).max() < 1e-3: break
            prev = cur
        # 2) layout on the visible ink (letters are added afterwards): gaps and margins become exact on the page
        li = {lt: self._letter_ink(lt) for lt in used}
        need = (max(v[2] for v in li.values()) + 0.12) if used else 0.0       # letter ink + 1.2 mm above the axes top
        prev = None
        for _ in range(max_iter):
            ext = self._ink_extents(); self._apply(ext, need)
            cur = np.array([ext[k] for k in sorted(ext)])
            if prev is not None and np.abs(cur - prev).max() < 0.005: break     # 0.05 mm at the final 600 dpi
            prev = cur
        else:
            raise RuntimeError('layout did not converge')
        lay = self.layout; W, H = lay['W'], lay['H']
        k = 0
        for r in range(self.nr):
            for c in range(self.nc):
                lt = self.letters[k]; k += 1
                if lt:
                    x, y, _, _ = lay['boxes'][r, c]; dx, dy, _ = li[lt]
                    self.fig.text((x - dx) / W, 1 - (y - dy) / H, lt, ha='left', va='top', fontsize=LETTER_PT, fontweight='bold')
        self._check_overlaps()
        self.fig.savefig(path, dpi=DPI, facecolor='white')
        plt.close(self.fig)
        return lay

    # ------------------------------------------------------------------ checks
    def _check_overlaps(self, tol_px=0.5):
        """No two text items may overlap, and every text item must lie inside a panel box."""
        rd = self._renderer(); lay = self.layout; H = lay['H']; dpi = self.fig.dpi
        items = []
        for t in _texts(self.fig):
            e = t.get_window_extent(rd)
            if e.width > 0 and e.height > 0: items.append((t.get_text(), e))
        bad = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i][1], items[j][1]
                if a.x0 + tol_px < b.x1 and b.x0 + tol_px < a.x1 and a.y0 + tol_px < b.y1 and b.y0 + tol_px < a.y1:
                    bad.append((items[i][0], items[j][0]))
        assert not bad, f'overlapping text items: {bad[:6]}'
        cm = lambda v: v / dpi / CM
        for name, e in items:
            x0, x1 = cm(e.x0), cm(e.x1); y0, y1 = H - cm(e.y1), H - cm(e.y0)
            hg = lay['gap'] / 2
            inside = any(bx - hg <= x0 and x1 <= bx + bw + hg and by - hg <= y0 and y1 <= by + bh + hg
                         for bx, by, bw, bh in lay['boxes'].values())
            assert inside, f'text outside every panel box: {name!r}'


def at_risk_table(ax, fitters, labels, ticks, header='At risk', fontsize=TICK_PT):
    """Number-at-risk table under a Kaplan-Meier axis. Counts are those of lifelines.add_at_risk_counts (default
    at_risk_count_from_start_of_period=False): at_risk minus removed at the last event time <= tick. Call after the
    x-axis label is set; positions are fixed offsets (points) from the axes, so the table follows any re-layout."""
    fig = ax.figure; fig.canvas.draw(); rd = fig.canvas.get_renderer()
    xl = ax.xaxis.label.get_window_extent(rd); pos = ax.get_window_extent(rd)
    base = -(pos.y0 - xl.y0) * 72 / fig.dpi - 3.0                      # points below the axes bottom
    line = fontsize * 1.35
    xy = blended_transform_factory(ax.transData, ax.transAxes)
    counts, first = [], []
    for i, f in enumerate(fitters):
        y = base - (i + 1) * line
        et = f.event_table; s = et.at_risk - et.removed
        row = []
        for t in ticks:
            sl = s.loc[:t]; n = int(sl.iloc[-1]) if len(sl) else 0; row.append(n)
            tx = ax.text(t, 0, str(n), transform=offset_copy(xy, fig=fig, x=0, y=y, units='points'), ha='center', va='top',
                         fontsize=fontsize, clip_on=False)
            if t == ticks[0]: first.append(tx)
        counts.append(row)
    fig.canvas.draw(); rd = fig.canvas.get_renderer(); pos = ax.get_window_extent(rd)
    over = max(0.0, max((pos.x0 - tx.get_window_extent(rd).x0) * 72 / fig.dpi for tx in first))   # first column left of the axes
    xo = -(over + 3.0)
    ax.text(0, 0, header, transform=offset_copy(ax.transAxes, fig=fig, x=xo, y=base, units='points'), ha='right', va='top',
            fontsize=fontsize, clip_on=False)
    for i, lab in enumerate(labels):
        ax.text(0, 0, lab, transform=offset_copy(ax.transAxes, fig=fig, x=xo, y=base - (i + 1) * line, units='points'),
                ha='right', va='top', fontsize=fontsize, clip_on=False)
    return counts


def lifelines_counts(fitters, ticks):
    """The counts that lifelines.add_at_risk_counts prints (closed-loop check of at_risk_table)."""
    from lifelines.plotting import add_at_risk_counts
    fig, ax = plt.subplots(); ax.set_xlim(min(ticks), max(ticks))
    add_at_risk_counts(*fitters, ax=ax, rows_to_show=['At risk'], xticks=list(ticks)); fig.canvas.draw()
    labs = [t.get_text() for t in fig.axes[-1].get_xticklabels()]; plt.close(fig)
    cols = []
    for i, s in enumerate(labs):
        lines = [ln for ln in s.split('\n') if ln.strip()]
        cols.append([int(ln.split()[-1]) for ln in (lines[1:] if i == 0 else lines)])   # the first label starts with 'At risk'
    return [list(r) for r in zip(*cols)]


def measure_png(path, layout, white=245):
    """Visible measurements on the saved image: whitespace between the contents of neighboring panel boxes
    (horizontal and vertical), outer margins and the content fraction of the canvas."""
    from PIL import Image
    a = np.asarray(Image.open(path).convert('L')) < white
    px = a.shape[1] / layout['W']                                        # pixels per cm
    content = {}
    for k, (x, y, w, h) in layout['boxes'].items():
        g2 = layout['gap'] / 2
        x0, y0 = max(0, int((x - g2) * px)), max(0, int((y - g2) * px))
        x1, y1 = min(a.shape[1], int(np.ceil((x + w + g2) * px))), min(a.shape[0], int(np.ceil((y + h + g2) * px)))
        sub = a[y0:y1, x0:x1]; ys, xs = np.where(sub)
        content[k] = (x0 + xs.min(), y0 + ys.min(), x0 + xs.max(), y0 + ys.max()) if len(xs) else None
    ys, xs = np.where(a)
    out = {'W_cm': layout['W'], 'H_cm': layout['H'],
           'margin_left_cm': xs.min() / px, 'margin_right_cm': (a.shape[1] - 1 - xs.max()) / px,
           'margin_top_cm': ys.min() / px, 'margin_bottom_cm': (a.shape[0] - 1 - ys.max()) / px}
    nr = 1 + max(r for r, _ in layout['boxes']); nc = 1 + max(c for _, c in layout['boxes'])
    hg = [(content[r, c + 1][0] - content[r, c][2] - 1) / px for r in range(nr) for c in range(nc - 1)]
    vg = [(content[r + 1, c][1] - content[r, c][3] - 1) / px for r in range(nr - 1) for c in range(nc)]
    out['box_gap_cm'] = layout['gap']
    out['gaps_detail'] = '; '.join([f'h{r}{c}-{r}{c + 1}={(content[r, c + 1][0] - content[r, c][2] - 1) / px:.3f}' for r in range(nr) for c in range(nc - 1)]
                                   + [f'v{r}{c}-{r + 1}{c}={(content[r + 1, c][1] - content[r, c][3] - 1) / px:.3f}' for r in range(nr - 1) for c in range(nc)])
    out['visible_gap_h_cm'] = (min(hg), max(hg)) if hg else None
    out['visible_gap_v_cm'] = (min(vg), max(vg)) if vg else None
    out['content_w_frac'] = (xs.max() - xs.min() + 1) / a.shape[1]
    out['content_h_frac'] = (ys.max() - ys.min() + 1) / a.shape[0]
    inner = (layout['W'] - 2 * layout['margin']) * (layout['H'] - 2 * layout['margin'])
    out['boxes_fill_inner'] = sum(w * h for (_, _, w, h) in layout['boxes'].values()) / inner
    return out


def legend_below(ax, **kw):
    """Legend centered under the x-axis label of ax (fixed offset in points, follows any re-layout)."""
    fig = ax.figure; fig.canvas.draw(); rd = fig.canvas.get_renderer()
    xl = ax.xaxis.label.get_window_extent(rd); pos = ax.get_window_extent(rd)
    off = -(pos.y0 - xl.y0) * 72 / fig.dpi - 2.0
    return ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0), borderaxespad=0,
                     bbox_transform=offset_copy(ax.transAxes, fig=fig, x=0, y=off, units='points'), **kw)
