# utils/visualize.py
"""
Visualization utilities.
All functions are safe to call even if matplotlib is not installed.

Fixed issues:
  1. Training curves: nan values filtered, high-scale configs shown separately
  2. Validation curves: nan values filtered, only real values plotted
  3. Qualitative comparison: shows all relevant configs not just L0 vs L4
  4. Color map extended for L5, L6, L7
"""

from pathlib import Path
from typing import Dict, List, Optional

try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend safe on Colab/headless
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _MPL = True
except ImportError:
    _MPL = False

import numpy as np


# Extended color map for all configs including L6, L7
COLORS = {
    "L0":  "#e74c3c",   # red
    "L1":  "#e67e22",   # orange
    "L2":  "#f1c40f",   # yellow
    "L3":  "#9b59b6",   # purple
    "L3c": "#2980b9",   # blue
    "L4":  "#27ae60",   # green
    "L5":  "#1abc9c",   # teal
    "L6":  "#2c3e50",   # dark
    "L7":  "#7f8c8d",   # grey
    "L_":  "#c0392b",   # dark red (L_ffl)
}

# Configs that use different loss scales — shown dashed to distinguish
HIGH_SCALE_CONFIGS = {
    "L3_spectral_only", "L_ffl",
    "L4_full_method", "L5_full_uw",
    "L7_full_adaptive_uw",
}


def _no_mpl(fn_name):
    print(f"  Skipping {fn_name}: matplotlib not available.")


def _save(fig, path: Path, dpi: int = 120):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def _get_color(exp_name: str) -> Optional[str]:
    """Get color for experiment name by matching prefix."""
    for prefix, color in COLORS.items():
        if exp_name.startswith(prefix):
            return color
    return None


def _clean_series(values: list, max_val: float = 100.0) -> list:
    """Remove nan and unreasonably large values from a loss series."""
    return [
        (i, v) for i, v in enumerate(values)
        if v == v and not (v != v) and v < max_val and v > 0
    ]


# ── EDA ───────────────────────────────────────────────────────────────────────

def plot_sample_grid(image_paths: List[str], results_dir: Path, n: int = 18):
    if not _MPL:
        return _no_mpl("plot_sample_grid")
    from PIL import Image
    rng  = np.random.RandomState(42)
    idxs = rng.choice(len(image_paths), min(n, len(image_paths)), replace=False)
    cols = 6
    rows = max(1, (len(idxs) + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 3))
    fig.suptitle("Sample Training Images", fontsize=13, fontweight="bold")
    for i, ax in enumerate(np.array(axes).flat):
        if i < len(idxs):
            try:
                img = Image.open(image_paths[idxs[i]]).convert("RGB").resize((256, 256))
                ax.imshow(np.array(img))
            except Exception:
                pass
        ax.axis("off")
    _save(fig, Path(results_dir) / "eda_sample_grid.png")


def plot_mask_examples(results_dir: Path, img_size: int = 256):
    if not _MPL:
        return _no_mpl("plot_mask_examples")
    from data.dataset import generate_freeform_mask, get_boundary_band
    bins = {"easy": (0.10, 0.20), "mid": (0.20, 0.35),
            "hard": (0.35, 0.50), "xhard": (0.50, 0.60)}
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle("Masks by Difficulty Level", fontsize=13, fontweight="bold")
    rng = np.random.RandomState(100)
    for col, (name, (lo, hi)) in enumerate(bins.items()):
        mask = None
        for _ in range(100):
            m = generate_freeform_mask(img_size, img_size, (lo, hi), rng=rng)
            if lo <= m.mean() <= hi:
                mask = m
                break
        if mask is None:
            mask = m
        band = get_boundary_band(mask)
        axes[0, col].imshow(mask, cmap="gray")
        axes[0, col].set_title(f"{name}\n{mask.mean()*100:.1f}%")
        axes[0, col].axis("off")
        axes[1, col].imshow(band, cmap="hot")
        axes[1, col].set_title("Boundary band")
        axes[1, col].axis("off")
        overlay = np.ones((img_size, img_size, 3))
        overlay[mask > 0.5] = [1, 0.3, 0.3]
        overlay[band > 0.5] = [0.3, 0.6, 1.0]
        axes[2, col].imshow(overlay)
        axes[2, col].set_title("Overlay")
        axes[2, col].axis("off")
    _save(fig, Path(results_dir) / "eda_masks.png")


def plot_mask_coverage(results_dir: Path, n: int = 300, img_size: int = 256):
    if not _MPL:
        return _no_mpl("plot_mask_coverage")
    from data.dataset import generate_freeform_mask
    rng = np.random.RandomState(42)
    covs = [generate_freeform_mask(img_size, img_size, rng=rng).mean() * 100
            for _ in range(n)]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(covs, bins=40, color="#8e44ad", edgecolor="white", alpha=0.8)
    for (name, lo, hi), c in zip(
        [("easy",10,20),("mid",20,35),("hard",35,50),("xhard",50,60)],
        ["#3498db","#27ae60","#e67e22","#e74c3c"]
    ):
        ax.axvspan(lo, hi, alpha=0.12, color=c, label=f"{name}")
    ax.set_xlabel("Mask Coverage (%)"); ax.set_ylabel("Count")
    ax.set_title(f"Mask Coverage Distribution (n={n})", fontweight="bold")
    ax.legend(fontsize=9)
    _save(fig, Path(results_dir) / "eda_coverage.png")


# ── Training Curves ────────────────────────────────────────────────────────────

def plot_training_curves(histories: Dict[str, dict], results_dir: Path,
                          title: str = "Training Curves"):
    """
    Fixed version:
      - Filters nan and unreasonably large values
      - Shows two subplots per row: comparable-scale and high-scale configs
      - Validation loss only plots real values (every 2 epochs)
      - Dashed lines for configs with different loss scales
    """
    if not _MPL:
        return _no_mpl("plot_training_curves")

    # Split configs into two groups by loss scale
    low_scale  = {k: v for k, v in histories.items()
                  if k not in HIGH_SCALE_CONFIGS}
    high_scale = {k: v for k, v in histories.items()
                  if k in HIGH_SCALE_CONFIGS}

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    def _plot_group(ax_train, ax_val, group, group_title, max_val=5.0):
        ax_train.set_title(f"{group_title} — Train Loss")
        ax_val.set_title(f"{group_title} — Val Loss")
        has_data = False

        for exp_name, h in group.items():
            color = _get_color(exp_name)
            ls    = "--" if exp_name in HIGH_SCALE_CONFIGS else "-"

            # Train loss
            train_pts = _clean_series(h.get("train_loss", []), max_val=100.0)
            if train_pts:
                xs, ys = zip(*train_pts)
                ax_train.plot(xs, ys, label=exp_name, color=color,
                              alpha=0.85, linestyle=ls, linewidth=1.8)
                has_data = True

            # Val loss — filter nan (every 2 epochs = alternating nan)
            val_pts = _clean_series(h.get("val_loss", []), max_val=100.0)
            if val_pts:
                xs, ys = zip(*val_pts)
                ax_val.plot(xs, ys, label=exp_name, color=color,
                            alpha=0.85, linestyle=ls, linewidth=1.8,
                            marker="o", markersize=4)

        for ax in [ax_train, ax_val]:
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(alpha=0.3)
            if has_data:
                ax.legend(fontsize=7, loc="upper right")

        if not has_data:
            for ax in [ax_train, ax_val]:
                ax.text(0.5, 0.5, "No training history",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=10, color="gray")

    _plot_group(axes[0, 0], axes[0, 1], low_scale,
                "L0-L3c, L6 (Comparable Scale)")
    _plot_group(axes[1, 0], axes[1, 1], high_scale,
                "L3, L4, L5, L7, FFL (High Loss Scale)")

    plt.tight_layout()
    _save(fig, Path(results_dir) / "training_curves.png")


# ── Ablation Table ─────────────────────────────────────────────────────────────

def plot_ablation_table(phase4_eval: Dict[str, dict], results_dir: Path,
                         title: str = "Ablation Study (Test Set)"):
    if not _MPL:
        return _no_mpl("plot_ablation_table")
    metrics = ["psnr", "ssim", "lpips", "hole_mae", "boundary_mae",
               "spectral_coherence", "fid"]
    display = ["PSNR↑", "SSIM↑", "LPIPS↓", "Hole-MAE↓", "B-MAE↓",
               "S-Coh↑", "FID↓"]
    rows, exp_names = [], list(phase4_eval.keys())
    for exp in exp_names:
        g = phase4_eval[exp].get("global", {})
        rows.append([exp] + [f"{g.get(m, float('nan')):.4f}" for m in metrics])
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(18, max(3, len(rows) * 0.9 + 1.5)))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=["Experiment"] + display,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.8)

    # Highlight best per metric (green) and worst (light red)
    higher_better = {"psnr", "ssim", "spectral_coherence"}
    for ci, m in enumerate(metrics, start=1):
        vals = []
        for e in exp_names:
            v = phase4_eval[e].get("global", {}).get(m, float("nan"))
            vals.append(v if v == v else float("nan"))
        valid = [v for v in vals if v == v]
        if not valid:
            continue
        best  = max(valid) if m in higher_better else min(valid)
        worst = min(valid) if m in higher_better else max(valid)
        for ri, v in enumerate(vals):
            if v != v:
                continue
            if abs(v - best) < 1e-6:
                tbl[ri + 1, ci].set_facecolor("#d5f5e3")
                tbl[ri + 1, ci].set_text_props(fontweight="bold")
            elif abs(v - worst) < 1e-6:
                tbl[ri + 1, ci].set_facecolor("#fde8e8")

    ax.set_title(title, fontweight="bold", fontsize=12, pad=20)
    _save(fig, Path(results_dir) / "ablation_table.png")


# ── Sensitivity Plot ───────────────────────────────────────────────────────────

def plot_sensitivity(sensitivity_results: dict, results_dir: Path):
    if not _MPL:
        return _no_mpl("plot_sensitivity")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Hyperparameter Sensitivity", fontsize=13, fontweight="bold")

    def _plot(ax, data, x_key, y_key, xlabel, ylabel, color):
        if not data:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color="gray")
            return
        xs = [d[x_key] for d in data]
        ys = [d[y_key] for d in data]
        ax.plot(xs, ys, "o-", color=color, linewidth=2, markersize=8)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_xticks(xs); ax.grid(alpha=0.3)

    _plot(axes[0], sensitivity_results.get("k_ablation", []),
          "k", "boundary_mae", "K (patches per image)", "Boundary-MAE↓", "#2980b9")
    axes[0].set_title("Sensitivity to K")

    _plot(axes[1], sensitivity_results.get("patch_ablation", []),
          "patch_size", "spectral_coherence",
          "Patch Size (px)", "Spectral Coherence↑", "#27ae60")
    axes[1].set_title("Sensitivity to Patch Size")

    _plot(axes[2], sensitivity_results.get("dilation_ablation", []),
          "dilation", "boundary_mae", "Dilation Radius r", "Boundary-MAE↓", "#e74c3c")
    axes[2].set_title("Sensitivity to Dilation r")

    _save(fig, Path(results_dir) / "sensitivity_analysis.png")


# ── Qualitative Comparison ─────────────────────────────────────────────────────

def plot_qualitative(model_l0, model_best, test_loader, device,
                     results_dir: Path, n_show: int = 4,
                     use_amp: bool = True,
                     best_label: str = "Best Method"):
    """L0 (Base) vs best method — with boundary error heatmaps."""
    if not _MPL:
        return _no_mpl("plot_qualitative")
    import torch
    ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
           if use_amp and torch.cuda.is_available()
           else torch.amp.autocast("cpu", dtype=torch.float32, enabled=False))

    model_l0.eval()
    model_best.eval()

    images, masks, boundaries = next(iter(test_loader))
    images     = images[:n_show].to(device)
    masks      = masks[:n_show].to(device)
    boundaries = boundaries[:n_show].to(device)
    masked_input = images * (1.0 - masks)

    with torch.no_grad(), ctx:
        pred_l0   = model_l0(masked_input, masks, boundaries)["output"].float()
        pred_best = model_best(masked_input, masks, boundaries)["output"].float()

    comp_l0   = images.float() * (1 - masks) + pred_l0   * masks
    comp_best = images.float() * (1 - masks) + pred_best * masks
    err_l0    = torch.abs(images.float() - comp_l0)   * boundaries
    err_best  = torch.abs(images.float() - comp_best) * boundaries

    cols = ["Original", "Masked Input", f"L0 (Base)",
            best_label, "Error L0", f"Error {best_label}"]
    fig, axes = plt.subplots(n_show, 6, figsize=(24, n_show * 4))
    for row in range(n_show):
        for col, t in enumerate([images[row], masked_input[row],
                                  comp_l0[row], comp_best[row],
                                  err_l0[row], err_best[row]]):
            arr = t.cpu().permute(1, 2, 0).numpy().clip(0, 1)
            if col >= 4:
                axes[row, col].imshow(arr.mean(-1), cmap="hot",
                                      vmin=0, vmax=0.25)
            else:
                axes[row, col].imshow(arr)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(cols[col], fontsize=9,
                                          fontweight="bold")
    fig.suptitle(f"L0 (Base) vs {best_label}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, Path(results_dir) / "qualitative_comparison.png")


def plot_qualitative_boundary_compare(models: Dict, test_loader, device,
                                       results_dir: Path, n_show: int = 4,
                                       use_amp: bool = True,
                                       save_name: str = "boundary_comparison.png",
                                       title: str = "Boundary Comparison"):
    """
    Compare multiple models side by side with boundary error column.
    Shows all configs passed in models dict.
    """
    if not _MPL:
        return _no_mpl("plot_qualitative_boundary_compare")
    import torch
    ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
           if use_amp and torch.cuda.is_available()
           else torch.amp.autocast("cpu", dtype=torch.float32, enabled=False))

    for m in models.values():
        m.eval()

    images, masks, boundaries = next(iter(test_loader))
    images     = images[:n_show].to(device)
    masks      = masks[:n_show].to(device)
    boundaries = boundaries[:n_show].to(device)
    masked_input = images * (1.0 - masks)

    preds = {}
    with torch.no_grad(), ctx:
        for name, model in models.items():
            preds[name] = model(masked_input, masks, boundaries)["output"].float()

    model_names = list(models.keys())
    # Columns: original, masked, then one per model
    n_cols = 2 + len(model_names)
    fig, axes = plt.subplots(n_show, n_cols,
                              figsize=(n_cols * 3.5, n_show * 3.5))
    if n_show == 1:
        axes = axes[np.newaxis, :]
    headers = ["Original", "Masked"] + model_names

    for row in range(n_show):
        items = [images[row].float(), masked_input[row].float()]
        for name in model_names:
            comp = (images[row].float() * (1 - masks[row]) +
                    preds[name][row] * masks[row])
            items.append(comp)

        for col, t in enumerate(items):
            arr = t.cpu().permute(1, 2, 0).numpy().clip(0, 1)
            axes[row, col].imshow(arr)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(headers[col], fontsize=9,
                                          fontweight="bold")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, Path(results_dir) / save_name)


def plot_baseline_comparison(comparison: Dict[str, Dict], results_dir: Path):
    if not _MPL or not comparison:
        return
    metrics = ["psnr", "ssim", "boundary_mae", "spectral_coherence"]
    display = ["PSNR↑", "SSIM↑", "B-MAE↓", "S-Coh↑"]
    methods = list(comparison.keys())
    x       = np.arange(len(methods))
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Method Comparison", fontsize=13, fontweight="bold")
    for ax, m, d in zip(axes, metrics, display):
        vals = [comparison[mn].get(m, 0) for mn in methods]
        ax.bar(x, vals, alpha=0.85, edgecolor="white",
               color=[_get_color(mn) or "#888" for mn in methods])
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=8)
        ax.set_title(d)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, Path(results_dir) / "baseline_comparison.png")


# ── Boundary Zoom ──────────────────────────────────────────────────────────────

def plot_boundary_zoom(models: Dict, test_loader, device,
                       results_dir: Path, n_show: int = 4,
                       use_amp: bool = True):
    """
    Full image (top row) + zoomed boundary crop (bottom row) for each model.
    Yellow box on full image shows the crop region.
    """
    if not _MPL:
        return _no_mpl("plot_boundary_zoom")
    import torch

    ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
           if use_amp and torch.cuda.is_available()
           else torch.amp.autocast("cpu", dtype=torch.float32, enabled=False))

    for m in models.values():
        m.eval()

    images, masks, boundaries = next(iter(test_loader))
    images     = images[:n_show].to(device)
    masks      = masks[:n_show].to(device)
    boundaries = boundaries[:n_show].to(device)
    masked_in  = images * (1.0 - masks)

    preds = {}
    with torch.no_grad(), ctx:
        for name, model in models.items():
            preds[name] = model(masked_in, masks, boundaries)["output"].float()

    model_names = list(models.keys())
    n_cols = 2 + len(model_names)   # masked input + GT + one per model
    fig, axes = plt.subplots(n_show * 2, n_cols,
                              figsize=(n_cols * 3.5, n_show * 6))
    if n_show == 1:
        axes = axes.reshape(2, n_cols)

    headers = ["Masked Input", "Ground Truth"] + model_names

    for row in range(n_show):
        band_np = boundaries[row, 0].cpu().numpy()
        ys, xs  = np.where(band_np > 0.5)
        if len(ys) == 0:
            continue
        cy, cx = int(ys.mean()), int(xs.mean())
        cs     = 48
        y0 = max(0, cy - cs); y1 = min(256, cy + cs)
        x0 = max(0, cx - cs); x1 = min(256, cx + cs)

        gt_np = images[row].float().cpu().permute(1, 2, 0).numpy().clip(0, 1)
        mi_np = masked_in[row].float().cpu().permute(1, 2, 0).numpy().clip(0, 1)

        items_full = [mi_np, gt_np]
        items_crop = [mi_np[y0:y1, x0:x1], gt_np[y0:y1, x0:x1]]

        for name in model_names:
            comp = (images[row].float() * (1 - masks[row]) +
                    preds[name][row] * masks[row])
            comp_np = comp.cpu().permute(1, 2, 0).numpy().clip(0, 1)
            items_full.append(comp_np)
            items_crop.append(comp_np[y0:y1, x0:x1])

        for col, (full, crop) in enumerate(zip(items_full, items_crop)):
            # Full image row
            axes[row * 2, col].imshow(full)
            axes[row * 2, col].axis("off")
            if row == 0:
                axes[row * 2, col].set_title(headers[col], fontsize=9,
                                              fontweight="bold")
            # Yellow crop box
            from matplotlib.patches import Rectangle
            rect = Rectangle((x0, y0), x1-x0, y1-y0,
                              linewidth=2, edgecolor="yellow", facecolor="none")
            axes[row * 2, col].add_patch(rect)

            # Crop row
            axes[row * 2 + 1, col].imshow(crop)
            axes[row * 2 + 1, col].axis("off")

    fig.suptitle("Full Image (top) + Boundary Zoom (bottom)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, Path(results_dir) / "boundary_zoom.png")


# ── Failure Cases ──────────────────────────────────────────────────────────────

def plot_failure_cases(model, test_loader, device, results_dir: Path,
                       n_show: int = 4, use_amp: bool = True):
    """Worst-performing samples by boundary MAE."""
    if not _MPL:
        return _no_mpl("plot_failure_cases")
    import torch

    ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
           if use_amp and torch.cuda.is_available()
           else torch.amp.autocast("cpu", dtype=torch.float32, enabled=False))

    model.eval()
    samples = []

    with torch.no_grad(), ctx:
        for images, masks, boundaries in test_loader:
            images     = images.to(device)
            masks      = masks.to(device)
            boundaries = boundaries.to(device)
            masked_in  = images * (1.0 - masks)
            pred  = model(masked_in, masks, boundaries)["output"].float()
            comp  = images.float() * (1 - masks) + pred * masks

            for i in range(images.shape[0]):
                band_px = boundaries[i, 0].cpu().numpy() > 0.5
                if not band_px.any():
                    continue
                img_np  = images[i].float().cpu().permute(1, 2, 0).numpy()
                comp_np = comp[i].cpu().permute(1, 2, 0).numpy().clip(0, 1)
                b_mae   = float(np.abs(img_np[band_px] - comp_np[band_px]).mean())
                samples.append((b_mae, img_np, comp_np, masks[i].cpu().numpy()[0]))

            if len(samples) >= 40:
                break

    samples.sort(key=lambda x: -x[0])
    worst = samples[:n_show]

    fig, axes = plt.subplots(n_show, 3, figsize=(12, n_show * 4))
    fig.suptitle("Failure Cases (Worst Boundary MAE)", fontsize=13, fontweight="bold")

    for row, (b_mae, img_np, comp_np, mask_np) in enumerate(worst):
        masked = img_np * (1 - mask_np[:, :, None])
        axes[row, 0].imshow(img_np.clip(0, 1))
        axes[row, 0].set_title("Ground Truth" if row == 0 else "")
        axes[row, 0].axis("off")
        axes[row, 1].imshow(masked.clip(0, 1))
        axes[row, 1].set_title("Masked Input" if row == 0 else "")
        axes[row, 1].axis("off")
        axes[row, 2].imshow(comp_np)
        axes[row, 2].set_title("Prediction" if row == 0 else "")
        axes[row, 2].set_xlabel(f"B-MAE={b_mae:.4f}", fontsize=9)
        axes[row, 2].axis("off")

    plt.tight_layout()
    _save(fig, Path(results_dir) / "failure_cases.png")
