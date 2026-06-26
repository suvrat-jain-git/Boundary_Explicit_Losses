# losses/losses.py
"""
Loss functions for boundary-aware inpainting.
All bfloat16 safe — float32 casts applied internally for VGG, Sobel, FFT.
Works on T4, V100, A100.

KEY FIX: sobel_gradient_magnitude uses a simple per-channel loop.
Grouped convolution (groups=C) fails on T4/V100 with bfloat16:
  RuntimeError: FIND was unable to find an engine to execute this computation
The loop version is safe on all hardware.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

from configs.config import LossConfig


# ── Perceptual Loss ────────────────────────────────────────────────────────────

class PerceptualLoss(nn.Module):
    """VGG16 relu3_2 feature loss. Always float32 internally."""

    def __init__(self):
        super().__init__()
        vgg = tvm.vgg16(weights=tvm.VGG16_Weights.DEFAULT).features[:16]
        self.feature_extractor = vgg.eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.register_buffer("mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dtype  = pred.dtype
        mean   = self.mean.float()
        std    = self.std.float()
        pred_n = (pred.float()   - mean) / std
        tgt_n  = (target.float() - mean) / std
        fp = self.feature_extractor(pred_n)
        with torch.no_grad():
            ft = self.feature_extractor(tgt_n)
        return F.l1_loss(fp, ft).to(dtype)


# ── Sobel Gradient ─────────────────────────────────────────────────────────────

def sobel_gradient_magnitude(img: torch.Tensor) -> torch.Tensor:
    """
    (B, C, H, W) -> (B, 1, H, W) max gradient magnitude.

    IMPORTANT: Uses a simple per-channel loop NOT grouped convolution.
    Grouped conv with bfloat16 fails on T4/V100 with FIND engine error.
    This loop version works on ALL GPU types and ALL precisions.
    Always returns float32.
    """
    img_f = img.float()
    B, C, H, W = img_f.shape

    sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                      device=img.device).view(1, 1, 3, 3)
    sy = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                      device=img.device).view(1, 1, 3, 3)

    grad_mag = torch.zeros(B, 1, H, W, device=img.device, dtype=torch.float32)
    for c in range(C):
        ch  = img_f[:, c:c+1]
        gx  = F.conv2d(ch, sx, padding=1)
        gy  = F.conv2d(ch, sy, padding=1)
        grad_mag = torch.max(grad_mag, torch.sqrt(gx**2 + gy**2 + 1e-8))

    return grad_mag


# ── Boundary Losses ────────────────────────────────────────────────────────────

class UniformBoundaryLoss(nn.Module):
    """Uniform L1 on boundary band. Ablation baseline L1."""

    def forward(self, pred, target, boundary_band):
        diff   = torch.abs(pred - target) * boundary_band
        n_band = boundary_band.sum().clamp(min=1.0)
        return diff.sum() / (n_band * 3)


class GradientWeightedBoundaryLoss(nn.Module):
    """
    Boundary L1 weighted by Sobel gradient magnitude.
    Normalized [0,1] per image. float32 cast handled by sobel function.
    """

    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target, boundary_band):
        with torch.no_grad():
            grad_w = sobel_gradient_magnitude(target)        # float32
            B      = grad_w.shape[0]
            max_v  = grad_w.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1)
            grad_w = grad_w / (max_v + self.epsilon)
            grad_w = (grad_w * boundary_band).to(pred.dtype)

        weighted = torch.abs(pred - target) * grad_w
        n_band   = boundary_band.sum().clamp(min=1.0)
        return weighted.sum() / (n_band * 3)


# ── Spectral Boundary Coherence Loss ──────────────────────────────────────────

class SpectralBoundaryCoherenceLoss(nn.Module):
    """
    FFT magnitude mismatch on boundary patches. LOCAL not global.
    float32 cast applied before FFT.
    """

    def __init__(self, patch_size=16, n_samples=8, spectral_mode="log_magnitude"):
        super().__init__()
        assert spectral_mode in ("magnitude", "log_magnitude", "complex")
        self.patch_size    = patch_size
        self.n_samples     = n_samples
        self.spectral_mode = spectral_mode

    def _collect(self, comp_f, target_f, boundary_band):
        B, C, H, W = comp_f.shape
        ps        = self.patch_size
        band_flat = boundary_band.view(B, -1)
        pc, pt    = [], []
        for b in range(B):
            nz = torch.nonzero(band_flat[b] > 0.5, as_tuple=False)
            if len(nz) < 4:
                continue
            n_draw  = min(self.n_samples, len(nz))
            indices = nz[torch.randperm(len(nz), device=comp_f.device)[:n_draw], 0]
            for idx in indices:
                y, x   = int(idx) // W, int(idx) % W
                y0, x0 = max(0, y - ps // 2), max(0, x - ps // 2)
                y1, x1 = min(H, y0 + ps), min(W, x0 + ps)
                if (y1 - y0) != ps or (x1 - x0) != ps:
                    continue
                pc.append(comp_f[b,   :, y0:y1, x0:x1])
                pt.append(target_f[b, :, y0:y1, x0:x1])
        return pc, pt

    def _repr(self, fft):
        if self.spectral_mode == "magnitude":
            return torch.abs(fft)
        elif self.spectral_mode == "log_magnitude":
            return torch.log(torch.abs(fft) + 1e-8)
        else:
            return torch.view_as_real(fft).flatten(-2)

    def forward(self, pred, target, mask, boundary_band):
        comp     = target * (1 - mask) + pred * mask
        comp_f   = comp.float()
        target_f = target.float()
        pc, pt   = self._collect(comp_f, target_f, boundary_band)
        if not pc:
            return pred.sum() * 0.0
        p_c = torch.stack(pc, dim=0)
        p_t = torch.stack(pt, dim=0)
        rc  = self._repr(torch.fft.fft2(p_c))
        rt  = self._repr(torch.fft.fft2(p_t)).detach()
        return F.l1_loss(rc, rt).to(pred.dtype)


# ── TV Loss ────────────────────────────────────────────────────────────────────

class TVLoss(nn.Module):
    """Total variation restricted to hole region."""

    def forward(self, pred, mask):
        h = pred * mask
        return (torch.abs(h[:, :, 1:, :] - h[:, :, :-1, :]).mean()
              + torch.abs(h[:, :, :, 1:] - h[:, :, :, :-1]).mean())


# ── ASBC Loss — paper novelty ──────────────────────────────────────────────────

class AdaptiveSpectralBoundaryCoherenceLoss(nn.Module):
    """
    ASBC Loss — paper novelty (L3c).
    Learns per-radial-band FFT weights end-to-end.
    Entropy regularizer prevents weight collapse.
    Always float32 for FFT stability.
    """

    def __init__(self, n_bands=4, patch_size=16, n_samples=8, reg_weight=0.01):
        super().__init__()
        self.n_bands    = n_bands
        self.patch_size = patch_size
        self.n_samples  = n_samples
        self.reg_weight = reg_weight

        self.log_band_weights = nn.Parameter(torch.zeros(n_bands))

        ps     = patch_size
        cy, cx = ps / 2.0, ps / 2.0
        yy, xx = torch.meshgrid(
            torch.arange(ps).float(), torch.arange(ps).float(), indexing="ij"
        )
        r     = torch.sqrt((yy - cy)**2 + (xx - cx)**2)
        r_max = r.max().item()
        masks = [
            ((r >= i * r_max / n_bands) & (r < (i+1) * r_max / n_bands)).float()
            for i in range(n_bands)
        ]
        self.register_buffer("band_masks", torch.stack(masks, dim=0))

    def get_band_weights(self):
        return torch.softmax(self.log_band_weights, dim=0).detach().cpu()

    def forward(self, pred, target, mask, boundary_band):
        B, C, H, W = pred.shape
        ps       = self.patch_size
        comp_f   = (target * (1 - mask) + pred * mask).float()
        target_f = target.float()
        band_flat = boundary_band.view(B, -1)

        pc, pt = [], []
        for b in range(B):
            nz = torch.nonzero(band_flat[b] > 0.5, as_tuple=False)
            if len(nz) < 4:
                continue
            n_draw  = min(self.n_samples, len(nz))
            indices = nz[torch.randperm(len(nz), device=pred.device)[:n_draw], 0]
            for idx in indices:
                y, x   = int(idx) // W, int(idx) % W
                y0, x0 = max(0, y - ps // 2), max(0, x - ps // 2)
                y1, x1 = min(H, y0 + ps), min(W, x0 + ps)
                if (y1 - y0) != ps or (x1 - x0) != ps:
                    continue
                pc.append(comp_f[b,   :, y0:y1, x0:x1])
                pt.append(target_f[b, :, y0:y1, x0:x1])

        if not pc:
            return pred.sum() * 0.0

        p_c   = torch.stack(pc, dim=0)
        p_t   = torch.stack(pt, dim=0)
        mag_c = torch.abs(torch.fft.fft2(p_c))
        mag_t = torch.abs(torch.fft.fft2(p_t)).detach()
        diff  = torch.abs(mag_c - mag_t)

        bm       = self.band_masks.unsqueeze(0).unsqueeze(0)
        per_band = (diff.unsqueeze(2) * bm).mean(dim=(0, 1, 3, 4))
        weights  = torch.softmax(self.log_band_weights, dim=0)
        wloss    = (per_band * weights).sum()
        entropy  = -(weights * (weights + 1e-8).log()).sum()
        return (wloss - self.reg_weight * entropy).to(pred.dtype)


# ── Uncertainty-Weighted Loss ──────────────────────────────────────────────────

class UncertaintyWeightedLoss(nn.Module):
    """
    Kendall et al. 2018. Replaces fixed scalar weights with learned log-variances.
    L = sum_i [ exp(-s_i) * L_i + s_i ]
    Uses 10x LR in trainer._build_optimizer. Fixes L4 underperformance.
    """

    def __init__(self, n_losses=6):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_losses))

    def forward(self, loss_list):
        assert len(loss_list) == len(self.log_vars)
        total = loss_list[0].new_zeros(1).squeeze()
        for i, loss in enumerate(loss_list):
            total = total + torch.exp(-self.log_vars[i]) * loss + self.log_vars[i]
        return total

    def get_effective_weights(self):
        return torch.exp(-self.log_vars).detach().cpu().tolist()


# ── Focal Frequency Loss — external baseline ───────────────────────────────────

class FocalFrequencyLoss(nn.Module):
    """FFL (Jiang et al. 2021) — global spectral. Used as baseline only."""

    def __init__(self, loss_weight=1.0, alpha=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha       = alpha

    def forward(self, pred, target):
        pred_f  = torch.fft.fft2(pred.float(),   norm="ortho")
        tgt_f   = torch.fft.fft2(target.float(), norm="ortho")
        pm, tm  = torch.abs(pred_f), torch.abs(tgt_f)
        weight  = ((pm - tm)**2 + 1e-8)
        weight  = (weight / weight.mean()) ** self.alpha
        loss    = (weight * torch.abs(pm - tm)).mean()
        return (self.loss_weight * loss).to(pred.dtype)


# ── InpaintingLossManager ─────────────────────────────────────────────────────

class InpaintingLossManager:
    """
    Combines all loss terms by loss_config string.

    Configs:
      "base"                    hole+valid+perceptual+tv
      "boundary_uniform"        base + uniform boundary L1
      "boundary_grad"           base + gradient boundary L1
      "spectral_only"           base + fixed spectral
      "adaptive_spectral"       base + ASBC
      "adaptive_spectral_noreg" base + ASBC no entropy (ablation)
      "ffl_only"                base + global FFL (baseline)
      "full"                    base + grad-boundary + spectral + MS heads
      "full_adaptive"           base + grad-boundary + ASBC + MS heads
      "full_uw"                 full but UW balancing handled in trainer
      "full_adaptive"          base + grad-boundary + ASBC + MS heads
      "full_adaptive_uw"       full_adaptive but with UW balancing
    """

    def __init__(self, cfg: LossConfig, device: torch.device):
        self.cfg    = cfg
        self.device = device
        self.skip_perceptual: bool = False

        self.perceptual       = PerceptualLoss().to(device)
        self.uniform_boundary = UniformBoundaryLoss()
        self.grad_boundary    = GradientWeightedBoundaryLoss()
        self.spectral         = SpectralBoundaryCoherenceLoss(
            patch_size    = cfg.spectral_patch_size,
            n_samples     = cfg.spectral_patches_k,
            spectral_mode = "log_magnitude" if cfg.spectral_use_log else "magnitude",
        ).to(device)
        self.adaptive_spectral = AdaptiveSpectralBoundaryCoherenceLoss(
            n_bands    = cfg.spectral_n_bands,
            patch_size = cfg.spectral_patch_size,
            n_samples  = cfg.spectral_patches_k,
            reg_weight = cfg.spectral_reg_weight,
        ).to(device)
        self.adaptive_spectral_noreg = AdaptiveSpectralBoundaryCoherenceLoss(
            n_bands    = cfg.spectral_n_bands,
            patch_size = cfg.spectral_patch_size,
            n_samples  = cfg.spectral_patches_k,
            reg_weight = 0.0,
        ).to(device)
        self.ffl = FocalFrequencyLoss().to(device)
        self.tv  = TVLoss()

    def compute(self, pred_dict, target, mask, boundary_band):
        pred       = pred_dict["output"]
        hole_mask  = mask
        valid_mask = 1.0 - mask
        losses     = {}
        cfg        = self.cfg

        # Base losses
        n_hole  = (hole_mask  * 3).sum().clamp(min=1.0)
        n_valid = (valid_mask * 3).sum().clamp(min=1.0)
        losses["hole_l1"]  = (torch.abs(pred - target) * hole_mask).sum()  / n_hole
        losses["valid_l1"] = (torch.abs(pred - target) * valid_mask).sum() / n_valid
        losses["perceptual"] = (
            pred.new_zeros(1).squeeze() if self.skip_perceptual
            else self.perceptual(pred, target)
        )
        losses["tv"] = self.tv(pred, mask)

        total = (
              cfg.hole_weight       * losses["hole_l1"]
            + cfg.valid_weight      * losses["valid_l1"]
            + cfg.perceptual_weight * losses["perceptual"]
            + cfg.tv_weight         * losses["tv"]
        )

        lc = cfg.loss_config

        if lc == "boundary_uniform":
            bl = self.uniform_boundary(pred, target, boundary_band)
            losses["boundary"] = bl
            total = total + cfg.boundary_weight * bl

        if lc in ("boundary_grad", "full", "full_adaptive", "full_uw", "full_adaptive_uw"):
            bl = self.grad_boundary(pred, target, boundary_band)
            losses["boundary"] = bl
            total = total + cfg.boundary_weight * bl

        if lc in ("spectral_only", "full", "full_uw"):
            sl = self.spectral(pred, target, mask, boundary_band)
            losses["spectral"] = sl
            total = total + cfg.spectral_weight * sl

        if lc in ("adaptive_spectral", "full_adaptive", "full_adaptive_uw"):
            sl = self.adaptive_spectral(pred, target, mask, boundary_band)
            losses["spectral"] = sl
            total = total + cfg.spectral_weight * sl

        if lc == "adaptive_spectral_noreg":
            sl = self.adaptive_spectral_noreg(pred, target, mask, boundary_band)
            losses["spectral"] = sl
            total = total + cfg.spectral_weight * sl

        if lc == "ffl_only":
            sl = self.ffl(pred, target)
            losses["spectral"] = sl
            total = total + cfg.spectral_weight * sl

        ms_outputs = pred_dict.get("ms_outputs", [])
        if lc in ("full", "full_adaptive", "full_uw", "full_adaptive_uw") and ms_outputs:
            ms_total = pred.new_zeros(1).squeeze()
            for ms_out in ms_outputs:
                scale   = ms_out.shape[2:]
                tgt_ds  = F.interpolate(target, size=scale,
                                        mode="bilinear", align_corners=False)
                band_ds = F.interpolate(boundary_band, size=scale, mode="nearest")
                ms_total = ms_total + self.grad_boundary(ms_out, tgt_ds, band_ds)
            losses["ms_boundary"] = ms_total
            total = total + cfg.boundary_weight * cfg.ms_weight * ms_total

        losses["total"] = total
        return losses
