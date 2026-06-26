# configs/config.py
"""
Central configuration.

PROFILES (pass --profile to main.py):
  default    → debug, fast, small data
  production → full dataset, 12 epochs, A100 settings
  smoke      → 5-epoch sanity check

SPEED CONTROL (SpeedConfig):
  test_fraction    : 0.001 = ~25 imgs (debug), 1.0 = full dataset (paper)
  epochs_full      : ALL phase-4 configs use this value (fair ablation)
  val_every_n_epochs: 2 saves ~30% time
  perc_every_n_steps: 4 saves ~40% time (VGG expensive)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class DataConfig:
    img_size:    int  = 256
    batch_size:  int  = 16
    num_workers: int  = 4
    pin_memory:  bool = True

    places365_dir: str = "./datasets/places365"
    celeba_dir:    str = "./datasets/celeba_hq"
    dtd_dir:       str = "./datasets/dtd"

    train_ratio: float = 0.70
    val_ratio:   float = 0.15

    mask_bins: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "easy":  (0.10, 0.20),
        "mid":   (0.20, 0.35),
        "hard":  (0.35, 0.50),
        "xhard": (0.50, 0.60),
    })

    n_val_masks:  int = 200
    n_test_masks: int = 500


@dataclass
class SpeedConfig:
    test_fraction:      float         = 0.001
    epochs_per_phase:   int           = 2
    epochs_full:        int           = 2      # ALL phase-4 configs (equal)
    val_every_n_epochs: int           = 1
    val_batches:        Optional[int] = 30
    perc_every_n_steps: int           = 1      # set 4 in production


@dataclass
class ModelConfig:
    backbone:                  str  = "resnet34"
    use_attention_skip:        bool = True
    use_gated_conv:            bool = True
    multiscale_output:         bool = True
    use_transformer_bottleneck:bool = True
    transformer_heads:         int  = 8
    transformer_layers:        int  = 2
    use_bilinear_upsample:     bool = True
    use_boundary_conditioning: bool = True


@dataclass
class LossConfig:
    hole_weight:       float = 6.0
    valid_weight:      float = 1.0
    perceptual_weight: float = 0.1
    boundary_weight:   float = 3.0
    boundary_dilation: int   = 5
    spectral_weight:   float = 1.0
    spectral_patches_k:  int = 8
    spectral_patch_size: int = 16
    spectral_use_log:   bool = True
    tv_weight:         float = 0.01
    ms_weight:         float = 0.5

    # ASBC
    spectral_n_bands:    int   = 4
    spectral_reg_weight: float = 0.01

    # Uncertainty-weighted loss
    use_uncertainty_weighting: bool = False

    # loss_config options:
    # "base"|"boundary_uniform"|"boundary_grad"|"spectral_only"
    # "adaptive_spectral"|"full"|"full_adaptive"|"full_uw"
    loss_config: str = "full"


@dataclass
class TrainConfig:
    seed:                  int   = 42
    lr_peak:               float = 3e-4
    lr_min:                float = 1e-6
    weight_decay:          float = 1e-4
    grad_clip:             float = 1.0
    warmup_epochs:         int   = 2
    patience:              int   = 8
    freeze_encoder_epochs: int   = 1
    use_amp:               bool  = True
    use_bf16:              bool  = True
    use_ema:               bool  = True
    ema_decay:             float = 0.999


@dataclass
class EvalConfig:
    k_values:         List[int]   = field(default_factory=lambda: [4, 8, 16, 32])
    patch_sizes:      List[int]   = field(default_factory=lambda: [8, 16, 32])
    dilation_values:  List[int]   = field(default_factory=lambda: [1, 3, 5, 7, 10, 15])
    spectral_weights: List[float] = field(default_factory=lambda: [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0])
    lama_checkpoint:         Optional[str] = None
    edge_connect_checkpoint: Optional[str] = None
    compute_fid: bool = True


@dataclass
class PathConfig:
    results_dir:      str = "./results"
    checkpoints_dir:  str = "./checkpoints"
    splits_dir:       str = "./splits"
    cached_masks_dir: str = "./cached_masks"
    logs_dir:         str = "./logs"


@dataclass
class Config:
    data:  DataConfig  = field(default_factory=DataConfig)
    speed: SpeedConfig = field(default_factory=SpeedConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss:  LossConfig  = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval:  EvalConfig  = field(default_factory=EvalConfig)
    paths: PathConfig  = field(default_factory=PathConfig)

    def make_dirs(self):
        for d in [self.paths.results_dir, self.paths.checkpoints_dir,
                  self.paths.splits_dir, self.paths.cached_masks_dir,
                  self.paths.logs_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)


def get_config() -> Config:
    return Config()


def get_production_config() -> Config:
    """
    A100 production run.
    epochs_full=12 x 6 configs x ~24 min/epoch = ~29h (fits Colab Pro+ limit).
    perc_every_n_steps=4 cuts VGG overhead by ~75%.
    """
    cfg = Config()
    cfg.data.batch_size          = 48
    cfg.data.num_workers         = 4
    cfg.data.n_val_masks         = 500
    cfg.data.n_test_masks        = 1000
    cfg.speed.test_fraction      = 1.0
    cfg.speed.epochs_per_phase   = 5
    cfg.speed.epochs_full        = 12
    cfg.speed.val_every_n_epochs = 2
    cfg.speed.val_batches        = None
    cfg.speed.perc_every_n_steps = 4
    cfg.train.patience           = 4
    cfg.train.use_ema            = True
    cfg.train.use_bf16           = True
    cfg.train.warmup_epochs      = 2
    return cfg


def get_smoke_config() -> Config:
    """5-epoch sanity check - runs in ~15 min on A100."""
    cfg = Config()
    cfg.data.batch_size          = 8
    cfg.data.num_workers         = 2
    cfg.speed.test_fraction      = 0.005
    cfg.speed.epochs_per_phase   = 2
    cfg.speed.epochs_full        = 5
    cfg.speed.val_every_n_epochs = 1
    cfg.speed.val_batches        = 10
    cfg.train.patience           = 5
    return cfg
