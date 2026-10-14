"""
config.py - Hardware detection and full model suite configurations
"""
import torch

def get_device() -> str:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def get_vram_gb() -> float:
    """Get available VRAM in GB."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / 1e9
    return 0

def auto_batch_size(base: int = 1024) -> int:
    """Scale batch size to available VRAM. Optimized for ADA 3500 12GB."""
    vram = get_vram_gb()
    if vram >= 16:
        return base
    elif vram >= 10: # ADA 3500 sits here
        return base // 2
    elif vram >= 4:
        return base // 4
    return 128

# Complete Model Suite Configuration
CONFIGS = {
    # Timeseries Domain (32-dim features)
    "ar": {
        "feature_dim": 32, "d_model": 256, "nhead": 8, "num_encoder_layers": 4, 
        "num_decoder_layers": 4, "dim_feedforward": 1024, "dropout": 0.1, "max_seq_len": 64,
    },
    "seasonal": {
        "feature_dim": 32, "d_model": 256, "nhead": 8, "num_encoder_layers": 4, 
        "num_decoder_layers": 4, "dim_feedforward": 1024, "dropout": 0.1, "max_seq_len": 64,
    },
    "stochastic": {
        "feature_dim": 32, "d_model": 256, "nhead": 8, "num_encoder_layers": 4, 
        "num_decoder_layers": 4, "dim_feedforward": 1024, "dropout": 0.1, "max_seq_len": 64,
    },
    
    # Algebraic Domain (48-dim features)
    "polynomial": {
        "feature_dim": 48, "d_model": 256, "nhead": 8, "num_encoder_layers": 4, 
        "num_decoder_layers": 4, "dim_feedforward": 1024, "dropout": 0.1, "max_seq_len": 64,
    },
    "rational": {
        "feature_dim": 48, "d_model": 256, "nhead": 8, "num_encoder_layers": 4, 
        "num_decoder_layers": 4, "dim_feedforward": 1024, "dropout": 0.1, "max_seq_len": 64,
    },
    "transcendental": {
        "feature_dim": 48, "d_model": 256, "nhead": 8, "num_encoder_layers": 4, 
        "num_decoder_layers": 4, "dim_feedforward": 1024, "dropout": 0.1, "max_seq_len": 64,
    },
    "trigonometric": {
        "feature_dim": 48, "d_model": 256, "nhead": 8, "num_encoder_layers": 4, 
        "num_decoder_layers": 4, "dim_feedforward": 1024, "dropout": 0.1, "max_seq_len": 96,
    },
    
    # Dynamical Domain (64-dim features)
    "linear_ode": {
        "feature_dim": 64, "d_model": 256, "nhead": 8, "num_encoder_layers": 4, 
        "num_decoder_layers": 4, "dim_feedforward": 1024, "dropout": 0.1, "max_seq_len": 64,
    },
    "nonlinear_ode": {
        "feature_dim": 64, "d_model": 256, "nhead": 8, "num_encoder_layers": 4, 
        "num_decoder_layers": 4, "dim_feedforward": 1024, "dropout": 0.1, "max_seq_len": 64,
    },
    "chaos": {
        "feature_dim": 64, "d_model": 512, "nhead": 8, "num_encoder_layers": 6, 
        "num_decoder_layers": 6, "dim_feedforward": 2048, "dropout": 0.1, "max_seq_len": 128,
    },
    
    # Discrete Domain (32-dim features)
    "recurrence": {
        "feature_dim": 32, "d_model": 256, "nhead": 8, "num_encoder_layers": 4, 
        "num_decoder_layers": 4, "dim_feedforward": 1024, "dropout": 0.1, "max_seq_len": 64,
    },
    "number_theory": {
        "feature_dim": 32, "d_model": 256, "nhead": 8, "num_encoder_layers": 4, 
        "num_decoder_layers": 4, "dim_feedforward": 1024, "dropout": 0.1, "max_seq_len": 64,
    },
    
    # Multivariate Domain (64-dim features)
    "physics_laws": {
        "feature_dim": 64, "d_model": 256, "nhead": 8, "num_encoder_layers": 4, 
        "num_decoder_layers": 4, "dim_feedforward": 1024, "dropout": 0.1, "max_seq_len": 64,
    },
}

TRAINING = {
    "learning_rate": 3e-4,
    "warmup_steps": 1000,
    "max_steps": 50000,
    "gradient_clip": 1.0,
    "log_interval": 100,
    "save_interval": 1000,
}

DOMAINS = {
    "ar": "timeseries",
    "seasonal": "timeseries",
    "stochastic": "timeseries",
    "polynomial": "algebraic",
    "rational": "algebraic",
    "transcendental": "algebraic",
    "trigonometric": "algebraic",
    "linear_ode": "dynamical",
    "nonlinear_ode": "dynamical",
    "chaos": "dynamical",
    "recurrence": "discrete",
    "number_theory": "discrete",
    "physics_laws": "multivariate",
}