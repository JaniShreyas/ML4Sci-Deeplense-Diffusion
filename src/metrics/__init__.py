from .roc_auc import create_roc_auc
from .accuracy import create_accuracy
from .roc import create_roc

# Metric Registry
METRIC_REGISTRY = {
    "roc_auc": create_roc_auc,
    "accuracy": create_accuracy,
    "roc": create_roc,
}

def get_metric(metric_cfg, config):
    name = metric_cfg.get("name")
    if name not in METRIC_REGISTRY:
        raise ValueError(f"Metric '{name}' not found. Available metrics: {list(METRIC_REGISTRY.keys())}")
    
    builder_fn = METRIC_REGISTRY[name]
    return builder_fn(config).to(config.device)