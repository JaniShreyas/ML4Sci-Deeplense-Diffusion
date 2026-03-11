# The auroc metric only returns the auc and not the fpr and tpr
# For that we require the roc metric from torchmetrics

from torchmetrics import ROC

def create_roc(config):
    if config.model.type == "classifier":
        if config.model.backbone.num_classes == 2:
            return ROC(task="binary")
        else:
            return ROC(task="multiclass", num_classes=config.model.backbone.num_classes)
    raise ValueError(f"ROC metric is not supported for model type {config.model.type}")