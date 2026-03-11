# Functions to add in metrics factory to use later for calculation
# Use torchmetrics

from torchmetrics import AUROC

# The ROC will be saved as an artifact in the pipeline later and AUC will be logged as a metric in MLflow
# Return them accordingly

def create_roc_auc(config):
    # For binary classification, set num_classes=1 and use sigmoid=True
    # For multi-class classification, set num_classes to the number of classes and use sigmoid=False
    if config.model.type == "classifier":
        if config.model.backbone.num_classes == 2:
            return AUROC(task="binary")
        else:
            return AUROC(task="multiclass", num_classes=config.model.backbone.num_classes)
    else:
        raise ValueError(f"ROC AUC metric is not applicable for model type '{config.model.type}'")