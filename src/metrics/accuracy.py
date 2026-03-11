from torchmetrics import Accuracy

def create_accuracy(config):
    if config.model.type == "classifier":
        return Accuracy(task="multiclass", num_classes=config.model.backbone.num_classes)
    else:
        raise ValueError(f"Accuracy metric is not applicable for model type '{config.model.type}'")