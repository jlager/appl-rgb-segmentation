import torch
from modules import(
    metrics
)
class Trainer:
    """
    Trainer class for training and validating a segmentation model.
    """
    
    def __init__(
        self, 
        model, 
        train_loader, 
        val_loader,
        optimizer, 
        criterion,
        scaler,
        device
        ):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scaler = scaler
        self.device = device

    # shared forward method for training and validation
    def forward(self, images, masks, auto_cast_dtype=None):

        images = images.to(self.device)
        masks = masks.to(self.device)

        if auto_cast_dtype is not None:
            with torch.amp.autocast('cuda', dtype=auto_cast_dtype):
                logits = self.model(images)
                loss = self.criterion(logits, masks)
        else:
            logits = self.model(images)
            loss = self.criterion(logits, masks)

        return logits, loss

    # backward pass for training
    def backward(self, loss):
        
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
            
    # shared batch iteration method for training and validation
    def shared_step(self, batch, stage):
        
        images = batch[0]
        assert images.ndim == 4, "Images should be a 4D tensor (batch_size, channels, height, width)"
        
        masks = batch[1]
        masks = masks
        assert masks.ndim == 3, "Masks should be a 3D tensor (batch_size, height, width)"
        assert masks.max() <= 1 and masks.min() >= 0, "Masks should be binary (0 or 1)"
        
        h, w = images.shape[2:]
        assert h % 32 == 0 and w % 32 == 0, "Image dimensions should be divisible by 32"

        # float32 will be good enough
        logits, loss = self.forward(images, masks, auto_cast_dtype=None)
        
        if stage == "train":
            self.backward(loss)
        
        # Get stats for metrics. Rates used for logging/printing
        tp, fp, fn, tn = metrics.get_stats(logits, masks, debug=False)
        tpr = metrics.recall_score(tp, fn, tn, fn, reduction="micro")
        fpr = metrics.false_positive_rate(fp, tn, tn, fn, reduction="micro")
        tnr = metrics.specificity(fp, tn, tn, fn, reduction="micro")
        fnr = metrics.false_negative_rate(fn, tp, tn, fn, reduction="micro")



        print(f"{stage} - Loss: {loss.item():.4f}, TPR: {tpr.item():.4f}, FPR: {fpr.item():.4f}, FNR: {fnr.item():.4f}, TNR: {tnr.item():.4f}")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        }
    
    # shared metric gathering for training and validation    
    def shared_epoch_end(self, outputs, stage):
        
        # Aggregate stats across all batches
        tp = torch.cat([out['tp'] for out in outputs])
        fp = torch.cat([out['fp'] for out in outputs])
        fn = torch.cat([out['fn'] for out in outputs])
        tn = torch.cat([out['tn'] for out in outputs])

        dataset_f1 = metrics.fbeta_score(tp, fp, fn, tn, beta=1.0, reduction="micro")

        metrics = {
            f"{stage}_dataset_f1": dataset_f1,
        }

        print(f"Metrics for {stage}:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.5f}")

    # training step
    def train(self):
        
        self.model.train()
        outputs = []
        
        for batch in self.train_loader:
            output = self.shared_step(batch, stage="train")
            outputs.append(output)
        
        # Metrics
        self.shared_epoch_end(outputs, stage="train")
    
    # Validation step
    def validate(self):
        
        self.model.eval()
        outputs = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                output = self.shared_step(batch, stage="val")
                outputs.append(output)
                        
        # Metrics
        self.shared_epoch_end(outputs, stage="val")
        
        print("")
        
    # Fit method to run training and validation for a specified number of epochs
    def fit(self, max_epochs=10):
        
        for epoch in range(max_epochs):
            print(f"Epoch {epoch + 1}/{max_epochs}")
            self.train()
            self.validate()
        
