import torch
import gc
import segmentation_models_pytorch as smp
import timm

from tqdm import tqdm

from modules import(
    get_stats,
    fbeta_score,
    sensitivity,
    false_positive_rate,
    specificity,
    false_negative_rate,
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
        device,
        autocast_dtype=None,
        patience=None,
        gradient_accumulation_steps=1
        ):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scaler = scaler
        self.device = device
        self.patience = patience
        self.gradient_accumulation_steps = gradient_accumulation_steps

        dtype_map = {
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
            'float32': torch.float32
        }
        self.autocast_dtype = dtype_map.get(autocast_dtype) if autocast_dtype in dtype_map else None

        self.step_idx = 0
        
    # shared forward method for training and validation
    def forward(self, images, masks):

        images = images.to(self.device)
        masks = masks.to(self.device)

        if self.autocast_dtype:
            with torch.autocast(device_type='cuda', dtype=self.autocast_dtype):
                logits = self.model(images)
                loss = self.criterion(logits, masks)
        else:
            logits = self.model(images)
            loss = self.criterion(logits, masks)

        return logits, loss

    # backward pass for training
    def backward(self, loss):
        
        assert torch.isfinite(loss).all(), f"Non-finite loss at step {self.step_idx}"
        self.scaler.scale(loss).backward()
        
        is_accumulation_step = (self.step_idx + 1) % self.gradient_accumulation_steps != 0
        is_last_batch = (self.step_idx + 1) == len(self.train_loader)
                
        # Update weights if we've accumulated enough gradients
        if (self.step_idx + 1) % self.gradient_accumulation_steps == 0 or is_last_batch:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            
    # shared batch iteration method for training and validation
    def shared_step(self, batch, stage):
        
        images = batch[0]
        assert images.ndim == 4, "Images should be a 4D tensor (batch_size, channels, height, width)"
        
        masks = batch[1]
        assert masks.ndim == 3, "Masks should be a 3D tensor (batch_size, height, width)"
        assert masks.max() <= 1 and masks.min() >= 0, "Masks should be binary (0 or 1)"
        
        h, w = images.shape[2:]
        assert h % 32 == 0 and w % 32 == 0, "Image dimensions should be divisible by 32"

        logits, loss = self.forward(images, masks)
        
        if stage == "train":
            self.backward(loss)
        
        self.step_idx += 1

        
        # Get total true positives, false positives, false negatives, and true negatives for batch
        tp, fp, fn, tn = get_stats(logits, masks, debug=False)
        
        # Get per batch F1
        f1 = fbeta_score(tp, fp, fn, tn, beta=1.0, reduction=None)


        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "f1": f1,
        }
    
    # shared metric gathering for training and validation    
    def shared_epoch_end(self, outputs, stage, verbose=False):
        
        # Aggregate stats across all batches
        loss = torch.stack([out['loss'] for out in outputs])
        tp = torch.cat([out['tp'] for out in outputs])
        fp = torch.cat([out['fp'] for out in outputs])
        fn = torch.cat([out['fn'] for out in outputs])
        tn = torch.cat([out['tn'] for out in outputs])

        metrics = {
            f"{stage}_dataset_loss": loss.mean().item(),
            f"{stage}_dataset_TPR": sensitivity(tp, fp, fn, tn, reduction="micro"),
            f"{stage}_dataset_FPR": false_positive_rate(tp, fp, fn, tn, reduction="micro"),
            f"{stage}_dataset_TNR": specificity(tp, fp, fn, tn, reduction="micro"),
            f"{stage}_dataset_FNR": false_negative_rate(tp, fp, fn, tn, reduction="micro"),
            f"{stage}_dataset_f1": fbeta_score(tp, fp, fn, tn, beta=1.0, reduction="micro")
        }

        if verbose:
            print(f"Metrics for {stage}:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.5f}")
            print("")
            
        self.step_idx = 0
        
        return metrics

    # training step
    def train(self, verbose=False):
        
        self.model.train()
        outputs = []
        pbar = tqdm(self.train_loader, desc=f"Training", )

        for batch in pbar:
            output = self.shared_step(batch, stage="train")
            current_loss = output['loss'].mean()
            current_f1 = output['f1'].mean()
            outputs.append(output)

            pbar.set_postfix(loss=f"{current_loss:.4e}", f1=f"{current_f1.item():.4f}")

        # Return metrics
        return self.shared_epoch_end(outputs, stage="train", verbose=verbose)
            
    # Validation step
    def validate(self, verbose=False):
        
        self.model.eval()
        outputs = []
        pbar = tqdm(self.val_loader, desc=f"Validation", )
        
        with torch.no_grad():
            for batch in pbar:
                output = self.shared_step(batch, stage="val")
                current_loss = output['loss'].mean()
                current_f1 = output['f1'].mean()
                outputs.append(output)


                pbar.set_postfix(loss=f"{current_loss:.4e}", f1=f"{current_f1:.4f}\n")

        # Return metrics
        return self.shared_epoch_end(outputs, stage="val", verbose=verbose)

    def early_stopping(self, delta, metrics, target_metric):
        
        """
        Check if early stopping criteria are met.
        If the target metric does not improve for 'patience' epochs,
        training will be stopped.
        """

        print(metrics)
        
        # Ensure the target metric is in the metrics dictionary
        assert target_metric in metrics, f"Target metric '{target_metric}' not found in metrics."
        
        # Get current metric value from passed metrics dict
        current_metric = metrics[target_metric]

        # Initialize es_best and count if not already set. Absence of one will imply the absence of the other
        if not hasattr(self, 'es_best'):
            self.es_best = current_metric
            self.count = 0
        
        # Check if current metric is better than the best seen so far
        elif current_metric > self.es_best + delta:
            print(f"Improvement detected: {current_metric:.4f} > {self.es_best:.4f}. Resetting early stopping counter.")
            self.es_best = current_metric
            self.count = 0
        else:
            self.count += 1
            print(f"Early stopping patience: {self.count}/{self.patience}")

        # If count exceeds patience, trigger early stopping
        if self.count >= self.patience:
            print("Early stopping triggered.")
            return True   
        
        return False     
        
    def terminate(self):
        """
        Clean up resources and finalize training.
        Called after all epochs are completed.
        """
        
        # Free CUDA memory
        torch.cuda.empty_cache()
        
        # Deref dataloaders and collect
        self.train_loader = None
        self.val_loader = None
        gc.collect()
        
        # Clear early stopping attributes if they exist
        if getattr(self, "patience", None) is not None:
            for attr in ("es_best", "count"):
                if hasattr(self, attr):
                    delattr(self, attr)
            
           
    # Fit method to run training and validation for a specified number of epochs
    def fit(self, max_epochs=10, verbose=False):
        try:
            for epoch in range(max_epochs):
                print(f"Epoch {epoch + 1}/{max_epochs}")
                train_metric = self.train(verbose=verbose)
                val_metric = self.validate(verbose=verbose)
                
                if self.patience is not None:
                    if self.early_stopping(delta=0.01, metrics=val_metric, target_metric="val_dataset_f1"):
                        print("Early stopping criteria met. Stopping training.")
                        break

        except Exception as e:
            print(f"Training interrupted by error: {str(e)}")
            raise e
        finally:
            print("Training finished. Shutting down")
            self.terminate()  # Clean up resources even if an error occurred

