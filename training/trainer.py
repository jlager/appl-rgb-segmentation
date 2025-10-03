import torch
import gc

from tqdm import tqdm
from typing import Optional
from torch.utils.tensorboard import SummaryWriter
from modules import(
    get_stats,
    fbeta_score,
    sensitivity,
    false_positive_rate,
    specificity,
    false_negative_rate,
)

class Trainer():
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
        scheduler,
        
        device,
        patience,
        autocast_dtype=None,
        gradient_accumulation_steps=1,
        lr_warmup_epochs=0,
        
        log_path: Optional[str] = None,
        ckpt_path: Optional[str] = None
        ):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scaler = scaler
        self.scheduler = scheduler

        self.device = device
        self.patience = patience
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.lr_warmup_epochs = lr_warmup_epochs

        # Matches autocast_dtype string to torch dtype and set self.autocast_dtype
        dtype_map = {
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
        }
        self.autocast_dtype = dtype_map.get(autocast_dtype) if autocast_dtype in dtype_map else None
        
        # If log_path specified... create TensorBoard SummaryWriter
        if log_path:
            self.writer = SummaryWriter(log_path)
            print(f"TensorBoard logging enabled at {log_path}")
        else:
            self.writer = None

        self.ckpt_path = ckpt_path
        
        self.current_epoch = 0


    def forward(self, images, masks):
        
        # Move data to device
        images = images.to(self.device)
        masks = masks.to(self.device)
        
        # If autocast dtype specified... use autocast context manager for mixed precision
        if self.autocast_dtype:
            with torch.amp.autocast(device_type=self.device.type, dtype=self.autocast_dtype):
                logits = self.model(images)
        else:
            logits = self.model(images)
        
        loss = self.criterion(logits, masks)

        return logits, loss

    def backward(self, loss):
        
        self.scaler.scale(loss).backward()
    
    def update(self, idx):

        # If we've accumulated enough gradients or it's the last batch... update weights
        is_last_batch = (idx + 1) == len(self.train_loader)
        if (idx + 1) % self.gradient_accumulation_steps == 0 or is_last_batch:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

    def shared_step(self, batch, stage, batch_idx=None):

        # Assert correct dimensions for images and masks
        images = batch[0]
        assert images.ndim == 4, "Images should be a 4D tensor (batch_size, channels, height, width)"
        
        masks = batch[1]
        assert masks.ndim == 3, "Masks should be a 3D tensor (batch_size, height, width)"
        assert masks.max() <= 1 and masks.min() >= 0, "Masks should be binary (0 or 1)"
        
        h, w = images.shape[2:]
        assert h % 32 == 0 and w % 32 == 0, "Image dimensions should be divisible by 32"

        # Forward pass
        logits, loss = self.forward(images, masks)

        # If training... scale loss to account for grad accumulation; Backward pass + updates
        if stage == "train":
            scaled_loss = loss / self.gradient_accumulation_steps  
            self.backward(scaled_loss)
            self.update(batch_idx)

        # If validating... designate a ROI/halo to ignore border artifacts when calculating stats during validation
        if stage == "val":
            halo = 64
            logits = logits[:, :, halo:-halo, halo:-halo]
            masks = masks[:, halo:-halo, halo:-halo]

        # Get total true positives, false positives, false negatives, true negatives and F1 for batch.
        with torch.no_grad():
            tp, fp, fn, tn = get_stats(logits, masks, debug=False)
            f1 = fbeta_score(tp, fp, fn, tn, beta=1.0, reduction=None)

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "f1": f1,
        }
        
    def save_best_model(self, metric, ckpt_path):

        # If no self.best_metric attribute exists... create it and save the model
        if not hasattr(self, 'best_metric'):
            self.best_metric = metric
            torch.save(self.model.state_dict(), ckpt_path)
            print(f"Initial model saved with {metric:.4f}")

        # Else... if recieved metric value > self.best_metric, save model
        elif metric > self.best_metric:
            self.best_metric = metric
            torch.save(self.model.state_dict(), ckpt_path)
            print(f"Model improved. Saved new best model with {metric:.4f}")

    def lr_warmup(self):
        
        # Get max_lr from optimizer's initial lr
        max_lr = self.optimizer.param_groups[0]['lr']
        
        # If on first epoch... set lr to 1% of max_lr.
        if self.current_epoch == 0:
            warmup_lr = max_lr * 0.01
            
        # Else... linearly scale lr based on current epoch
        else:
            progress = min(self.current_epoch / self.lr_warmup_epochs, 1.0)   # Increment factor determined by epoch progress
            warmup_lr = max_lr * progress

        # Set lr for all param groups
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = warmup_lr
        
        
    def scheduler_step(self, metric=None):

        # If scheduler exists...
        if self.scheduler:
            
            # If metric is provided... step with metric
            if metric is not None:
                self.scheduler.step(metric)
     
            # Else... step normally
            else:
                self.scheduler.step()


    def early_stopping(self, delta, metrics, target_metric):
            
        # Assert target metric is in metrics dict and grab
        assert target_metric in metrics, f"Target metric '{target_metric}' not found in metrics."
        current_metric = metrics[target_metric]

        # If no self.es_best attribute exists... create it and initialize count
        if not hasattr(self, 'es_best'):
            self.es_best = current_metric
            self.count = 0
        
        # Else, if current metric improved beyond delta threshold... set new best and reset count
        elif current_metric > self.es_best + delta:
            print(f"Improvement detected: {current_metric:.4f} > {self.es_best:.4f}. Resetting early stopping counter.")
            self.es_best = current_metric
            self.count = 0
            
        # Else... increment count
        else:
            self.count += 1
            print(f"Early stopping patience: {self.count}/{self.patience}")

        # If count exceeds patience... trigger early stopping
        if self.count >= self.patience:
            print("Early stopping triggered.")
            return True   
        
        return False    

    def shared_epoch_end(self, outputs, stage, verbose=False):
        
        # Aggregate stats across all batches
        loss = torch.stack([out['loss'] for out in outputs])
        tp = torch.cat([out['tp'] for out in outputs])
        fp = torch.cat([out['fp'] for out in outputs])
        fn = torch.cat([out['fn'] for out in outputs])
        tn = torch.cat([out['tn'] for out in outputs])

        # Store metrics
        metrics = {
            f"{stage}_dataset_loss": loss.mean().item(),
            f"{stage}_dataset_TPR": sensitivity(tp, fp, fn, tn, reduction="micro"),
            f"{stage}_dataset_FPR": false_positive_rate(tp, fp, fn, tn, reduction="micro"),
            f"{stage}_dataset_TNR": specificity(tp, fp, fn, tn, reduction="micro"),
            f"{stage}_dataset_FNR": false_negative_rate(tp, fp, fn, tn, reduction="micro"),
            f"{stage}_dataset_f1": fbeta_score(tp, fp, fn, tn, beta=1.0, reduction="micro")
        }

        # If TensorBoard writer exists... log metrics
        if self.writer:
            for key, value in metrics.items():
                tag = key.replace(f"{stage}_dataset_", "")
                self.writer.add_scalar(f"{stage}/{tag}", value, self.current_epoch)        
        
        # If verbose... print metrics to console
        if verbose:
            print(f"Metrics for {stage}:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.5f}")
            print("")

        # If in validation stage... save checkpoint if target metric improved during validation and check for early stopping   
        if stage == "val" and self.ckpt_path:
            self.save_best_model(metrics[f"{stage}_dataset_f1"], self.ckpt_path)
            
            if self.patience is not None:
                if self.early_stopping(delta=0.01, metrics=metrics, target_metric=f"{stage}_dataset_f1"):
                    print("Early stopping criteria met. Stopping training.")
                    raise StopIteration
                            
    def train(self, verbose=False):
        
        # Training mode; bookkeeping for metrics; progress bar
        self.model.train()
        outputs = []
        pbar = tqdm(self.train_loader, desc=f"Training", )

        # For each index, batch in the training dataloader...
        for idx, batch in enumerate(pbar):
            
            # Obtain dictionary of output tensors for current batch. Move tensors to CPU and detach. Store in outputs list
            output = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                      for k, v in self.shared_step(batch, "train", idx).items()}
            outputs.append(output)

            # Loss and F1 are aggregated over all batches into outputs[]. Take the mean of these and report in pbar
            current_loss = output['loss'].mean()
            current_f1 = output['f1'].mean()
            pbar.set_postfix(loss=f"{current_loss:.4e}", f1=f"{current_f1.item():.4f}")

        # Return metrics
        return self.shared_epoch_end(outputs, stage="train", verbose=verbose)
            
 
    def validate(self, verbose=False):
         
        # Similar to train() except we do backward pass or updates, hence, no need to pass batch idx into self.shared_step()
        self.model.eval()
        outputs = []
        pbar = tqdm(self.val_loader, desc=f"Validation", )

        with torch.inference_mode():
            for batch in pbar:
                output = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v 
                         for k, v in self.shared_step(batch, "val").items()}
                outputs.append(output)
                
                current_loss = output['loss'].mean()
                current_f1 = output['f1'].mean()
                pbar.set_postfix(loss=f"{current_loss:.4e}", f1=f"{current_f1:.4f}")

        return self.shared_epoch_end(outputs, stage="val", verbose=verbose)
                   
                   
    def fit(self, max_epochs=10, verbose=False):
        
        try:
            # For each epoch in the range of max_epochs...
            for epoch in range(max_epochs):
                
                # Display epoch progress; train model; validate model; increment epoch counter
                print(f"Epoch {epoch + 1}/{max_epochs}")
                self.train(verbose=verbose)
                self.validate(verbose=verbose)
                
                # If self.current_epoch < self.lr_warmup_epochs... step lr warmup
                if self.current_epoch < self.lr_warmup_epochs:
                    self.lr_warmup(max_lr=self.max_lr)

                # If self.current_epoch > self.lr_warmup_epochs and scheduler exists... step scheduler
                if self.scheduler and epoch >= self.lr_warmup_epochs:
                    self.scheduler_step()
                    
                self.current_epoch += 1

        except Exception as e:
            print(f"Training interrupted by error: {str(e)}")
            raise e
        finally:
            print("Training finished. Shutting down")
