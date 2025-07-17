import torch
from typing import Optional

def get_stats(logits, targets, threshold=0.5, debug=False):
    # Logits: [B, C, H, W]; Targets: [B, H, W]

    probs = torch.softmax(logits, dim=1).float()        # Softmax prob values to range [0, 1]; Dim 1 holds our foreground data
    targets = targets.float()                           # Casting to float
    
    # Thresholding
    assert threshold >=0, f"Threshold must be >= 0. Got {threshold}"
    preds = (probs[:, 1, :, :] > threshold).to(targets.device) # Get our predictions above the threshold
    preds = preds.float()                    

    # Flatten
    B = preds.shape[0]                                      # Get batch size
    preds = preds.view(B, -1)                               # Flatten to [B, H*W]
    targets = targets.view(B, -1)                           # Flatten to [B, H*W]
    
    # Calculate the rates for these instead of the overall pixel counts
    TP = (preds * targets).sum(dim=1)
    FP = (preds * (1 - targets)).sum(dim=1)
    FN = ((1 - preds) * targets).sum(dim=1)
    TN = ((1 - preds) * (1 - targets)).sum(dim=1)


    return TP, FP, FN, TN

def _compute_metric(metric_fn, tp, fp, fn, tn, reduction, **metric_kwargs):
            
    # Global sum over all samples and all classes.
    # Treats the whole dataset as one big pool of pixels.
    if reduction == "micro":
        tp = tp.sum()
        fp = fp.sum()
        fn = fn.sum()
        tn = tn.sum()
        score = metric_fn(tp, fp, fn, tn, **metric_kwargs)

    # per sample/batch
    elif reduction is None:
        score = metric_fn(tp, fp, fn, tn, **metric_kwargs)
        
    else:
        raise ValueError(
            f"reduction` should be in [micro, None]. got {reduction}"
        )

    return score

"""
METRIC FUNCTIONS
"""

def _fbeta_score(tp, fp, fn, tn, beta=1):
    beta_tp = (1 + beta**2) * tp
    beta_fn = (beta**2) * fn
    score = beta_tp / (beta_tp + beta_fn + fp)
    return score

def _sensitivity(tp, fp, fn, tn):
    """Sensitivity (True Positive Rate)"""
    return tp / (tp + fn + 1e-8)

def _specificity(tp, fp, fn, tn):
    """Specificity (True Negative Rate)"""
    return tn / (tn + fp + 1e-8)

def _false_negative_rate(tp, fp, fn, tn):
    return 1 - _sensitivity(tp, fp, fn, tn)


def _false_positive_rate(tp, fp, fn, tn):
    return 1 - _specificity(tp, fp, fn, tn)


"""
METRICS
"""

# F1 Score with beta
def fbeta_score(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    beta: float = 1.0,
    reduction: Optional[str] = None,
) -> torch.Tensor:
    """F beta score"""
    return _compute_metric(
        _fbeta_score,
        tp,
        fp,
        fn,
        tn,
        beta=beta,
        reduction=reduction,
    )

def sensitivity(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    reduction: Optional[str] = None,
) -> torch.Tensor:
    """Sensitivity (TPR)"""
    return _compute_metric(
        _sensitivity,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
    )
    
def false_positive_rate(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    reduction: Optional[str] = None,
) -> torch.Tensor:
    """False Positive Rate (FPR)"""
    return _compute_metric(
        _false_positive_rate,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
    )
    
def specificity(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    reduction: Optional[str] = None,
) -> torch.Tensor:
    """Specificity (TNR)"""
    return _compute_metric(
        _specificity,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
    )

def false_negative_rate(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    reduction: Optional[str] = None,
) -> torch.Tensor:
    """False Negative Rate (FNR)"""
    return _compute_metric(
        _false_negative_rate,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
    )
