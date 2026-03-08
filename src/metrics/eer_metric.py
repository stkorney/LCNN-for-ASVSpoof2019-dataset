import torch
import numpy as np

from src.metrics.base_metric import BaseMetric


def compute_det_curve(target_scores, nontarget_scores):
    """
    Compute DET curve from scores.
    """
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - \
        (np.arange(1, n_scores + 1) - tar_trial_sums)

    # false rejection rates
    frr = np.concatenate(
        (np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums /
                          nontarget_scores.size))  # false acceptance rates
    # Thresholds are the sorted scores
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))

    return frr, far, thresholds


def compute_eer(bonafide_scores, spoof_scores):
    """
    Compute Equal Error Rate from scores.
    """
    frr, far, thresholds = compute_det_curve(bonafide_scores, spoof_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]


class EERMetric(BaseMetric):
    """
    Metric for computing Equal Error Rate (EER).
    """

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        """
        Return placeholder value for EER (0.0).
        Actual EER needs to be computed separately after collecting all predictions.
        
        Args:
            logits (Tensor): model output logits.
            labels (Tensor): ground-truth labels.
        Returns:
            eer (float): placeholder value (0.0)
        """
        return 0.0


def compute(logits_list, labels_list):
    """
    Compute EER from lists of logits and labels.
    This should be called after processing all batches.
    
    Args:
        logits_list: list of logit tensors from all batches
        labels_list: list of label tensors from all batches
    Returns:
        eer (float): equal error rate as percentage
    """

    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    
    probs = torch.softmax(logits, dim=1)
    bonafide_scores = probs[:, 1].detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    
    bonafide_pred_scores = bonafide_scores[labels == 1]
    spoof_pred_scores = bonafide_scores[labels == 0]
    
    if len(bonafide_pred_scores) == 0 or len(spoof_pred_scores) == 0:
        return 100.0
    
    eer, _ = compute_eer(bonafide_pred_scores, spoof_pred_scores)
    return eer * 100

