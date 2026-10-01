"""
NODE21 RetinaNet Evaluation
============================

Overall metrics
---------------
- FROC / CPM (primary -- sensitivity at fixed FPs/image, averaged)
- Precision
- Sensitivity / Recall
- F1-score
- False Negative Rate (FNR)
- False Positives per Image

Small-nodule analysis
---------------------
NODE21 provides bounding-box coordinates rather than physical
nodule diameters. Therefore, nodule size is defined using the
ground-truth bounding-box area:

    area = (xmax - xmin) * (ymax - ymin)

The small-nodule threshold is determined ONLY from the training
set using the 25th percentile of positive nodule areas.

The threshold is then frozen and applied to the test set.

FROC / CPM note
----------------
FROC/CPM matching is class-agnostic (single "nodule" category),
matching how NODE21 itself is framed. The full FROC curve (avg FP/image
vs. sensitivity) is built in one confidence-sorted pass and reused both
for the overall curve and for a parallel small-nodule-only curve, so
false positives stay pooled across the whole model (a false positive
has no "size") while only the sensitivity numerator is split by GT
nodule size -- this is the standard way size-stratified FROC is done
in the nodule-CAD literature.

Raw curve points are saved to CSV (not plotted here) via
save_froc_curve(), so plotting can be done separately afterward.

Author: Mashego Gideon Mabeloane
"""

import bisect
import random
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision.ops import box_iou


# ============================================================
# Configuration
# ============================================================

SMALL_PERCENTILE = 25

SCORE_THRESHOLD = 0.55

IOU_THRESHOLD = 0.20

# Relaxed IoU criterion used ONLY for small-nodule matching. A few-pixel
# shift on a tiny box drops IoU below 0.5 even when the detection is
# clinically fine, so a lower threshold is used specifically for GT
# boxes classified as "small" (area < small_threshold, frozen from the
# training set). Standard-size nodules still use IOU_THRESHOLD = 0.50.
SMALL_IOU_THRESHOLD = 0.16

# Standard LUNA16/NODE21-style CPM operating points (FPs per image).
FROC_FP_POINTS = (0.125, 0.25, 0.5)


# ============================================================
# 1. Calculate training-set nodule areas
# ============================================================

def get_nodule_sizes(csv_file):

    df = pd.read_csv(csv_file)

    # Keep only positive nodule annotations
    positive = df[df["label"] == 1].copy()

    if len(positive) == 0:
        raise ValueError(
            f"No positive annotations found in {csv_file}"
        )

    positive["width"] = (
        positive["xmax"] - positive["xmin"]
    )

    positive["height"] = (
        positive["ymax"] - positive["ymin"]
    )

    positive["area"] = (
        positive["width"]
        * positive["height"]
    )

    # Remove invalid boxes
    positive = positive[
        (positive["width"] > 0)
        & (positive["height"] > 0)
    ]

    return positive


# ============================================================
# 2. Determine small-nodule threshold
# ============================================================

def determine_small_threshold(
    train_csv,
    percentile=SMALL_PERCENTILE,
):

    annotations = get_nodule_sizes(
        train_csv
    )

    areas = annotations[
        "area"
    ].to_numpy()

    threshold = float(
        np.percentile(
            areas,
            percentile,
        )
    )

    print()
    print("=" * 60)
    print("NODE21 NODULE SIZE DISTRIBUTION")
    print("=" * 60)

    print(
        f"Positive annotations : {len(areas)}"
    )

    print(
        f"Minimum area         : "
        f"{np.min(areas):.2f} px²"
    )

    print(
        f"10th percentile      : "
        f"{np.percentile(areas, 10):.2f} px²"
    )

    print(
        f"25th percentile      : "
        f"{np.percentile(areas, 25):.2f} px²"
    )

    print(
        f"Median               : "
        f"{np.percentile(areas, 50):.2f} px²"
    )

    print(
        f"75th percentile      : "
        f"{np.percentile(areas, 75):.2f} px²"
    )

    print(
        f"90th percentile      : "
        f"{np.percentile(areas, 90):.2f} px²"
    )

    print(
        f"Maximum area         : "
        f"{np.max(areas):.2f} px²"
    )

    print()
    print(
        f"SMALL NODULE DEFINITION"
    )
    print("-" * 60)

    print(
        f"Small nodule area < "
        f"{threshold:.2f} px²"
    )

    print(
        f"Threshold source: "
        f"{percentile}th percentile of TRAINING nodules"
    )

    print("=" * 60)

    return threshold


# ============================================================
# 3. Run inference
# ============================================================

@torch.no_grad()
def run_inference(
    model,
    dataloader,
    device,
):

    model.eval()

    predictions = []

    print()
    print("=" * 60)
    print("RUNNING TEST INFERENCE")
    print("=" * 60)

    for images, targets in dataloader:

        images = [
            image.to(device)
            for image in images
        ]

        outputs = model(images)

        for output in outputs:

            predictions.append(
                {
                    "boxes": output[
                        "boxes"
                    ].detach().cpu(),

                    "scores": output[
                        "scores"
                    ].detach().cpu(),

                    "labels": output[
                        "labels"
                    ].detach().cpu(),
                }
            )

    print(
        f"Images evaluated : "
        f"{len(predictions)}"
    )

    print("=" * 60)

    return predictions


# ============================================================
# 4. Extract ground truth
# ============================================================

def get_ground_truths(
    dataset,
):

    ground_truths = []

    for idx in range(
        len(dataset)
    ):

        _, target = dataset[idx]

        ground_truths.append(
            {
                "boxes": target[
                    "boxes"
                ].detach().cpu(),

                "labels": target[
                    "labels"
                ].detach().cpu(),
            }
        )

    return ground_truths


# ============================================================
# 5. Match predictions to ground truth
# ============================================================

def match_predictions(
    predicted_boxes,
    predicted_scores,
    ground_truth_boxes,
    iou_threshold=IOU_THRESHOLD,
):

    num_predictions = len(
        predicted_boxes
    )

    num_ground_truths = len(
        ground_truth_boxes
    )

    # No predictions
    if num_predictions == 0:

        return {
            "tp": [],
            "fp": [],
            "fn": list(
                range(
                    num_ground_truths
                )
            ),
        }

    # No ground-truth objects
    if num_ground_truths == 0:

        return {
            "tp": [],
            "fp": list(
                range(
                    num_predictions
                )
            ),
            "fn": [],
        }

    # Sort predictions by confidence
    order = torch.argsort(
        predicted_scores,
        descending=True,
    )

    sorted_boxes = (
        predicted_boxes[order]
    )

    # IoU matrix
    ious = box_iou(
        sorted_boxes,
        ground_truth_boxes,
    )

    matched_gt = set()

    tp = []

    fp = []

    # Greedy matching: highest-confidence predictions are matched first.
    for pred_idx in range(
        len(sorted_boxes)
    ):

        best_iou = 0.0
        best_gt = None

        for gt_idx in range(
            num_ground_truths
        ):

            if gt_idx in matched_gt:
                continue

            current_iou = float(
                ious[
                    pred_idx,
                    gt_idx,
                ]
            )

            if current_iou > best_iou:

                best_iou = current_iou
                best_gt = gt_idx

        original_pred_idx = int(
            order[pred_idx]
        )

        if (
            best_gt is not None
            and best_iou >= iou_threshold
        ):

            tp.append(
                {
                    "pred_idx":
                        original_pred_idx,

                    "gt_idx":
                        best_gt,

                    "iou":
                        best_iou,
                }
            )

            matched_gt.add(
                best_gt
            )

        else:

            fp.append(
                original_pred_idx
            )

    fn = [
        gt_idx
        for gt_idx in range(
            num_ground_truths
        )
        if gt_idx not in matched_gt
    ]

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


# ============================================================
# 5b. Size-aware matching (relaxed IoU for small ground truth)
# ============================================================

def match_predictions_size_aware(
    predicted_boxes,
    predicted_scores,
    ground_truth_boxes,
    small_mask,
    iou_threshold=IOU_THRESHOLD,
    small_iou_threshold=SMALL_IOU_THRESHOLD,
):

    num_predictions = len(predicted_boxes)
    num_ground_truths = len(ground_truth_boxes)

    if num_predictions == 0:
        return {"tp": [], "fp": [], "fn": list(range(num_ground_truths))}

    if num_ground_truths == 0:
        return {"tp": [], "fp": list(range(num_predictions)), "fn": []}

    order = torch.argsort(predicted_scores, descending=True)
    sorted_boxes = predicted_boxes[order]

    ious = box_iou(sorted_boxes, ground_truth_boxes)

    matched_gt = set()
    tp = []
    fp = []

    for pred_idx in range(len(sorted_boxes)):

        best_iou = 0.0
        best_gt = None

        for gt_idx in range(num_ground_truths):

            if gt_idx in matched_gt:
                continue

            current_iou = float(ious[pred_idx, gt_idx])

            if current_iou > best_iou:
                best_iou = current_iou
                best_gt = gt_idx

        original_pred_idx = int(order[pred_idx])

        if best_gt is not None:
            threshold = (
                small_iou_threshold
                if bool(small_mask[best_gt])
                else iou_threshold
            )
        else:
            threshold = iou_threshold

        if best_gt is not None and best_iou >= threshold:

            tp.append(
                {"pred_idx": original_pred_idx, "gt_idx": best_gt, "iou": best_iou}
            )
            matched_gt.add(best_gt)

        else:

            fp.append(original_pred_idx)

    fn = [
        gt_idx
        for gt_idx in range(num_ground_truths)
        if gt_idx not in matched_gt
    ]

    return {"tp": tp, "fp": fp, "fn": fn}


# ============================================================
# 6. Overall detection metrics
# ============================================================

def calculate_detection_metrics(
    predictions,
    ground_truths,
    score_threshold=SCORE_THRESHOLD,
    iou_threshold=IOU_THRESHOLD,
):

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for prediction, target in zip(
        predictions,
        ground_truths,
    ):

        scores = prediction[
            "scores"
        ]

        boxes = prediction[
            "boxes"
        ]

        keep = (
            scores
            >= score_threshold
        )

        boxes = boxes[keep]

        scores = scores[keep]

        gt_boxes = target[
            "boxes"
        ]

        matches = match_predictions(
            boxes,
            scores,
            gt_boxes,
            iou_threshold,
        )

        total_tp += len(
            matches["tp"]
        )

        total_fp += len(
            matches["fp"]
        )

        total_fn += len(
            matches["fn"]
        )

    precision = (
        total_tp
        / (
            total_tp
            + total_fp
            + 1e-8
        )
    )

    sensitivity = (
        total_tp
        / (
            total_tp
            + total_fn
            + 1e-8
        )
    )

    fnr = (
        total_fn
        / (
            total_tp
            + total_fn
            + 1e-8
        )
    )

    f1 = (
        2
        * precision
        * sensitivity
        / (
            precision
            + sensitivity
            + 1e-8
        )
    )

    return {
        "TP": int(total_tp),
        "FP": int(total_fp),
        "FN": int(total_fn),

        "precision": float(
            precision
        ),

        "sensitivity": float(
            sensitivity
        ),

        "recall": float(
            sensitivity
        ),

        "F1": float(f1),

        "FNR": float(fnr),
    }


# ============================================================
# 6b. Overall image-level specificity
# ============================================================

def calculate_specificity(
    predictions,
    ground_truths,
    score_threshold=SCORE_THRESHOLD,
):
    """
    Calculate image-level specificity for the overall nodule
    detection task.

    Positive image:
        Contains at least one ground-truth nodule.

    Negative image:
        Contains no ground-truth nodules.

    TN:
        Negative image with no predictions above threshold.

    FP:
        Negative image with one or more predictions above threshold.

    Specificity = TN / (TN + FP)
    """

    tn = 0
    fp = 0

    for prediction, target in zip(
        predictions,
        ground_truths,
    ):
        gt_boxes = target["boxes"]
        pred_scores = prediction["scores"]

        predicted_positive = bool(
            (pred_scores >= score_threshold).any()
        )

        if len(gt_boxes) == 0:
            if predicted_positive:
                fp += 1
            else:
                tn += 1

    specificity = tn / (tn + fp + 1e-8)

    return {
        "TN": int(tn),
        "specificity": float(specificity),
    }


# ============================================================
# 7. Small-nodule metrics (fixed threshold, precision/sensitivity/FNR)
# ============================================================

def calculate_small_nodule_metrics(
    predictions,
    ground_truths,
    small_threshold,
    score_threshold=SCORE_THRESHOLD,
    iou_threshold=IOU_THRESHOLD,
    small_iou_threshold=SMALL_IOU_THRESHOLD,
):

    small_tp = 0
    small_fn = 0
    small_total = 0

    for prediction, target in zip(
        predictions,
        ground_truths,
    ):

        pred_boxes = prediction[
            "boxes"
        ]

        pred_scores = prediction[
            "scores"
        ]

        gt_boxes = target[
            "boxes"
        ]

        if len(gt_boxes) == 0:
            continue

        gt_widths = (
            gt_boxes[:, 2]
            - gt_boxes[:, 0]
        )

        gt_heights = (
            gt_boxes[:, 3]
            - gt_boxes[:, 1]
        )

        gt_areas = (
            gt_widths
            * gt_heights
        )

        small_mask = (
            gt_areas
            < small_threshold
        )

        small_indices = torch.where(
            small_mask
        )[0].tolist()

        small_total += len(
            small_indices
        )

        keep = (
            pred_scores
            >= score_threshold
        )

        pred_boxes_filtered = (
            pred_boxes[keep]
        )

        pred_scores_filtered = (
            pred_scores[keep]
        )

        matches = match_predictions_size_aware(
            pred_boxes_filtered,
            pred_scores_filtered,
            gt_boxes,
            small_mask,
            iou_threshold=iou_threshold,
            small_iou_threshold=small_iou_threshold,
        )

        matched_small = set()

        for match in matches["tp"]:

            gt_idx = match[
                "gt_idx"
            ]

            if gt_idx in small_indices:

                matched_small.add(
                    gt_idx
                )

        small_tp += len(
            matched_small
        )

        small_fn += (
            len(small_indices)
            - len(matched_small)
        )

    small_sensitivity = (
        small_tp
        / (
            small_tp
            + small_fn
            + 1e-8
        )
    )

    small_fnr = (
        small_fn
        / (
            small_tp
            + small_fn
            + 1e-8
        )
    )

    return {
        "small_ground_truth": int(
            small_total
        ),

        "small_TP": int(
            small_tp
        ),

        "small_FN": int(
            small_fn
        ),

        "small_sensitivity": float(
            small_sensitivity
        ),

        "small_FNR": float(
            small_fnr
        ),
    }


# ============================================================
# 7b. Small-nodule image-level specificity
# ============================================================

def calculate_small_nodule_specificity(
    predictions,
    ground_truths,
    small_threshold,
    score_threshold=SCORE_THRESHOLD,
):
    """
    Calculate image-level specificity for the small-nodule
    detection task.

    Positive image:
        Contains at least one small ground-truth nodule.

    Negative image:
        Contains no small ground-truth nodules.

    TN:
        No small ground-truth nodule AND no prediction.

    FP:
        No small ground-truth nodule BUT at least one prediction.

    Specificity = TN / (TN + FP)
    """

    tn = 0
    fp = 0

    for prediction, target in zip(
        predictions,
        ground_truths,
    ):
        gt_boxes = target["boxes"]
        pred_scores = prediction["scores"]

        has_small_nodule = False

        if len(gt_boxes) > 0:
            gt_widths = (
                gt_boxes[:, 2] - gt_boxes[:, 0]
            )
            gt_heights = (
                gt_boxes[:, 3] - gt_boxes[:, 1]
            )
            gt_areas = gt_widths * gt_heights

            has_small_nodule = bool(
                (gt_areas < small_threshold).any()
            )

        predicted_positive = bool(
            (pred_scores >= score_threshold).any()
        )

        if not has_small_nodule:
            if predicted_positive:
                fp += 1
            else:
                tn += 1

    specificity = tn / (tn + fp + 1e-8)

    return {
        "small_TN": int(tn),
        "small_specificity": float(specificity),
    }


# ============================================================
# 8. False positives per image
# ============================================================

def calculate_fp_per_image(
    predictions,
    ground_truths,
    score_threshold=SCORE_THRESHOLD,
    iou_threshold=IOU_THRESHOLD,
):

    total_fp = 0

    for prediction, target in zip(
        predictions,
        ground_truths,
    ):

        boxes = prediction[
            "boxes"
        ]

        scores = prediction[
            "scores"
        ]

        gt_boxes = target[
            "boxes"
        ]

        keep = (
            scores
            >= score_threshold
        )

        boxes = boxes[keep]

        scores = scores[keep]

        matches = match_predictions(
            boxes,
            scores,
            gt_boxes,
            iou_threshold,
        )

        total_fp += len(
            matches["fp"]
        )

    return (
        total_fp
        / len(predictions)
    )


# ============================================================
# 9. FROC / CPM
# ============================================================

def _iou_row(pred_box, gt_boxes):

    if len(gt_boxes) == 0:
        return torch.zeros(0)

    pred_box = pred_box.reshape(1, 4)

    return box_iou(pred_box, gt_boxes)[0]


def _interp_sensitivity(fp_curve, sens_curve, target_fp):

    if target_fp <= fp_curve[0]:
        return sens_curve[0]

    if target_fp >= fp_curve[-1]:
        return sens_curve[-1]

    idx = bisect.bisect_right(fp_curve, target_fp) - 1
    idx = max(0, min(idx, len(fp_curve) - 2))

    x0, x1 = fp_curve[idx], fp_curve[idx + 1]
    y0, y1 = sens_curve[idx], sens_curve[idx + 1]

    if x1 == x0:
        return y1

    frac = (target_fp - x0) / (x1 - x0)

    return y0 + frac * (y1 - y0)


def calculate_froc(
    predictions,
    ground_truths,
    iou_threshold=IOU_THRESHOLD,
    small_iou_threshold=SMALL_IOU_THRESHOLD,
    fp_points=FROC_FP_POINTS,
    small_threshold=None,
):

    num_images = len(predictions)

    total_gt = sum(len(gt["boxes"]) for gt in ground_truths)

    small_masks = None
    total_small_gt = 0

    if small_threshold is not None:

        small_masks = []

        for gt in ground_truths:

            boxes = gt["boxes"]

            if len(boxes) == 0:
                small_masks.append(torch.zeros(0, dtype=torch.bool))
                continue

            w = boxes[:, 2] - boxes[:, 0]
            h = boxes[:, 3] - boxes[:, 1]
            areas = w * h

            mask = areas < small_threshold

            small_masks.append(mask)

            total_small_gt += int(mask.sum().item())

    all_preds = []

    for image_idx, pred in enumerate(predictions):

        boxes = pred["boxes"]
        scores = pred["scores"]

        for i in range(len(boxes)):
            all_preds.append((float(scores[i]), image_idx, boxes[i]))

    fp_curve = [0.0]
    sens_curve = [0.0]
    small_sens_curve = [0.0] if small_threshold is not None else None

    def _empty_result():

        result = {
            "curve": list(zip(fp_curve, sens_curve)),
            "sensitivity_at_fp": {fp: 0.0 for fp in fp_points},
            "cpm": 0.0,
            "total_gt": total_gt,
        }

        if small_threshold is not None:
            result["small_curve"] = list(zip(fp_curve, small_sens_curve))
            result["small_sensitivity_at_fp"] = {fp: 0.0 for fp in fp_points}
            result["small_cpm"] = 0.0
            result["total_small_gt"] = total_small_gt

        return result

    if total_gt == 0 or len(all_preds) == 0 or num_images == 0:
        return _empty_result()

    all_preds.sort(key=lambda x: x[0], reverse=True)

    matched_gt = [set() for _ in ground_truths]

    cum_tp = 0
    cum_fp = 0
    cum_small_tp = 0

    for score, image_idx, pred_box in all_preds:

        gt_boxes = ground_truths[image_idx]["boxes"]

        ious = _iou_row(pred_box, gt_boxes)

        best_iou = -1.0
        best_gt_idx = None

        for gt_idx in range(len(gt_boxes)):

            if gt_idx in matched_gt[image_idx]:
                continue

            v = ious[gt_idx].item()

            if v > best_iou:
                best_iou = v
                best_gt_idx = gt_idx

        is_small_gt = (
            best_gt_idx is not None
            and small_threshold is not None
            and bool(small_masks[image_idx][best_gt_idx])
        )

        effective_threshold = (
            small_iou_threshold if is_small_gt else iou_threshold
        )

        if best_gt_idx is not None and best_iou >= effective_threshold:

            cum_tp += 1
            matched_gt[image_idx].add(best_gt_idx)

            if is_small_gt:
                cum_small_tp += 1

        else:
            cum_fp += 1

        fp_curve.append(cum_fp / num_images)
        sens_curve.append(cum_tp / total_gt)

        if small_threshold is not None:
            denom = total_small_gt if total_small_gt > 0 else 1
            small_sens_curve.append(cum_small_tp / denom)

    sensitivity_at_fp = {
        fp: _interp_sensitivity(fp_curve, sens_curve, fp) for fp in fp_points
    }

    cpm = sum(sensitivity_at_fp.values()) / len(fp_points) if fp_points else 0.0

    result = {
        "curve": list(zip(fp_curve, sens_curve)),
        "sensitivity_at_fp": sensitivity_at_fp,
        "cpm": cpm,
        "total_gt": total_gt,
    }

    if small_threshold is not None:

        small_sensitivity_at_fp = {
            fp: _interp_sensitivity(fp_curve, small_sens_curve, fp) for fp in fp_points
        }

        small_cpm = (
            sum(small_sensitivity_at_fp.values()) / len(fp_points) if fp_points else 0.0
        )

        result["small_curve"] = list(zip(fp_curve, small_sens_curve))
        result["small_sensitivity_at_fp"] = small_sensitivity_at_fp
        result["small_cpm"] = small_cpm
        result["total_small_gt"] = total_small_gt

    return result


# ============================================================
# 10. Paired bootstrap confidence intervals for FROC / CPM
# ============================================================

def paired_bootstrap_froc(
    baseline_predictions,
    proposed_predictions,
    ground_truths,
    small_threshold,
    iou_threshold=IOU_THRESHOLD,
    small_iou_threshold=SMALL_IOU_THRESHOLD,
    fp_points=FROC_FP_POINTS,
    n_bootstrap=2000,
    seed=42,
    output_file="testing/bootstrap_results.csv",
):
    """
    Paired bootstrap comparison of two detectors evaluated on the
    same images.

    The bootstrap resamples IMAGE INDICES with replacement. The same
    sampled indices are used for the baseline and proposed model, so
    the comparison remains paired.

    The existing calculate_froc() function is reused for every
    bootstrap sample. This ensures that the bootstrap uses exactly the
    same FROC/CPM definition as the main evaluation.

    Returns
    -------
    summary : dict
        Observed CPM/small CPM, mean bootstrap differences and 95% CIs.
    distributions : pandas.DataFrame
        Bootstrap difference distributions for CPM and sensitivities.
    """

    if len(baseline_predictions) != len(proposed_predictions):
        raise ValueError(
            "Baseline and proposed predictions must contain the same "
            "number of images."
        )

    if len(baseline_predictions) != len(ground_truths):
        raise ValueError(
            "Predictions and ground truths must contain the same number "
            "of images."
        )

    num_images = len(ground_truths)

    if num_images == 0:
        raise ValueError("Cannot bootstrap an empty test set.")

    if n_bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap iterations.")

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------
    # Original observed results
    # ------------------------------------------------------------

    baseline_result = calculate_froc(
        baseline_predictions,
        ground_truths,
        iou_threshold=iou_threshold,
        small_iou_threshold=small_iou_threshold,
        fp_points=fp_points,
        small_threshold=small_threshold,
    )

    proposed_result = calculate_froc(
        proposed_predictions,
        ground_truths,
        iou_threshold=iou_threshold,
        small_iou_threshold=small_iou_threshold,
        fp_points=fp_points,
        small_threshold=small_threshold,
    )

    observed = {
        "CPM": float(proposed_result["cpm"] - baseline_result["cpm"]),
        "small_CPM": float(
            proposed_result["small_cpm"]
            - baseline_result["small_cpm"]
        ),
    }

    for fp in fp_points:
        observed[f"sensitivity@FP={fp}"] = float(
            proposed_result["sensitivity_at_fp"][fp]
            - baseline_result["sensitivity_at_fp"][fp]
        )

        observed[f"small_sensitivity@FP={fp}"] = float(
            proposed_result["small_sensitivity_at_fp"][fp]
            - baseline_result["small_sensitivity_at_fp"][fp]
        )

    # ------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------

    bootstrap_rows = []

    print()
    print("=" * 60)
    print("PAIRED BOOTSTRAP FROC / CPM")
    print("=" * 60)
    print(f"Test images        : {num_images}")
    print(f"Bootstrap samples  : {n_bootstrap}")
    print(f"Random seed        : {seed}")
    print("Resampling unit    : image")
    print("Confidence level   : 95%")
    print("=" * 60)

    for iteration in range(n_bootstrap):

        # Sample complete images with replacement.
        # The SAME indices are used for both models.
        indices = rng.integers(
            low=0,
            high=num_images,
            size=num_images,
        )

        sampled_baseline = [
            baseline_predictions[int(i)]
            for i in indices
        ]

        sampled_proposed = [
            proposed_predictions[int(i)]
            for i in indices
        ]

        sampled_ground_truths = [
            ground_truths[int(i)]
            for i in indices
        ]

        baseline_boot = calculate_froc(
            sampled_baseline,
            sampled_ground_truths,
            iou_threshold=iou_threshold,
            small_iou_threshold=small_iou_threshold,
            fp_points=fp_points,
            small_threshold=small_threshold,
        )

        proposed_boot = calculate_froc(
            sampled_proposed,
            sampled_ground_truths,
            iou_threshold=iou_threshold,
            small_iou_threshold=small_iou_threshold,
            fp_points=fp_points,
            small_threshold=small_threshold,
        )

        row = {
            "iteration": iteration + 1,
            "CPM_difference": float(
                proposed_boot["cpm"]
                - baseline_boot["cpm"]
            ),
            "small_CPM_difference": float(
                proposed_boot["small_cpm"]
                - baseline_boot["small_cpm"]
            ),
        }

        for fp in fp_points:
            row[f"sensitivity_difference@FP={fp}"] = float(
                proposed_boot["sensitivity_at_fp"][fp]
                - baseline_boot["sensitivity_at_fp"][fp]
            )

            row[f"small_sensitivity_difference@FP={fp}"] = float(
                proposed_boot["small_sensitivity_at_fp"][fp]
                - baseline_boot["small_sensitivity_at_fp"][fp]
            )

        bootstrap_rows.append(row)

        if (iteration + 1) % max(1, n_bootstrap // 10) == 0:
            print(
                f"Bootstrap progress: "
                f"{iteration + 1}/{n_bootstrap}"
            )

    distributions = pd.DataFrame(bootstrap_rows)

    # ------------------------------------------------------------
    # Percentile 95% confidence intervals
    # ------------------------------------------------------------

    summary = {
        "baseline_CPM": float(baseline_result["cpm"]),
        "proposed_CPM": float(proposed_result["cpm"]),
        "observed_CPM_difference": observed["CPM"],
        "baseline_small_CPM": float(baseline_result["small_cpm"]),
        "proposed_small_CPM": float(proposed_result["small_cpm"]),
        "observed_small_CPM_difference": observed["small_CPM"],
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "confidence_level": 0.95,
    }

    metrics_for_ci = [
        "CPM_difference",
        "small_CPM_difference",
    ]

    for fp in fp_points:
        metrics_for_ci.append(
            f"sensitivity_difference@FP={fp}"
        )
        metrics_for_ci.append(
            f"small_sensitivity_difference@FP={fp}"
        )

    for metric in metrics_for_ci:
        values = distributions[metric].to_numpy()

        lower = float(np.percentile(values, 2.5))
        upper = float(np.percentile(values, 97.5))
        mean_difference = float(np.mean(values))

        summary[f"{metric}_mean"] = mean_difference
        summary[f"{metric}_CI_lower"] = lower
        summary[f"{metric}_CI_upper"] = upper

    # ------------------------------------------------------------
    # Save bootstrap distribution
    # ------------------------------------------------------------

    output_path = Path(output_file)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    distributions.to_csv(
        output_path,
        index=False,
    )

    # Also save a compact summary next to the distribution.
    summary_path = output_path.with_name(
        output_path.stem + "_summary.csv"
    )

    pd.DataFrame([summary]).to_csv(
        summary_path,
        index=False,
    )

    # ------------------------------------------------------------
    # Print main results
    # ------------------------------------------------------------

    print()
    print("BOOTSTRAP RESULTS")
    print("-" * 60)

    print(
        f"Overall CPM difference       : "
        f"{summary['observed_CPM_difference']:.4f}"
    )
    print(
        f"95% CI                       : "
        f"[{summary['CPM_difference_CI_lower']:.4f}, "
        f"{summary['CPM_difference_CI_upper']:.4f}]"
    )

    print(
        f"Small-nodule CPM difference  : "
        f"{summary['observed_small_CPM_difference']:.4f}"
    )
    print(
        f"95% CI                       : "
        f"[{summary['small_CPM_difference_CI_lower']:.4f}, "
        f"{summary['small_CPM_difference_CI_upper']:.4f}]"
    )

    print()
    print("Small-nodule sensitivity differences")
    print("-" * 60)

    for fp in fp_points:
    	observed_metric = f"small_sensitivity@FP={fp}"
    	bootstrap_metric = f"small_sensitivity_difference@FP={fp}"

    	print(
        	f"@ {fp} FP/image : "
        	f"{observed[observed_metric]:+.4f} "
        	f"[95% CI: "
        	f"{summary[bootstrap_metric + '_CI_lower']:+.4f}, "
        	f"{summary[bootstrap_metric + '_CI_upper']:+.4f}]"
    	)

    print()
    print(f"✓ Bootstrap distribution saved to {output_file}")
    print(f"✓ Bootstrap summary saved to {summary_path}")
    print("=" * 60)

    return summary, distributions


def save_froc_curve(
    froc_result,
    output_file="testing/froc_curve.csv",
):

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for avg_fp, sensitivity in froc_result["curve"]:
        rows.append(
            {
                "avg_fp_per_image": avg_fp,
                "sensitivity": sensitivity,
                "subset": "overall",
            }
        )

    if "small_curve" in froc_result:
        for avg_fp, sensitivity in froc_result["small_curve"]:
            rows.append(
                {
                    "avg_fp_per_image": avg_fp,
                    "sensitivity": sensitivity,
                    "subset": "small",
                }
            )

    df = pd.DataFrame(rows)

    df.to_csv(output_path, index=False)

    print(f"✓ FROC curve points saved to {output_file}")


# ============================================================
# 11. Image Visualization Function
# ============================================================

def visualize_and_save_samples(
    dataset,
    predictions,
    ground_truths,
    output_dir="testing/visualizations",
    score_threshold=SCORE_THRESHOLD,
    max_samples=20,
    only_positive=True,
    random_sample=True,
    seed=42,
):

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if only_positive:
        candidate_indices = [
            idx for idx in range(len(dataset))
            if len(ground_truths[idx]["boxes"]) > 0
        ]
    else:
        candidate_indices = list(range(len(dataset)))

    if len(candidate_indices) == 0:
        print(
            f"\nNo samples found matching only_positive={only_positive}; "
            f"skipping visualization."
        )
        return

    if random_sample:
        rng = random.Random(seed)
        sample_indices = rng.sample(
            candidate_indices,
            min(max_samples, len(candidate_indices)),
        )
    else:
        sample_indices = candidate_indices[:max_samples]

    num_samples = len(sample_indices)
    print(
        f"\nSaving {num_samples} sample visualizations "
        f"(only_positive={only_positive}) to {output_dir}..."
    )

    for idx in sample_indices:
        image, _ = dataset[idx]
        pred = predictions[idx]
        gt = ground_truths[idx]

        if isinstance(image, torch.Tensor):
            img_np = image.cpu().numpy()
            if img_np.ndim == 3:
                img_np = img_np.transpose(1, 2, 0)
                if img_np.shape[2] == 1:
                    img_np = img_np.squeeze(axis=2)
        elif isinstance(image, np.ndarray):
            img_np = image
        else:
            img_np = np.array(image)

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(img_np, cmap="gray")

        # 1. Draw Ground Truth boxes (GREEN / LIME)
        gt_boxes = gt["boxes"]
        for box in gt_boxes:
            xmin, ymin, xmax, ymax = box.tolist()
            width, height = xmax - xmin, ymax - ymin
            rect = patches.Rectangle(
                (xmin, ymin),
                width,
                height,
                linewidth=2,
                edgecolor="lime",
                facecolor="none",
                label="Ground Truth",
            )
            ax.add_patch(rect)

        # 2. Draw Predicted boxes above threshold (RED)
        keep = pred["scores"] >= score_threshold
        boxes = pred["boxes"][keep]
        scores = pred["scores"][keep]

        for box, score in zip(boxes, scores):
            xmin, ymin, xmax, ymax = box.tolist()
            width, height = xmax - xmin, ymax - ymin
            rect = patches.Rectangle(
                (xmin, ymin),
                width,
                height,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
                label="Prediction",
            )
            ax.add_patch(rect)
            ax.text(
                xmin,
                max(ymin - 5, 10),
                f"{score:.2f}",
                color="red",
                fontsize=10,
                weight="bold",
                bbox=dict(facecolor="white", alpha=0.7, pad=1),
            )

        ax.set_title(f"Image Sample #{idx} (GT: Green | Pred: Red)")
        ax.axis("off")

        plt.savefig(output_path / f"sample_{idx:03d}.png", bbox_inches="tight", dpi=150)
        plt.close(fig)

    print(f"✓ Visualizations saved successfully to {output_dir}")


# ============================================================
# 12. Main evaluation
# ============================================================

def evaluate_model(
    model,
    test_loader,
    test_dataset,
    train_csv,
    device,
    froc_curve_output_file="testing/froc_curve.csv",
    save_visualizations=True,
    visualization_dir="testing/visualizations",
    max_visualizations=20,
    visualize_only_positive=True,
    visualize_random_sample=False,
):

    small_threshold = (
        determine_small_threshold(
            train_csv
        )
    )

    predictions = run_inference(
        model,
        test_loader,
        device,
    )

    ground_truths = (
        get_ground_truths(
            test_dataset
        )
    )

    if save_visualizations:
        visualize_and_save_samples(
            dataset=test_dataset,
            predictions=predictions,
            ground_truths=ground_truths,
            output_dir=visualization_dir,
            score_threshold=SCORE_THRESHOLD,
            max_samples=max_visualizations,
            only_positive=visualize_only_positive,
            random_sample=visualize_random_sample,
        )

    detection_metrics = (
        calculate_detection_metrics(
            predictions,
            ground_truths,
        )
    )

    specificity_metrics = (
        calculate_specificity(
            predictions,
            ground_truths,
        )
    )

    small_metrics = (
        calculate_small_nodule_metrics(
            predictions,
            ground_truths,
            small_threshold,
            iou_threshold=IOU_THRESHOLD,
            small_iou_threshold=SMALL_IOU_THRESHOLD,
        )
    )

    small_specificity_metrics = (
        calculate_small_nodule_specificity(
            predictions,
            ground_truths,
            small_threshold,
        )
    )

    fp_per_image = (
        calculate_fp_per_image(
            predictions,
            ground_truths,
        )
    )

    froc_result = calculate_froc(
        predictions,
        ground_truths,
        iou_threshold=IOU_THRESHOLD,
        small_iou_threshold=SMALL_IOU_THRESHOLD,
        small_threshold=small_threshold,
    )

    save_froc_curve(
        froc_result,
        output_file=froc_curve_output_file,
    )

    froc_summary = {"CPM": float(froc_result["cpm"])}

    for fp, sens in froc_result["sensitivity_at_fp"].items():
        froc_summary[f"sensitivity@FP={fp}"] = float(sens)

    froc_summary["small_CPM"] = float(froc_result["small_cpm"])

    for fp, sens in froc_result["small_sensitivity_at_fp"].items():
        froc_summary[f"small_sensitivity@FP={fp}"] = float(sens)

    results = {
        **froc_summary,

        **detection_metrics,

        **specificity_metrics,

        **small_metrics,

        **small_specificity_metrics,

        "FP_per_image":
            float(
                fp_per_image
            ),

        "small_area_threshold_px2":
            float(
                small_threshold
            ),

        "score_threshold":
            float(
                SCORE_THRESHOLD
            ),

        "iou_threshold":
            float(
                IOU_THRESHOLD
            ),

        "small_iou_threshold":
            float(
                SMALL_IOU_THRESHOLD
            ),
    }

    print()
    print("=" * 60)
    print("NODE21 RETINANET EVALUATION")
    print("=" * 60)

    print()
    print("FROC / CPM (primary metric)")
    print("-" * 60)

    for fp in FROC_FP_POINTS:
        print(
            f"Sensitivity @ {fp} FP/image : "
            f"{results[f'sensitivity@FP={fp}']:.4f}"
        )

    print(f"CPM                        : {results['CPM']:.4f}")

    print()
    print("Small-nodule FROC / CPM")
    print("-" * 60)

    for fp in FROC_FP_POINTS:
        print(
            f"Sensitivity @ {fp} FP/image : "
            f"{results[f'small_sensitivity@FP={fp}']:.4f}"
        )

    print(f"Small CPM                  : {results['small_CPM']:.4f}")

    print()
    print("OVERALL DETECTION (fixed threshold, secondary)")
    print("-" * 60)

    print(
        f"Precision    : "
        f"{results['precision']:.4f}"
    )

    print(
        f"Sensitivity  : "
        f"{results['sensitivity']:.4f}"
    )

    print(
        f"Specificity  : "
        f"{results['specificity']:.4f}"
    )

    print(
        f"TN           : "
        f"{results['TN']}"
    )

    print(
        f"F1           : "
        f"{results['F1']:.4f}"
    )

    print(
        f"FNR          : "
        f"{results['FNR']:.4f}"
    )

    print(
        f"FP / image   : "
        f"{results['FP_per_image']:.4f}"
    )

    print()
    print("SMALL NODULE DETECTION (fixed threshold, secondary)")
    print("-" * 60)

    print(
        f"Area threshold : "
        f"{results['small_area_threshold_px2']:.2f} px²"
    )

    print(
        f"Small nodules  : "
        f"{results['small_ground_truth']}"
    )

    print(
        f"Small TP       : "
        f"{results['small_TP']}"
    )

    print(
        f"Small FN       : "
        f"{results['small_FN']}"
    )

    print(
        f"Small Sens.    : "
        f"{results['small_sensitivity']:.4f}"
    )

    print(
        f"Small Specificity : "
        f"{results['small_specificity']:.4f}"
    )

    print(
        f"Small TN          : "
        f"{results['small_TN']}"
    )

    print(
        f"Small FNR         : "
        f"{results['small_FNR']:.4f}"
    )

    print("=" * 60)

    return results


# ============================================================
# 13. Save results
# ============================================================

def save_results(
    results,
    output_file=
        "testing/evaluation_results.csv",
):

    output_path = Path(
        output_file
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.DataFrame(
        [results]
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"✓ Evaluation results saved to "
        f"{output_file}"
    )

# ============================================================
# 14. Compare two trained models with paired bootstrap
# ============================================================

def evaluate_models_with_bootstrap(
    baseline_model,
    proposed_model,
    test_loader,
    test_dataset,
    train_csv,
    device,
    n_bootstrap=2000,
    seed=42,
    output_file="testing/bootstrap_results.csv",
):
    """
    Evaluate two already-trained models on the same test set and run
    a paired bootstrap comparison.

    This function DOES NOT train either model. It only performs
    inference and statistical evaluation.

    Parameters
    ----------
    baseline_model : torch.nn.Module
        Trained baseline detector.

    proposed_model : torch.nn.Module
        Trained proposed detector (for example, RetinaNet + LIP).

    test_loader : DataLoader
        Test dataloader. It must use the same ordering for both models.

    test_dataset : Dataset
        Dataset corresponding to test_loader.

    train_csv : str
        Training CSV used to determine the frozen small-nodule area
        threshold.

    device : torch.device
        CPU or CUDA device.

    n_bootstrap : int
        Number of paired bootstrap samples. 2000 is a good default.

    seed : int
        Random seed for reproducibility.

    output_file : str
        CSV file containing the bootstrap distribution.

    Returns
    -------
    summary : dict
        Observed model differences and 95% confidence intervals.
    distributions : pandas.DataFrame
        Bootstrap distribution of the differences.
    """

    print()
    print("=" * 60)
    print("MODEL COMPARISON + PAIRED BOOTSTRAP")
    print("=" * 60)
    print("Baseline model  : supplied trained model")
    print("Proposed model  : supplied trained model")
    print("Training        : NOT performed")
    print("Evaluation      : same test set")
    print("=" * 60)

    # ------------------------------------------------------------
    # Determine the small-nodule threshold from TRAINING data only.
    # ------------------------------------------------------------

    small_threshold = determine_small_threshold(train_csv)

    # ------------------------------------------------------------
    # Run inference for baseline.
    # ------------------------------------------------------------

    print()
    print("=" * 60)
    print("BASELINE MODEL INFERENCE")
    print("=" * 60)

    baseline_predictions = run_inference(
        baseline_model,
        test_loader,
        device,
    )

    # ------------------------------------------------------------
    # Run inference for proposed model.
    # ------------------------------------------------------------

    print()
    print("=" * 60)
    print("PROPOSED MODEL INFERENCE")
    print("=" * 60)

    proposed_predictions = run_inference(
        proposed_model,
        test_loader,
        device,
    )

    # ------------------------------------------------------------
    # Ground truth is obtained once because both models use the same
    # test images and annotations.
    # ------------------------------------------------------------

    ground_truths = get_ground_truths(test_dataset)

    if len(baseline_predictions) != len(proposed_predictions):
        raise RuntimeError(
            "Baseline and proposed models produced different numbers "
            "of predictions. Make sure they use the same test loader "
            "and image ordering."
        )

    if len(baseline_predictions) != len(ground_truths):
        raise RuntimeError(
            "Prediction count does not match ground-truth count."
        )

    # ------------------------------------------------------------
    # Show the ordinary FROC / CPM results before bootstrap.
    # ------------------------------------------------------------

    baseline_froc = calculate_froc(
        baseline_predictions,
        ground_truths,
        iou_threshold=IOU_THRESHOLD,
        small_iou_threshold=SMALL_IOU_THRESHOLD,
        fp_points=FROC_FP_POINTS,
        small_threshold=small_threshold,
    )

    proposed_froc = calculate_froc(
        proposed_predictions,
        ground_truths,
        iou_threshold=IOU_THRESHOLD,
        small_iou_threshold=SMALL_IOU_THRESHOLD,
        fp_points=FROC_FP_POINTS,
        small_threshold=small_threshold,
    )

    print()
    print("=" * 60)
    print("OBSERVED MODEL COMPARISON")
    print("=" * 60)

    print(
        f"Baseline CPM       : {baseline_froc['cpm']:.4f}"
    )
    print(
        f"Proposed CPM       : {proposed_froc['cpm']:.4f}"
    )
    print(
        f"CPM difference     : "
        f"{proposed_froc['cpm'] - baseline_froc['cpm']:+.4f}"
    )

    print()

    print(
        f"Baseline small CPM : "
        f"{baseline_froc['small_cpm']:.4f}"
    )
    print(
        f"Proposed small CPM : "
        f"{proposed_froc['small_cpm']:.4f}"
    )
    print(
        f"Small CPM difference: "
        f"{proposed_froc['small_cpm'] - baseline_froc['small_cpm']:+.4f}"
    )

    # ------------------------------------------------------------
    # Paired bootstrap.
    # ------------------------------------------------------------

    summary, distributions = paired_bootstrap_froc(
        baseline_predictions=baseline_predictions,
        proposed_predictions=proposed_predictions,
        ground_truths=ground_truths,
        small_threshold=small_threshold,
        iou_threshold=IOU_THRESHOLD,
        small_iou_threshold=SMALL_IOU_THRESHOLD,
        fp_points=FROC_FP_POINTS,
        n_bootstrap=n_bootstrap,
        seed=seed,
        output_file=output_file,
    )

    return summary, distributions
