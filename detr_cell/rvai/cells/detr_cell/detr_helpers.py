from pathlib import Path
import os
import glob
import numpy as np
import torch
import logging
from torch.autograd import Variable

def get_model_dir():
    cache_dir = Path.home() / ".rvai"
    cache_dir.mkdir(exist_ok=True)

    model_dir = Path(cache_dir) / "models"
    model_dir.mkdir(exist_ok=True)
    model_dir = Path(model_dir) / "detr"
    model_dir.mkdir(exist_ok=True)
    return model_dir

def clear_folder(folder):
    files = glob.glob(f'{folder}/*')
    for f in files:
        os.remove(f)

def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def convert_detections(detections, classes, keep_prob=0.7):
    #detections should be list of dictionaries
    all_detections = []
    for d in detections:
        # keep only predictions with 0.7+ confidence
        #TODO dynamic threshold is it even needed?
        #print(d['scores'].softmax(-1).numpy())
        #print(d['scores'].numpy())
        p = d['scores'].numpy()
        #print(f'max prob is {np.max(p)}')
        b = d['boxes'].numpy()
        l = d['labels'].numpy()
        #keep = p > keep_prob
        #p = p[keep]
        #b = b[keep]
        #l = l[keep]
        #print(f'Number of outputs after threshold filter: {l.shape}')
        d = {}
        for label in range(len(classes)):
            d[label] = np.array([])
        for p2, l2, b2 in zip(p,l,b):
            label = l2
            values = np.append(b2, p2)
            d[label] = np.append(d[label], [values])
        for label in range(len(classes)):
            d[label] = np.reshape(d[label], (-1, 5))
        all_detections.append(d)
    return all_detections

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return b

def convert_annotations(annotations, classes):
    all_annotations = []
    for ann in annotations:
        a = {}
        for label in range(len(classes)):
            a[label] = np.array([])
        size = ann['orig_size'].numpy()
        for b, l in zip(ann['boxes'].numpy(), ann['labels'].numpy()):
            box = box_cxcywh_to_xyxy(b)
            img_h, img_w = size
            scale_fct = [img_w, img_h, img_w, img_h]
            box = np.multiply(box, scale_fct)   

            a[l] = np.append(a[l], [box])
        for label in range(len(classes)):
            a[label] = np.reshape(a[label], (-1, 4))
        all_annotations.append(a)
    return all_annotations

def compute_overlap(
    boxes,
    query_boxes
):
    """
    Args
        a: (N, 4) ndarray of float
        b: (K, 4) ndarray of float
    Returns
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float64)
    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0]) *
            (query_boxes[k, 3] - query_boxes[k, 1])
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) 
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) 
                )
                if ih > 0:
                    ua = np.float64(
                        (boxes[n, 2] - boxes[n, 0]) *
                        (boxes[n, 3] - boxes[n, 1]) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps

def calculate_map(targets, outputs, classes):
    all_detections = convert_detections(outputs, classes)
    all_annotations = convert_annotations(targets, classes)
    average_precisions = {}

    # loop over all classes
    for label in range(len(classes)):

        #TODO label should be class_to_index

        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0

        #loop over all annotations
        for i in range(len(targets)):
            detections           = all_detections[i][label]
            annotations          = all_annotations[i][label]
            num_annotations      += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    continue
                
                overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]
                if max_overlap >= 0.5 and assigned_annotation not in detected_annotations: #TODO iou_threshold
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices         = np.argsort(-scores)
        #false_positives = [x for y, x in sorted(zip(indices, false_positives))]
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)

        # compute recall and precision
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision  = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

    return average_precisions