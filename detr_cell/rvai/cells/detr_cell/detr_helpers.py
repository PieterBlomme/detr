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
    # keep only predictions with 0.7+ confidence
    #TODO dynamic threshold 
    probas = detections['pred_logits'].softmax(-1)[:, :, :-1].numpy()
    # convert boxes from [0; 1] to image scales
    # TODO im.size should be dynamic .
    bboxes_scaled = rescale_bboxes(detections['pred_boxes'], (704,576)).numpy()

    #detections should be list of dictionaries
    all_detections = []
    for p, b in zip(probas, bboxes_scaled):
        keep = np.max(p, -1) > keep_prob
        p = p[keep]
        b = b[keep]
        d = {}
        for label in range(len(classes)):
            d[label] = np.array([])
        for p2, b2 in zip(p,b):
            label = np.argmax(p2)
            values = np.append(b2, p2[label])
            d[label] = np.append(d[label], [values])
        for label in range(len(classes)):
            d[label] = np.reshape(d[label], (-1, 5))
        all_detections.append(d)
    return all_detections

def convert_annotations(annotations, classes):
    all_annotations = []
    for ann in annotations:
        a = {}
        for label in range(len(classes)):
            a[label] = np.array([])
        for b, l in zip(ann['boxes'].numpy(), ann['labels'].numpy()):
            a[l] = np.append(a[l], [b])
        for label in range(len(classes)):
            a[label] = np.reshape(a[label], (-1, 4))
        all_annotations.append(a)
    return all_annotations

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bboxes, size):
    final_b = []
    for out_bbox in out_bboxes:
        img_w, img_h = size
        b = box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        final_b.append(b)
    return torch.stack(final_b, dim=0)

def calculate_map(annotations, outputs, classes):
    all_detections = convert_detections(outputs, classes)
    all_annotations = convert_annotations(annotations, classes)
    average_precisions = {}
    # loop over all classes
    for label in range(len(classes)):

        #TODO label should be class_to_index

        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0

        #loop over all annotations
        for i in range(len(annotations)):
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

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
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