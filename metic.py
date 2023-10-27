import numpy as np
from skimage import measure
import os
from PIL import Image
import json
from pycocotools import mask as maskUtils
from scipy.optimize import linear_sum_assignment
from collections import OrderedDict
import logging
from datetime import datetime
def remap_label(pred, by_size=False):
    """Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3] 
    not [0, 2, 4, 6]. The ordering of instances (which one comes first) 
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID.

    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger nuclei has smaller id (on-top)

    """
    pred_id = list(np.unique(pred))
    pred_id.remove(0)
    if len(pred_id) == 0:
        return pred  # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred


def get_fast_aji(true, pred):
    """
    AJI version distributed by MoNuSeg, has no permutation problem but suffered from
    over-penalisation similar to DICE2
    Fast computation requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4]
    not [2, 3, 6, 10]. Please call `remap_label` before hand and `by_size` flag has no
    effect on the result.
    """
    true = np.copy(true)  # ? do we need this
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [None, ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [None, ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_inter = np.zeros([len(true_id_list) - 1,
                               len(pred_id_list) - 1], dtype=np.float64)
    pairwise_union = np.zeros([len(true_id_list) - 1,
                               len(pred_id_list) - 1], dtype=np.float64)
    # 多检
    pairwise_FP = np.zeros([len(true_id_list) - 1,
                            len(pred_id_list) - 1], dtype=np.float64)
    # 漏检
    pairwise_FN = np.zeros([len(true_id_list) - 1,
                            len(pred_id_list) - 1], dtype=np.float64)

    # caching pairwise
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            pairwise_inter[true_id - 1, pred_id - 1] = inter
            pairwise_union[true_id - 1, pred_id - 1] = total - inter

            pairwise_FP[true_id - 1, pred_id - 1] = p_mask.sum() - inter
            pairwise_FN[true_id - 1, pred_id - 1] = t_mask.sum() - inter
    #
    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    # pair of pred that give highest iou for each true, dont care
    # about reusing pred instance multiple times
    paired_pred = np.argmax(pairwise_iou, axis=1)
    pairwise_iou = np.max(pairwise_iou, axis=1)
    # exlude those dont have intersection
    paired_true = np.nonzero(pairwise_iou > 0.0)[0]
    paired_pred = paired_pred[paired_true]
    # print(paired_true.shape, paired_pred.shape)

    overall_inter = (pairwise_inter[paired_true, paired_pred]).sum()
    overall_union = (pairwise_union[paired_true, paired_pred]).sum()

    overall_FP = (pairwise_FP[paired_true, paired_pred]).sum()
    overall_FN = (pairwise_FN[paired_true, paired_pred]).sum()

    #
    paired_true = (list(paired_true + 1))  # index to instance ID
    paired_pred = (list(paired_pred + 1))
    # add all unpaired GT and Prediction into the union
    unpaired_true = np.array(
        [idx for idx in true_id_list[1:] if idx not in paired_true])
    unpaired_pred = np.array(
        [idx for idx in pred_id_list[1:] if idx not in paired_pred])

    less_pred = 0
    more_pred = 0

    for true_id in unpaired_true:
        less_pred += true_masks[true_id].sum()
        overall_union += true_masks[true_id].sum()
    for pred_id in unpaired_pred:
        more_pred += pred_masks[pred_id].sum()
        overall_union += pred_masks[pred_id].sum()
    #
    aji_score = overall_inter / overall_union
    fm = overall_union - overall_inter
    # print('\t [ana_FP = {:.4f}, ana_FN = {:.4f}, ana_less = {:.4f}, ana_more = {:.4f}]'.format((overall_FP / fm),(overall_FN / fm),(less_pred / fm),(more_pred / fm)))

    # return aji_score, overall_FP / fm, overall_FN / fm, less_pred / fm, more_pred / fm
    return aji_score


def get_dice_1(true, pred):
    """
        Traditional dice
    """
    # cast to binary 1st
    true = np.copy(true)
    pred = np.copy(pred)
    true[true > 0] = 1
    pred[pred > 0] = 1
    inter = true * pred
    denom = true + pred
    return 2.0 * np.sum(inter) / np.sum(denom)


def get_fast_pq(true, pred, match_iou=0.5):
    """`match_iou` is the IoU threshold level to determine the pairing between
    GT instances `p` and prediction instances `g`. `p` and `g` is a pair
    if IoU > `match_iou`. However, pair of `p` and `g` must be unique 
    (1 prediction instance to 1 GT instance mapping).

    If `match_iou` < 0.5, Munkres assignment (solving minimum weight matching
    in bipartite graphs) is caculated to find the maximal amount of unique pairing. 

    If `match_iou` >= 0.5, all IoU(p,g) > 0.5 pairing is proven to be unique and
    the number of pairs is also maximal.    
    
    Fast computation requires instance IDs are in contiguous orderding 
    i.e [1, 2, 3, 4] not [2, 3, 6, 10]. Please call `remap_label` beforehand 
    and `by_size` flag has no effect on the result.

    Returns:
        [dq, sq, pq]: measurement statistic

        [paired_true, paired_pred, unpaired_true, unpaired_pred]: 
                      pairing information to perform measurement
                    
    """
    assert match_iou >= 0.0, "Cant' be negative"

    true = np.copy(true)
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [
        None,
    ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [
        None,
    ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_iou = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )

    # caching pairwise iou
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            iou = inter / (total - inter)
            pairwise_iou[true_id - 1, pred_id - 1] = iou
    #
    if match_iou >= 0.5:
        paired_iou = pairwise_iou[pairwise_iou > match_iou]
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1  # index is instance id - 1
        paired_pred += 1  # hence return back to original
    else:  # * Exhaustive maximal unique pairing
        #### Munkres pairing with scipy library
        # the algorithm return (row indices, matched column indices)
        # if there is multiple same cost in a row, index of first occurence
        # is return, thus the unique pairing is ensure
        # inverse pair to get high IoU as minimum
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        ### extract the paired cost and remove invalid pair
        paired_iou = pairwise_iou[paired_true, paired_pred]

        # now select those above threshold level
        # paired with iou = 0.0 i.e no intersection => FP or FN
        paired_true = list(paired_true[paired_iou > match_iou] + 1)
        paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
        paired_iou = paired_iou[paired_iou > match_iou]

    # get the actual FP and FN
    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    # print(paired_iou.shape, paired_true.shape, len(unpaired_true), len(unpaired_pred))

    #
    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    # get the F1-score i.e DQ
    dq = tp / (tp + 0.5 * fp + 0.5 * fn)
    # get the SQ, no paired has 0 iou so not impact
    sq = paired_iou.sum() / (tp + 1.0e-6)

    return [dq, sq, dq * sq], [paired_true, paired_pred, unpaired_true, unpaired_pred]
def get_metric(predictions,logger):

	npy_data=OrderedDict()
	for prediction in predictions:
			image_id=prediction['image_id']
			instances=prediction['instances']
			h=instances[0]['segmentation']['size'][0]
			w=instances[0]['segmentation']['size'][1]
			instance_mask = np.zeros((h, w), dtype=np.uint16)
			instance_id = 1  # 从1开始计数，0表示背景
			for instance in instances:
					rle = {
					'counts': instance['segmentation']['counts'],
					'size': instance['segmentation']['size']}
					binary_mask = maskUtils.decode(rle)
					instance_mask[binary_mask > 0] = instance_id
					instance_id += 1
			npy_data[image_id]=instance_mask
	result_Dice = 0
	result_AJI = 0
	result_PQ=0
	datasets_path = os.environ.get('DETECTRON2_DATASETS')
	true_path = os.path.join(datasets_path,"coco/val_lab")
	for true_name in sorted(os.listdir(true_path)):
			true = os.path.join(true_path, true_name)
			true = np.array(Image.open(true))
			image_id=int(true_name.split(".")[0])
			pred=npy_data[image_id]
			remap_label(true)
			true = measure.label(true)
			pred = measure.label(pred)
			result_Dice += get_dice_1(true, pred)
			result_AJI += get_fast_aji(true, pred)
			result_PQ+=get_fast_pq(true, pred, match_iou=0.5)[0][2]
	result_AJI = result_AJI/len(os.listdir(true_path))
	result_Dice = result_Dice/len(os.listdir(true_path))
	result_PQ = result_PQ/len(os.listdir(true_path))


	# 设置日志级别为 INFO
	# logging.basicConfig(level=logging.INFO, format="%(message)s")

	# logger = logging.getLogger()

	header = "\n\nEvaluation Medical_metirc for segm: \n"
	table_header = "|   PQ   |  AJI   |  Dice  |  \n|:------:|:------:|:------:|"
	table_content = "| {:.3f} | {:.3f} | {:.3f} |".format(result_PQ*100, result_AJI*100, result_Dice*100)
	footer = "\n\n"

	logger.info(header + table_header + "\n" + table_content + footer)

