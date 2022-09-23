# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from fairseq.dataclass.configs import FairseqDataclass

import torch
import torch.nn as nn
from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion

import torch.nn.functional as F
from graphormer.utils.bbox_ops import *
from graphormer.utils.det_util import *
from scipy.optimize import linear_sum_assignment

from graphormer.tasks.svg_detection import SVGDetectionConfig
import torchvision
import numpy as np

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, labels=()):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = x[:, :4]

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        #tgt_ids = torch.cat([v["y_ids"] for v in targets])
        tgt_ids = torch.cat([v for v in targets["y_ids"]])
        tgt_bbox = torch.cat([v for v in targets["y_bbox"]])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, box_xyxy_to_cxcywh(tgt_bbox), p=1)

        # Compute the giou cost betwen boxes
        #cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), tgt_bbox)

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v) for v in targets["y_bbox"]]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]




@register_criterion("detr_loss", dataclass=SVGDetectionConfig)
class DetrLoss(FairseqCriterion):
    """
    Implementation for the L1 loss (MAE loss) used in graphormer model training.
    """
    def __init__(self, task, cfg):
        super().__init__(task)
        self.matcher =  HungarianMatcher()
        self.num_classes = cfg.num_classes
        self.weight_dict = {"loss_ce": 1, "loss_bbox": 5, "loss_giou":2}

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        sample_size = sample["nsamples"]
        #print(sample['target']['y_ids'][0].size())
        width = sample["net_input"]["batched_data"]['width']
        height = sample["net_input"]["batched_data"]['height']
        
        num_boxes = sum(t.size(0) for t in sample['target']['y_ids'])
        #print(num_boxes)
        #raise SystemExit

        with torch.no_grad():
            natoms = sample["net_input"]["batched_data"]["x"].shape[1]

        out = model(**sample["net_input"])
        indices = self.matcher(out, sample['target'])

        loss = self.loss_boxes(out, sample['target'], indices) #, num_boxes)
        loss.update(self.loss_labels(out, sample['target'], indices)) # , num_boxes))
        
        
        ret = loss['loss_bbox'] #0
        for k in loss:
            ret += self.weight_dict[k] * loss[k]

        
        if self.training:
            ap_dict = self.accuracy(out, sample['target'], height, width)
            ap = {}
            ap['ap50'] = ap_dict['0.5000']
            ap['ap75'] = ap_dict['0.7500']
            ap['ap95'] = ap_dict['0.9500']
            ap['appall'] = ap_dict['all']
        else:
            ap = {'ap50':-1, 'ap75':-1, 'ap95':-1, 'apall':-1}
        logging_output = {
            "loss": ret, #loss['loss_bbox'], #ret,
            "sample_size": num_boxes,
            "batch_size": out['pred_boxes'].size(0), 
            "ap": ap
            #"nsentences": num_boxes,
            #"ntokens": natoms,
        }

        
        return ret, num_boxes, logging_output

    @torch.no_grad()
    def accuracy(self, output, target, height, width):
        """Computes the precision@k for the specified values of k"""
        sample_metrics = [[] for i in  range(10)]

        out_bbox = output["pred_boxes"].clone()
        out_prob = output["pred_logits"].clone()
        out_prob = F.softmax(out_prob)
        scores, labels = out_prob[..., :-1].max(-1)
        
        boxes = box_cxcywh_to_xyxy(output["pred_boxes"])

        target_bbox = target['y_bbox']
        target_label = target['y_ids']

        #print(target_bbox)
        #print(out_bbox.size(), scores.size(), len(target_bbox), len(target_label))

        for i in range(0, len(target_bbox)):
            o_bbox = out_bbox[i]
            o_prob = out_prob[i]
            o_score = scores[i].unsqueeze(1)
            o_label = labels[i].unsqueeze(1)
            t_bbox = target_bbox[i]
            t_label = target_label[i].unsqueeze(1)
            #print(t_bbox.size(), t_label.size())

            w = width[i]
            h = height[i]

            o_bbox[:, 0] *= w
            o_bbox[:, 2] *= w
            o_bbox[:, 1] *= h
            o_bbox[:, 3] *= h

            t_bbox[:, 0] *= w
            t_bbox[:, 2] *= w
            t_bbox[:, 1] *= h
            t_bbox[:, 3] *= h

            
            #pred = torch.cat([o_bbox, o_score, o_score], dim = 1)
            pred = torch.cat([o_bbox, o_score, o_prob], dim = 1)
            target = torch.cat((torch.zeros((t_bbox.size(0), 1)).cuda(), t_label, t_bbox), dim = 1)
            outputs = non_max_suppression(pred.unsqueeze(0), conf_thres=0.0, iou_thres=0.5)
            outputs = [x.cpu() for x in outputs]
            #print('outputs', outputs)
            iou_ths = np.linspace(0.5, 0.95, 10)

            for i_th, th in enumerate(iou_ths):
                #print(get_batch_statistics(outputs, target, iou_threshold=th))
                sample_metrics[i_th] += get_batch_statistics(outputs, target.cpu(), iou_threshold=th)
                
                
        AP_total = 0
        APs = {}
        for i in range(0, len(iou_ths)):
            if len(sample_metrics[i]) == 0:  # no detections over whole validation set.
                return None
            
            true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics[i]))]
            precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels.cpu())
            #print(AP)
            #test_value = test_metric(out.max(dim=1)[1], gt, opt.n_classes)
            #opt.test_values.update(test_value, opt.batch_size)
            #print(test_loss)
            #output_str += 'Epoch: [{0}]\t Iter: [{1}]\t''MAP@{2:.2f}: {3:.4f}\t'.format(
            #    opt.epoch, opt.iter, iou_ths[i], np.mean(AP))
            #output_str += 'Top1 Acc@{0:.2f}:{1:.4f}\t'.format(iou_ths[i], n_true * 1.0 / n_total)
            #output_str += '\n'
            APs['%.4f'%iou_ths[i]] = np.sum(AP)
            AP_total += np.sum(AP)
        
        APs['all'] = AP_total / len(iou_ths)
        
        return APs

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=6)
    
        for key in logging_outputs[0]['ap']:
            ap_sum = sum(log['ap'][key] for log in logging_outputs)
            sample_size = sum(log.get("batch_size", 0) for log in logging_outputs)
            metrics.log_scalar(key, ap_sum / sample_size, sample_size)
        

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

    def _get_src_permutation_idx(self, indices):
            # permute predictions following indices
            batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
            src_idx = torch.cat([src for (src, _) in indices])
            return batch_idx, src_idx

    def loss_boxes(self, outputs, targets, indices): #, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
            targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
            The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t[i] for t, (_, i) in zip(targets['y_bbox'], indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, box_xyxy_to_cxcywh(target_boxes), reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() #/ num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            target_boxes))
        losses['loss_giou'] = loss_giou.sum() #/ num_boxes
        
        return losses

    def loss_labels(self, outputs, targets, indices, log=True):#, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(targets["y_ids"], indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, reduction = 'sum')
        losses = {'loss_ce': loss_ce}

        #if log:
            # TODO this should probably be a separate loss, not hacked in this one here
        #    losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

