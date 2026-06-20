from typing import Any

import torch
from torchmetrics import Metric
from torchvision.ops import box_iou
from transformers import AutoModel, AutoTokenizer


class HOISoftMapMetric(Metric):
    def __init__(
        self,
        *args,
        num_interactions: int,
        objects_verbs_to_interaction_id: dict[tuple[int, int], int],
        num_annotations_per_interaction: list[int],
        verbs_name: list[tuple[str, str]],
        iou_threshold: float | None = 0.5,
        ap_method: str = "11P",
        verb_similarity_threshold: float = 0.5,
        **kwargs,
    ):
        """Compute the mean Average Precision (mAP) for Human-Object Interactions (HOI)."""
        super().__init__(*args, **kwargs)

        assert ap_method in ["AUC", "11P", "INT"], "Invalid ap_method"
        assert (
            len(num_annotations_per_interaction) == num_interactions
        ), "Invalid number of interactions"

        self.num_interactions = num_interactions
        self.num_annotations_per_interaction = num_annotations_per_interaction
        self.verbs_name = verbs_name
        self.iou_threshold = iou_threshold
        self.ap_method = ap_method
        self.verb_similarity_threshold = verb_similarity_threshold

        self.add_state(
            "objects_verbs_to_interaction_id",
            default=objects_verbs_to_interaction_id,
            dist_reduce_fx=None,
        )
        self.add_state("all_pred_scores", default=[], dist_reduce_fx="cat")
        self.add_state("all_pred_interactions_id", default=[], dist_reduce_fx="cat")
        self.add_state("all_pred_tgt_matchings", default=[], dist_reduce_fx="cat")

        # Per-interaction score / matching buckets, filled inside `compute()` after
        # the torchmetrics dist-cat reduction over `all_pred_*`.
        self.output = [[] for _ in range(num_interactions)]
        self.labels = [[] for _ in range(num_interactions)]

        # Load Sentence-BERT model for computing the similarity between verb labels
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.sentence_bert_processor = AutoTokenizer.from_pretrained(model_name)
        self.sentence_bert_model = AutoModel.from_pretrained(model_name)
        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.float16 if device != "cpu" else torch.float32
            self.sentence_bert_model.to(device=device, dtype=dtype)
        else:
            device = "cpu"

        # NOTE: not keeping track of the no_interaction
        self.tgt_verb_embeds = self.get_embeddings(verbs_name)

    def get_embeddings(self, text: list[str]) -> torch.Tensor:
        def _mean_pooling(text_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            # The first element of model_output contains all token embeddings
            token_embeddings = text_embeds[0]
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )

        text_inputs = self.sentence_bert_processor(
            text, padding=True, truncation=True, return_tensors="pt"
        )
        text_inputs = text_inputs.to(device=self.sentence_bert_model.device)
        with torch.no_grad():
            text_embeds = self.sentence_bert_model(**text_inputs)
            text_embeds = _mean_pooling(text_embeds, text_inputs["attention_mask"])

        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        return text_embeds

    def update(self, preds: dict[Any], tgts: dict[Any]):
        # Keep track of the scores, predictions and labels for each item in the batch
        all_pred_scores = []
        all_pred_interactions_id = []
        all_pred_tgt_matchings = []

        # Get target keys
        tgt_keys = tgts.keys()
        for tgt_key in tgt_keys:
            tgt = tgts.get(tgt_key)
            pred = preds.get(tgt_key)

            # If preds is empty, skip the iteration
            if not pred:
                continue

            pred_human_bboxes = pred["humans_bbox"]
            pred_object_bboxes = pred["objects_bbox"]
            pred_object_ids = pred[
                "objects_id"
            ]  # NOTE: assuming that the triplets make sense for the ground truth object.
            pred_verb_labels = pred["verbs_label"]
            pred_prior_scores = pred["prior_scores"]

            if len(pred_verb_labels) == 0:
                continue

            pred_verb_embeds = self.get_embeddings(pred_verb_labels)

            tgt_pred_sim = torch.matmul(
                self.tgt_verb_embeds, pred_verb_embeds.T
            )  # (num_gt_verbs, num_pred_verbs)

            # Mask predictions with similarity lower than the threshold
            tgt_pred_sim[tgt_pred_sim < self.verb_similarity_threshold] = 0

            new_pred_human_bboxes = []
            new_pred_object_bboxes = []
            new_pred_object_ids = []
            new_pred_prior_scores = []
            new_pred_verb_ids = []
            new_pred_scores = []
            for tgt_idx, pred_idx in torch.nonzero(tgt_pred_sim).tolist():
                new_pred_human_bboxes.append(pred_human_bboxes[pred_idx])
                new_pred_object_bboxes.append(pred_object_bboxes[pred_idx])
                new_pred_object_ids.append(pred_object_ids[pred_idx])
                new_pred_prior_scores.append(pred_prior_scores[pred_idx])
                new_pred_verb_ids.append(tgt_idx)
                new_pred_scores.append(pred_prior_scores[pred_idx])

            if len(new_pred_scores) == 0:
                continue

            # Get the indices of the maximum similarity
            pred_human_bboxes = torch.stack(new_pred_human_bboxes).to(self.device)
            pred_object_bboxes = torch.stack(new_pred_object_bboxes).to(self.device)
            pred_object_ids = torch.stack(new_pred_object_ids).to(self.device)
            pred_prior_scores = torch.stack(new_pred_prior_scores).to(self.device)
            pred_verb_ids = torch.tensor(new_pred_verb_ids).to(self.device)
            pred_scores = torch.stack(new_pred_scores).to(self.device)

            pred_interactions_id = self.objects_verbs_to_interaction_id[
                (pred_object_ids, pred_verb_ids)
            ]

            # Remove invalid interactions, i.e. those with verb_id = -1
            pred_human_bboxes = pred_human_bboxes[pred_interactions_id != -1]
            pred_object_bboxes = pred_object_bboxes[pred_interactions_id != -1]
            pred_scores = pred_scores[pred_interactions_id != -1]
            pred_object_ids = pred_object_ids[pred_interactions_id != -1]
            pred_verb_ids = pred_verb_ids[pred_interactions_id != -1]
            pred_interactions_id = pred_interactions_id[pred_interactions_id != -1]

            # Get interactions id
            if not pred_verb_ids.any():
                continue

            tgt_human_bboxes = tgt["humans_bbox"]
            tgt_object_bboxes = tgt["objects_bbox"]
            tgt_verbs_ids = tgt["verbs_id"]
            tgt_object_ids = tgt["objects_id"]
            tgt_interactions_id = self.objects_verbs_to_interaction_id[
                (tgt_object_ids, tgt_verbs_ids)
            ]

            # This should never trigger. If it does, it means that there's a bug somewhere
            assert (pred_interactions_id != -1).all() and (
                tgt_interactions_id != -1
            ).all(), "Invalid interaction ids"

            # 1 means that the prediction is a true positive
            # 0 means that it is a false positive
            pred_tgt_matchings = torch.zeros(len(pred_scores), device=self.device)

            for pred_interaction_id in pred_interactions_id.unique():
                tgt_idxs = torch.nonzero(tgt_interactions_id == pred_interaction_id).squeeze(1)
                pred_idxs = torch.nonzero(pred_interactions_id == pred_interaction_id).squeeze(1)

                if len(tgt_idxs) == 0:
                    continue

                pred_tgt_matchings[pred_idxs] = self._match(
                    tgt_human_bboxes[tgt_idxs].view(-1, 4),
                    tgt_object_bboxes[tgt_idxs].view(-1, 4),
                    pred_human_bboxes[pred_idxs].view(-1, 4),
                    pred_object_bboxes[pred_idxs].view(-1, 4),
                )

            all_pred_scores.append(pred_scores)
            all_pred_interactions_id.append(pred_interactions_id)
            all_pred_tgt_matchings.append(pred_tgt_matchings)

        all_pred_scores = (
            torch.cat(all_pred_scores) if all_pred_scores else torch.tensor([], device=self.device)
        )
        all_pred_interactions_id = (
            torch.cat(all_pred_interactions_id).long()
            if all_pred_interactions_id
            else torch.tensor([], device=self.device, dtype=torch.long)
        )
        all_pred_tgt_matchings = (
            torch.cat(all_pred_tgt_matchings)
            if all_pred_tgt_matchings
            else torch.tensor([], device=self.device, dtype=torch.float)
        )

        self.all_pred_scores.append(all_pred_scores)
        self.all_pred_interactions_id.append(all_pred_interactions_id)
        self.all_pred_tgt_matchings.append(all_pred_tgt_matchings)

    def compute(self, *args, **kwargs):
        if isinstance(self.all_pred_interactions_id, list):
            # It means this was executed in a single GPU
            all_pred_interactions_id = torch.cat(self.all_pred_interactions_id)
            all_pred_scores = torch.cat(self.all_pred_scores)
            all_pred_tgt_matchings = torch.cat(self.all_pred_tgt_matchings)
        else:
            all_pred_interactions_id = self.all_pred_interactions_id
            all_pred_scores = self.all_pred_scores
            all_pred_tgt_matchings = self.all_pred_tgt_matchings

        for int_id in all_pred_interactions_id.unique():
            samples_idx = torch.nonzero(all_pred_interactions_id == int_id).squeeze(1)
            self.output[int_id.item()] += all_pred_scores[samples_idx].tolist()
            self.labels[int_id.item()] += all_pred_tgt_matchings[samples_idx].tolist()

        output = [
            torch.tensor(self.output[idx], device=self.device, dtype=torch.float64)
            for idx in range(self.num_interactions)
        ]
        labels = [
            torch.tensor(self.labels[idx], device=self.device, dtype=torch.float64)
            for idx in range(self.num_interactions)
        ]

        aps, max_recs, max_precs = self._compute_ap(output, labels)
        return aps, max_recs, max_precs

    def _match(
        self,
        tgt_human_bboxes,
        tgt_object_bboxes,
        pred_human_bboxes,
        pred_object_bboxes,
        pred_scores=None,
    ):
        iou_human = box_iou(tgt_human_bboxes, pred_human_bboxes)
        iou_object = box_iou(tgt_object_bboxes, pred_object_bboxes)
        iou = torch.min(iou_human, iou_object)

        max_iou, max_idx = iou.max(dim=0)

        if pred_scores is None:
            pred_scores = max_iou

        # Assign each detection to the best matching ground truth
        match = torch.zeros_like(iou)
        match[max_idx, torch.arange(iou.shape[1])] = max_iou

        # Threshold the matches
        match = match > self.iou_threshold
        pred_tgt_matching = torch.zeros_like(pred_scores)

        # Determine true positive
        for _, m in enumerate(match):
            match_idx = torch.nonzero(m).squeeze(1)
            if len(match_idx) == 0:
                continue

            match_scores = pred_scores[match_idx]
            pred_tgt_matching[match_idx[match_scores.argmax()]] = 1

        return pred_tgt_matching

    def _compute_ap(self, output, labels):
        ap = torch.zeros(len(output), device=self.device, dtype=output[0].dtype)
        max_rec = torch.zeros_like(ap)
        max_prec = torch.zeros_like(ap)

        if self.ap_method == "11P":
            ap_method_fn = self._compute_per_class_ap_with_11_point_interpolation
        elif self.ap_method == "INT":
            ap_method_fn = self._compute_per_class_ap_with_interpolation
        else:
            ap_method_fn = self._compute_per_class_ap_as_auc

        for idx in range(len(output)):
            ap[idx], max_rec[idx], max_prec[idx] = self._compute_ap_for_each(
                (
                    idx,
                    output[idx],
                    labels[idx],
                    self.num_annotations_per_interaction[idx],
                    ap_method_fn,
                )
            )

        return ap, max_rec, max_prec

    def _compute_ap_for_each(self, items):
        idx, output, labels, num_annotations, ap_method_fn = items

        if labels.sum() > num_annotations:
            print(f"Class {idx}: number of true positives larger than that of ground truth")

            return 0, 0, 0

        if len(output) and len(labels):
            prec, rec = self._compute_pr_for_each(output, labels, num_annotations)
            return ap_method_fn(prec, rec), rec[-1], prec[-1]
        else:
            # print(f"WARNING: Collected results are empty. Return zero AP for class {idx}")
            return 0, 0, 0

    def _compute_pr_for_each(self, output, labels, num_annotations):
        order = torch.argsort(output, descending=True)

        tp = labels[order]
        fp = 1 - tp
        tp = torch.cumsum(tp, dim=0)
        fp = torch.cumsum(fp, dim=0)

        prec = tp / (tp + fp)
        rec = tp / num_annotations

        # If nan, set to 0
        prec[torch.isnan(prec)] = 0
        rec[torch.isnan(rec)] = 0

        return prec, rec

    def _compute_per_class_ap_as_auc(self, prec, rec):
        ap = 0
        max_rec = rec[-1]

        for i in range(prec.numel()):
            # Stop when maximum recall is reached
            if rec[i] >= max_rec:
                break

            d_x = rec[i] - rec[i - 1]

            # Skip when negative example is registered
            if d_x == 0:
                continue

            ap += prec[i] * rec[i] if i == 0 else 0.5 * (prec[i] + prec[i - 1]) * d_x

        return ap

    def _compute_per_class_ap_with_11_point_interpolation(self, prec, rec):
        ap = 0
        for t in torch.linspace(0, 1, 11, dtype=prec.dtype, device=self.device):
            idxs = torch.nonzero(rec >= t).squeeze()
            if idxs.numel():
                ap += prec[idxs].max() / 11

        return ap

    def _compute_per_class_ap_with_interpolation(self, prec, rec):
        ap = 0
        max_rec = rec[-1]
        for idx in range(prec.numel()):
            # Stop when maximum recall is reached
            if rec[idx] >= max_rec:
                break

            d_x = rec[idx] - rec[idx - 1]

            # Skip when negative example is registered
            if d_x == 0:
                continue

            # Compute interpolated precision
            max_ = prec[idx:].max()
            ap += (
                max_ * rec[idx]
                if idx == 0
                else 0.5 * (max_ + torch.max(prec[idx - 1], max_)) * d_x
            )

        return ap
