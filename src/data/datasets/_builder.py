import argparse
import json
import os
import random
from pathlib import Path

import datasets
import numpy as np
import torch
from torchvision.ops import batched_nms, box_iou
from tqdm import tqdm

from src.data.datasets._hicodet import HICODET
from src.data.datasets._vg_hoi import VGHOI


def set_seed(seed: int) -> None:
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generator_instances(args, dataset, split, detector):
    # Autoincrement the index for each row
    generator_idx = 0

    # Set the parameters for the detector
    min_instances = 3
    max_instances = 15
    nms_threshold = 0.5
    score_threshold = 0.2

    for sample_idx in range(len(dataset)):
        sample = dataset[sample_idx]
        image_filename = sample["images_filename"]

        if detector != "gt":
            boxes = sample["detector_boxes"]
            scores = sample["detector_scores"]
            labels = sample["detector_labels"]

            # Apply NMS
            keep = batched_nms(boxes, scores, labels, nms_threshold)
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            # Filter out low scoring boxes
            keep = torch.nonzero(scores >= score_threshold).squeeze(1)

            # Separate humans and objects and keep only the top instances
            human_mask = labels == 0
            human_idx = torch.nonzero(human_mask).squeeze(1)
            object_idx = torch.nonzero(human_mask == 0).squeeze(1)
            num_humans = human_mask[keep].sum()
            num_objects = len(keep) - num_humans

            # Keep the number of human instances in a specified interval
            if num_humans < min_instances:
                keep_humans = scores[human_idx].argsort(descending=True)[:min_instances]
                keep_humans = human_idx[keep_humans]
            elif num_humans > max_instances:
                keep_humans = scores[human_idx].argsort(descending=True)[:max_instances]
                keep_humans = human_idx[keep_humans]
            else:
                keep_humans = torch.nonzero(human_mask[keep]).squeeze(1)
                keep_humans = keep[keep_humans]

            # Keep the number of object instances in a specified interval
            if num_objects < min_instances:
                keep_objects = scores[object_idx].argsort(descending=True)[:min_instances]
                keep_objects = object_idx[keep_objects]
            elif num_objects > max_instances:
                keep_objects = scores[object_idx].argsort(descending=True)[:max_instances]
                keep_objects = object_idx[keep_objects]
            else:
                keep_objects = torch.nonzero(human_mask[keep] == 0).squeeze(1)
                keep_objects = keep[keep_objects]

            keep = torch.cat([keep_humans, keep_objects])
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            # Create pairings
            is_human = labels == 0
            num_humans = torch.sum(is_human)
            num_boxes = len(boxes)

            # Permute human instances to the top
            if not torch.all(labels[:num_humans] == 0):
                humans_idx = torch.nonzero(is_human).squeeze(1)
                objects_idx = torch.nonzero(is_human == 0).squeeze(1)
                keep = torch.cat([humans_idx, objects_idx])

                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

            # Skip image when there are no valid human-object pairs
            if num_humans == 0 or num_boxes <= 1:
                continue

            # Get the pairwise indices
            humans_idx, objects_idx = torch.meshgrid(
                torch.arange(num_boxes), torch.arange(num_boxes)
            )

            # Valid human-object pairs
            humans_idx, objects_idx = torch.nonzero(
                torch.logical_and(humans_idx != objects_idx, humans_idx < num_humans)
            ).unbind(1)

            humans_bbox = boxes[humans_idx].long()
            humans_score = scores[humans_idx]
            objects_bbox = boxes[objects_idx].long()
            objects_score = scores[objects_idx]
            objects_id = labels[objects_idx].long()
            verbs_id = torch.tensor([-1] * len(humans_bbox))  # This is not used
        else:
            humans_bbox = sample["humans_bbox"]
            humans_score = torch.ones(len(humans_bbox))
            objects_bbox = sample["objects_bbox"]
            objects_score = torch.ones(len(objects_bbox))
            objects_id = sample["objects_id"]
            verbs_id = sample["verbs_id"]

        processed_interaction_idxs = []
        for interaction_idx, (
            verb_id,
            object_id,
            human_bbox,
            human_score,
            object_bbox,
            object_score,
        ) in enumerate(
            zip(
                verbs_id,
                objects_id,
                humans_bbox,
                humans_score,
                objects_bbox,
                objects_score,
            )
        ):
            object_id = object_id.item()
            human_bbox = human_bbox.tolist()
            human_score = human_score.item()
            object_bbox = object_bbox.tolist()
            object_score = object_score.item()

            if detector != "gt":
                valid_interaction_ids = dataset.objects_to_interactions[object_id].tolist()
                valid_interaction_ids_shuffled = [
                    valid_interaction_ids[i] for i in torch.randperm(len(valid_interaction_ids))
                ]
                valid_verb_ids_shuffled = [
                    dataset.interactions_id[valid_interaction_id][1]
                    for valid_interaction_id in valid_interaction_ids_shuffled
                ]

                yield {
                    "index": generator_idx,
                    "image_filename": image_filename,
                    "human_bbox": human_bbox,
                    "human_score": human_score,
                    "object_bbox": object_bbox,
                    "object_score": object_score,
                    "object_id": object_id,
                    "object_label": dataset.objects_name[object_id],
                    "verb_ids": [],
                    "verb_labels": [],
                    "interaction_ids": [],
                    "interaction_labels": [],
                    "valid_interaction_ids": valid_interaction_ids_shuffled,
                    "valid_interaction_labels": [
                        dataset.interactions_name[valid_interaction_id]
                        for valid_interaction_id in valid_interaction_ids_shuffled
                    ],
                    "valid_verb_ids": valid_verb_ids_shuffled,
                    "valid_verb_labels": [
                        dataset.verbs_name[valid_verb_id]
                        for valid_verb_id in valid_verb_ids_shuffled
                    ],
                }
                generator_idx += 1
            else:
                # Skip if already processed
                if interaction_idx in processed_interaction_idxs:
                    continue

                verb_id = verb_id.item()
                verb_label = dataset.verbs_name[verb_id]

                # Skip if no interaction
                if verb_label == "no interaction":
                    candidate_overlapping_interaction_idxs = [interaction_idx]
                    candidate_overlapping_humans_bbox = torch.tensor([human_bbox])
                    candidate_overlapping_objects_bbox = torch.tensor([object_bbox])
                else:
                    candidate_overlapping_interaction_idxs = [
                        i
                        for i, (o_id, _) in enumerate(
                            zip(sample["objects_id"], sample["objects_bbox"])
                        )
                        if o_id == object_id and i not in processed_interaction_idxs
                    ]
                    candidate_overlapping_humans_bbox = torch.tensor(
                        [
                            sample["humans_bbox"][i].tolist()
                            for i in candidate_overlapping_interaction_idxs
                        ]
                    )
                    candidate_overlapping_objects_bbox = torch.tensor(
                        [
                            sample["objects_bbox"][i].tolist()
                            for i in candidate_overlapping_interaction_idxs
                        ]
                    )

                humans_iou = box_iou(
                    torch.tensor(human_bbox).unsqueeze(0), candidate_overlapping_humans_bbox
                ).squeeze()
                objects_iou = box_iou(
                    torch.tensor(object_bbox).unsqueeze(0), candidate_overlapping_objects_bbox
                ).squeeze()
                interactions_iou = torch.min(humans_iou, objects_iou)

                # Get idxs of interactions_iou with iou > 0.5 (remember these are indexes of candidate_overlapping_interaction_idxs, not the candidate indexes)
                overlapping_interaction_idxs = torch.where(interactions_iou > 0.5)[0].tolist()
                overlapping_interaction_idxs = [
                    candidate_overlapping_interaction_idxs[i] for i in overlapping_interaction_idxs
                ]

                # Get the verb ids of interactions that are overlapping
                overlapping_verb_ids = [
                    sample["verbs_id"][i].item() for i in overlapping_interaction_idxs
                ]
                overlapping_interaction_ids = dataset.objects_verbs_to_interaction_id[
                    object_id, overlapping_verb_ids
                ].tolist()

                human_bbox = sample["humans_bbox"][overlapping_interaction_idxs].long()
                human_bbox = [
                    human_bbox[:, 0].min().item(),
                    human_bbox[:, 1].min().item(),
                    human_bbox[:, 2].max().item(),
                    human_bbox[:, 3].max().item(),
                ]
                object_bbox = sample["objects_bbox"][overlapping_interaction_idxs].long()
                object_bbox = [
                    object_bbox[:, 0].min().item(),
                    object_bbox[:, 1].min().item(),
                    object_bbox[:, 2].max().item(),
                    object_bbox[:, 3].max().item(),
                ]

                # Check if all interactions are valid, i.e. their index is not "null" (-1)
                assert_all_interactions_valid = all(
                    [
                        overlapping_interaction_id != -1
                        for overlapping_interaction_id in overlapping_interaction_ids
                    ]
                )
                assert assert_all_interactions_valid, "All interactions must be valid"

                # Check if the object is the same for all overlapping interactions. This is critical, as the following is based on this assumption
                assert_all_object_equal = set(
                    [
                        dataset.interactions_id[overlapping_interaction_id][0]
                        for overlapping_interaction_id in overlapping_interaction_ids
                    ]
                ) == {object_id}
                assert (
                    assert_all_object_equal
                ), "The object should be the same for all overlapping interactions"

                object_id = dataset.interactions_id[overlapping_interaction_ids[0]][0]
                valid_interaction_ids = dataset.objects_to_interactions[object_id].tolist()

                valid_interaction_ids_shuffled = [
                    valid_interaction_ids[i] for i in torch.randperm(len(valid_interaction_ids))
                ]
                valid_verb_ids_shuffled = [
                    dataset.interactions_id[valid_interaction_id][1]
                    for valid_interaction_id in valid_interaction_ids_shuffled
                ]

                yield {
                    "index": generator_idx,
                    "image_filename": image_filename,
                    "human_bbox": human_bbox,
                    "human_score": human_score,
                    "object_bbox": object_bbox,
                    "object_score": object_score,
                    "object_id": object_id,
                    "object_label": dataset.objects_name[object_id],
                    "verb_ids": overlapping_verb_ids,
                    "verb_labels": [
                        dataset.verbs_name[overlapping_verb_id]
                        for overlapping_verb_id in overlapping_verb_ids
                    ],
                    "interaction_ids": overlapping_interaction_ids,
                    "interaction_labels": [
                        dataset.interactions_name[overlapping_interaction_id]
                        for overlapping_interaction_id in overlapping_interaction_ids
                    ],
                    "valid_interaction_ids": valid_interaction_ids_shuffled,
                    "valid_interaction_labels": [
                        dataset.interactions_name[valid_interaction_id]
                        for valid_interaction_id in valid_interaction_ids_shuffled
                    ],
                    "valid_verb_ids": valid_verb_ids_shuffled,
                    "valid_verb_labels": [
                        dataset.verbs_name[valid_verb_id]
                        for valid_verb_id in valid_verb_ids_shuffled
                    ],
                }
                generator_idx += 1

                processed_interaction_idxs += overlapping_interaction_idxs


def build_dataset(args, split, detector):
    print(f"Building {args.dataset_name} dataset ({split}) with {detector} detector...")

    if detector == "gt":
        detector_path = None
    elif detector == "detr_r50":
        if args.dataset_name == "hicodet":
            detector_path = args.dataset_path / "detr_r50_preds" / split
        else:
            raise ValueError(
                f"Detector {detector} is not supported for {args.dataset_name} dataset."
            )
    elif detector == "gdino":
        if args.dataset_name == "vghoi":
            detector_path = args.dataset_path / "gdino_vg_anno_xyxy.json"
        else:
            raise ValueError(
                f"Detector {detector} is not supported for {args.dataset_name} dataset."
            )
    else:
        raise ValueError(f"Unknown detector {detector}.")

    # Set the dataset class based on the dataset name
    dataset_cls = None
    if args.dataset_name == "hicodet":
        dataset_cls = HICODET
    elif args.dataset_name == "vghoi":
        dataset_cls = VGHOI
    else:
        raise ValueError(f"Unknown dataset {args.dataset_name}.")

    dataset = dataset_cls(root_dir=args.dataset_path, split=split)
    dataset.setup(detector_path=detector_path)

    generator_fn = generator_instances
    features = {
        "index": datasets.Value("int32"),
        "image_filename": datasets.Value("string"),
        "human_bbox": datasets.Sequence(datasets.Value("int32")),
        "human_score": datasets.Value("float32"),
        "object_bbox": datasets.Sequence(datasets.Value("int32")),
        "object_score": datasets.Value("float32"),
        "object_id": datasets.Value("int32"),
        "object_label": datasets.Value("string"),
        "verb_ids": datasets.Sequence(datasets.Value("int32")),
        "verb_labels": datasets.Sequence(datasets.Value("string")),
        "interaction_ids": datasets.Sequence(datasets.Value("int32")),
        "interaction_labels": datasets.Sequence(datasets.Value("string")),
        "valid_interaction_ids": datasets.Sequence(datasets.Value("int32")),
        "valid_interaction_labels": datasets.Sequence(datasets.Value("string")),
        "valid_verb_ids": datasets.Sequence(datasets.Value("int32")),
        "valid_verb_labels": datasets.Sequence(datasets.Value("string")),
    }

    # Build the dataset rows
    dataset_rows = []
    for row in tqdm(generator_fn(args, dataset, split, detector)):
        dataset_rows.append(row)

    # Create the dataset
    hf_dataset = datasets.Dataset.from_list(dataset_rows, features=datasets.Features(features))

    # Build the metadata
    meta = {
        "objects": dataset.objects_name,
        "verbs": dataset.verbs_name,
        "interactions": dataset.interactions_name,
    }

    return hf_dataset, meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_username",
        type=str,
        required=True,
        help="HuggingFace username where the dataset will be pushed",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["hicodet", "vghoi"],
        required=True,
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the root directory of the dataset",
        required=True,
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["instance"],
        default="instance",
        help="Type of dataset to build. Currently only 'instance' (one row per human-object pair) is supported.",
    )
    parser.add_argument(
        "--dataset_splits",
        nargs="+",
        help="Dataset splits to build.",
        required=True,
    )
    parser.add_argument(
        "--detectors",
        default=["gt"],
        nargs="+",
        help="List of detectors to use. If 'gt', use the ground truth annotations.",
    )
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    # Ensure dataset exists
    args.dataset_path = Path(args.dataset_path)
    assert args.dataset_path.exists(), f"Data directory {args.dataset_path} does not exist."

    # Ensure HF_TOKEN is set
    if "HF_TOKEN" not in os.environ:
        raise ValueError("Please set the HF_TOKEN environment variable.")

    # Print the arguments, for debugging
    print(vars(args))

    # Set the seed, for reproducibility
    set_seed(args.seed)

    # Build the datasets
    hf_datasets = {}
    metas = {}
    for split in args.dataset_splits:
        for detector in args.detectors:
            key = f"{split}_{detector}"
            hf_datasets[key], metas[split] = build_dataset(args, split, detector)

    # Push the datasets to the hub
    hf_datasets = datasets.DatasetDict(hf_datasets)
    hf_datasets.push_to_hub(
        f"{args.hf_username}/{args.dataset_name}_{args.dataset_type}_seed_{args.seed}",
        private=True,
    )

    # Save the metadata
    with open(
        f"artifacts/datasets/{args.dataset_name}_{args.dataset_type}_meta.json",
        "w",
    ) as f:
        json.dump(metas, f, indent=4)
