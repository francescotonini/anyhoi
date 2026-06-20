import argparse
import io
import json
import math
import re
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from src.data.datasets._hicodet import HICODET
from src.data.datasets._vg_hoi import VGHOI
from src.data.metrics._hoi_soft_map import HOISoftMapMetric


def clean_text(text: str) -> str:
    """Strip punctuation and lowercase a string."""
    return re.sub(r"[^a-zA-Z0-9\s]", "", text).strip().lower()


def process_results(args: argparse.Namespace, dataset: HICODET | VGHOI) -> dict:
    """Convert raw jsonl predictions into per-image tensors expected by the metric."""
    with open(args.results_path) as f:
        raw_results = [json.loads(line) for line in f]

    results: dict = {}

    for raw_result in tqdm(raw_results, desc="Processing results"):
        filename = raw_result["image_filename"]
        human_bbox = raw_result["human_bbox"]
        human_score = math.pow(raw_result["human_score"], 2.8)
        object_bbox = raw_result["object_bbox"]
        object_score = math.pow(raw_result["object_score"], 2.8)
        object_id = raw_result["object_id"]
        prior_score = human_score * object_score
        object_embedding = args.get_embeddings_fn(
            [clean_text(dataset.objects_name[object_id])]
        )[0]

        if filename not in results:
            results[filename] = {
                "humans_bbox": [],
                "humans_score": [],
                "humans_id": [],
                "objects_bbox": [],
                "objects_score": [],
                "objects_id": [],
                "verbs_label": [],
                "prior_scores": [],
            }

        for triplet in raw_result["output_sg"][: args.top_k]:
            pred_subject_label, pred_verb_label, pred_object_label = triplet

            if pred_subject_label not in ["person", "man", "woman", "people", "child", "kid"]:
                continue

            if pred_verb_label in ["is", "has"]:
                continue

            if not pred_object_label:
                continue

            pred_object_embedding = args.get_embeddings_fn([pred_object_label])[0]
            object_similarity = object_embedding @ pred_object_embedding

            if object_similarity < args.object_similarity_threshold:
                continue

            results[filename]["humans_bbox"].append(human_bbox)
            results[filename]["humans_score"].append(human_score)
            results[filename]["objects_bbox"].append(object_bbox)
            results[filename]["objects_score"].append(object_score)
            results[filename]["objects_id"].append(object_id)
            results[filename]["verbs_label"].append(pred_verb_label)
            results[filename]["prior_scores"].append(prior_score)

    for filename in results:
        for key in results[filename]:
            if key == "verbs_label":
                continue
            results[filename][key] = torch.tensor(results[filename][key])

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate open-set mAP for HOI predictions")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["hicodet", "vghoi"],
        default="hicodet",
    )
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Path to the dataset root directory (e.g. /path/to/hicodet).",
    )
    parser.add_argument("--results_path", type=str, required=True)
    parser.add_argument("--object_similarity_threshold", type=float, default=0.6)
    parser.add_argument("--verb_similarity_threshold", type=float, default=None)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=None)
    args = parser.parse_args()

    args.results_path = Path(args.results_path)
    assert args.results_path.exists() and args.results_path.is_file(), "Invalid results path"

    verb_labels: list = []
    if args.dataset == "hicodet":
        dataset = HICODET(root_dir=args.dataset_root, split=args.dataset_split)
        dataset.setup()
        assert dataset.verbs_name.index("no interaction") == 57

        verbs_path = Path("artifacts") / "verbs" / "hicodet.json"
        if verbs_path.exists():
            with open(verbs_path) as f:
                meta = json.load(f)
                verb_labels = meta["verbs_ing"]
            assert len(verb_labels) == len(dataset.verbs_name)
            assert dataset.verbs_name.index("no interaction") == verb_labels.index(
                "not interacting with"
            ), "Verb index mismatch"
        else:
            verb_labels = list(dataset.verbs_name)
    elif args.dataset == "vghoi":
        dataset = VGHOI(root_dir=args.dataset_root, split=args.dataset_split)
        dataset.setup()
        verb_labels = dataset.verbs_name

    verb_similarity_thresholds = [0.6, 0.7, 0.8, 0.9, 0.95]
    if args.verb_similarity_threshold:
        verb_similarity_thresholds = [args.verb_similarity_threshold]
    print(f"Using verb similarity threshold: {verb_similarity_thresholds}")

    metrics = {}
    for thr in verb_similarity_thresholds:
        metric = HOISoftMapMetric(
            num_interactions=dataset.num_interactions,
            objects_verbs_to_interaction_id=dataset.objects_verbs_to_interaction_id,
            num_annotations_per_interaction=dataset.num_annotations_per_interaction,
            verbs_name=verb_labels,
            verb_similarity_threshold=thr,
            iou_threshold=args.iou_threshold,
        )
        args.get_embeddings_fn = metric.get_embeddings
        metrics[thr] = metric

    tgts: dict = {}
    for idx in tqdm(range(len(dataset)), desc="Loading dataset"):
        sample = dataset[idx]
        del sample["images_tensor"]
        del sample["images_pil"]
        tgts[sample["images_filename"]] = sample

    for sample_filename, preds in process_results(args, dataset).items():
        for metric in metrics.values():
            metric.update({sample_filename: preds}, {sample_filename: tgts[sample_filename]})

    if args.dataset == "hicodet":
        rows = [
            "verb_similarity_threshold,map,recall,precision,"
            "map_rare,recall_rare,precision_rare,"
            "map_non_rare,recall_non_rare,precision_non_rare"
        ]
    else:
        rows = ["verb_similarity_threshold,map,recall,precision"]

    all_aps = []
    for thr, metric in metrics.items():
        aps, max_recs, max_precs = metric.compute()
        all_aps.append(aps.tolist())

        if args.dataset == "hicodet":
            ids_full = dataset.rare_interactions_id + dataset.non_rare_interactions_id
            recall_full = max_recs[ids_full].mean()
            recall_rare = max_recs[dataset.rare_interactions_id].mean()
            recall_non_rare = max_recs[dataset.non_rare_interactions_id].mean()
            map_full = aps[ids_full].mean()
            map_rare = aps[dataset.rare_interactions_id].mean()
            map_non_rare = aps[dataset.non_rare_interactions_id].mean()
            precision_full = max_precs[ids_full].mean()
            precision_rare = max_precs[dataset.rare_interactions_id].mean()
            precision_non_rare = max_precs[dataset.non_rare_interactions_id].mean()

            print(f"Verb similarity threshold: {thr}")
            print(f"mAP: {map_full}  (rare: {map_rare}, non-rare: {map_non_rare})")
            print(f"Recall: {recall_full}  (rare: {recall_rare}, non-rare: {recall_non_rare})")
            print(f"Precision: {precision_full}")

            rows.append(
                f"{thr},{map_full},{recall_full},{precision_full},"
                f"{map_rare},{recall_rare},{precision_rare},"
                f"{map_non_rare},{recall_non_rare},{precision_non_rare}"
            )
        else:
            recall_full = max_recs.mean()
            precision_full = max_precs.mean()
            map_full = aps.mean()
            print(f"Verb similarity threshold: {thr}")
            print(f"mAP: {map_full}  Recall: {recall_full}  Precision: {precision_full}")
            rows.append(f"{thr},{map_full},{recall_full},{precision_full}")

    df = pd.read_csv(io.StringIO("\n".join(rows)))
    # Paper reports mAP averaged across verb-similarity thresholds; append an
    # "avg" row with the column-wise mean across all per-threshold rows.
    if len(df) > 1:
        avg_row = df.drop(columns=["verb_similarity_threshold"]).mean()
        avg_row["verb_similarity_threshold"] = "avg"
        df = pd.concat([df, pd.DataFrame([avg_row])[df.columns]], ignore_index=True)
        print("Threshold-averaged mAP:", avg_row["map"])
    df.to_csv(args.results_path.parent / f"{args.results_path.stem}_open_map.csv", index=False)
    pd.DataFrame(all_aps).to_csv(
        args.results_path.parent / f"{args.results_path.stem}_all_aps.csv", index=False
    )
