import argparse
import json
from pathlib import Path

import multiprocess
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

sentence_bert_processor = None
sentence_bert_model = None
object_labels = None
verb_labels = None
verb_ing_labels = None
interaction_labels = None


def remove_ing(verb: str) -> str:
    """Strip trailing 'ing' from each word in a verb phrase."""
    pieces = verb.split(" ")
    return " ".join(p[:-3] if p.endswith("ing") else p for p in pieces)


def get_embeddings(text: list[str], rank: int | None = None) -> torch.Tensor:
    """Encode a batch of strings with sentence-transformers/all-MiniLM-L6-v2."""
    global sentence_bert_processor, sentence_bert_model

    def _mean_pooling(text_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = text_embeds[0]
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

    if sentence_bert_processor is None or sentence_bert_model is None:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        sentence_bert_processor = AutoTokenizer.from_pretrained(model_name)
        sentence_bert_model = AutoModel.from_pretrained(model_name)
        if torch.cuda.is_available():
            device = f"cuda:{rank % torch.cuda.device_count()}" if rank is not None else "cuda"
            dtype = torch.float16 if device != "cpu" else torch.float32
            sentence_bert_model.to(device=device, dtype=dtype)

    text_inputs = sentence_bert_processor(text, padding=True, truncation=True, return_tensors="pt")
    text_inputs = text_inputs.to(device=sentence_bert_model.device)
    with torch.no_grad():
        text_embeds = sentence_bert_model(**text_inputs)
        text_embeds = _mean_pooling(text_embeds, text_inputs["attention_mask"])

    return text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)


def get_reference_interactions(sample: dict, **kwargs) -> dict:
    """Build the textual reference interactions for each ground-truth interaction id."""
    args = kwargs.get("args")
    reference_template = args.reference_template
    reference_interactions = []
    for interaction_id in sample["interaction_ids"]:
        object_label, verb_label = interaction_labels[interaction_id]
        object_label = object_labels[object_labels.index(object_label)].replace("_", " ")
        verb_label = verb_ing_labels[verb_labels.index(verb_label)]
        reference_interactions.append(
            reference_template.replace("{verb_label}", verb_label).replace(
                "{object_label}", object_label
            )
        )
    sample["reference_interactions"] = reference_interactions
    return sample


def get_predicted_interactions(sample: dict, **kwargs) -> dict:
    """Filter predicted triplets and turn them into textual interactions."""
    args = kwargs.get("args")
    object_embedding = get_embeddings(
        [object_labels[sample["object_id"]].replace("_", " ")]
    )[0]

    if args.top_k is not None:
        sample["output_sg"] = sample["output_sg"][: args.top_k]

    predicted_template = args.predicted_template
    predicted_interactions = []
    for pred_subject_label, pred_verb_label, pred_object_label in sample["output_sg"]:
        if not pred_object_label:
            continue
        if pred_subject_label not in ["person", "man", "woman", "people", "child", "kid"]:
            continue
        if pred_verb_label in ["is", "has"]:
            continue

        pred_object_embedding = get_embeddings([pred_object_label])[0]
        if object_embedding @ pred_object_embedding < args.object_similarity_threshold:
            continue

        predicted_interactions.append(
            predicted_template.replace("{object_label}", pred_object_label)
            .replace("{verb_label}", pred_verb_label)
            .replace("{subject_label}", pred_subject_label)
        )

    sample["predicted_interactions"] = predicted_interactions
    return sample


def get_recall_score(sample: dict, rank: int, **kwargs) -> dict:  # noqa: ARG001
    """Compute the per-sample semantic recall."""
    if "interaction_ids" not in sample or len(sample["predicted_interactions"]) == 0:
        sample["all_similarity_scores"] = torch.tensor([0.0])
        sample["mean_semantic_recall_score"] = 0.0
        return sample

    reference_embeddings = get_embeddings(sample["reference_interactions"], rank=rank)
    predicted_embeddings = get_embeddings(sample["predicted_interactions"], rank=rank)
    sim = reference_embeddings @ predicted_embeddings.T

    num_tgts = len(sample["reference_interactions"])
    sample["all_similarity_scores"] = sim.max(dim=1).values
    sample["mean_semantic_recall_score"] = (sample["all_similarity_scores"].sum() / num_tgts).item()
    return sample


if __name__ == "__main__":
    multiprocess.set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["hicodet", "vghoi"], default="hicodet")
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--results_path", type=str, required=True)
    parser.add_argument("--object_similarity_threshold", type=float, default=0.6)
    parser.add_argument("--reference_template", type=str, default="{verb_label}")
    parser.add_argument("--predicted_template", type=str, default="{verb_label}")
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    args.results_path = Path(args.results_path)
    assert args.results_path.exists() and args.results_path.is_file(), "Invalid path"

    if args.dataset == "hicodet":
        verbs_path = Path("artifacts") / "verbs" / "hicodet.json"
        meta_path = Path("artifacts") / "datasets" / "hicodet_instance_meta.json"
    else:
        verbs_path = Path("artifacts") / "verbs" / "vghoi.json"
        meta_path = Path("artifacts") / "datasets" / "vghoi_instance_meta.json"

    with open(meta_path) as f:
        meta = json.load(f)[args.dataset_split]
        object_labels = meta["objects"]
        verb_labels = meta["verbs"]
        interaction_labels = meta["interactions"]

    if verbs_path.exists():
        with open(verbs_path) as f:
            verbs_meta = json.load(f)
        verb_ing_labels = verbs_meta.get("verbs_ing") or verbs_meta.get("verbs")
    else:
        verb_ing_labels = list(verb_labels)

    results = load_dataset("json", data_files=[str(args.results_path)])["train"]
    assert "output_sg" in results[0], "Invalid results file, missing output_sg"
    assert "interaction_ids" in results[0], "Invalid results file, missing interaction_ids"

    if args.limit is not None:
        print(f"Limiting the number of samples to {args.limit}")
        results = results.select(range(args.limit))

    results = results.map(get_reference_interactions, num_proc=args.num_proc, fn_kwargs={"args": args})
    results = results.map(get_predicted_interactions, num_proc=args.num_proc, fn_kwargs={"args": args})
    results = results.map(get_recall_score, num_proc=args.num_proc, with_rank=True, fn_kwargs={"args": args})

    mean_recall = torch.tensor(results["mean_semantic_recall_score"]).mean().item()
    print(f"Mean semantic recall score: {mean_recall}")

    out_path = args.results_path.parent / f"{args.results_path.stem}_semantic_recall.csv"
    with open(out_path, "w") as f:
        f.write("mean_semantic_recall_score\n")
        f.write(f"{mean_recall}\n")
