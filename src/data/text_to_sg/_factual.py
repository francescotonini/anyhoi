import argparse
import re
import tempfile
from pathlib import Path

import torch
from datasets import load_dataset
from multiprocess import set_start_method
from tqdm import tqdm

factual_text_to_sg_model = None
factual_text_to_sg_processor = None


def factual_text_to_sg(sample: dict, rank: int | None = None, **kwargs) -> dict:
    """Return a list of triplets representing the scene graph of the input text."""
    input_column = kwargs.pop("input_column", "text")
    output_column = kwargs.pop("output_column", f"{input_column}_sg")

    global factual_text_to_sg_model
    global factual_text_to_sg_processor
    if factual_text_to_sg_model is None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        model_name = "lizhuang144/flan-t5-base-VG-factual-sg"
        factual_text_to_sg_processor = AutoTokenizer.from_pretrained(model_name)
        factual_text_to_sg_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    if rank is not None or torch.cuda.is_available():
        device = f"cuda:{(rank or 0)% torch.cuda.device_count()}"
        dtype = torch.float16 if device != "cpu" else torch.float32
        factual_text_to_sg_model.to(device=device, dtype=dtype)
    else:
        device = "cpu"

    if input_column not in sample:
        raise ValueError(f"{input_column} missing in dataset")

    sentences = sample[input_column].split(".")
    sentences = [x.strip() for x in sentences if x.strip() != ""]

    text_inputs = factual_text_to_sg_processor(
        [f"Generate Scene Graph: {x}" for x in sentences],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    text_inputs = text_inputs.to(device=factual_text_to_sg_model.device)

    with torch.no_grad():
        sg_ids = factual_text_to_sg_model.generate(
            text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
            use_cache=True,
            decoder_start_token_id=factual_text_to_sg_processor.pad_token_id,
            num_beams=1,
            max_length=200,
            early_stopping=True,
        )

    sg_raw_triplets = factual_text_to_sg_processor.batch_decode(
        sg_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    # Remove leading/trailing spaces and split the triplets
    sg_triplets = [
        re.findall(r"\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^)]+?)\s*\)", x)
        for x in sg_raw_triplets
    ]
    sample[output_column] = [t[0] for t in sg_triplets if len(t) > 0]

    return sample


if __name__ == "__main__":
    set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=str, help="Path to the results file")
    parser.add_argument("--results_output_path", type=str, help="Path to the results file")
    parser.add_argument("--shards_size", type=int, help="Size of the dataset shards", default=1000)
    parser.add_argument("--num_proc", type=int, help="Num of parallel processes", default=1)
    parser.add_argument(
        "--tmp_dir", type=str, help="Path to the temporary directory to store dataset shards"
    )
    args = parser.parse_args()

    args.results_path = Path(args.results_path)
    assert args.results_path.exists() and args.results_path.is_file(), "Invalid results path"

    # Print the arguments, for debugging
    print(args)

    # If output_path is none, use the same path as the input and add _sg to the name
    if args.results_output_path is None:
        args.results_output_path = (
            args.results_path.parent / f"{args.results_path.stem}_sg{args.results_path.suffix}"
        )

    # If tmp_dir is None, create a temporary directory with mktemp
    if args.tmp_dir is None:
        args.tmp_dir = Path(tempfile.mkdtemp())
    else:
        args.tmp_dir = Path(args.tmp_dir)

    # Make sure the temporary directory is empty
    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    # Load the results
    results_dataset = load_dataset("json", data_files=str(args.results_path))["train"]

    # Calculate the number of shards
    num_shards = max(1, len(results_dataset) // args.shards_size)

    shard_datasets = []
    for shard_idx in tqdm(range(num_shards), desc="Processing shards"):
        # Check if the shard already exists
        shard_path = args.tmp_dir / f"shard_{shard_idx}.json"
        if shard_path.exists():
            print(f"Shard {shard_idx} already exists")

            shard_datasets.append(shard_path)
            continue

        # Process the shard
        shard_dataset = results_dataset.shard(num_shards, shard_idx)

        # Perfect exact match: check if the answer includes all target interactions, no synonyms
        shard_dataset = shard_dataset.map(
            factual_text_to_sg,
            batched=False,
            fn_kwargs={
                "input_column": "output",
                "output_column": "output_sg",
            },
            num_proc=args.num_proc,
            with_rank=True,
        )

        # Save the shard
        shard_dataset.to_json(shard_path)
        shard_datasets.append(shard_path)

    # Load all the shards
    results_dataset = load_dataset("json", data_files=[str(s) for s in shard_datasets])["train"]

    # Save the results
    results_dataset.to_json(args.results_output_path)

    # Delete the temporary directory
    for shard_path in shard_datasets:
        shard_path.unlink()
    args.tmp_dir.rmdir()
