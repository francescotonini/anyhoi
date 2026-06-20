import json
import os
from pathlib import Path
from typing import Any

import datasets
import yaml
from PIL import Image, ImageDraw, ImageFilter

__all__ = [
    "doc_to_visual",
    "doc_to_visual_red_circle",
    "doc_to_visual_reverse_blur",
    "doc_to_text",
    "doc_to_target",
    "process_results",
    "create_results_file",
]


with open(Path(__file__).parent / "_default_template_yaml") as f:
    raw_data = f.readlines()
    safe_data = []
    for _, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))


with open(Path(__file__).parent / "assets" / "hicodet_instance_meta.json") as f:
    meta = json.load(f)
    split = config["test_split"].split("_")[0]
    OBJECTS = meta[split]["objects"]


def doc_to_visual(doc: dict[str, Any]) -> list[Image.Image]:
    """Convert an image from a HICODET document to RGB format.

    Args:
    ----
        doc (dict): A dictionary containing HICODET document data with image information.

    """
    filename = os.path.join(os.environ["HICODET_IMAGES_DIR"], doc["image_filename"])
    human_bbox = doc["human_bbox"]
    object_bbox = doc["object_bbox"]
    union_bbox = tuple(
        [
            min(human_bbox[0], object_bbox[0]),
            min(human_bbox[1], object_bbox[1]),
            max(human_bbox[2], object_bbox[2]),
            max(human_bbox[3], object_bbox[3]),
        ]
    )

    image = Image.open(filename).convert("RGB")
    image = image.crop(union_bbox)

    return [image]


def doc_to_visual_red_circle(doc: dict[str, Any]) -> list[Image.Image]:
    """Convert an image from a HICODET document to RGB format with a red circle.

    Args:
    ----
        doc (dict): A dictionary containing HICODET document data with image information.

    """
    filename = os.path.join(os.environ["HICODET_IMAGES_DIR"], doc["image_filename"])
    human_bbox = doc["human_bbox"]
    object_bbox = doc["object_bbox"]

    image = Image.open(filename).convert("RGB")

    # Draw a red circle around the human and object
    draw = ImageDraw.Draw(image)
    draw.ellipse(human_bbox, outline="red", width=3)
    draw.ellipse(object_bbox, outline="red", width=3)

    return [image]


def doc_to_visual_reverse_blur(doc: dict[str, Any]) -> list[Image.Image]:
    """Convert an image from a HICODET document to RGB format with a reverse blur effect.

    Args:
    ----
        doc (dict): A dictionary containing HICODET document data with image information.

    """
    filename = os.path.join(os.environ["HICODET_IMAGES_DIR"], doc["image_filename"])
    human_bbox = doc["human_bbox"]
    object_bbox = doc["object_bbox"]

    image = Image.open(filename).convert("RGB")

    # Blur the image except for the human and object
    image_blur = image.filter(ImageFilter.GaussianBlur(5))
    image_blur.paste(image.crop(human_bbox), human_bbox)
    image_blur.paste(image.crop(object_bbox), object_bbox)

    return [image_blur]


def doc_to_text(doc: dict[str, Any], model_specific_kwargs: dict[str, str] | None = None) -> str:
    """Convert a HICODET document to a question string."""
    object_id = doc["object_id"]
    object_label = OBJECTS[object_id]

    question = model_specific_kwargs["question_template"].replace("{object_label}", object_label)

    return question


def doc_to_target(doc: dict[str, Any], model_specific_kwargs: dict[str, str] | None = None) -> str:
    """Convert a HICODET document to an empty target string."""
    return ""


def process_results(doc: dict[str, Any], results: list[str]) -> dict[str, dict[str, dict | str]]:
    """Process prediction results.

    Always stores the generations as a list under ``outputs``; ``output`` keeps the
    first generation for human inspection of greedy runs.
    """
    doc["output"] = results[0]
    doc["outputs"] = list(results)

    return {"create_results_file": doc}


def create_results_file(items: list[dict], args: dict) -> None:
    """Create a JSON file containing HICODET triplets for evaluation.

    Runs the FACTUAL text-to-SG backend on every generation in ``outputs`` and
    concatenates the resulting triplets into a single ``output_sg`` list per pair,
    so the written ``results.jsonl`` is ready for ``eval_hoi_map.py`` and
    ``eval_semantic_recall.py``.
    """
    # pytype: disable=attribute-error
    if not args.output_path:
        return

    from src.data.text_to_sg import factual_text_to_sg

    def per_generation_sg(sample: dict, rank: int | None = None) -> dict:
        triplets: list = []
        for generation in sample["outputs"]:
            tmp = factual_text_to_sg(
                {"_gen": generation},
                rank=rank,
                input_column="_gen",
                output_column="_gen_sg",
            )
            triplets.extend(tmp["_gen_sg"])
        sample["output_sg"] = triplets
        return sample

    data = datasets.Dataset.from_list(items)
    data = data.map(per_generation_sg, with_rank=True)
    data.set_format("torch")

    data.to_json(Path(args.output_path) / "results.jsonl")
