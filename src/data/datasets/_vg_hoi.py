import json
import os

import numpy as np
import torch
import torchvision
from PIL import Image

from src.data.datasets._base import BaseDataset


class VGHOI(BaseDataset):
    """A dataset for human-object interaction."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(name="VGHOI", *args, **kwargs)

        assert self.split in ["test"], f"Unknown VGHOI split {self.split}"
        assert self.targets_transforms is None, "Targets transforms are not supported."

    def setup(self, **kwargs) -> None:
        detector_path = kwargs.get("detector_path")
        if detector_path is not None:
            with open(detector_path) as f:
                gdino_boxes = json.load(f)
        else:
            gdino_boxes = []

        self._images_dir = os.path.join(self.root_dir, "vg", "VG_100K")
        self._annotations_path = os.path.join(
            self.root_dir, "vg_hicodet_anno_xyxy_v1.json"
        )
        self._meta_path = os.path.join(self.root_dir, "meta.json")

        # Process annotations
        with open(self._annotations_path) as f:
            self._raw_annotations = json.load(f)
        with open(self._meta_path) as f:
            self._meta = json.load(f)

        self._objects_name = self._meta["objects"]
        self._verbs_name = self._meta["verbs"]
        self._interactions_name = [
            (object_label, verb_label) for (verb_label, object_label) in self._meta["interactions"]
        ]

        assert self._objects_name.index("person") == 0, "Person class must be the first class."

        self._annotations = []
        self._filenames = []

        idxs = list(range(len(self._raw_annotations)))
        idxs_empty = []
        present_interaction_ids = []

        # Get gdino boxes
        gdino_anno = {}
        for gdino_box in gdino_boxes:
            keep = [
                i
                for i in range(len(gdino_box["labels"]))
                if gdino_box["labels"][i] in self.objects_name
            ]
            gdino_box["labels_g"] = [
                self.objects_name.index(label) for label in np.array(gdino_box["labels"])[keep]
            ]
            gdino_box["scores_g"] = np.array(gdino_box["scores"], dtype=np.float32)[keep]
            gdino_box["boxes_g"] = np.array(gdino_box["boxes"], dtype=np.float32)[keep]
            gdino_anno[gdino_box["filename"]] = gdino_box

        # Processing samples
        for sample_idx in idxs:
            sample = self._raw_annotations[sample_idx]
            annotations = sample["annotations"]
            filename = f"{sample['image_id']}.jpg"
            self._filenames.append(filename)

            humans_bbox = []
            objects_bbox = []
            objects_id = []
            verbs_id = []
            interactions_id = []

            for annotation in annotations:
                object_name = annotation["obj"]
                verb_name = annotation["verb"]
                human_bbox = annotation["box_h"]
                object_bbox = annotation["box_o"]

                humans_bbox.append(human_bbox)
                objects_bbox.append(object_bbox)
                objects_id.append(self._objects_name.index(object_name))
                verbs_id.append(self._verbs_name.index(verb_name))
                interactions_id.append(self._interactions_name.index((object_name, verb_name)))

            present_interaction_ids.extend(interactions_id)

            if len(humans_bbox) == 0:
                idxs_empty.append(sample_idx)

            if filename in gdino_anno:
                labels_g = gdino_anno[filename]["labels_g"]
                scores_g = gdino_anno[filename]["scores_g"]
                boxes_g = gdino_anno[filename]["boxes_g"]
                if len(labels_g) == 0 and sample_idx not in idxs_empty:
                    idxs_empty.append(sample_idx)
            else:
                labels_g = []
                scores_g = []
                boxes_g = []

            self._annotations.append(
                {
                    "boxes_h": humans_bbox,
                    "boxes_o": objects_bbox,
                    "object": objects_id,
                    "verb": verbs_id,
                    "hoi": interactions_id,
                    "boxes_g": boxes_g,
                    "labels_g": labels_g,
                    "scores_g": scores_g,
                }
            )

        self._interactions_name = [
            self._interactions_name[i] for i in list(set(present_interaction_ids))
        ]

        # Get image idxs and remove empty idxs
        for idx_empty in idxs_empty:
            idxs.remove(idx_empty)

        num_annotations = [0 for _ in range(len(self._interactions_name))]
        for idx, annotation in enumerate(self._annotations):
            for hoi in annotation["hoi"]:
                num_annotations[hoi] += 1

        self._int_obj_verb_id_matrix = [
            [
                self._interactions_name.index(interaction_name),
                self._objects_name.index(interaction_name[0]),
                self._verbs_name.index(interaction_name[1]),
            ]
            for interaction_name in self._interactions_name
        ]
        self._num_annotations_per_interaction = num_annotations
        self._idxs = idxs

        self._objects_to_interactions = [
            self.objects_verbs_to_interaction_id[obj_id][
                self.objects_verbs_to_interaction_id[obj_id] != -1
            ]
            for obj_id in range(self.num_objects)
        ]

    def __len__(self):
        return len(self._idxs)

    def __getitem__(self, idx):
        idx = self._idxs[idx]
        annotation = self._annotations[idx]
        image_filename = self._filenames[idx]
        image_filepath = os.path.join(self._images_dir, image_filename)

        image_pil = Image.open(image_filepath).convert("RGB")
        image_size = torch.tensor(image_pil.size)

        verbs_id = []
        objects_id = []
        humans_bbox = []
        objects_bbox = []
        for human_bbox, object_bbox, object_id, verb_id in zip(
            annotation["boxes_h"],
            annotation["boxes_o"],
            annotation["object"],
            annotation["verb"],
        ):
            verbs_id.append(verb_id)
            objects_id.append(object_id)
            humans_bbox.append(human_bbox)
            objects_bbox.append(object_bbox)

        verbs_id = torch.tensor(verbs_id, dtype=torch.long)
        objects_id = torch.tensor(objects_id, dtype=torch.long)
        humans_bbox = torch.tensor(humans_bbox, dtype=torch.float)
        objects_bbox = torch.tensor(objects_bbox, dtype=torch.float)

        target = {
            "images_filename": image_filename,
            "images_filepath": image_filepath,
            "images_size": image_size,
            "verbs_id": verbs_id,
            "objects_id": objects_id,
            "humans_bbox": humans_bbox,
            "objects_bbox": objects_bbox,
            # detector annotations
            "detector_boxes": torch.tensor(annotation["boxes_g"], dtype=torch.float),
            "detector_labels": torch.tensor(annotation["labels_g"], dtype=torch.long),
            "detector_scores": torch.tensor(annotation["scores_g"], dtype=torch.float),
        }

        if self.transforms:
            image_pil, target = self.transforms(image_pil, target)

        if self.images_transforms:
            image_tensor = self.images_transforms(image_pil)
        else:
            image_tensor = torchvision.transforms.functional.to_tensor(image_pil)

        target["images_tensor"] = image_tensor
        target["images_pil"] = image_pil

        return target
