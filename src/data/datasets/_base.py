from typing import Any

import torch
import torchvision
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(
        self,
        name: str,
        root_dir: str,
        split: str | None = None,
        transforms: torchvision.transforms.Compose | None = None,
        images_transforms: torchvision.transforms.Compose | None = None,
        targets_transforms: torchvision.transforms.Compose | None = None,
    ) -> None:
        """Base dataset class.

        Args:
            name (str): The name of the dataset.
            root_dir (str): The root directory where the dataset is stored.
            split (str): The split of the dataset to use.
            transforms (Optional[torchvision.transforms.Compose], optional): Transformation applied to the image and the target. Default to None.
            images_transforms (Optional[torchvision.transforms.Compose], optional): Transformation applied to the image only. Default to None.
            targets_transforms (Optional[torchvision.transforms.Compose], optional): Transformation applied to the targets only. Default to None.

        """
        self._name = name
        self._root_dir = root_dir
        self._split = split
        self._transforms = transforms
        self._images_transforms = images_transforms
        self._targets_transforms = targets_transforms
        self._text_transforms = None

        self._interactions = []
        self._interactions_id = []
        self._interactions_name = []
        self._objects = []
        self._objects_id = []
        self._objects_name = []
        self._verbs = []
        self._verbs_id = []
        self._verbs_name = []
        self._int_obj_verb_id_matrix = []
        self._num_annotations_per_interaction = []
        self._rare_int_ids = []
        self._non_rare_int_ids = []
        self._objects_verbs_to_interaction_id = None
        self._num_annotations_per_object = None
        self._num_annotations_per_verb = None
        self._objects_to_interactions = []
        self._objects_to_verbs = []
        self._verbs_to_interactions = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def root_dir(self) -> str:
        return self._root_dir

    @property
    def split(self) -> str | None:
        return self._split

    @property
    def transforms(self) -> torchvision.transforms.Compose:
        return self._transforms

    @transforms.setter
    def transforms(self, value: torchvision.transforms.Compose):
        self._transforms = value

    @property
    def images_transforms(self) -> torchvision.transforms.Compose:
        return self._images_transforms

    @images_transforms.setter
    def images_transforms(self, value: torchvision.transforms.Compose):
        self._images_transforms = value

    @property
    def text_transforms(self) -> torchvision.transforms.Compose:
        return self._text_transforms

    @property
    def targets_transforms(self) -> torchvision.transforms.Compose:
        return self._targets_transforms

    @targets_transforms.setter
    def targets_transforms(self, value: torchvision.transforms.Compose):
        self._targets_transforms = value

    @property
    def person_idx(self) -> int:
        return self.objects_name.index("person")

    @property
    def objects(self) -> list[tuple[int, str]]:
        if len(self._objects) == 0:
            self._objects = list(zip(self.objects_id, self.objects_name))

        return self._objects

    @property
    def num_objects(self) -> int:
        return len(self.objects)

    @property
    def objects_name(self) -> list[str]:
        return self._objects_name

    @property
    def objects_id(self) -> list[int]:
        if len(self._objects_id) == 0:
            self._objects_id = list(range(len(self.objects_name)))

        return self._objects_id

    @property
    def verbs(self) -> list[tuple[int, str]]:
        if len(self._verbs) == 0:
            self._verbs = list(zip(self.verbs_id, self.verbs_name))

        return self._verbs

    @property
    def num_verbs(self) -> int:
        return len(self.verbs_name)

    @property
    def verbs_name(self) -> list[str]:
        # This gets filled in the setup method
        return self._verbs_name

    @property
    def verbs_id(self) -> list[int]:
        if len(self._verbs_id) == 0:
            self._verbs_id = list(range(len(self.verbs_name)))

        return self._verbs_id

    @property
    def int_obj_verbs_id_matrix(self) -> list[tuple[int, int, int]]:
        """Class correspondence matrix in zero-based index [ [int_idx, obj_idx, verb_idx], ... ]

        Returns:
            list[list[3]]

        """
        return self._int_obj_verb_id_matrix

    @property
    def objects_verbs_to_interaction_id(self) -> list[list[int]]:
        """The interaction classes corresponding to an object-verb pair.

        Returns:
            List[List[int]]

        """
        if self._objects_verbs_to_interaction_id is None:
            self._objects_verbs_to_interaction_id = torch.full(
                (self.num_objects, self.num_verbs), -1
            )
            for int_id, obj_id, verb_id in self._int_obj_verb_id_matrix:
                self._objects_verbs_to_interaction_id[obj_id, verb_id] = int_id

        return self._objects_verbs_to_interaction_id

    @property
    def num_annotations_per_object(self) -> list[int]:
        """Number of annotated box pairs for each object class.

        Returns:
            list[80]

        """
        if self._num_annotations_per_object is None:
            self._num_annotations_per_object = [0 for _ in range(self.num_objects)]

            for int_id, obj_id, _ in self._int_obj_verb_id_matrix:
                self._num_annotations_per_object[obj_id] += self._num_annotations_per_interaction[
                    int_id
                ]

        return self._num_annotations_per_object

    @property
    def num_annotations_per_verb(self) -> list[int]:
        """Number of annotated box pairs for each verb class.

        Returns:
            list[117]

        """
        if self._num_annotations_per_verb is None:
            self._num_annotations_per_verb = [0 for _ in range(self.num_verbs)]

            for int_id, _, verb_id in self._int_obj_verb_id_matrix:
                self._num_annotations_per_verb[verb_id] += self._num_annotations_per_interaction[
                    int_id
                ]

        return self._num_annotations_per_verb

    def object_to_verbs(self, object_id) -> list[int]:
        if len(self._objects_to_verbs) == 0:
            self._objects_to_verbs = [
                [verb_id for _, o_id, verb_id in self.int_obj_verbs_id_matrix if o_id == obj_id]for obj_id in self.objects_id
            ]

        return self._objects_to_verbs[object_id]

    def verb_to_objects(self, verb_id) -> list[int]:
        if len(self._verbs_to_objects) == 0:
            self._verbs_to_objects = [
                [obj_id for _, obj_id, v_id in self.int_obj_verbs_id_matrix if v_id == verb_id]
                for verb_id in self.verbs_id
            ]

        return self._verbs_to_objects[verb_id]

    @property
    def objects_to_interactions(self) -> list[list[int]]:
        return self._objects_to_interactions

    @property
    def objects_to_verbs(self) -> list[list[int]]:
        if len(self._objects_to_verbs) == 0:
            self._objects_to_verbs = [
                [verb_id for _, o_id, verb_id in self.int_obj_verbs_id_matrix if o_id == obj_id]for obj_id in self.objects_id
            ]

        return self._objects_to_verbs

    @property
    def verbs_to_interactions(self) -> list[list[int]]:
        return self._verbs_to_interactions

    @property
    def interactions_id(self) -> list[tuple[int, int]]:
        if len(self._interactions_id) == 0:
            self._interactions_id = [
                (obj_id, verb_id) for _, obj_id, verb_id in self.int_obj_verbs_id_matrix
            ]

        return self._interactions_id

    @property
    def interactions_name(self) -> list[tuple[str, str]]:
        if len(self._interactions_name) == 0:
            self._interactions_name = [
                (self.objects_name[obj_id], self.verbs_name[verb_id])
                for obj_id, verb_id in self.interactions_id
            ]

        return self._interactions_name

    @property
    def interactions_to_verbs(self) -> list[list[int]]:
        if len(self._interactions_to_verbs) == 0:
            self._interactions_to_verbs = torch.tensor(
                [verb_id for _, verb_id in self.interactions_id]
            )

        return self._interactions_to_verbs

    @property
    def interactions_to_objects(self) -> list[list[int]]:
        if len(self._interactions_to_objects) == 0:
            self._interactions_to_objects = torch.tensor(
                [obj_id for obj_id, _ in self.interactions_id]
            )

        return self._interactions_to_objects

    @property
    def interactions(self) -> list[tuple[tuple[int, int], tuple[str, str]]]:
        if len(self._interactions) == 0:
            self._interactions = list(zip(self.interactions_id, self.interactions_name))

        return self._interactions

    @property
    def interactions_descriptions(self) -> dict[str, list[str]]:
        return self._interactions_descriptions

    @property
    def num_interactions(self) -> int:
        return len(self.interactions)

    @property
    def num_annotations_per_interaction(self) -> list[int]:
        return self._num_annotations_per_interaction.copy()

    @property
    def rare_interactions_id(self) -> list[int]:
        return self._rare_int_ids

    @property
    def non_rare_interactions_id(self) -> list[int]:
        return self._non_rare_int_ids

    def setup(self, **kwargs) -> None:
        raise NotImplementedError("You must implement the setup method.")

    def __len__(self) -> int:
        raise NotImplementedError("You must implement the __len__ method.")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        raise NotImplementedError("You must implement the __getitem__ method.")
