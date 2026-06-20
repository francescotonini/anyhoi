# AnyHOI: Towards Unconstrained Human-Object Interaction
This is the official repo of the paper ["Towards Unconstrained Human-Object Interaction"](https://arxiv.org/pdf/2604.14069), accepted at IEEE FG 2026.

## Setup
### Install dependencies

```bash
# clone project
git clone https://github.com/francescotonini/anyhoi
cd anyhoi

# (recommended) use uv to set up the python version
# and to install the required dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --frozen

# (alternative) use conda + pip
conda create --name py3.12 python=3.12
conda activate py3.12
python -m venv .venv
.venv/bin/python3 -m pip install -e .

# activate virtual environment
source .venv/bin/activate
```

If your GPUs support **FlashAttention**, install the corresponding extra:
```bash
uv sync --frozen --extra nvidia --no-build-isolation
```

### Environment variables
Copy the example file and fill in your values:

```bash
cp .env.example .env
nano .env
```

The relevant variables are:

| Variable             | Required for                                                     |
| -------------------- | ---------------------------------------------------------------- |
| `HF_USERNAME`        | Resolving the dataset paths in the task YAMLs (`${HF_USERNAME}`) |
| `HF_TOKEN`           | Pulling private datasets and pushing built datasets to the Hub   |
| `HICODET_IMAGES_DIR` | Resolving HICODET image paths during evaluation                  |
| `VGHOI_IMAGES_DIR`   | Resolving VGHOI image paths during evaluation                    |

### Build the HuggingFace datasets
The repository ships HuggingFace dataset builders for HICODET and VGHOI.
The builders take the raw dataset directory on disk and push a packaged
HuggingFace dataset under `${HF_USERNAME}/<dataset>_instance_seed_42`.

```bash
python -m src.data.datasets._builder \
    --hf_username "$HF_USERNAME" \
    --dataset_name hicodet \
    --dataset_path <path-to-hicodet> \
    --dataset_type instance \
    --dataset_splits train test \
    --detectors gt detr_r50

python -m src.data.datasets._builder \
    --hf_username "$HF_USERNAME" \
    --dataset_name vghoi \
    --dataset_path <path-to-vghoi> \
    --dataset_type instance \
    --dataset_splits test \
    --detectors gt gdino
```

A wrapper script bundles both invocations:

```bash
bash scripts/build_datasets.sh "$HF_USERNAME" <path-to-hicodet> <path-to-vghoi>
```

`HF_TOKEN` must be set for the upload step. The task YAMLs under
`src/data/tasks/_relationships/` reference the resulting Hub paths via
`${HF_USERNAME}/...`.

## Run the experiments
You can launch a single evaluation directly:

```bash
python eval_model.py --help
python eval_model.py --model qwen2-vl-7b --tasks hicodet_openset_gt
```

| Option         | Description                                              |
| -------------- | -------------------------------------------------------- |
| `--models`     | Comma-separated list of models to evaluate.              |
| `--tasks`      | Comma-separated list of tasks to evaluate on.            |
| `--limit`      | Limit the number of samples per task.                    |
| `--model-args` | Extra comma-separated arguments for the models.          |
| `--no-samples` | Disable saving of sample predictions to disk.            |
| `--output`     | Output directory for results (default: `logs/schedule`). |

The HOI tasks automatically parse each model output into `(subject, verb, object)` triplets using the text-to-scene-graph backend.
The resulting `results.jsonl` therefore already contains an `output_sg` column ready for the evaluation scripts below.

## Evaluate the results

The two HOI metrics consume the `results.jsonl` written by each evaluation run; it
already contains the `output_sg` (triplets) and `interaction_ids` columns.

### Open-set mAP

```bash
python eval_hoi_map.py \
    --dataset hicodet \
    --dataset_root <path-to-hicodet> \
    --results_path logs/schedule/.../results.jsonl
```

Produces `<results>_open_map.csv` (mAP / recall / precision at several verb-similarity
thresholds, with rare / non-rare breakdowns for HICODET) and `<results>_all_aps.csv`
(per-interaction APs).

### Semantic recall

```bash
python eval_semantic_recall.py \
    --dataset hicodet \
    --results_path logs/schedule/.../results.jsonl
```

Produces `<results>_semantic_recall.csv` with the mean semantic recall score.

---

## Citation
```bibtex
@inproceedings{tonini2026towards,
  title={Towards Unconstrained Human-Object Interaction},
  author={Tonini, Francesco and Conti, Alessandro and Vaquero, Lorenzo and Beyan, Cigdem and Ricci, Elisa},
  journal={IEEE International Conference on Automatic Face and Gesture Recognition (FG)},
  year={2026}
}
```

## Acknowledgements
This codebase builds on [`altndrr/lmms-owc`](https://github.com/altndrr/lmms-owc), which we use as the open-world evaluation backbone.
