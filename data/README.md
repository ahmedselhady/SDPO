# Data Generation

## Setup

Add repository directory to `PYTHONPATH` env variable:

```
export PYTHONPATH=/users/$USER/SDPO:$PYTHONPATH
```

## Benchmark datasets

* LiveCodeBench v6: `python data/load_dataset.py --dataset_name livecodebench/code_generation_lite-v6 --output_path datasets/lcb_v6.json`

## Generalization Experiments

* tooluse: the json files are already provided in `datasets/tooluse/train.json` and `datasets/tooluse/test.json`
* sciknoweval Biology: `python data/load_dataset.py --dataset_name Biology --output_path datasets/sciknoweval/biology/biology.json`
* sciknoweval Chemistry: `python data/load_dataset.py --dataset_name Chemistry --output_path datasets/sciknoweval/chemistry/chemistry.json`
* sciknoweval Material: `python data/load_dataset.py --dataset_name Material --output_path datasets/sciknoweval/material/material.json`
* sciknoweval Physics: `python data/load_dataset.py --dataset_name Physics --output_path datasets/sciknoweval/physics/physics.json`


## LiveCodeBench

Create a train/test split of the tests by running:
```bash
python data/split_tests.py \
    --json_path datasets/lcb_v6.json \
    --output_dir datasets/lcb_v6
```

## SciKnowEval
Create a train/test split of the data by running:
```bash
python data/split_tasks.py \
    --json_path datasets/biology.json \
    --output_dir datasets/sciknoweval/biology

python data/split_tasks.py \
    --json_path datasets/chemistry.json \
    --output_dir datasets/sciknoweval/chemistry

python data/split_tasks.py \
    --json_path datasets/material.json \
    --output_dir datasets/sciknoweval/material

python data/split_tasks.py \
    --json_path datasets/physics.json \
    --output_dir datasets/sciknoweval/physics
```

## Prepocessing
Our implementation uses the `parquet` format for the data. To preprocess the data, run the following command:
```bash
python data/preprocess.py \
    --data_source DATASET_PATH
```
`DATASET_PATH` should contain the `train.json` and `test.json` files.

## mAceReason-Math for multilingual SDPO

From the `RL/SDPO` directory, prepare the parallel English and translated prompts from Hugging Face:

```bash
python data/prepare_macereason_math.py \
    --dataset ahmedselhady/mAceReason-Math \
    --output-dir datasets/mAceReason-Math
```

The script loads `train` and `test` for all 14 released language codes by default and writes `train.parquet` and `test.parquet`. It accepts per-language dataset configs or a combined dataset with a language column. It preserves the stable source problem ID as `parallel_id`, adds the language code, creates verl chat-format `prompt` messages, and stores the answer under `reward_model.ground_truth`. It checks that each parallel group has exactly one row per requested language and rejects duplicate language/problem pairs.

If any requested language has an empty solution for a problem, the script excludes that `parallel_id` from every language in that split and reports the excluded groups. Empty problems and duplicate language/problem pairs still raise errors.

The SDPO config uses a batch size of 28 rows: each problem contributes 14 language variants, so two complete parallel groups fit in each generation batch. If selecting a different set of languages, adjust `data.train_batch_size` to a multiple of the number of rows per parallel group and keep the resulting generated batch divisible by `actor_rollout_ref.actor.ppo_mini_batch_size`.

Point `data.train_files` and `data.val_files` in your run config at `datasets/mAceReason-Math/train.parquet` and `datasets/mAceReason-Math/test.parquet`. The dataset is released under CC BY-NC-ND 4.0; review its terms before redistributing converted files.

Submit the Hitz training job after the Parquet files exist:

```bash
sbatch scripts/hitz/train_sdpo_macereason.slurm
```

The job uses `Qwen/Qwen3-8B` by default. Override the model or experiment label at submission with, for example, `sbatch --export=ALL,MODEL_PATH=Qwen/Qwen3-14B,EXPERIMENT=mace14b scripts/hitz/train_sdpo_macereason.slurm` from the `RL/SDPO` directory.

For a smaller first run, submit the second job script:

```bash
sbatch scripts/hitz/train_sdpo_macereason_500.slurm
```

It prepares a deterministic sample of 500 `parallel_id`s from the training split (500 rows per language, 7,000 total), keeps the full test split, and trains one epoch on two GPUs. It defaults to `nvidia/Nemotron-Research-Reasoning-Qwen-1.5B`; use `sbatch --export=ALL,MODEL_PATH=Qwen/Qwen3-8B scripts/hitz/train_sdpo_macereason_500.slurm` to override it.

## Test-Time Training on LiveCodeBench Questions (TBD)
