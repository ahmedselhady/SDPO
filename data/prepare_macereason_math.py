#!/usr/bin/env python3
"""Convert mAceReason-Math into SDPO/verl parquet files."""

import argparse
from collections import defaultdict
import math
from pathlib import Path
import random

from datasets import Dataset, get_dataset_config_names, load_dataset


LANGUAGES = ["bn", "de", "en", "es", "fr", "it", "ja", "ko", "pt", "ru", "sw", "te", "th", "zh"]
LANGUAGE_NAMES = {
    "bengali": "bn",
    "chinese": "zh",
    "english": "en",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "japanese": "ja",
    "korean": "ko",
    "portuguese": "pt",
    "russian": "ru",
    "spanish": "es",
    "swahili": "sw",
    "thai": "th",
    "telugu": "te",
}


def canonical_language(value) -> str:
    language = str(value).strip().lower().replace("_", "-")
    return LANGUAGE_NAMES.get(language, language.split("-")[0])


def first_column(columns, candidates, requested=None):
    if requested:
        if requested not in columns:
            raise ValueError(f"Requested column {requested!r} is missing; available columns: {columns}")
        return requested
    return next((column for column in candidates if column in columns), None)


def load_language_split(dataset_name, language, split, config_names, combined_cache):
    if language in config_names:
        return load_dataset(dataset_name, name=language, split=split), language

    if "combined" not in combined_cache:
        combined_cache["combined"] = load_dataset(dataset_name, split=split)
    dataset = combined_cache["combined"]
    language_column = first_column(dataset.column_names, ["language", "lang", "language_code"])
    if language_column is None:
        raise ValueError(
            f"Dataset {dataset_name!r} has no config {language!r} and no language column. "
            f"Available configs: {config_names}; columns: {dataset.column_names}"
        )
    filtered = dataset.filter(lambda row: canonical_language(row[language_column]) == language)
    if len(filtered) == 0:
        raise ValueError(f"No rows for language {language!r} in {dataset_name!r} split {split!r}")
    return filtered, language_column


def extract_problem(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        for message in reversed(value):
            if isinstance(message, dict) and message.get("role") == "user":
                return str(message.get("content", ""))
    if isinstance(value, dict):
        return str(value.get("content", ""))
    return ""


def convert_split(
    dataset_name, split, languages, id_column, problem_column, solution_column,
    max_groups=None, seed=42, percentage=None,
):
    config_names = set(get_dataset_config_names(dataset_name))
    combined_cache = {}
    rows = []
    invalid_solution_groups = defaultdict(set)

    for language in languages:
        source, _language_column = load_language_split(dataset_name, language, split, config_names, combined_cache)
        columns = source.column_names
        problem_key = first_column(columns, ["problem", "question", "prompt"], problem_column)
        solution_key = first_column(columns, ["solution", "answer", "ground_truth", "target"], solution_column)
        source_id_key = first_column(
            columns,
            ["original_idx", "original_id", "source_id", "problem_id", "id", "idx", "index"],
            id_column,
        )
        if source_id_key is None:
            raise ValueError(
                f"No source problem ID found for {language}/{split}. Set --id-column to a stable ID column; "
                "parallel rows must not be aligned by row order implicitly."
            )

        for example in source:
            problem = extract_problem(example[problem_key])
            solution = example[solution_key]
            parallel_id = str(example[source_id_key])
            if solution is None or not str(solution).strip():
                invalid_solution_groups[parallel_id].add(language)
                continue
            if not problem.strip():
                raise ValueError(f"Empty problem for {language}/{split}, ID={parallel_id!r}")
            rows.append(
                {
                    "data_source": "math",
                    "prompt": [{"role": "user", "content": problem}],
                    "reward_model": {"style": "math", "ground_truth": str(solution)},
                    "extra_info": {"index": parallel_id, "source": "mAceReason-Math"},
                    "parallel_id": parallel_id,
                    "language": language,
                }
            )

    if invalid_solution_groups:
        invalid_ids = set(invalid_solution_groups)
        rows = [row for row in rows if row["parallel_id"] not in invalid_ids]
        examples = [
            f"{parallel_id} ({', '.join(sorted(invalid_solution_groups[parallel_id]))})"
            for parallel_id in sorted(invalid_ids)[:5]
        ]
        print(
            f"{split}: excluded {len(invalid_ids):,} parallel groups with empty solutions; "
            f"examples (ID [languages]): {examples}"
        )

    group_languages = defaultdict(set)
    seen_pairs = set()
    for row in rows:
        key = (row["parallel_id"], row["language"])
        if key in seen_pairs:
            raise ValueError(f"Duplicate parallel_id/language pair: {key}")
        seen_pairs.add(key)
        group_languages[row["parallel_id"]].add(row["language"])

    expected_languages = set(languages)
    incomplete = [(group_id, sorted(found)) for group_id, found in group_languages.items() if found != expected_languages]
    if incomplete:
        examples = incomplete[:5]
        raise ValueError(
            f"{len(incomplete)} parallel groups do not contain exactly the requested languages "
            f"{sorted(expected_languages)}. Examples: {examples}"
        )

    if percentage is not None:
        if not group_languages:
            raise ValueError(f"No usable parallel groups in split {split!r}")
        max_groups = math.ceil(len(group_languages) * percentage / 100)

    if max_groups is not None:
        if "en" in languages:
            group_ids = list(dict.fromkeys(row["parallel_id"] for row in rows if row["language"] == "en"))
        else:
            group_ids = sorted(group_languages)
        if max_groups > len(group_ids):
            raise ValueError(f"Requested {max_groups} groups but split {split!r} has only {len(group_ids)}")
        selected_group_ids = set(random.Random(seed).sample(group_ids, max_groups))
        rows = [row for row in rows if row["parallel_id"] in selected_group_ids]
        group_languages = {group_id: group_languages[group_id] for group_id in selected_group_ids}

    rows.sort(key=lambda row: (row["parallel_id"], languages.index(row["language"])))
    output_language_counts = {
        language: sum(row["language"] == language for row in rows)
        for language in languages
    }
    print(f"{split}: {len(rows):,} rows, {len(group_languages):,} parallel groups; per-language={output_language_counts}")
    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="ahmedselhady/mAceReason-Math")
    parser.add_argument("--output-dir", default="datasets/mAceReason-Math")
    parser.add_argument("--languages", default=",".join(LANGUAGES), help="Comma-separated language codes")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--id-column", default=None, help="Stable source ID column (auto-detected by default)")
    parser.add_argument("--problem-column", default=None, help="Problem text column (auto-detected by default)")
    parser.add_argument("--solution-column", default=None, help="Answer column (auto-detected by default)")
    subset = parser.add_mutually_exclusive_group()
    subset.add_argument(
        "--max-train-groups",
        type=int,
        default=None,
        help="Randomly select this many parallel problem IDs from train; each selected ID retains all languages",
    )
    subset.add_argument(
        "--train-percentage", type=float,
        help="Percentage of usable train problem IDs to sample, rounded up; retain every requested language",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for train-group subsampling")
    args = parser.parse_args()

    languages = [canonical_language(language) for language in args.languages.split(",") if language.strip()]
    if not languages:
        parser.error("--languages must contain at least one language code")
    if len(languages) != len(set(languages)):
        raise ValueError(f"Duplicate languages requested: {languages}")
    if args.max_train_groups is not None and args.max_train_groups <= 0:
        raise ValueError("--max-train-groups must be a positive integer")
    if args.train_percentage is not None and not 0 < args.train_percentage <= 100:
        parser.error("--train-percentage must be greater than 0 and at most 100")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for source_split, output_name in ((args.train_split, "train.parquet"), (args.test_split, "test.parquet")):
        dataset = convert_split(
            args.dataset,
            source_split,
            languages,
            args.id_column,
            args.problem_column,
            args.solution_column,
            max_groups=args.max_train_groups if source_split == args.train_split else None,
            seed=args.seed,
            percentage=args.train_percentage if source_split == args.train_split else None,
        )
        output_path = output_dir / output_name
        dataset.to_parquet(str(output_path))
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
