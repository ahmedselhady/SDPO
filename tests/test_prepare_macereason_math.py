from datasets import Dataset

from data import prepare_macereason_math


def test_convert_split_excludes_parallel_groups_with_empty_solutions(monkeypatch, capsys):
    examples = {
        "en": [
            {"original_idx": 1, "problem": "Valid problem", "solution": "Valid answer"},
            {"original_idx": 2, "problem": "Another problem", "solution": "Another answer"},
        ],
        "ko": [
            {"original_idx": 1, "problem": "유효한 문제", "solution": "유효한 답"},
            {"original_idx": 2, "problem": "답이 없는 문제", "solution": ""},
        ],
    }

    monkeypatch.setattr(prepare_macereason_math, "get_dataset_config_names", lambda _name: ["en", "ko"])
    monkeypatch.setattr(
        prepare_macereason_math,
        "load_dataset",
        lambda _name, name, split: Dataset.from_list(examples[name]),
    )

    result = prepare_macereason_math.convert_split(
        "dataset",
        "train",
        ["en", "ko"],
        id_column=None,
        problem_column=None,
        solution_column=None,
    )

    assert result["parallel_id"] == ["1", "1"]
    output = capsys.readouterr().out
    assert "excluded 1 parallel groups with empty solutions" in output
    assert "2 (ko)" in output
