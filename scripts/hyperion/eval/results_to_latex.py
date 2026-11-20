#!/usr/bin/env python3
"""Export lm-evaluation-harness scores as LaTeX tabular fragments."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[3]

# Each pattern matches a language-level result, including MMLU-ProX group scores.
TASKS = (
    (r"macereason_math_([a-z]{2,3})", "macereason_math", "", "math_verify,none"),
    (r"macereason_math_([a-z]{2,3})_train", "macereason_math_train", "", "math_verify,none"),
    (r"mmlu_prox_([a-z]{2,3})", "mmlu_prox", "", "exact_match,custom-extract"),
    (r"mgpqa_main_native_cot_zeroshot_([a-z]{2,3})", "mgpqa", "Native CoT", "exact_match,flexible-extract"),
    (r"mgpqa_main_english_cot_zeroshot_([a-z]{2,3})", "mgpqa", "English CoT", "exact_match,flexible-extract"),
)

LATEX_ESCAPE = str.maketrans({
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
})


def load_scores(results_dir):
    scores = defaultdict(dict)
    # lm-eval embeds a sortable ISO timestamp in each result filename.
    for path in sorted(results_dir.rglob("results_*.json"), key=lambda p: (p.name, p)):
        with path.open(encoding="utf-8") as source:
            report = json.load(source)
        model = report.get("model_name") or path.parent.name.replace("__", "/")
        results = {**report.get("results", {}), **(report.get("groups") or {})}
        for task, metrics in results.items():
            for pattern, benchmark, variant, metric in TASKS:
                match = re.fullmatch(pattern, task)
                if match is None:
                    continue
                score = metrics.get(metric)
                if type(score) in (int, float) and math.isfinite(score):
                    scores[benchmark, variant, metric][model, match[1]] = score
                break
    return scores


def render_table(scores, variant, metric):
    models = sorted({model for model, _ in scores})
    languages = sorted({language for _, language in scores}, key=lambda lang: (lang != "en", lang))
    end_row = r" \\"
    columns = len(languages) + 2
    lines = [f"% {metric} (percent)", r"\begin{tabular}{l" + "r" * (columns - 1) + "}", r"\hline"]
    if variant:
        lines.append(r"\multicolumn{" + str(columns) + "}{l}{" + variant + "}" + end_row)
    lines.extend([" & ".join(["Model"] + languages + ["Avg"]) + end_row, r"\hline"])
    for model in models:
        language_scores = [scores.get((model, language)) for language in languages]
        average = mean(score for score in language_scores if score is not None)
        cells = [model.translate(LATEX_ESCAPE)] + [
            "--" if score is None else f"{score * 100:.2f}"
            for score in language_scores + [average]
        ]
        lines.append(" & ".join(cells) + end_row)
    lines.extend([r"\hline", r"\end{tabular}"])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "eval_results")
    parser.add_argument("--output-dir", type=Path, help="Defaults to RESULTS_DIR/tables.")
    args = parser.parse_args()

    scores = load_scores(args.results_dir)
    if not scores:
        parser.error(f"No supported benchmark scores found under {args.results_dir}")

    tables = defaultdict(list)
    for _, benchmark, variant, metric in TASKS:
        cells = scores.get((benchmark, variant, metric))
        if cells:
            tables[benchmark].append(render_table(cells, variant, metric))

    output_dir = args.output_dir or args.results_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    for benchmark, fragments in tables.items():
        path = output_dir / f"{benchmark}.tex"
        path.write_text("\n\n".join(fragments) + "\n", encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
