"""Reward for localized mAceReason prompts that request a boxed final answer."""

from verl.utils.chat_template import strip_thinking_trace
from verl.utils.reward_score.math_dapo import last_boxed_only_string, remove_boxed


def is_macereason(data_source, extra_info):
    return data_source == "macereason_math" or (
        data_source == "math" and (extra_info or {}).get("source") == "mAceReason-Math"
    )


def compute_score(solution_str, ground_truth, extra_info=None):
    final_answer = strip_thinking_trace(solution_str)
    final_box = last_boxed_only_string(final_answer)
    pred = remove_boxed(final_box).strip() if final_box else ""
    ground_truth = str(ground_truth).strip()
    correct = bool(pred and pred == ground_truth)
    if pred and not correct:
        from math_verify import LatexExtractionConfig, parse, verify

        latex = [LatexExtractionConfig(boxed_match_priority=0)]
        gold_box = last_boxed_only_string(ground_truth)
        gold_answer = remove_boxed(gold_box) if gold_box else ground_truth.strip("$").strip()
        gold = parse(r"\boxed{" + gold_answer + "}", extraction_config=latex, fallback_mode="no_fallback")
        prediction = parse(final_box, extraction_config=latex, fallback_mode="no_fallback")
        correct = bool(gold and prediction and verify(gold, prediction))

    truncated = bool((extra_info or {}).get("truncated", False))
    repeated_final_box = bool(final_box and solution_str.count(final_box) >= 3)
    feedback = ""
    if truncated:
        feedback = "Your response was truncated because it exceeded the maximum length."
    elif not pred:
        feedback = r"Give the final answer exactly once in \boxed{...} on the last line."

    return {
        "score": float(correct),
        "acc": float(correct),
        "pred": pred,
        "incorrect_format": int(not pred),
        "truncated": int(truncated),
        "truncated_and_missing_answer": int(truncated and not pred),
        "sdpo_reference_eligible": correct and not truncated and not repeated_final_box,
        "feedback": feedback,
    }
