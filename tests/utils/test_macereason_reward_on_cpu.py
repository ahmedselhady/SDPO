import pytest

from verl.utils.chat_template import format_full_reasoning_reference, strip_thinking_trace
from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.feedback import compute_score


@pytest.mark.parametrize("score", [default_compute_score, compute_score])
@pytest.mark.parametrize("data_source", ["math", "macereason_math"])
def test_macereason_scores_boxed_answers_in_both_reward_paths(score, data_source):
    info = {"source": "mAceReason-Math"}

    for text in [r"The result is \boxed{42}.", r"Respuesta: \boxed{42}"]:
        result = score(data_source, text, "42", info)
        assert result["score"] == 1.0
        assert result["sdpo_reference_eligible"]

    result = score(data_source, r"A draft says \boxed{42}</think>Respuesta: \boxed{41}", "42", info)
    assert result["score"] == 0.0
    assert not result["sdpo_reference_eligible"]

    result = score(data_source, r"Calculation </think>Respuesta: \boxed{42}", "42", {**info, "truncated": True})
    assert not result["sdpo_reference_eligible"]


def test_macereason_verifies_latex_equivalence():
    result = default_compute_score("macereason_math", r"Respuesta: \boxed{\frac{1}{2}}", "0.5")
    assert result["score"] == 1.0


@pytest.mark.parametrize(
    "text,expected",
    [
        ("<think>private\nreasoning</think>\nFinal", "Final"),
        ("private\nreasoning</think>\nFinal", "Final"),
        ("<think>unfinished reasoning", ""),
        ("Visible reasoning and final answer", "Visible reasoning and final answer"),
    ],
)
def test_strip_thinking_trace(text, expected):
    assert strip_thinking_trace(text) == expected


@pytest.mark.parametrize("opening", ["", "<think>"])
def test_full_reference_preserves_both_reasoning_sections(opening):
    text = opening + r"Subtract 2 from both sides.</think>Therefore, x is 3. \boxed{3}"
    assert format_full_reasoning_reference(text) == (
        "Reference reasoning:\nSubtract 2 from both sides.\n\n"
        "Reference final response:\n" r"Therefore, x is 3. \boxed{3}"
    )


def test_repeated_boxes_in_thinking_are_ineligible_as_full_references():
    text = r"\boxed{42} \boxed{42}</think>Final answer: \boxed{42}"
    result = default_compute_score("macereason_math", text, "42")
    assert result["acc"] == 1.0
    assert not result["sdpo_reference_eligible"]
