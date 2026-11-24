# Copyright 2025 Bytedance Ltd. and/or its affiliates
import logging
import os
import re

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def strip_thinking_trace(text: str) -> str:
    """Keep the visible answer after a reasoning trace, including an implicit opening tag."""
    visible = text.rsplit("</think>", 1)[-1]
    return visible.split("<think>", 1)[0].strip()


def format_full_reasoning_reference(text: str) -> str:
    """Keep the reasoning and final response while removing chat-template markers."""
    reasoning, separator, final_response = text.rpartition("</think>")
    if not separator:
        return text.replace("<think>", "").strip()
    reasoning = reasoning.replace("<think>", "").replace("</think>", "").strip()
    final_response = final_response.replace("<think>", "").strip()
    if not reasoning:
        return final_response
    return f"Reference reasoning:\n{reasoning}\n\nReference final response:\n{final_response}"


# English references can repeat the answer inside the reasoning trace.
_ANSWER_DISCLOSURE = re.compile(
    r"\\(?:boxed|fbox)\b|\b(?:final|correct)\s+(?:answer|result|solution|value)\b"
    r"|\b(?:the\s+)?(?:answer|result)\s*(?:is\b|[:=])"
    r"|^[ \t]*(?:answer|conclusion)[ \t]*[:：]"
    r"|\b(?:therefore|thus|hence|so|in conclusion)\b[,:]?[ \t]+"
    r"(?:[\w ]*\b(?:candidate|answer|result|solution|minimum|maximum|total)\s+(?:is|are|was|equals)\b"
    r"|(?:the\s+)?[a-z](?:_\w+)?(?:\(\s*\d+\s*\))?\s*=\s*[-+]?\d)",
    re.IGNORECASE | re.MULTILINE,
)

_STEP_BY_STEP = {
    "bn": "উদাহরণের যুক্তি দিকনির্দেশনা হিসেবে ব্যবহার করে সমস্যাটি ধাপে ধাপে বাংলায় সমাধান করুন।",
    "de": "Nutze die Beispielbegründung als Orientierung und löse die Aufgabe Schritt für Schritt auf Deutsch.",
    "en": "Use the reference reasoning as guidance and solve the problem step by step in English.",
    "es": "Usa el razonamiento de referencia como guía y resuelve el problema paso a paso en español.",
    "fr": "Sers-toi du raisonnement de référence comme guide et résous le problème étape par étape en français.",
    "it": "Usa il ragionamento di riferimento come guida e risolvi il problema passo dopo passo in italiano.",
    "ja": "参考の推論を手がかりにして、問題の解き方を日本語で順を追って説明してください。",
    "ko": "참고 풀이를 지침으로 삼아 문제를 한국어로 단계별로 해결하세요.",
    "pt": "Usa o raciocínio de referência como guia e resolve o problema passo a passo em português.",
    "ru": "Используйте приведённое рассуждение как подсказку и решите задачу шаг за шагом на русском языке.",
    "sw": "Tumia maelezo ya mfano kama mwongozo na utatue tatizo hatua kwa hatua kwa Kiswahili.",
    "te": "ఉదాహరణలోని తర్కాన్ని మార్గదర్శకంగా తీసుకుని సమస్యను తెలుగులో దశలవారీగా పరిష్కరించండి.",
    "th": "ใช้เหตุผลจากตัวอย่างเป็นแนวทาง แล้วแก้โจทย์ทีละขั้นตอนเป็นภาษาไทย",
    "zh": "以参考推理为指导，用中文逐步解答这道题。",
}


def extract_reference_reasoning(text: str) -> str:
    """Keep a closed reasoning trace up to the first explicit English answer cue."""
    reasoning, separator, _ = text.partition("</think>")
    if not separator:
        return ""
    reasoning = reasoning.rsplit("<think>", 1)[-1]
    disclosure = _ANSWER_DISCLOSURE.search(reasoning)
    if disclosure:
        # Drop the whole line, including any value written before the answer cue.
        reasoning = reasoning[: reasoning.rfind("\n", 0, disclosure.start()) + 1]
    return reasoning.strip()


def reference_reasoning_reminder(language_code: str, language_label: str) -> str:
    return _STEP_BY_STEP.get(language_code) or (
        f"Use the reference reasoning as guidance and solve the problem step by step in {language_label}."
    )


def initialize_system_prompt(tokenizer, **apply_chat_template_kwargs) -> list[int]:
    """
    Initialize system prompt tokens for chat templates that support them.

    Args:
        tokenizer: The tokenizer with a chat template
        **apply_chat_template_kwargs: Additional arguments for apply_chat_template

    Returns:
        List of token IDs for the system prompt, or empty list if not supported
    """
    token1 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True, return_dict=False
    )
    token2 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}] * 2, add_generation_prompt=False, tokenize=True, return_dict=False
    )
    # get system prompt tokens
    system_prompt = token1[: -(len(token2) - len(token1))]
    return system_prompt


def extract_system_prompt_and_generation(tokenizer):
    token1 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True
    )
    token2 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}] * 2, add_generation_prompt=False, tokenize=True
    )
    # get system prompt tokens
    system_prompt = token1[: -(len(token2) - len(token1))]
    # get generate prompt tokens
    token3 = tokenizer.apply_chat_template([{"role": "user", "content": ""}], add_generation_prompt=True, tokenize=True)
    generate_prompt = token3[len(token1) :]

    return system_prompt, generate_prompt
