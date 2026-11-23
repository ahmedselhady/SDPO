# Copyright 2025 Bytedance Ltd. and/or its affiliates
import logging
import os

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
