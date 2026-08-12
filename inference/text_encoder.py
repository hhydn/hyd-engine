import torch
import transformers

from typing import cast
from transformers import PreTrainedModel, PreTrainedTokenizerBase


_state: dict[str, object | None] = {
    "text_encoder": None,
    "text_encoder_2": None,
    "tokenizer": None,
    "tokenizer_2": None,
    "prompt": None,
}


def on_prompt_changed(prompt: str) -> None:
    _state["prompt"] = prompt


def on_text_encoder_loaded(text_encoder: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> None:
    _state["text_encoder"] = text_encoder
    _state["tokenizer"] = tokenizer


def on_text_encoder_2_loaded(text_encoder_2: PreTrainedModel, tokenizer_2: PreTrainedTokenizerBase) -> None:
    _state["text_encoder_2"] = text_encoder_2
    _state["tokenizer_2"] = tokenizer_2


def get_conditioning(batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    text_encoder: PreTrainedModel = cast(PreTrainedModel, _state["text_encoder"])
    text_encoder_2: PreTrainedModel = cast(PreTrainedModel, _state["text_encoder_2"])
    tokenizer: PreTrainedTokenizerBase = cast(PreTrainedTokenizerBase, _state["tokenizer"])
    tokenizer_2: PreTrainedTokenizerBase = cast(PreTrainedTokenizerBase, _state["tokenizer_2"])
    prompt: str = cast(str, _state["prompt"])

    input_ids: torch.Tensor = _tokenize(tokenizer, prompt, batch)
    input_ids_2: torch.Tensor = _tokenize(tokenizer_2, prompt, batch)

    return _encode(text_encoder, text_encoder_2, input_ids, input_ids_2)


def _tokenize(tokenizer: PreTrainedTokenizerBase, prompt: str, batch: int) -> torch.Tensor:
    tokens: transformers.BatchEncoding = tokenizer([prompt] * batch, padding="max_length", truncation=True, return_tensors="pt")
    input_ids: torch.Tensor = cast(torch.Tensor, tokens["input_ids"]).to("cuda")

    return input_ids


@torch.inference_mode()
def _encode(text_encoder: PreTrainedModel, text_encoder_2: PreTrainedModel, input_ids: torch.Tensor, input_ids_2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    output = text_encoder(input_ids, output_hidden_states=True)
    output_2 = text_encoder_2(input_ids_2, output_hidden_states=True)

    prompt_embeds: torch.Tensor = torch.cat((output.hidden_states[-2], output_2.hidden_states[-2]), dim=-1)
    pooled_prompt_embeds: torch.Tensor = output_2.text_embeds

    return prompt_embeds, pooled_prompt_embeds