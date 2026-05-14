from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .prompts import build_image_prompt
from .utils import PROJECT_ROOT, load_yaml


@dataclass
class GenerationConfig:
    max_new_tokens: int = 1024
    do_sample: bool = False
    temperature: float = 0.0


class VLMInferencer:
    """Lazy Qwen2.5-VL wrapper for single-image inference."""

    def __init__(self, config_path: str | Path | None = None, model_name: str | None = None) -> None:
        self.config_path = Path(config_path) if config_path else PROJECT_ROOT / "configs" / "model_config.yaml"
        self.config = load_yaml(self.config_path)
        self.model_name = model_name or self.config.get("model", {}).get("name", "Qwen/Qwen2.5-VL-7B-Instruct")
        generation = self.config.get("generation", {})
        self.generation = GenerationConfig(
            max_new_tokens=int(generation.get("max_new_tokens", 1024)),
            do_sample=bool(generation.get("do_sample", False)),
            temperature=float(generation.get("temperature", 0.0)),
        )
        self.model: Any | None = None
        self.processor: Any | None = None

    def load_model(self) -> None:
        if self.model is not None and self.processor is not None:
            return

        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        model_cfg = self.config.get("model", {})
        vision_cfg = self.config.get("vision", {})

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=model_cfg.get("torch_dtype", "auto"),
            device_map=model_cfg.get("device_map", "auto"),
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            min_pixels=int(vision_cfg.get("min_pixels", 200704)),
            max_pixels=int(vision_cfg.get("max_pixels", 1003520)),
        )

    def answer_image(
        self,
        image_path: str | Path,
        question: str | None = None,
        structured: bool = True,
        max_new_tokens: int | None = None,
        prompt_override: str | None = None,
    ) -> str:
        self.load_model()
        assert self.model is not None
        assert self.processor is not None

        from qwen_vl_utils import process_vision_info

        prompt = prompt_override or build_image_prompt(question, structured=structured)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens or self.generation.max_new_tokens,
            "do_sample": self.generation.do_sample,
        }
        if self.generation.do_sample:
            generation_kwargs["temperature"] = self.generation.temperature

        generated_ids = self.model.generate(**inputs, **generation_kwargs)
        trimmed_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def answer_text(self, prompt: str, max_new_tokens: int | None = None) -> str:
        self.load_model()
        assert self.model is not None
        assert self.processor is not None

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], padding=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens or self.generation.max_new_tokens,
            "do_sample": self.generation.do_sample,
        }
        if self.generation.do_sample:
            generation_kwargs["temperature"] = self.generation.temperature

        generated_ids = self.model.generate(**inputs, **generation_kwargs)
        trimmed_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


_DEFAULT_INFERENCER: VLMInferencer | None = None


def get_default_inferencer(config_path: str | Path | None = None, model_name: str | None = None) -> VLMInferencer:
    global _DEFAULT_INFERENCER
    requested_name = model_name or None
    should_create = _DEFAULT_INFERENCER is None
    if _DEFAULT_INFERENCER is not None and requested_name and requested_name != _DEFAULT_INFERENCER.model_name:
        should_create = True
    if should_create:
        _DEFAULT_INFERENCER = VLMInferencer(config_path=config_path, model_name=model_name)
    return _DEFAULT_INFERENCER
