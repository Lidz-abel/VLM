from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .prompts import build_image_prompt
from .utils import PROJECT_ROOT, assert_gpu_ready, get_bool_env, load_yaml, safe_stem


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

        runtime_cfg = self.config.get("runtime", {})
        assert_gpu_ready(
            min_free_gb=float(runtime_cfg.get("min_free_gb", 18.0)),
            require_single_visible=bool(runtime_cfg.get("require_single_visible_gpu", True)),
        )

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


class MockVLMInferencer(VLMInferencer):
    """No-GPU inferencer for testing app, PDF, RAG, and note flows."""

    def __init__(self, config_path: str | Path | None = None, model_name: str | None = None) -> None:
        super().__init__(config_path=config_path, model_name=model_name or "mock-vlm")

    def load_model(self) -> None:
        self.model = "mock"
        self.processor = "mock"

    def answer_image(
        self,
        image_path: str | Path,
        question: str | None = None,
        structured: bool = True,
        max_new_tokens: int | None = None,
        prompt_override: str | None = None,
    ) -> str:
        image_name = safe_stem(image_path)
        prompt_text = prompt_override or build_image_prompt(question, structured=structured)
        if "【本页摘要】" in prompt_text:
            return f"""【本页主题】
Mock 页面解析：{image_name}

【OCR 转写内容】
这是 MOCK_VLM=1 生成的占位 OCR 文本，用于在没有空闲 GPU 时测试 PDF 渲染、逐页 JSON 保存、RAG 和笔记流程。

【关键概念】
- Mock 模式
- PDF 页面解析
- 无 GPU 流程验证

【图表/公式/代码说明】
无明显图表/公式/代码；真实结果需要在 GPU 空闲后使用 Qwen2.5-VL 重新解析。

【本页摘要】
本页是 {image_name} 的模拟解析结果。它不代表真实图片内容，只用于验证工程链路是否正常。
"""
        if not structured:
            return f"这是 {image_name} 的 Mock 回答。真实图片内容需要 GPU 空闲后加载 Qwen2.5-VL 验证。"
        return f"""【图片类型】
other：Mock 模式无法判断真实图片类型。

【识别内容】
这是 MOCK_VLM=1 生成的占位识别内容，图片文件名为 {image_name}。

【核心知识点】
- Mock 模式
- 单图问答链路
- 结构化输出格式

【分析过程】
当前未加载真实 Qwen2.5-VL 模型，因此不分析图片视觉内容。该输出仅用于测试前端、Prompt 拼接、Markdown 渲染和后续流程。

【最终答案】
Mock 推理成功。等 GPU 空闲后关闭 MOCK_VLM，再用真实模型验证图片理解效果。
"""

    def answer_text(self, prompt: str, max_new_tokens: int | None = None) -> str:
        return """这是 MOCK_VLM=1 生成的文本回答。

当前回答只用于验证 RAG 问答链路、上下文拼接和页面引用展示。真实答案需要在 GPU 空闲后使用 Qwen2.5-VL 或文本模型重新生成。"""


_DEFAULT_INFERENCER: VLMInferencer | None = None


def get_default_inferencer(config_path: str | Path | None = None, model_name: str | None = None) -> VLMInferencer:
    global _DEFAULT_INFERENCER
    if get_bool_env("MOCK_VLM", False):
        if _DEFAULT_INFERENCER is None or not isinstance(_DEFAULT_INFERENCER, MockVLMInferencer):
            _DEFAULT_INFERENCER = MockVLMInferencer(config_path=config_path, model_name=model_name)
        return _DEFAULT_INFERENCER

    requested_name = model_name or None
    should_create = _DEFAULT_INFERENCER is None
    if _DEFAULT_INFERENCER is not None and requested_name and requested_name != _DEFAULT_INFERENCER.model_name:
        should_create = True
    if should_create:
        _DEFAULT_INFERENCER = VLMInferencer(config_path=config_path, model_name=model_name)
    return _DEFAULT_INFERENCER
