"""Lazy local VLM adapter for offline VQA experiments."""
from __future__ import annotations

from pathlib import Path
import json
import re


class LocalVLM:
    # Bound visual tokens so one 1080p keyframe cannot monopolize generation
    # latency or fragment the same GPU used by the offline benchmark.
    MIN_PIXELS = 256 * 28 * 28
    MAX_PIXELS = 1280 * 28 * 28
    MAX_BATCH_NEW_TOKENS = 160
    MAX_EVIDENCE_FRAMES = 12
    STRUCTURED_FIELDS = ("answer", "grounding_score", "answer_confidence", "abstain")

    @classmethod
    def _structured_prompt(cls, prompt: str) -> str:
        """Return the single structured-output contract used by local VLM calls.

        Callers may provide a prompt that already contains this contract.  In
        that case it is returned unchanged so the answer layer never presents
        contradictory plain-text and JSON instructions to the model.
        """
        contract_marker = "Return ONLY valid JSON with exactly these fields"
        if contract_marker in prompt:
            return prompt
        return prompt.rstrip() + (
            "\n\nReturn ONLY valid JSON with exactly these fields: "
            '{"answer":"short answer", "grounding_score":0.0, '
            '"answer_confidence":0.0, "abstain":false}. '
            "Scores must be numbers from 0 to 1. Set abstain=true if evidence is insufficient."
        )

    @classmethod
    def _parse_metadata(cls, raw) -> dict:
        """Parse one model response into the stable answer-layer record."""
        payload = raw if isinstance(raw, dict) else None
        if payload is None:
            text = str(raw or "").strip()
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if match:
                try:
                    candidate = json.loads(match.group(0))
                except json.JSONDecodeError:
                    candidate = None
                if isinstance(candidate, dict):
                    payload = candidate
            if payload is None:
                return {
                    "answer": text,
                    "grounding_score": 0.0,
                    "answer_confidence": 0.0,
                    "abstain": not bool(text),
                    "parse_failed": True,
                }

        answer = str(payload.get("answer") or "").strip()
        try:
            grounding = max(0.0, min(1.0, float(payload.get("grounding_score", 0.0))))
        except (TypeError, ValueError):
            grounding = 0.0
        try:
            confidence = max(0.0, min(1.0, float(payload.get("answer_confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        abstain_value = payload.get("abstain", False)
        if isinstance(abstain_value, str):
            abstain = abstain_value.strip().casefold() in {"1", "true", "yes", "y"}
        else:
            abstain = bool(abstain_value)
        if not answer:
            abstain = True
        return {
            "answer": answer,
            "grounding_score": grounding,
            "answer_confidence": confidence,
            "abstain": abstain,
            "parse_failed": False,
        }

    def __init__(self, model_path: str | Path, *, load_in_4bit: bool = False):
        self.model_path = str(model_path)
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.processor = None

    def _ensure(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        quantization_config = None
        if self.load_in_4bit:
            if not torch.cuda.is_available():
                raise RuntimeError("4-bit VLM loading requires CUDA")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map="auto",
            quantization_config=quantization_config,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            min_pixels=self.MIN_PIXELS,
            max_pixels=self.MAX_PIXELS,
        )

    def answer(self, image_path: str, prompt: str, max_new_tokens: int = 128) -> str:
        self._ensure()
        import torch
        from qwen_vl_utils import process_vision_info

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt},
        ]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, videos=videos, padding=True,
            return_tensors="pt"
        ).to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated = output[:, inputs.input_ids.shape[1]:]
        answer = self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        del generated, output, inputs, images, videos
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return answer

    def answer_batch(self, image_paths: list[str], prompt: str,
                     max_new_tokens: int = 160) -> list[str]:
        """Run independent one-image requests in one model forward pass.

        Putting all images into one conversation and asking Qwen for a JSON
        list is not a safe batching contract: the model can merge, omit, or
        duplicate items.  A real batch is a list of independent conversations
        with one image each, so decoding preserves one output per input.
        """
        if not image_paths:
            return []
        self._ensure()
        import torch
        from qwen_vl_utils import process_vision_info

        bounded_tokens = max(1, min(int(max_new_tokens), self.MAX_BATCH_NEW_TOKENS))
        conversations = [
            [{"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ]}]
            for image_path in image_paths
        ]
        texts = [
            self.processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            for conversation in conversations
        ]
        images, videos = process_vision_info(conversations)
        inputs = self.processor(
            text=texts, images=images, videos=videos, padding=True,
            return_tensors="pt"
        ).to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=bounded_tokens)
        generated = output[:, inputs.input_ids.shape[1]:]
        answers = [answer.strip() for answer in self.processor.batch_decode(
            generated, skip_special_tokens=True
        )]
        del generated, output, inputs, images, videos
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if len(answers) != len(image_paths):
            raise RuntimeError(
                "local VLM batch returned an unexpected number of responses: "
                f"expected {len(image_paths)}, received {len(answers)}"
            )
        return answers

    def answer_with_metadata(self, image_path: str, prompt: str,
                             max_new_tokens: int = 160) -> dict:
        """Return a parsed answer/grounding record for the routed Q&A path."""
        raw = self.answer(
            image_path,
            self._structured_prompt(prompt),
            max_new_tokens=max_new_tokens,
        )
        return self._parse_metadata(raw)

    def answer_frames(self, image_paths: list[str], prompt: str,
                      max_new_tokens: int = 128) -> str:
        """Answer using several independently sampled frames as evidence."""
        image_paths = list(image_paths[:self.MAX_EVIDENCE_FRAMES])
        if not image_paths:
            return ""
        self._ensure()
        import torch
        from qwen_vl_utils import process_vision_info

        content = [{"type": "text", "text": (
            "These frames are from the same video and are provided as evidence. "
            "Use all frames, but do not assume an object is present when it is "
            "not visible. Answer briefly and directly.\n" + prompt)}]
        for index, image_path in enumerate(image_paths, 1):
            content.append({"type": "text", "text": f"FRAME {index}"})
            content.append({"type": "image", "image": image_path})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = self.processor(text=[text], images=images, videos=videos,
                                padding=True, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated = output[:, inputs.input_ids.shape[1]:]
        answer = self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        del generated, output, inputs, images, videos
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return answer

    def answer_frames_with_metadata(self, image_paths: list[str], prompt: str,
                                    max_new_tokens: int = 160) -> dict:
        """Answer from a bounded multi-frame evidence set using the shared schema.

        This is a real multi-image request.  If a caller has only a
        single-image model, the pipeline owns the deterministic per-frame
        fallback; this method never pretends that independent calls are one
        multi-frame observation.
        """
        paths = list(image_paths[:self.MAX_EVIDENCE_FRAMES])
        if not paths:
            return {
                "answer": "", "grounding_score": 0.0,
                "answer_confidence": 0.0, "abstain": True,
                "parse_failed": False, "reason": "no_evidence_frames",
            }
        raw = self.answer_frames(
            paths,
            self._structured_prompt(prompt),
            max_new_tokens=max_new_tokens,
        )
        return self._parse_metadata(raw)

    def choose_frame(self, image_a: str, image_b: str, question: str) -> str:
        """Choose A/B for two same-video frames; returns ``A``, ``B``, or ``TIE``."""
        self._ensure()
        import torch
        from qwen_vl_utils import process_vision_info

        messages = [{"role": "user", "content": [
            {"type": "text", "text": (
                "Two frames are from the same video. Choose the frame that "
                "best supports answering the question. Return only A, B, or TIE.\n"
                f"Question: {question}\nFRAME A:" )},
            {"type": "image", "image": image_a},
            {"type": "text", "text": "FRAME B:"},
            {"type": "image", "image": image_b},
        ]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = self.processor(text=[text], images=images, videos=videos,
                                padding=True, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=4)
        answer = self.processor.batch_decode(output[:, inputs.input_ids.shape[1]:],
                                             skip_special_tokens=True)[0].strip().upper()
        if answer.startswith("A"):
            return "A"
        if answer.startswith("B"):
            return "B"
        return "TIE"
