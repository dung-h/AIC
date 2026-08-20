import inspect
import glob
import unittest
import os
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pandas as pd


class OfflineDefaultsTests(unittest.TestCase):
    def test_runtime_policy_is_single_source_for_service_defaults(self):
        from src.runtime_policy import RuntimePolicy

        with patch.dict(os.environ, {
            "HCMAI_TRAKE_MODE": "visual",
            "HCMAI_VKIS_SELECTOR": "hybrid0.7",
            "VQA_VISUAL_SELECTOR_POLICY": "legacy",
        }, clear=False):
            policy = RuntimePolicy.from_env()
        self.assertEqual(policy.trake_mode, "visual")
        self.assertEqual(policy.vkis_selector, "hybrid0.7")
        self.assertEqual(policy.vqa_visual_selector_policy, "legacy")

    def test_runtime_policy_allows_anchor_preserving_only_as_explicit_policy(self):
        from src.runtime_policy import RuntimePolicy

        self.assertEqual(
            RuntimePolicy(vqa_visual_selector_policy="anchor_preserving").vqa_visual_selector_policy,
            "anchor_preserving",
        )

    def test_service_passes_policy_selector_to_ranked_vqa(self):
        from src.runtime_policy import RuntimePolicy
        from src.service.runtime import RetrievalRuntime

        class FakePipeline:
            def __init__(self):
                self.kwargs = None

            def vqa_ranked(self, *_args, **kwargs):
                self.kwargs = kwargs
                return {"query": "q", "question": "a", "answers": [], "status": "ok"}

        runtime = RetrievalRuntime(policy=RuntimePolicy(vqa_visual_selector_policy="legacy"))
        fake = FakePipeline()
        runtime._pipeline = fake
        runtime.search_vqa("scene", "what?", 1, 1, 1, 1)
        self.assertEqual(fake.kwargs["visual_selector_policy"], "legacy")

    def test_ranked_vqa_does_not_silently_switch_model_after_first_request(self):
        from src.pipelines.hcmai_pipeline import HCMAIPipeline

        pipeline = HCMAIPipeline()
        pipeline._vqa_ranked = object()
        pipeline._vqa_ranked_spec = (os.path.abspath("models/model-a"), False)
        with self.assertRaisesRegex(RuntimeError, "different local model"):
            pipeline._ensure_vqa_ranked("models/model-b")

    def test_codabench_offline_policy_disables_remote_flags(self):
        from src.runtime_policy import RuntimePolicy
        from src.pipelines.codabench_submit import _submission_policy

        policy = RuntimePolicy(kis_remote_translation=True, trake_remote_embeddings=True)
        locked = _submission_policy(policy, "qa", True)
        self.assertFalse(locked.kis_remote_translation)
        self.assertFalse(locked.trake_remote_embeddings)
        self.assertTrue(_submission_policy(policy, "kis", False).kis_remote_translation)

    def test_service_metadata_lookup_fails_closed_on_noncanonical_frame(self):
        from src.service.runtime import RetrievalRuntime

        runtime = RetrievalRuntime()
        runtime._kis = SimpleNamespace(km=SimpleNamespace(
            video_id=["V1"], kf_n=[3], frame_idx=[42], pts_time=[1.5]
        ))
        with self.assertRaisesRegex(RuntimeError, "canonical"):
            runtime._kis_result_metadata("V1", 999)
        with self.assertRaisesRegex(RuntimeError, "canonical"):
            runtime.kis_frame_time("V1", 3, 999)

        import pandas as pd
        runtime._vkis = SimpleNamespace(vmap=pd.DataFrame({
            "video_id": ["V1"], "kf_n": [3], "frame_idx": [42], "pts_time": [1.5]
        }))
        with self.assertRaisesRegex(RuntimeError, "canonical"):
            runtime.vkis_frame_metadata("V1", 999)

    def test_production_entrypoint_defaults_follow_measured_paths(self):
        from src.pipelines.hcmai_pipeline import HCMAIPipeline
        from src.pipelines.vkis_pipeline import VKISPipeline

        pipeline = HCMAIPipeline()
        self.assertEqual(pipeline._default_trake_mode, "visual")
        self.assertEqual(pipeline._default_vkis_selector, "hybrid0.5")
        self.assertEqual(inspect.signature(VKISPipeline.search_clip).parameters["agg"].default,
                         "hybrid0.5")

    def test_vqa_service_question_type_is_a_supported_contract(self):
        from src.pipelines.vqa_pipeline_v3 import VQAPipelineV3

        self.assertIn("question_type", inspect.signature(VQAPipelineV3.ranked_answers).parameters)

    def test_vqa_rejects_unknown_routing_labels_instead_of_falling_back(self):
        from src.pipelines.hcmai_pipeline import HCMAIPipeline

        pipeline = HCMAIPipeline()
        with self.assertRaisesRegex(ValueError, "unsupported question_type"):
            pipeline.vqa_ranked("scene", "what?", question_type="spken_fact")
        with self.assertRaisesRegex(ValueError, "unsupported required modality"):
            pipeline.vqa_ranked("scene", "what?", required_modalities="visual,deepgram")

    def test_trake_rejects_invalid_events_before_loading_models(self):
        from src.pipelines.hcmai_pipeline import HCMAIPipeline

        pipeline = HCMAIPipeline()
        with self.assertRaisesRegex(ValueError, "non-empty sequence"):
            pipeline.trake([], topk=1)
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            pipeline.trake(["first", "  "], topk=1)

    def test_trake_boundary_rejects_incomplete_or_unordered_backend_paths(self):
        from src.pipelines.hcmai_pipeline import HCMAIPipeline

        bad = [{"video_id": "V1", "path": [{"frame_idx": 8}, {"frame_idx": 7}]}]
        with self.assertRaisesRegex(RuntimeError, "strictly time-ordered"):
            HCMAIPipeline._validate_trake_results(bad, 2)
        with self.assertRaisesRegex(RuntimeError, "incomplete path"):
            HCMAIPipeline._validate_trake_results(
                [{"video_id": "V1", "path": [{"frame_idx": 8}]}], 2
            )

    def test_trake_boundary_rejects_empty_backend_results(self):
        from src.pipelines.hcmai_pipeline import HCMAIPipeline
        with self.assertRaisesRegex(RuntimeError, "no ranked answers"):
            HCMAIPipeline._validate_trake_results([], 1)

    def test_kis_default_translation_is_offline(self):
        from src.pipelines.kis_fusion_retriever import KISFusionRetriever

        self.assertFalse(inspect.signature(KISFusionRetriever).parameters["translate"].default)

    def test_kis_remote_translation_fails_closed_without_credentials(self):
        from src.pipelines.kis_fusion_retriever import KISFusionRetriever
        from src.core.providers import ProviderConfig

        obj = object.__new__(KISFusionRetriever)
        obj.translate_on = True
        obj.text_provider = ProviderConfig("text", "", "", "")
        obj._translate_cache = {}
        with self.assertRaises(RuntimeError):
            obj.translate("một cảnh")

    def test_trake_default_embedder_is_offline(self):
        from src.pipelines import trake_pipeline

        # TRAKE production deliberately refuses legacy per-pack discovery;
        # the fixture must inject the shared global-index boundary instead.
        fake_index = SimpleNamespace(
            chunks=pd.DataFrame({
                "vid": ["K01_V001"], "start": [0.0], "end": [1.0],
                "frame_idx": [1], "kf_n": [1], "chunk": ["x"],
            }),
            embeddings=np.zeros((1, 1024), dtype=np.float32),
            no_speech_videos=set(),
            diagnostics=lambda: {"scope": "full_corpus"},
        )
        with patch.object(trake_pipeline, "SharedASRGlobalIndex", return_value=fake_index), \
             patch.object(trake_pipeline, "get_text_embedder") as get_embedder:
            trake_pipeline.TrakePipeline()
            get_embedder.assert_called_once_with("offline")

    def test_trake_remote_mode_requires_credentials(self):
        from src.pipelines import trake_pipeline
        from src.core.providers import ProviderConfig

        with patch.object(
            trake_pipeline, "provider_for",
            return_value=ProviderConfig("embedding", "", "", ""),
        ):
            with self.assertRaises(RuntimeError):
                trake_pipeline.TrakePipeline(online=True)


if __name__ == "__main__":
    unittest.main()
