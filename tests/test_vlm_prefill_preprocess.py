"""Regression tests for VLM text-only prefill preprocessing fast paths.

Long multi-turn prompts (~200K tokens) paid a multi-second TTFT tax before
the first token because the text-only chat path (a) converted the tokenized
``mx.array`` to a Python list via ``tolist()`` (per-element Python recursion,
~30 us/token) and (b) tokenized the exact same rendered prompt twice — once
in ``preflight_chat`` for the memory check, again in ``_prepare_vision_inputs``.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest

from omlx.engine.vlm import (
    VLMBatchedEngine,
    _ids_tensor_to_list,
    _text_prompt_sha256,
)


def test_ids_tensor_to_list_matches_native_tolist():
    """The numpy fast path must return the exact ids mx.tolist() returns."""
    ids = list(range(5000))
    assert _ids_tensor_to_list(mx.array([ids])) == ids
    assert _ids_tensor_to_list(mx.array(ids)) == ids


def test_ids_tensor_to_list_falls_back_without_array_interface():
    """Objects that np.asarray wraps as object dtype use native tolist."""

    class NoArrayInterface:
        ndim = 1

        def tolist(self):
            return [1, 2, 3]

    assert _ids_tensor_to_list(NoArrayInterface()) == [1, 2, 3]


def _engine_with_template(prompt: str) -> VLMBatchedEngine:
    engine = VLMBatchedEngine(model_name="test-vlm")
    engine._loaded = True
    engine._vlm_model = MagicMock()
    engine._vlm_model.config.model_type = "test"
    proc_tok = MagicMock()
    proc_tok.apply_chat_template.return_value = prompt
    engine._processor = SimpleNamespace(chat_template=None, tokenizer=proc_tok)
    engine._tokenizer = MagicMock()
    engine._tokenizer.apply_chat_template.return_value = prompt
    engine._tokenizer.encode.return_value = list(range(len(prompt)))
    return engine


def test_vlm_text_only_reuses_preflight_token_ids():
    """After preflight_chat encodes a prompt, the chat path must not re-tokenize it."""
    prompt = "hello world, this is a cached prompt"
    engine = _engine_with_template(prompt)
    msgs = [{"role": "user", "content": "x"}]

    # Seed the stash exactly as preflight_chat would, for the rendered prompt.
    engine._preflight_prompt_ids = (_text_prompt_sha256(prompt), list(range(10)))

    with patch("mlx_vlm.utils.prepare_inputs") as mock_prep:
        token_ids, embeds, extra, img_hash, start, ranges = (
            engine._prepare_vision_inputs(messages=msgs, images=[], audio=None)
        )
        mock_prep.assert_not_called()
    assert token_ids == list(range(10))
    assert embeds is None and extra is None and img_hash is None


def test_vlm_text_only_retokens_on_prompt_mismatch():
    """A stale stash for a different prompt must not be reused."""
    prompt = "the real prompt rendered now"
    engine = _engine_with_template(prompt)
    engine._preflight_prompt_ids = (_text_prompt_sha256("a different prompt"), [9, 9])

    fake = {"input_ids": mx.array([[1, 2, 3]])}
    with patch("mlx_vlm.utils.prepare_inputs", return_value=fake) as mock_prep:
        token_ids, *_ = engine._prepare_vision_inputs(
            messages=[{"role": "user", "content": "x"}],
            images=[],
            audio=None,
        )
        mock_prep.assert_called_once()
    assert token_ids == [1, 2, 3]


@pytest.mark.asyncio
async def test_preflight_chat_stashes_token_ids():
    """preflight_chat must populate the reuse stash with (sha256, ids)."""
    prompt = "preflight prompt content"
    engine = _engine_with_template(prompt)
    engine._preflight_prompt_ids = None
    engine._engine = SimpleNamespace(engine=SimpleNamespace(scheduler=MagicMock()))
    engine._preflight_or_raise_with_eviction = MagicMock(
        side_effect=lambda *a, **k: asyncio.sleep(0)
    )
    with patch(
        "omlx.engine.vlm.extract_images_from_messages",
        return_value=([{"role": "user", "content": prompt}], [], []),
    ), patch("omlx.engine.vlm._count_image_tokens_real", return_value=0):
        await engine.preflight_chat([{"role": "user", "content": prompt}])

    assert engine._preflight_prompt_ids is not None
    sha, ids = engine._preflight_prompt_ids
    assert isinstance(sha, str) and len(sha) == 64
    assert ids == list(range(len(prompt)))
