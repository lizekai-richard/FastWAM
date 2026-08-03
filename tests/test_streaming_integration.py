from __future__ import annotations

import torch

from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler,
)


def _training_shell(num_slots: int = 4, chunk_size: int = 2) -> FastWAM:
    model = FastWAM.__new__(FastWAM)
    torch.nn.Module.__init__(model)
    model.streaming_action_num_slots = num_slots
    model.streaming_action_chunk_size = chunk_size
    model.train_action_scheduler = WanContinuousFlowMatchScheduler(
        num_train_timesteps=1000,
        shift=5.0,
    )
    return model


def _inverse_phi(sigma: torch.Tensor, shift: float) -> torch.Tensor:
    return sigma / (shift - (shift - 1.0) * sigma)


def test_streaming_training_batch_uses_schedule_suffixes_and_slot_times() -> None:
    torch.manual_seed(0)
    model = _training_shell()
    action = torch.randn(2, 8, 3)
    action_is_pad = torch.zeros(2, 8, dtype=torch.bool)
    action_is_pad[1, -1] = True

    stream = model._build_streaming_training_batch(action, action_is_pad)

    assert stream["noisy"].shape == (2, 32, 3)
    assert stream["timestep"].shape == (2, 32)
    assert stream["valid"].shape == (2, 32)
    assert stream["valid"][0].sum().item() == 2 + 4 + 6 + 8
    assert stream["valid"][1].sum().item() == 2 + 4 + 6 + 7

    timestep = stream["timestep"].reshape(2, 4, 4, 2)
    torch.testing.assert_close(timestep[..., 0], timestep[..., 1])
    sigma = timestep[..., 0].float() / 1000.0
    raw_u = _inverse_phi(sigma, shift=5.0)

    for config_idx in range(4):
        num_real = config_idx + 1
        first_stage = 4 - num_real
        for slot_idx in range(num_real):
            stage = first_stage + slot_idx
            values = raw_u[:, config_idx, slot_idx]
            assert torch.all(values >= stage / 4.0)
            assert torch.all(values < (stage + 1) / 4.0)


def test_streaming_training_mask_isolates_configs_padding_and_future_slots() -> None:
    model = _training_shell()
    valid = torch.tensor(
        [[
            1, 1, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 0,
        ]],
        dtype=torch.bool,
    )
    mask = model._build_streaming_training_action_mask(valid)
    assert mask.shape == (1, 32, 32)

    # Config branches never see each other.
    assert not mask[0, :8, 8:].any()
    assert not mask[0, 8:16, :8].any()
    # Within a slot attention is bidirectional; clean queries cannot see a
    # later/noisier slot, while later queries can see clean keys.
    assert mask[0, 24:26, 24:26].all()
    assert not mask[0, 24:26, 26:28].any()
    assert mask[0, 26:28, 24:26].all()
    # The final padded token is neither a query nor a key in action attention.
    assert not mask[0, 31, :].any()
    assert not mask[0, :, 31].any()


def _mot_attention_shell(num_heads: int = 2) -> MoT:
    mot = MoT.__new__(MoT)
    torch.nn.Module.__init__(mot)
    mot.num_heads = num_heads
    mot.mot_checkpoint_mixed_attn = False
    mot.eval()
    return mot


def test_mot_batched_mask_matches_legacy_broadcast_and_rejects_wrong_batch() -> None:
    torch.manual_seed(1)
    mot = _mot_attention_shell()
    q = torch.randn(2, 5, 8)
    k = torch.randn(2, 5, 8)
    v = torch.randn(2, 5, 8)
    mask_2d = torch.tril(torch.ones(5, 5, dtype=torch.bool))
    mask_3d = mask_2d.unsqueeze(0).expand(2, -1, -1)

    legacy = mot._mixed_attention(q, k, v, mask_2d)
    batched = mot._mixed_attention(q, k, v, mask_3d)
    torch.testing.assert_close(batched, legacy)

    wrong_batch = mask_2d.unsqueeze(0).expand(3, -1, -1)
    try:
        mot._mixed_attention(q, k, v, wrong_batch)
    except ValueError as exc:
        assert "batch" in str(exc)
    else:
        raise AssertionError("Expected malformed mask batch to be rejected.")


class _DummyActionExpert(torch.nn.Module):
    action_dim = 1


class _DummyVideoExpert(torch.nn.Module):
    video_attention_mask_mode = "first_frame_causal"
    fuse_vae_embedding_in_latents = False

    def pre_dit(self, x, timestep, context, context_mask, action, **kwargs):
        del timestep, action, kwargs
        batch_size = x.shape[0]
        return {
            "tokens": torch.zeros(batch_size, 1, 4),
            "freqs": torch.zeros(1, 1, 1, dtype=torch.complex64),
            "t_mod": torch.zeros(batch_size, 6, 4),
            "context": context,
            "context_mask": context_mask[:, None, :],
            "meta": {"tokens_per_frame": 1},
        }

    def build_video_to_video_mask(self, video_seq_len, **kwargs):
        del kwargs
        return torch.ones(video_seq_len, video_seq_len, dtype=torch.bool)


class _DummyVAE(torch.nn.Module):
    def encode(self, images, **kwargs):
        del kwargs
        image = images[0]
        return [torch.zeros(1, 1, 1, 1, device=image.device)]


class _DummyMoT(torch.nn.Module):
    def prefill_video_cache(self, **kwargs):
        del kwargs
        return []


def _inference_shell() -> FastWAM:
    model = FastWAM.__new__(FastWAM)
    torch.nn.Module.__init__(model)
    model.streaming_action_enabled = True
    model.streaming_action_num_slots = 4
    model.streaming_action_chunk_size = 2
    model.device = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.proprio_dim = None
    model.proprio_encoder = None
    model.video_expert = _DummyVideoExpert()
    model.action_expert = _DummyActionExpert()
    model.vae = _DummyVAE()
    model.mot = _DummyMoT()
    model.infer_action_scheduler = WanContinuousFlowMatchScheduler(
        num_train_timesteps=1000,
        shift=5.0,
    )
    model._predict_action_noise_with_cache = lambda **kwargs: torch.ones_like(
        kwargs["latents_action"]
    )
    return model


def test_public_streaming_api_keeps_state_external_and_emits_on_nth_call() -> None:
    model = _inference_shell()
    generator = torch.Generator(device="cpu").manual_seed(7)
    image = torch.zeros(1, 3, 16, 16)
    context = torch.zeros(1, 2, 4)
    context_mask = torch.ones(1, 2, dtype=torch.bool)

    state = None
    actions = []
    for _ in range(4):
        result = model.infer_action_streaming(
            prompt=None,
            input_image=image,
            action_horizon=8,
            streaming_state=state,
            action_generator=generator,
            context=context,
            context_mask=context_mask,
        )
        actions.append(result["action"])
        state = result["streaming_state"]

    assert actions[:3] == [None, None, None]
    assert actions[3] is not None and actions[3].shape == (2, 1)
    assert state is not None
    assert state.steps.item() == 4
    assert state.valid_mask.all()
    assert state.buffer.dtype == torch.float32
    assert state.schedule_shift == 5.0
    # The model object itself owns no rollout buffer or step counter.
    assert not hasattr(model, "streaming_action_state")

    try:
        model.infer_action_stream_step(
            prompt=None,
            input_image=image,
            state=state,
            new_noise=torch.zeros(1, 2, 1),
            context=context,
            context_mask=context_mask,
            sigma_shift=2.0,
        )
    except ValueError as exc:
        assert "Cannot change `sigma_shift`" in str(exc)
    else:
        raise AssertionError("Expected a reused state to reject a new sigma shift.")


def test_streaming_public_api_compile_fails_fast_instead_of_graph_breaking() -> None:
    model = _inference_shell()
    try:
        model._compile_infer_action_entrypoint()
    except RuntimeError as exc:
        assert "not supported" in str(exc)
    else:
        raise AssertionError("Expected whole-method streaming compile to be rejected.")


def test_streaming_constructor_requires_one_train_infer_shift_contract() -> None:
    try:
        FastWAM(
            video_expert=torch.nn.Linear(2, 2),
            action_expert=torch.nn.Linear(2, 2),
            mot=torch.nn.Linear(2, 2),
            vae=torch.nn.Linear(2, 2),
            text_dim=4,
            streaming_action_enabled=True,
            action_train_shift=5.0,
            action_infer_shift=3.0,
        )
    except ValueError as exc:
        assert "identical action train/infer shifts" in str(exc)
    else:
        raise AssertionError("Expected mismatched streaming shifts to be rejected.")
