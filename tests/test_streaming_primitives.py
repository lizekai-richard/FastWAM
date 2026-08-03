import torch

from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler,
)


def _tiny_action_dit() -> ActionDiT:
    return ActionDiT(
        hidden_dim=16,
        action_dim=4,
        ffn_dim=32,
        text_dim=8,
        freq_dim=8,
        eps=1e-6,
        num_heads=2,
        attn_head_dim=8,
        num_layers=1,
    ).eval()


@torch.no_grad()
def test_action_dit_shared_and_repeated_per_token_timesteps_match() -> None:
    torch.manual_seed(0)
    model = _tiny_action_dit()
    action_tokens = torch.randn(2, 3, 4)
    context = torch.randn(2, 2, 8)
    shared_timestep = torch.tensor([125.0, 750.0])
    per_token_timestep = shared_timestep[:, None].expand(-1, action_tokens.shape[1])

    shared_output = model(action_tokens, shared_timestep, context)
    per_token_output = model(action_tokens, per_token_timestep, context)

    torch.testing.assert_close(per_token_output, shared_output, rtol=1e-5, atol=1e-6)


def test_streaming_slot_schedule_is_shifted_reverse_inference_schedule() -> None:
    scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=1000, shift=5.0)
    inference_t, inference_delta = scheduler.build_inference_schedule(
        num_inference_steps=4,
        device=torch.device("cpu"),
        dtype=torch.float64,
        shift_override=2.0,
    )

    slot_t, slot_delta = scheduler.build_streaming_slot_schedule(
        num_slots=4,
        device=torch.device("cpu"),
        dtype=torch.float64,
        shift_override=2.0,
    )

    torch.testing.assert_close(slot_t, inference_t.flip(0))
    torch.testing.assert_close(slot_delta, inference_delta.flip(0))
    assert torch.all(slot_t[1:] > slot_t[:-1])
    assert not torch.allclose(
        slot_delta, torch.full_like(slot_delta, slot_delta.mean())
    )


def test_step_per_token_broadcasts_shared_and_batched_deltas() -> None:
    sample = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)
    model_output = torch.ones_like(sample)
    shared_delta = torch.tensor([-0.1, -0.2, -0.3])
    batched_delta = torch.stack((shared_delta, shared_delta * 2.0))

    shared_result = WanContinuousFlowMatchScheduler.step_per_token(
        model_output, shared_delta, sample
    )
    batched_result = WanContinuousFlowMatchScheduler.step_per_token(
        model_output, batched_delta, sample
    )

    torch.testing.assert_close(
        shared_result, sample + shared_delta.reshape(1, 3, 1)
    )
    torch.testing.assert_close(
        batched_result, sample + batched_delta.reshape(2, 3, 1)
    )


def test_scheduler_rejects_nonfinite_shift() -> None:
    try:
        WanContinuousFlowMatchScheduler(shift=float("nan"))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected a non-finite scheduler shift to be rejected.")
