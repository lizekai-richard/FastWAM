"""Stateless tensor helpers for FlashVLA-style streaming action decoding.

The streaming buffer is laid out from the cleanest (oldest) slot to the
noisiest (newest) slot::

    [slot_0(C tokens), ..., slot_{N-1}(C tokens)]

All buffer transitions in this module are functional: inputs are never
modified in place, and action buffers are kept in FP32 independently of the
model compute dtype.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real

import torch


__all__ = [
    "StreamingActionState",
    "append_cold_start_noise",
    "clean_to_noisy_token_stages",
    "cold_start_token_values",
    "cold_start_valid_mask",
    "emit_shift_append_action_buffer",
    "initialize_streaming_action_state",
    "slot_block_causal_action_mask",
    "update_action_buffer",
    "validate_streaming_config",
]


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"`{name}` must be an integer, got {type(value).__name__}")
    value = int(value)
    if value <= 0:
        raise ValueError(f"`{name}` must be > 0, got {value}")
    return value


def validate_streaming_config(
    num_slots: int,
    chunk_size: int,
    action_horizon: int,
) -> tuple[int, int, int]:
    """Validate and return the canonical ``(N, C, H)`` configuration.

    ``N`` is the number of streaming slots, ``C`` is the number of action
    tokens in each slot, and ``H`` must be exactly ``N * C``.
    """

    num_slots = _positive_int(num_slots, "num_slots (N)")
    chunk_size = _positive_int(chunk_size, "chunk_size (C)")
    action_horizon = _positive_int(action_horizon, "action_horizon (H)")
    expected_horizon = num_slots * chunk_size
    if action_horizon != expected_horizon:
        raise ValueError(
            "`action_horizon (H)` must equal `num_slots (N) * chunk_size (C)`: "
            f"expected {expected_horizon}, got {action_horizon}"
        )
    return num_slots, chunk_size, action_horizon


def _optional_batch_size(batch_size: int | None) -> int | None:
    if batch_size is None:
        return None
    return _positive_int(batch_size, "batch_size")


def _normalize_steps(
    step: int | torch.Tensor,
    *,
    batch_size: int | None,
    device: torch.device,
) -> tuple[torch.Tensor, bool]:
    """Return steps as ``[B]`` int64 and whether the input was unbatched."""

    batch_size = _optional_batch_size(batch_size)
    if isinstance(step, torch.Tensor):
        if step.ndim not in (0, 1):
            raise ValueError(
                f"`step` must be a scalar or 1D [B] tensor, got shape {tuple(step.shape)}"
            )
        if step.dtype == torch.bool or torch.is_floating_point(step) or torch.is_complex(step):
            raise TypeError(f"`step` must have an integer dtype, got {step.dtype}")
        steps = step.to(device=device, dtype=torch.int64)
        unbatched = step.ndim == 0 and batch_size is None
        if step.ndim == 0:
            size = 1 if batch_size is None else batch_size
            steps = steps.reshape(1).expand(size)
        elif batch_size is not None and step.shape[0] != batch_size:
            raise ValueError(
                f"`step` batch length must equal batch_size={batch_size}, got {step.shape[0]}"
            )
    else:
        if isinstance(step, bool) or not isinstance(step, Integral):
            raise TypeError(f"`step` must be an integer or tensor, got {type(step).__name__}")
        unbatched = batch_size is None
        size = 1 if batch_size is None else batch_size
        steps = torch.full((size,), int(step), device=device, dtype=torch.int64)
    return steps, unbatched


def clean_to_noisy_token_stages(
    noisy_to_clean_slot_stages: torch.Tensor,
    chunk_size: int,
    *,
    batch_size: int | None = None,
) -> torch.Tensor:
    """Map a noisy-to-clean slot schedule to clean-to-noisy action tokens.

    Args:
        noisy_to_clean_slot_stages: One value per slot, shape ``[N]``, in the
            conventional inference-schedule order (noisiest to cleanest).
        chunk_size: Number of action tokens in each slot (``C``).
        batch_size: If provided, expand the result to ``[B, N*C]``.

    Returns:
        Stage values in buffer order, shape ``[N*C]`` or ``[B, N*C]``. Values
        are repeated for all tokens within a slot.
    """

    if not isinstance(noisy_to_clean_slot_stages, torch.Tensor):
        raise TypeError("`noisy_to_clean_slot_stages` must be a torch.Tensor")
    if noisy_to_clean_slot_stages.ndim != 1:
        raise ValueError(
            "`noisy_to_clean_slot_stages` must have shape [N], got "
            f"{tuple(noisy_to_clean_slot_stages.shape)}"
        )
    num_slots = int(noisy_to_clean_slot_stages.shape[0])
    chunk_size = _positive_int(chunk_size, "chunk_size (C)")
    validate_streaming_config(num_slots, chunk_size, num_slots * chunk_size)

    token_stages = noisy_to_clean_slot_stages.flip(0).repeat_interleave(chunk_size)
    batch_size = _optional_batch_size(batch_size)
    if batch_size is not None:
        token_stages = token_stages.unsqueeze(0).expand(batch_size, -1)
    return token_stages


def cold_start_token_values(
    clean_to_noisy_slot_values: torch.Tensor,
    chunk_size: int,
    step: int | torch.Tensor,
    *,
    padding_value: int | float | torch.Tensor | None = None,
    batch_size: int | None = None,
) -> torch.Tensor:
    """Build left-aligned cold-start token values from a slot schedule.

    This helper is deliberately value-agnostic and can be used for both the
    per-slot timestep schedule and the per-slot Euler delta schedule. At
    zero-based cold-start ``step=k``, ``k+1`` slots are active and receive the
    suffix of the clean-to-noisy schedule::

        step 0: [value_noisiest, P, ..., P]
        step 1: [value_{N-2}, value_noisiest, P, ..., P]
        ...
        step N-1: [value_cleanest, ..., value_noisiest]

    ``step`` may be scalar or ``[B]``. A scalar result is ``[N*C]`` unless
    ``batch_size`` is supplied; batched results are ``[B, N*C]``.
    """

    if not isinstance(clean_to_noisy_slot_values, torch.Tensor):
        raise TypeError("`clean_to_noisy_slot_values` must be a torch.Tensor")
    if clean_to_noisy_slot_values.ndim != 1:
        raise ValueError(
            "`clean_to_noisy_slot_values` must have shape [N], got "
            f"{tuple(clean_to_noisy_slot_values.shape)}"
        )

    num_slots = int(clean_to_noisy_slot_values.shape[0])
    chunk_size = _positive_int(chunk_size, "chunk_size (C)")
    validate_streaming_config(num_slots, chunk_size, num_slots * chunk_size)
    steps, unbatched = _normalize_steps(
        step,
        batch_size=batch_size,
        device=clean_to_noisy_slot_values.device,
    )

    num_real = (steps + 1).clamp(min=0, max=num_slots)
    slot_index = torch.arange(num_slots, device=clean_to_noisy_slot_values.device)
    source_index = num_slots - num_real[:, None] + slot_index[None, :]
    source_index = source_index.clamp(min=0, max=num_slots - 1)
    slot_values = clean_to_noisy_slot_values.unsqueeze(0).expand(steps.shape[0], -1)
    active_values = slot_values.gather(1, source_index)
    active_mask = slot_index[None, :] < num_real[:, None]

    if padding_value is None:
        padding = clean_to_noisy_slot_values[-1]
    else:
        padding = torch.as_tensor(
            padding_value,
            device=clean_to_noisy_slot_values.device,
            dtype=clean_to_noisy_slot_values.dtype,
        )
        if padding.numel() != 1:
            raise ValueError("`padding_value` must be scalar")
        padding = padding.reshape(())

    cold_slot_values = torch.where(active_mask, active_values, padding)
    token_values = cold_slot_values.repeat_interleave(chunk_size, dim=1)
    return token_values[0] if unbatched else token_values


def slot_block_causal_action_mask(
    num_slots: int,
    chunk_size: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return a boolean slot-block-causal mask with shape ``[N*C, N*C]``.

    Rows are queries and columns are keys. Tokens within the same slot attend
    bidirectionally, while a later (noisier) slot may attend every earlier
    (cleaner) slot. Earlier slots cannot attend later slots.
    """

    num_slots = _positive_int(num_slots, "num_slots (N)")
    chunk_size = _positive_int(chunk_size, "chunk_size (C)")
    horizon = num_slots * chunk_size
    validate_streaming_config(num_slots, chunk_size, horizon)
    token_slots = torch.arange(num_slots, device=device).repeat_interleave(chunk_size)
    query_slots = token_slots[:, None]
    key_slots = token_slots[None, :]
    return key_slots <= query_slots


def cold_start_valid_mask(
    step: int | torch.Tensor,
    num_slots: int,
    chunk_size: int,
    *,
    batch_size: int | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the real-token prefix mask for a zero-based cold-start step.

    Step 0 has one valid slot, step ``N-1`` has all ``N`` slots, negative
    steps have no valid slots, and later steps remain fully valid.
    """

    num_slots = _positive_int(num_slots, "num_slots (N)")
    chunk_size = _positive_int(chunk_size, "chunk_size (C)")
    horizon = num_slots * chunk_size
    validate_streaming_config(num_slots, chunk_size, horizon)

    if device is None and isinstance(step, torch.Tensor):
        target_device = step.device
    else:
        target_device = torch.device("cpu") if device is None else torch.device(device)
    steps, unbatched = _normalize_steps(
        step,
        batch_size=batch_size,
        device=target_device,
    )
    num_real = (steps + 1).clamp(min=0, max=num_slots)
    token_slots = torch.arange(num_slots, device=target_device).repeat_interleave(chunk_size)
    valid_mask = token_slots[None, :] < num_real[:, None]
    return valid_mask[0] if unbatched else valid_mask


def _validate_fp32_buffer(buffer: torch.Tensor) -> tuple[int, int, int]:
    if not isinstance(buffer, torch.Tensor):
        raise TypeError("`buffer` must be a torch.Tensor")
    if buffer.ndim != 3:
        raise ValueError(f"`buffer` must have shape [B, H, A], got {tuple(buffer.shape)}")
    batch_size, horizon, action_dim = (int(size) for size in buffer.shape)
    if batch_size <= 0 or horizon <= 0 or action_dim <= 0:
        raise ValueError(f"`buffer` dimensions must be non-zero, got {tuple(buffer.shape)}")
    if buffer.dtype != torch.float32:
        raise TypeError(f"`buffer` must be FP32, got {buffer.dtype}")
    return batch_size, horizon, action_dim


def _normalize_valid_mask(
    valid_mask: torch.Tensor,
    *,
    batch_size: int,
    horizon: int,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(valid_mask, torch.Tensor):
        raise TypeError("`valid_mask` must be a torch.Tensor")
    if valid_mask.dtype != torch.bool:
        raise TypeError(f"`valid_mask` must have bool dtype, got {valid_mask.dtype}")
    if valid_mask.device != device:
        raise ValueError(
            f"`valid_mask` and buffer must share a device, got {valid_mask.device} and {device}"
        )
    if valid_mask.shape == (horizon,):
        return valid_mask.unsqueeze(0).expand(batch_size, -1)
    if valid_mask.shape != (batch_size, horizon):
        raise ValueError(
            "`valid_mask` must have shape [H] or [B, H], got "
            f"{tuple(valid_mask.shape)} for buffer {(batch_size, horizon)}"
        )
    return valid_mask


def _validate_new_noise(
    new_noise: torch.Tensor,
    *,
    batch_size: int,
    chunk_size: int,
    action_dim: int,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(new_noise, torch.Tensor):
        raise TypeError("`new_noise` must be a torch.Tensor")
    expected_shape = (batch_size, chunk_size, action_dim)
    if new_noise.shape != expected_shape:
        raise ValueError(
            f"`new_noise` must have shape {expected_shape}, got {tuple(new_noise.shape)}"
        )
    if new_noise.device != device:
        raise ValueError(
            f"`new_noise` and buffer must share a device, got {new_noise.device} and {device}"
        )
    if not torch.is_floating_point(new_noise):
        raise TypeError(f"`new_noise` must have a floating dtype, got {new_noise.dtype}")
    return new_noise.to(dtype=torch.float32)


@dataclass(frozen=True, slots=True)
class StreamingActionState:
    """Caller-owned state carried between streaming inference calls."""

    buffer: torch.Tensor
    valid_mask: torch.Tensor
    steps: torch.Tensor
    schedule_shift: float | None = None

    def __post_init__(self) -> None:
        batch_size, horizon, _ = _validate_fp32_buffer(self.buffer)
        if self.valid_mask.shape != (batch_size, horizon):
            raise ValueError(
                "`valid_mask` must have shape [B, H] matching buffer, got "
                f"{tuple(self.valid_mask.shape)} and {tuple(self.buffer.shape)}"
            )
        if self.valid_mask.dtype != torch.bool:
            raise TypeError(f"`valid_mask` must have bool dtype, got {self.valid_mask.dtype}")
        if self.steps.shape != (batch_size,):
            raise ValueError(
                f"`steps` must have shape [B]={batch_size}, got {tuple(self.steps.shape)}"
            )
        if self.steps.dtype != torch.int64:
            raise TypeError(f"`steps` must have int64 dtype, got {self.steps.dtype}")
        if self.valid_mask.device != self.buffer.device or self.steps.device != self.buffer.device:
            raise ValueError("`buffer`, `valid_mask`, and `steps` must share a device")
        if self.schedule_shift is not None:
            if isinstance(self.schedule_shift, bool) or not isinstance(self.schedule_shift, Real):
                raise TypeError(
                    "`schedule_shift` must be a real number or None, got "
                    f"{type(self.schedule_shift).__name__}"
                )
            schedule_shift = float(self.schedule_shift)
            if not math.isfinite(schedule_shift) or schedule_shift <= 0:
                raise ValueError(
                    "`schedule_shift` must be finite and > 0 when provided, got "
                    f"{schedule_shift}"
                )
            object.__setattr__(self, "schedule_shift", schedule_shift)


def initialize_streaming_action_state(
    initial_noise: torch.Tensor,
    num_slots: int,
    chunk_size: int,
    *,
    schedule_shift: float | None = None,
) -> StreamingActionState:
    """Create step-0 padded state with noise in slot 0 and zeros elsewhere."""

    num_slots = _positive_int(num_slots, "num_slots (N)")
    chunk_size = _positive_int(chunk_size, "chunk_size (C)")
    horizon = num_slots * chunk_size
    validate_streaming_config(num_slots, chunk_size, horizon)
    if not isinstance(initial_noise, torch.Tensor):
        raise TypeError("`initial_noise` must be a torch.Tensor")
    if initial_noise.ndim != 3:
        raise ValueError(
            f"`initial_noise` must have shape [B, C, A], got {tuple(initial_noise.shape)}"
        )
    batch_size, noise_chunk, action_dim = (int(size) for size in initial_noise.shape)
    if batch_size <= 0 or action_dim <= 0:
        raise ValueError(f"`initial_noise` dimensions must be non-zero, got {tuple(initial_noise.shape)}")
    if noise_chunk != chunk_size:
        raise ValueError(
            f"`initial_noise` chunk dimension must equal C={chunk_size}, got {noise_chunk}"
        )
    if not torch.is_floating_point(initial_noise):
        raise TypeError(f"`initial_noise` must have a floating dtype, got {initial_noise.dtype}")

    first_slot = initial_noise.to(dtype=torch.float32)
    padding = torch.zeros(
        (batch_size, horizon - chunk_size, action_dim),
        dtype=torch.float32,
        device=initial_noise.device,
    )
    buffer = torch.cat((first_slot, padding), dim=1)
    valid_mask = cold_start_valid_mask(
        0,
        num_slots,
        chunk_size,
        batch_size=batch_size,
        device=initial_noise.device,
    )
    steps = torch.zeros((batch_size,), dtype=torch.int64, device=initial_noise.device)
    return StreamingActionState(
        buffer=buffer,
        valid_mask=valid_mask,
        steps=steps,
        schedule_shift=schedule_shift,
    )


def update_action_buffer(
    buffer: torch.Tensor,
    token_delta: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply an externally computed per-token update as ``x_next = x + delta``.

    ``token_delta`` must have shape ``[B, H, A]``. It may use a lower-precision
    model dtype; the addition and returned buffer are always FP32. If supplied,
    ``valid_mask`` prevents padded cold-start tokens from changing.
    """

    batch_size, horizon, action_dim = _validate_fp32_buffer(buffer)
    if not isinstance(token_delta, torch.Tensor):
        raise TypeError("`token_delta` must be a torch.Tensor")
    if token_delta.shape != buffer.shape:
        raise ValueError(
            f"`token_delta` must match buffer shape {tuple(buffer.shape)}, got {tuple(token_delta.shape)}"
        )
    if token_delta.device != buffer.device:
        raise ValueError(
            f"`token_delta` and buffer must share a device, got {token_delta.device} and {buffer.device}"
        )
    if not torch.is_floating_point(token_delta):
        raise TypeError(f"`token_delta` must have a floating dtype, got {token_delta.dtype}")

    updated = buffer + token_delta.to(dtype=torch.float32)
    if valid_mask is None:
        return updated
    normalized_mask = _normalize_valid_mask(
        valid_mask,
        batch_size=batch_size,
        horizon=horizon,
        device=buffer.device,
    )
    return torch.where(normalized_mask.unsqueeze(-1), updated, buffer)


def append_cold_start_noise(
    buffer: torch.Tensor,
    new_noise: torch.Tensor,
    valid_mask: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append noise to the first invalid cold-start slot without shifting.

    The valid mask is expected to be slot-aligned. A full buffer is returned
    unchanged, making this function safe when the caller tensor-selects between
    cold-start and steady-state paths.
    """

    batch_size, horizon, action_dim = _validate_fp32_buffer(buffer)
    chunk_size = _positive_int(chunk_size, "chunk_size (C)")
    if horizon % chunk_size != 0:
        raise ValueError(f"Buffer horizon H={horizon} must be divisible by C={chunk_size}")
    num_slots = horizon // chunk_size
    validate_streaming_config(num_slots, chunk_size, horizon)
    normalized_mask = _normalize_valid_mask(
        valid_mask,
        batch_size=batch_size,
        horizon=horizon,
        device=buffer.device,
    )
    noise_fp32 = _validate_new_noise(
        new_noise,
        batch_size=batch_size,
        chunk_size=chunk_size,
        action_dim=action_dim,
        device=buffer.device,
    )

    slot_valid = normalized_mask.reshape(batch_size, num_slots, chunk_size).all(dim=2)
    invalid_slots = ~slot_valid
    has_invalid = invalid_slots.any(dim=1)
    first_invalid = invalid_slots.to(dtype=torch.int64).argmax(dim=1)
    slot_index = torch.arange(num_slots, device=buffer.device)
    target_slots = (slot_index[None, :] == first_invalid[:, None]) & has_invalid[:, None]
    target_tokens = target_slots.repeat_interleave(chunk_size, dim=1)
    tiled_noise = (
        noise_fp32[:, None, :, :]
        .expand(-1, num_slots, -1, -1)
        .reshape(batch_size, horizon, action_dim)
    )
    next_buffer = torch.where(target_tokens.unsqueeze(-1), tiled_noise, buffer)
    next_valid_mask = normalized_mask | target_tokens
    return next_buffer, next_valid_mask


def emit_shift_append_action_buffer(
    updated_buffer: torch.Tensor,
    new_noise: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Emit the clean slot, shift left, and append one new noisy slot.

    Returns:
        ``(emitted, next_buffer)`` with shapes ``[B, C, A]`` and
        ``[B, N*C, A]`` respectively. Both outputs are FP32.
    """

    batch_size, horizon, action_dim = _validate_fp32_buffer(updated_buffer)
    chunk_size = _positive_int(chunk_size, "chunk_size (C)")
    if horizon % chunk_size != 0:
        raise ValueError(f"Buffer horizon H={horizon} must be divisible by C={chunk_size}")
    num_slots = horizon // chunk_size
    validate_streaming_config(num_slots, chunk_size, horizon)
    noise_fp32 = _validate_new_noise(
        new_noise,
        batch_size=batch_size,
        chunk_size=chunk_size,
        action_dim=action_dim,
        device=updated_buffer.device,
    )

    emitted = updated_buffer[:, :chunk_size, :].clone()
    next_buffer = torch.cat((updated_buffer[:, chunk_size:, :], noise_fp32), dim=1)
    return emitted, next_buffer
