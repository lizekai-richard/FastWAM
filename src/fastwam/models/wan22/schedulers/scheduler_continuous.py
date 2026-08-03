import math

import torch


class WanContinuousFlowMatchScheduler:
    """Continuous-time Flow-Matching scheduler with shift-based sampling."""

    def __init__(self, num_train_timesteps: int = 1000, shift: float = 5.0, eps: float = 1e-10):
        if num_train_timesteps <= 0:
            raise ValueError(f"`num_train_timesteps` must be positive, got {num_train_timesteps}")
        shift = float(shift)
        if not math.isfinite(shift) or shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}")
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = shift
        self.eps = float(eps)
        self._y_min, self._weight_norm_const = self._precompute_training_weight_stats()

    @staticmethod
    def _phi(u: torch.Tensor, shift: float) -> torch.Tensor:
        return shift * u / (1.0 + (shift - 1.0) * u)

    def _precompute_training_weight_stats(self) -> tuple[float, float]:
        steps = self.num_train_timesteps
        u_grid = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)[:-1]
        t_grid = self._phi(u_grid, self.shift) * float(steps)
        y_grid = torch.exp(-2.0 * ((t_grid - (steps / 2.0)) / steps) ** 2)
        y_min = float(y_grid.min().item())
        y_shifted_grid = y_grid - y_min
        norm_const = float(y_shifted_grid.mean().item())
        return y_min, norm_const

    def sample_training_t(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError(f"`batch_size` must be positive, got {batch_size}")
        u = torch.rand((batch_size,), device=device, dtype=torch.float32)
        sigma = self._phi(u, self.shift)
        timestep = sigma * float(self.num_train_timesteps)
        return timestep.to(dtype=dtype)

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        t = timestep.to(dtype=torch.float32)
        steps = float(self.num_train_timesteps)
        y = torch.exp(-2.0 * ((t - (steps / 2.0)) / steps) ** 2)
        y_shifted = y - self._y_min
        weight = y_shifted / (self._weight_norm_const + self.eps)
        if weight.numel() == 1:
            return weight.reshape(())
        return weight

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        sigma = (timestep / float(self.num_train_timesteps)).to(
            original_samples.device, dtype=original_samples.dtype
        )
        if sigma.ndim == 0:
            return (1 - sigma) * original_samples + sigma * noise
        sigma = sigma.view(-1, *([1] * (original_samples.ndim - 1)))
        return (1 - sigma) * original_samples + sigma * noise

    @staticmethod
    def training_target(sample: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        del timestep
        return noise - sample

    def build_inference_schedule(
        self,
        num_inference_steps: int,
        device: torch.device,
        dtype: torch.dtype,
        shift_override: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be positive, got {num_inference_steps}")
        shift = self.shift if shift_override is None else float(shift_override)
        if not math.isfinite(shift) or shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}")

        u_steps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device, dtype=torch.float32)
        sigma_steps = self._phi(u_steps, shift)
        timesteps = sigma_steps[:-1] * float(self.num_train_timesteps)
        deltas = sigma_steps[1:] - sigma_steps[:-1]
        return timesteps.to(dtype=dtype), deltas.to(dtype=dtype)

    def build_streaming_slot_schedule(
        self,
        num_slots: int,
        device: torch.device,
        dtype: torch.dtype,
        shift_override: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build one denoising timestep and delta for each streaming slot.

        Slots are ordered from cleanest to noisiest, the opposite of the
        sequential inference traversal.  The delta at each slot remains paired
        with its timestep and advances that slot toward the next cleaner state.
        """
        timesteps, deltas = self.build_inference_schedule(
            num_inference_steps=num_slots,
            device=device,
            dtype=dtype,
            shift_override=shift_override,
        )
        return timesteps.flip(0), deltas.flip(0)

    @staticmethod
    def step(model_output: torch.Tensor, delta: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        delta = delta.to(sample.device, dtype=sample.dtype)
        if delta.ndim == 0:
            return sample + model_output * delta
        delta = delta.view(-1, *([1] * (sample.ndim - 1)))
        return sample + model_output * delta

    @staticmethod
    def step_per_token(
        model_output: torch.Tensor,
        delta_per_token: torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """Apply token-specific Euler deltas to samples shaped ``[B, T, ...]``."""
        if sample.ndim < 2:
            raise ValueError(
                f"`sample` must have shape [B, T, ...], got {tuple(sample.shape)}"
            )
        if model_output.shape != sample.shape:
            raise ValueError(
                "`model_output` shape must match `sample`, "
                f"got {tuple(model_output.shape)} vs {tuple(sample.shape)}"
            )

        batch_size, num_tokens = sample.shape[:2]
        if delta_per_token.ndim == 1:
            if delta_per_token.shape[0] != num_tokens:
                raise ValueError(
                    f"1D `delta_per_token` must have shape [T]=[{num_tokens}], "
                    f"got {tuple(delta_per_token.shape)}"
                )
            delta_shape = (1, num_tokens) + (1,) * (sample.ndim - 2)
        elif delta_per_token.ndim == 2:
            if delta_per_token.shape != (batch_size, num_tokens):
                raise ValueError(
                    "2D `delta_per_token` must have shape [B, T]="
                    f"[{batch_size}, {num_tokens}], got {tuple(delta_per_token.shape)}"
                )
            delta_shape = (batch_size, num_tokens) + (1,) * (sample.ndim - 2)
        else:
            raise ValueError(
                "`delta_per_token` must have shape [T] or [B, T], "
                f"got {tuple(delta_per_token.shape)}"
            )

        delta = delta_per_token.to(sample.device, dtype=sample.dtype).reshape(delta_shape)
        return sample + model_output * delta
