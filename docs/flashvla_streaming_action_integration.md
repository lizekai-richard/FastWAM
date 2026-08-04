# FlashVLA-style streaming action integration

This integration is opt-in. The default FastWAM path remains the legacy
multi-step action denoiser:

```yaml
model:
  streaming_action:
    enabled: false
    num_slots: 8
    chunk_size: 4
```

`8 x 4` preserves FastWAM's current 32-action horizon. Changing either value
changes the training and checkpoint contract; it is not an inference-only
tuning knob.

The current implementation intentionally targets base `FastWAM` only.
`FastWAMJoint` and `FastWAMIDM` reject this switch until their paired video and
action streaming/noising contract is implemented separately.

## Runtime contract

Streaming state belongs to the rollout, not to the model singleton. Each
episode/environment owns an independent state with:

- an FP32 rolling action buffer `[B, num_slots * chunk_size, action_dim]`;
- a boolean validity mask `[B, num_slots * chunk_size]`;
- an `int64` step counter `[B]`;
- the effective scalar `schedule_shift`, which prevents a rollout from silently
  changing schedules while buffered latents are mid-denoise;
- an episode-owned random generator used for initial and appended noise.

Reset the state and generator at an episode boundary. Do not store them globally
on `FastWAM`, share them between environments, or recreate the generator from
the same seed on every inference call.

The benchmark uses the following explicit model protocol:

```python
result = model.infer_action_streaming(
    ..., streaming_state=state, action_generator=episode_generator
)
# Either:
#   {"action": action_or_none, "streaming_state": next_state}
# Or:
#   ({"action": action_or_none}, next_state)
```

For a two-stage latency report the model must also expose
`infer_action_streaming_profiled(..., latency_recorder=recorder)` and record
exactly `video_kv_prefill` and `action_prediction`. Missing APIs or stages are
errors; the benchmark does not infer a breakdown from end-to-end latency.

## Cold start and scheduling

At reset, only the noisiest slot is initialized and the remaining slots are
padded. Cold start fills/refines one additional slot per inference call. The
first `num_slots - 1` calls may return `action=None`; the controller must issue
a task-safe hold chunk of `chunk_size` environment actions so cold start keeps
the same control-call cadence as steady state. The `num_slots`-th call produces
the first real chunk.
The latency benchmark discards this cold-start phase and reports steady-state
latency separately.

FastWAM uses a shifted flow schedule (`infer_shift=5`), so slot timesteps and
deltas must come from the scheduler's streaming slot schedule. Do not replace
them with uniform `1 / num_slots` Euler steps copied from the original
FlashVLA Pi0.5 implementation. Streaming mode also requires equal action
training and inference shifts so slot identities retain one schedule contract.

## Checkpoints

An existing legacy FastWAM checkpoint is only an initialization. Correct
streaming behavior requires training or fine-tuning with staggered per-slot
timesteps, the streaming action mask, padded cold-start configurations, and
the same slot schedule used at inference. A latency run cannot establish task
quality or checkpoint compatibility; closed-loop RoboTwin/LIBERO evaluation is
required before deployment.

New checkpoints record the objective version, `num_slots`, `chunk_size`, the
training/inference shifts, and action timestep scale. Loading checks those
fields against an enabled streaming config. RoboTwin deployment additionally
rejects a legacy checkpoint in streaming mode and rejects a streaming-trained
checkpoint when the runtime switch was accidentally left off. The latency
benchmark remains able to run a legacy checkpoint for implementation timing
only and labels that result as not quality-compatible.

Accelerate full-state resumes record and validate the same contract in
`trainer_state.json` before restoring model/optimizer state. Older state
directories without that metadata cannot resume a streaming run; use their
`.pt` weights as initialization instead.

The first integration stage intentionally freezes the pretrained video expert
and trains ActionDiT plus the optional proprio encoder. Streaming training uses
the clean first-frame video KV prefix only and has no video reconstruction
objective; freezing avoids unused trainable video-head parameters in DDP/ZeRO
while preserving the existing visual representation. Jointly adapting the
video prefix would require an explicit auxiliary video loss and is outside this
base FastWAM action-streaming stage.

Trainer validation continues to render the legacy joint video rollout as a
frozen-video diagnostic, but suppresses its action L1/L2 because that one-shot
action path is not the streaming policy. Use streaming validation loss and
closed-loop evaluation for action quality.

## Latency benchmark

Legacy two-stage report:

```bash
python experiments/benchmark/benchmark_inference_latency.py \
  --num-views 3 --inference-mode legacy --latency-breakdown
```

Streaming steady-state report, after a streaming-trained checkpoint and the
runtime API above are available:

```bash
python experiments/benchmark/benchmark_inference_latency.py \
  --num-views 3 --inference-mode streaming --latency-breakdown \
  --checkpoint /path/to/streaming-trained.pt
```

Add `--torch-compile` to benchmark FlashVLA-style compiled streaming action
kernels. The benchmark primes every cold-start phase and steady state before
recording canonical latency, so first-use compilation is excluded.

RoboTwin deployment is enabled end to end with the regular Hydra entrypoint:

```bash
python experiments/robotwin/eval_robotwin_single.py \
  ckpt=/path/to/streaming-trained.pt \
  EVALUATION.task_name=click_alarmclock \
  model.streaming_action.enabled=true \
  model.torch_compile_infer_action=true
```

Both reports use the same boundaries:

1. `video_kv_prefill`: image transfer, VAE encode, conditioning, video token
   preparation, and video KV-cache materialization.
2. `action_prediction`: legacy full denoising loop, or one streaming rolling-
   buffer prediction/update. Output device-to-host transfer is excluded.

## `torch.compile` boundary

Legacy inference retains whole-method `infer_action` compilation. Streaming
mode follows FlashVLA's different structure: the public
`infer_action_streaming` dispatcher remains eager, while fixed-shape
`_streaming_action_cold_start_kernel` and
`_streaming_action_steady_kernel` methods are compiled separately. The two
kernels cover ActionDiT/MoT action prediction, the FP32 scheduler update, and
the cold append or steady emit/shift/append transition. Caller-owned random
sampling, state validation, prompt/proprio preparation, and output D2H remain
outside the graph.

Wan VAE encoding and video KV prefill intentionally remain eager. The current
VAE mutates Python feature-cache lists during `encode`, so placing it inside a
CUDA graph would introduce graph breaks or unsafe replay. This boundary also
lets the two-stage profiler put CUDA events around the same compiled action
kernel used by canonical streaming latency.

The default streaming compile settings match FlashVLA: `max-autotune`, static
shape specialization, and CUDA graphs enabled. Batch size, image resolution,
context length, proprio presence, action horizon, and dtype should therefore
stay fixed during a deployment. RoboTwin performs `num_slots + 1` dummy calls
at startup to capture one cold graph and warm two steady invocations, then
resets rollout state and RNG before the first episode.
