from __future__ import annotations

import unittest

import torch

from fastwam.models.wan22.streaming_action import (
    StreamingActionState,
    append_cold_start_noise,
    clean_to_noisy_token_stages,
    cold_start_token_values,
    cold_start_valid_mask,
    emit_shift_append_action_buffer,
    initialize_streaming_action_state,
    slot_block_causal_action_mask,
    update_action_buffer,
    validate_streaming_config,
)


class StreamingActionConfigTest(unittest.TestCase):
    def test_validates_n_c_h(self) -> None:
        self.assertEqual(validate_streaming_config(4, 2, 8), (4, 2, 8))

        for args in ((0, 2, 0), (4, -1, -4), (4, 2, 7)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                validate_streaming_config(*args)

        with self.assertRaises(TypeError):
            validate_streaming_config(True, 2, 2)

    def test_maps_noisy_to_clean_schedule_into_clean_to_noisy_tokens(self) -> None:
        noisy_to_clean = torch.tensor([40.0, 30.0, 20.0, 10.0])
        expected = torch.tensor([10.0, 10.0, 20.0, 20.0, 30.0, 30.0, 40.0, 40.0])
        actual = clean_to_noisy_token_stages(noisy_to_clean, chunk_size=2)
        torch.testing.assert_close(actual, expected)

        batched = clean_to_noisy_token_stages(noisy_to_clean, chunk_size=2, batch_size=3)
        self.assertEqual(batched.shape, (3, 8))
        torch.testing.assert_close(batched, expected.expand(3, -1))

    def test_cold_start_values_use_left_aligned_schedule_suffix(self) -> None:
        clean_to_noisy = torch.tensor([10.0, 20.0, 30.0, 40.0])
        expected_by_step = (
            [40.0, 40.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            [30.0, 30.0, 40.0, 40.0, -1.0, -1.0, -1.0, -1.0],
            [20.0, 20.0, 30.0, 30.0, 40.0, 40.0, -1.0, -1.0],
            [10.0, 10.0, 20.0, 20.0, 30.0, 30.0, 40.0, 40.0],
        )
        for step, expected in enumerate(expected_by_step):
            with self.subTest(step=step):
                actual = cold_start_token_values(
                    clean_to_noisy,
                    chunk_size=2,
                    step=step,
                    padding_value=-1.0,
                )
                torch.testing.assert_close(actual, torch.tensor(expected))

        batched = cold_start_token_values(
            clean_to_noisy,
            chunk_size=2,
            step=torch.tensor([0, 2]),
            padding_value=-1.0,
        )
        self.assertEqual(batched.shape, (2, 8))
        torch.testing.assert_close(batched[0], torch.tensor(expected_by_step[0]))
        torch.testing.assert_close(batched[1], torch.tensor(expected_by_step[2]))


class StreamingActionMaskTest(unittest.TestCase):
    def test_slot_block_causal_mask_has_bidirectional_diagonal_blocks(self) -> None:
        mask = slot_block_causal_action_mask(num_slots=3, chunk_size=2)
        expected = torch.tensor(
            [
                [1, 1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
            ],
            dtype=torch.bool,
        )
        self.assertEqual(mask.dtype, torch.bool)
        self.assertTrue(torch.equal(mask, expected))

    def test_cold_start_valid_mask_supports_scalar_and_batched_steps(self) -> None:
        torch.testing.assert_close(
            cold_start_valid_mask(0, num_slots=4, chunk_size=2),
            torch.tensor([1, 1, 0, 0, 0, 0, 0, 0], dtype=torch.bool),
        )
        torch.testing.assert_close(
            cold_start_valid_mask(2, num_slots=4, chunk_size=2),
            torch.tensor([1, 1, 1, 1, 1, 1, 0, 0], dtype=torch.bool),
        )
        batched = cold_start_valid_mask(
            torch.tensor([0, 3]),
            num_slots=4,
            chunk_size=2,
        )
        self.assertEqual(batched.shape, (2, 8))
        self.assertEqual(batched[0].sum().item(), 2)
        self.assertEqual(batched[1].sum().item(), 8)


class StreamingActionStateTest(unittest.TestCase):
    def test_initial_state_is_fp32_noise_plus_zero_padding(self) -> None:
        initial_noise = torch.tensor([[[1.0], [2.0]]], dtype=torch.float64)
        state = initialize_streaming_action_state(
            initial_noise,
            num_slots=4,
            chunk_size=2,
            schedule_shift=5,
        )

        self.assertEqual(state.buffer.dtype, torch.float32)
        self.assertEqual(state.buffer.shape, (1, 8, 1))
        torch.testing.assert_close(state.buffer[0, :2, 0], torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(state.buffer[0, 2:, 0], torch.zeros(6))
        self.assertTrue(torch.equal(state.valid_mask, torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0]], dtype=torch.bool)))
        self.assertTrue(torch.equal(state.steps, torch.zeros(1, dtype=torch.int64)))
        self.assertEqual(state.schedule_shift, 5.0)

    def test_state_validates_optional_schedule_shift(self) -> None:
        state = initialize_streaming_action_state(
            torch.zeros((1, 2, 1)),
            num_slots=2,
            chunk_size=2,
        )
        self.assertIsNone(state.schedule_shift)

        for schedule_shift in (0.0, -1.0, float("nan"), float("inf"), float("-inf")):
            with self.subTest(schedule_shift=schedule_shift), self.assertRaises(ValueError):
                StreamingActionState(
                    buffer=state.buffer,
                    valid_mask=state.valid_mask,
                    steps=state.steps,
                    schedule_shift=schedule_shift,
                )

        with self.assertRaises(TypeError):
            StreamingActionState(
                buffer=state.buffer,
                valid_mask=state.valid_mask,
                steps=state.steps,
                schedule_shift=True,
            )

        with self.assertRaises(TypeError):
            StreamingActionState(
                buffer=state.buffer,
                valid_mask=state.valid_mask,
                steps=state.steps.to(torch.int32),
                schedule_shift=None,
            )

    def test_masked_update_is_fp32_and_does_not_mutate_inputs(self) -> None:
        buffer = torch.zeros((1, 4, 1), dtype=torch.float32)
        delta = torch.ones((1, 4, 1), dtype=torch.float64)
        valid_mask = torch.tensor([1, 1, 0, 0], dtype=torch.bool)
        buffer_before = buffer.clone()
        delta_before = delta.clone()

        updated = update_action_buffer(buffer, delta, valid_mask)

        self.assertEqual(updated.dtype, torch.float32)
        torch.testing.assert_close(updated[0, :, 0], torch.tensor([1.0, 1.0, 0.0, 0.0]))
        torch.testing.assert_close(buffer, buffer_before)
        torch.testing.assert_close(delta, delta_before)

    def test_n4_c2_full_cold_start_then_emit_shift_append_sequence(self) -> None:
        num_slots, chunk_size = 4, 2
        noises = [
            torch.tensor([[[10.0], [11.0]]], dtype=torch.float64),
            torch.tensor([[[20.0], [21.0]]], dtype=torch.float64),
            torch.tensor([[[30.0], [31.0]]], dtype=torch.float64),
            torch.tensor([[[40.0], [41.0]]], dtype=torch.float64),
            torch.tensor([[[50.0], [51.0]]], dtype=torch.float64),
        ]
        state = initialize_streaming_action_state(noises[0], num_slots, chunk_size)
        one = torch.ones_like(state.buffer)

        expected_cold_buffers = (
            [11.0, 12.0, 20.0, 21.0, 0.0, 0.0, 0.0, 0.0],
            [12.0, 13.0, 21.0, 22.0, 30.0, 31.0, 0.0, 0.0],
            [13.0, 14.0, 22.0, 23.0, 31.0, 32.0, 40.0, 41.0],
        )
        for cold_call, expected in enumerate(expected_cold_buffers):
            updated = update_action_buffer(state.buffer, one, state.valid_mask)
            next_buffer, next_valid_mask = append_cold_start_noise(
                updated,
                noises[cold_call + 1],
                state.valid_mask,
                chunk_size,
            )
            state = StreamingActionState(
                buffer=next_buffer,
                valid_mask=next_valid_mask,
                steps=state.steps + 1,
            )
            with self.subTest(cold_call=cold_call + 1):
                torch.testing.assert_close(state.buffer[0, :, 0], torch.tensor(expected))
                self.assertEqual(state.valid_mask.sum().item(), (cold_call + 2) * chunk_size)
                self.assertEqual(state.steps.item(), cold_call + 1)

        # The Nth call sees a full buffer, advances every stage, emits slot 0,
        # shifts the other slots toward clean, and appends a new noisy slot.
        self.assertTrue(state.valid_mask.all())
        updated = update_action_buffer(state.buffer, one, state.valid_mask)
        emitted, next_buffer = emit_shift_append_action_buffer(
            updated,
            noises[4],
            chunk_size,
        )
        state = StreamingActionState(
            buffer=next_buffer,
            valid_mask=torch.ones_like(state.valid_mask),
            steps=state.steps + 1,
        )

        torch.testing.assert_close(emitted[0, :, 0], torch.tensor([14.0, 15.0]))
        torch.testing.assert_close(
            state.buffer[0, :, 0],
            torch.tensor([23.0, 24.0, 32.0, 33.0, 41.0, 42.0, 50.0, 51.0]),
        )
        self.assertEqual(emitted.dtype, torch.float32)
        self.assertEqual(state.buffer.dtype, torch.float32)
        self.assertEqual(state.steps.item(), num_slots)

    def test_buffer_helpers_reject_non_fp32_state(self) -> None:
        buffer = torch.zeros((1, 4, 1), dtype=torch.float64)
        with self.assertRaises(TypeError):
            update_action_buffer(buffer, torch.zeros_like(buffer))


if __name__ == "__main__":
    unittest.main()
