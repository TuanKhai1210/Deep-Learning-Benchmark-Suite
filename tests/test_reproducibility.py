import json
import random
import unittest

import numpy as np
import torch

from dlbench.common.reproducibility import (
    collect_environment,
    seed_everything,
    seed_worker,
)


class ReproducibilityTests(unittest.TestCase):
    def test_seed_everything_uses_requested_seed(self):
        # 1. Gọi hàm cần kiểm tra với seed
        seed = 67
        seed_everything(seed, deterministic=False)
        actual_torch = torch.rand(1)
        actual_python = random.random()
        actual_numpy = np.random.rand(1)
        # 2. Tạo kết quả mong đợi bằng API trực tiếp
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        expected_torch = torch.rand(1)
        expected_python = random.random()
        expected_numpy = np.random.rand(1)

        # 3. Báo lỗi nếu không khớp
        self.assertEqual(actual_python, expected_python)
        self.assertTrue(torch.equal(actual_torch, expected_torch))
        self.assertTrue(np.array_equal(actual_numpy, expected_numpy))

    def test_seed_everything_toggles_deterministic_algorithms(self):
        previous_state = torch.are_deterministic_algorithms_enabled()

        try:
            seed_everything(69, deterministic=True)
            self.assertTrue(torch.are_deterministic_algorithms_enabled())

            seed_everything(69, deterministic=False)
            self.assertFalse(torch.are_deterministic_algorithms_enabled())
        finally:
            torch.use_deterministic_algorithms(previous_state)

    def test_seed_worker(self):
        torch.manual_seed(2**32 + 70)
        seed_worker(1)
        actual_numpy = np.random.rand(3)
        actual_python = random.random()

        np.random.seed(70)
        random.seed(70)

        expected_python = random.random()
        expected_numpy = np.random.rand(3)

        self.assertEqual(actual_python, expected_python)
        self.assertTrue(np.array_equal(actual_numpy, expected_numpy))

    def test_collect_environment(self):
        environment = collect_environment()
        previous_state = torch.are_deterministic_algorithms_enabled()

        try:
            seed_everything(69, deterministic=True)
            env_on = collect_environment()

            seed_everything(69, deterministic=False)
            env_off = collect_environment()

            self.assertTrue(env_on["deterministic_algorithms"])
            self.assertFalse(env_off["deterministic_algorithms"])
        finally:
            torch.use_deterministic_algorithms(previous_state)

        restored = json.loads(json.dumps(environment, allow_nan=False))
        self.assertEqual(restored, environment)

        environment = collect_environment()
        if not environment["cuda_available"]:
            self.assertEqual(environment["gpus"], [])
