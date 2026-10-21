import json
import random
import unittest

import numpy as np
import torch

from dlbench.common.reproducibility import (
    RNG_STATE_KEYS,
    capture_rng_state,
    collect_environment,
    restore_rng_state,
    seed_everything,
    seed_worker,
)


class ReproducibilityTests(unittest.TestCase):
    def test_rng_state_roundtrip_restores_all_generators(self):
        previous_state = capture_rng_state()
        previous_deterministic = (
            torch.are_deterministic_algorithms_enabled()
        )

        try:
            seed_everything(69, deterministic=False)
            saved_state = capture_rng_state()

            expected_python = random.random()
            expected_numpy = np.random.rand(3)
            expected_torch = torch.rand(3)
            expected_cuda = [
                torch.rand(3, device=f"cuda:{index}")
                for index in range(torch.cuda.device_count())
            ]

            seed_everything(999, deterministic=False)
            restore_rng_state(saved_state)

            self.assertEqual(random.random(), expected_python)
            self.assertTrue(
                np.array_equal(np.random.rand(3), expected_numpy)
            )
            self.assertTrue(torch.equal(torch.rand(3), expected_torch))
            for index, expected in enumerate(expected_cuda):
                self.assertTrue(
                    torch.equal(
                        torch.rand(3, device=f"cuda:{index}"),
                        expected,
                    )
                )
            self.assertEqual(set(saved_state), RNG_STATE_KEYS)
        finally:
            restore_rng_state(previous_state)
            torch.use_deterministic_algorithms(previous_deterministic)

    def test_restore_rng_state_rejects_missing_keys(self):
        with self.assertRaises(ValueError):
            restore_rng_state({})

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

    def test_collect_environment_precision_fields(self):
        environment = collect_environment()
        self.assertIn("precision", environment)
        precision = environment["precision"]
        self.assertIn("float32_matmul_precision", precision)
        self.assertIn(precision["float32_matmul_precision"], ("highest", "high", "medium"))
        self.assertIn("cuda_matmul_allow_tf32", precision)
        if environment["cuda_available"]:
            self.assertIsInstance(precision["cuda_matmul_allow_tf32"], bool)
        else:
            self.assertIsNone(precision["cuda_matmul_allow_tf32"])

    def test_collect_environment_cudnn_fields(self):
        environment = collect_environment()
        self.assertIn("cudnn", environment)
        cudnn = environment["cudnn"]
        self.assertIn("available", cudnn)
        self.assertIn("enabled", cudnn)
        self.assertIn("version", cudnn)
        self.assertIn("allow_tf32", cudnn)
        self.assertIn("benchmark", cudnn)
        self.assertIn("deterministic", cudnn)
        if cudnn["available"]:
            self.assertIsInstance(cudnn["version"], int)
            self.assertIsInstance(cudnn["allow_tf32"], bool)
            self.assertIsInstance(cudnn["benchmark"], bool)
            self.assertIsInstance(cudnn["deterministic"], bool)
        else:
            self.assertIsNone(cudnn["version"])

    def test_collect_environment_system_ram(self):
        environment = collect_environment()
        self.assertIn("system_ram", environment)
        ram = environment["system_ram"]
        self.assertIn("total_bytes", ram)
        self.assertIn("total_gb", ram)
        if ram["total_bytes"] is not None:
            self.assertIsInstance(ram["total_bytes"], int)
            self.assertGreater(ram["total_bytes"], 0)
            self.assertIsInstance(ram["total_gb"], float)
            self.assertGreater(ram["total_gb"], 0)

    def test_collect_environment_gpu_vram_and_arch(self):
        environment = collect_environment()
        if not environment["cuda_available"]:
            return
        for gpu in environment["gpus"]:
            self.assertIn("total_memory_bytes", gpu)
            self.assertIn("total_memory_gb", gpu)
            self.assertIn("compute_capability", gpu)
            self.assertIn("multi_processor_count", gpu)
            self.assertIsInstance(gpu["total_memory_bytes"], int)
            self.assertGreater(gpu["total_memory_bytes"], 0)
            self.assertIsInstance(gpu["total_memory_gb"], float)
            self.assertGreater(gpu["total_memory_gb"], 0)
            self.assertRegex(gpu["compute_capability"], r"^\d+\.\d+$")
            self.assertIsInstance(gpu["multi_processor_count"], int)
            self.assertGreater(gpu["multi_processor_count"], 0)

