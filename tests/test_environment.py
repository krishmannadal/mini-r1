"""
Mini-R1 — Environment Tests
============================
Minimal pytest suite to verify that the development environment is correctly
configured before any training code is written.

These tests are intentionally simple — they validate preconditions, not model logic.

Usage:
    pytest tests/test_environment.py -v
"""

import pathlib
import sys

import pytest


# ---------------------------------------------------------------------------
# 1. Core dependency tests
# ---------------------------------------------------------------------------

class TestDependencies:
    """Verify that required packages are importable."""

    def test_python_version(self):
        """We require Python 3.10+ for modern syntax and typing features."""
        assert sys.version_info >= (3, 10), (
            f"Python 3.10+ required, got {sys.version_info.major}.{sys.version_info.minor}"
        )

    def test_torch_import(self):
        """PyTorch must be importable — it's the only ML framework we use."""
        import torch  # noqa: F811
        assert torch.__version__, "PyTorch imported but version string is empty"

    def test_yaml_import(self):
        """PyYAML is needed to load configs/base.yaml."""
        import yaml  # noqa: F811
        assert yaml.__version__, "PyYAML imported but version string is empty"


# ---------------------------------------------------------------------------
# 2. CUDA availability tests
# ---------------------------------------------------------------------------

class TestCUDA:
    """Verify GPU access. These tests are skipped if CUDA is unavailable,
    so they won't break CI or CPU-only development — but they MUST pass
    before training."""

    @pytest.fixture(autouse=True)
    def _require_cuda(self):
        """Skip all tests in this class if CUDA is not available."""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available — install CUDA-enabled PyTorch")

    def test_cuda_device_count(self):
        """At least one GPU must be visible to PyTorch."""
        import torch
        assert torch.cuda.device_count() >= 1, "No CUDA devices found"

    def test_gpu_name(self):
        """Sanity check: the GPU should have a non-empty name string."""
        import torch
        name = torch.cuda.get_device_name(0)
        assert len(name) > 0, "GPU name is empty"
        print(f"GPU: {name}")

    def test_tensor_on_cuda(self):
        """Verify we can allocate a tensor on GPU memory.
        This is the most basic proof that the CUDA runtime works end-to-end:
        driver -> runtime -> memory allocation."""
        import torch
        x = torch.randn(64, 64, device="cuda")
        assert x.device.type == "cuda"
        del x
        torch.cuda.empty_cache()

    def test_cuda_matmul(self):
        """Verify GPU compute works, not just memory allocation.
        Matrix multiplication exercises the tensor cores / CUDA cores
        that we'll rely on for every forward and backward pass."""
        import torch
        a = torch.randn(128, 128, device="cuda")
        b = torch.randn(128, 128, device="cuda")
        c = a @ b
        assert c.shape == (128, 128)
        assert c.device.type == "cuda"
        del a, b, c
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 3. Configuration tests
# ---------------------------------------------------------------------------

class TestConfig:
    """Verify the project config file exists and has the expected structure."""

    CONFIG_PATH = pathlib.Path(__file__).parent.parent / "configs" / "base.yaml"

    def test_config_file_exists(self):
        """base.yaml must exist — all future scripts will load it."""
        assert self.CONFIG_PATH.exists(), f"Config not found at {self.CONFIG_PATH}"

    def test_config_loads(self):
        """Verify the YAML is syntactically valid and contains expected top-level keys."""
        import yaml
        with open(self.CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)

        assert isinstance(config, dict), "Config should parse to a dict"

        # These keys are the minimal contract for Task 1
        expected_keys = ["project", "seed", "device", "dtype", "paths"]
        for key in expected_keys:
            assert key in config, f"Missing required config key: '{key}'"

    def test_config_paths_section(self):
        """The paths section must define output directories for future tasks."""
        import yaml
        with open(self.CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)

        paths = config["paths"]
        for dir_key in ["checkpoints", "outputs", "logs"]:
            assert dir_key in paths, f"Missing paths.{dir_key} in config"
