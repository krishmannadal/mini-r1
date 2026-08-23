"""
Mini-R1 -- GPU & Environment Diagnostic Script
===============================================
Verifies that the runtime environment is correctly configured for GPU training.
Run this BEFORE any training to catch issues early (wrong PyTorch build, missing
CUDA, driver mismatch, etc.).

Usage:
    python scripts/gpu_check.py
"""

import sys
import time


def separator(title: str) -> None:
    """Print a visual section separator for readability."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def check_python() -> None:
    """Report Python version. We need 3.10+ for modern type hints and match statements."""
    separator("Python Environment")
    print(f"  Python version  : {sys.version}")
    print(f"  Python executable: {sys.executable}")

    major, minor = sys.version_info[:2]
    if major < 3 or minor < 10:
        print("  [WARN] Python 3.10+ recommended for this project.")
    else:
        print("  [OK] Python version OK")


def check_pytorch() -> dict:
    """
    Import PyTorch and report its version and build configuration.
    Returns a dict with 'torch' module and 'cuda_available' flag for downstream checks.
    """
    separator("PyTorch")
    try:
        import torch
    except ImportError:
        print("  [FAIL] PyTorch is NOT installed.")
        print("  Install with: pip install torch --index-url https://download.pytorch.org/whl/cu124")
        return {"torch": None, "cuda_available": False}

    print(f"  PyTorch version : {torch.__version__}")

    # The '+cpu' suffix means CUDA kernels are missing from this build
    if "+cpu" in torch.__version__:
        print("  [FAIL] This is a CPU-ONLY PyTorch build!")
        print("  You must reinstall with CUDA support.")
        print("  Run: pip install torch --index-url https://download.pytorch.org/whl/cu124")
        return {"torch": torch, "cuda_available": False}

    cuda_available = torch.cuda.is_available()
    print(f"  CUDA available  : {cuda_available}")

    if cuda_available:
        print(f"  CUDA version    : {torch.version.cuda}")
        print(f"  cuDNN version   : {torch.backends.cudnn.version()}")
        print(f"  cuDNN enabled   : {torch.backends.cudnn.enabled}")
    else:
        print("  [FAIL] CUDA is not available even though PyTorch has CUDA build.")
        print("    Possible causes: no NVIDIA driver, driver mismatch, or WSL issue.")

    return {"torch": torch, "cuda_available": cuda_available}


def check_gpu(torch_module) -> None:
    """Report GPU hardware details: name, memory, compute capability, device count."""
    separator("GPU Hardware")
    torch = torch_module

    device_count = torch.cuda.device_count()
    print(f"  CUDA device count: {device_count}")

    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        total_mem_gb = props.total_memory / (1024 ** 3)
        print(f"\n  --- Device {i} ---")
        print(f"  Name             : {props.name}")
        print(f"  Compute capability: {props.major}.{props.minor}")
        print(f"  Total memory     : {total_mem_gb:.2f} GB")
        print(f"  Multi-processors : {props.multi_processor_count}")

        # Ada Lovelace = SM 8.9, Ampere = SM 8.x, Turing = SM 7.5
        if props.major >= 8:
            print("  [OK] Architecture supports both FP16 and BF16")
        elif props.major == 7 and props.minor >= 5:
            print("  [OK] FP16 supported; BF16 may not be natively supported")
        else:
            print(f"  [WARN] Older GPU architecture (SM {props.major}.{props.minor})")


def check_tensor_allocation(torch_module) -> bool:
    """
    Verify that we can actually allocate a tensor on the GPU.
    This catches driver/runtime mismatches that pass the 'is_available' check
    but fail when real GPU memory is requested.
    """
    separator("CUDA Tensor Allocation")
    torch = torch_module

    try:
        # Allocate a small tensor on GPU -- if this fails, nothing else will work
        x = torch.randn(256, 256, device="cuda")
        print(f"  [OK] Allocated tensor on CUDA: shape={x.shape}, dtype={x.dtype}")

        # Verify the tensor is actually on the GPU, not silently on CPU
        assert x.device.type == "cuda", "Tensor ended up on CPU!"
        print(f"  [OK] Tensor confirmed on device: {x.device}")

        del x
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"  [FAIL] Failed to allocate tensor on CUDA: {e}")
        return False


def check_matmul(torch_module) -> bool:
    """
    Run a matrix multiplication on GPU to verify compute kernels work.
    This is the fundamental operation inside every Transformer layer
    (attention scores, linear projections, FFN), so if this fails, training will fail.
    """
    separator("CUDA Matrix Multiplication")
    torch = torch_module

    try:
        size = 1024
        a = torch.randn(size, size, device="cuda")
        b = torch.randn(size, size, device="cuda")

        # Warm-up pass: first CUDA kernel launch has overhead from JIT compilation
        _ = a @ b
        torch.cuda.synchronize()

        # Timed pass
        start = time.perf_counter()
        c = a @ b
        torch.cuda.synchronize()  # Wait for GPU to finish before timing
        elapsed_ms = (time.perf_counter() - start) * 1000

        print(f"  [OK] Matmul ({size}x{size}) completed in {elapsed_ms:.2f} ms")
        print(f"  Result shape: {c.shape}, device: {c.device}")

        del a, b, c
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"  [FAIL] Matmul failed: {e}")
        return False


def check_mixed_precision(torch_module) -> None:
    """
    Test FP16 and BF16 autocast independently.

    Why both? FP16 has been standard for mixed-precision training for years
    (wider hardware support), but BF16 has better numerical range (same exponent
    bits as FP32) which avoids loss scaling entirely. Modern GPUs (Ampere+)
    support both, but we must verify at runtime, not assume.

    We do NOT choose the final dtype here -- that is a design decision for a future task.
    """
    separator("Mixed Precision Support")
    torch = torch_module

    # --- FP16 ---
    print("\n  [FP16 - float16]")
    try:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            a = torch.randn(256, 256, device="cuda")
            b = a @ a
            # autocast should have promoted matmul to use FP16 kernels
            print(f"    [OK] FP16 autocast works")
            print(f"    Output dtype inside autocast: {b.dtype}")
        del a, b
    except Exception as e:
        print(f"    [FAIL] FP16 autocast failed: {e}")

    # --- BF16 ---
    print("\n  [BF16 - bfloat16]")

    # First check PyTorch's built-in detection
    bf16_supported = torch.cuda.is_bf16_supported()
    print(f"    torch.cuda.is_bf16_supported(): {bf16_supported}")

    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            a = torch.randn(256, 256, device="cuda")
            b = a @ a
            print(f"    [OK] BF16 autocast works")
            print(f"    Output dtype inside autocast: {b.dtype}")
        del a, b
    except Exception as e:
        print(f"    [FAIL] BF16 autocast failed: {e}")

    # --- Summary ---
    print("\n  [Summary]")
    print("    Both FP16 and BF16 availability reported above.")
    print("    Final dtype selection will be made in a future task.")

    torch.cuda.empty_cache()


def check_memory_snapshot(torch_module) -> None:
    """Report current GPU memory usage after all tests, so we know our baseline footprint."""
    separator("GPU Memory After Tests")
    torch = torch_module

    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)

    print(f"  Allocated : {allocated:.1f} MB")
    print(f"  Reserved  : {reserved:.1f} MB")
    print(f"  Total VRAM: {total:.1f} MB")
    print(f"  Free      : {total - reserved:.1f} MB (approximate)")


def main() -> None:
    print("+============================================================+")
    print("|           Mini-R1 -- Environment Diagnostic                |")
    print("+============================================================+")

    check_python()
    result = check_pytorch()

    torch = result["torch"]
    if torch is None:
        print("\n[FAIL] PyTorch not installed. Cannot continue GPU checks.")
        sys.exit(1)

    if not result["cuda_available"]:
        print("\n[FAIL] CUDA not available. GPU checks skipped.")
        print("  If you have an NVIDIA GPU, reinstall PyTorch with CUDA support:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu124")
        sys.exit(1)

    check_gpu(torch)

    alloc_ok = check_tensor_allocation(torch)
    if not alloc_ok:
        print("\n[FAIL] Cannot allocate tensors on GPU. Stopping.")
        sys.exit(1)

    matmul_ok = check_matmul(torch)
    if not matmul_ok:
        print("\n[FAIL] GPU matmul failed. Stopping.")
        sys.exit(1)

    check_mixed_precision(torch)
    check_memory_snapshot(torch)

    separator("Result")
    print("  [OK] All GPU checks passed.")
    print("  Environment is ready for training.")


if __name__ == "__main__":
    main()
