import sys
import platform
import torch

print("=" * 80)
print("PYTHON ENVIRONMENT")
print("=" * 80)
print(f"Python executable: {sys.executable}")
print(f"Python version:    {sys.version}")
print(f"Platform:          {platform.platform()}")

print("\n" + "=" * 80)
print("PYTORCH / CUDA")
print("=" * 80)
print(f"PyTorch version:   {torch.__version__}")
print(f"CUDA available:    {torch.cuda.is_available()}")
print(f"CUDA version:      {torch.version.cuda}")

if torch.cuda.is_available():
    print(f"GPU count:         {torch.cuda.device_count()}")
    print(f"GPU name:          {torch.cuda.get_device_name(0)}")
    print(f"Current device:    {torch.cuda.current_device()}")

    x = torch.randn(2000, 2000, device="cuda")
    y = torch.randn(2000, 2000, device="cuda")
    z = x @ y
    torch.cuda.synchronize()

    print(f"GPU tensor test:   PASSED")
    print(f"Tensor device:     {z.device}")
else:
    print("GPU tensor test:   SKIPPED — CUDA not available")

print("\nDone.")