import time
import torch

x = torch.randn(5000, 5000, device="cuda")
y = torch.randn(5000, 5000, device="cuda")

torch.cuda.synchronize()
t0 = time.time()

for _ in range(20):
    z = x @ y

torch.cuda.synchronize()

print("Elapsed:", time.time() - t0)