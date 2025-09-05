import torch
print(torch.__version__)          # Should show: 2.8.0+cpu
print(torch.cuda.is_available())  # Should be False (expected for CPU-only)