import sys

import torch
import transformers
import sentence_transformers
import fastapi
import uvicorn
import httpx
import pytest

print("Python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("sentence-transformers:", sentence_transformers.__version__)
print("fastapi:", fastapi.__version__)
print("uvicorn:", uvicorn.__version__)
print("httpx:", httpx.__version__)
print("pytest:", pytest.__version__)

# Tiny sanity check that PyTorch actually computes something
x = torch.tensor([1.0, 2.0, 3.0])
print("torch test (should be 6.0):", x.sum().item())