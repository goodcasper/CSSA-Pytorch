# Assuming your model is saved in cssa_module.py
from cssa_module import CSSA
import torch

# Dummy input feature maps from two modalities (e.g., RGB and IR)
RGB = torch.randn(4, 64, 32, 32)  # modality A
IR = torch.randn(4, 64, 32, 32)  # modality B

# Initialize CSSA fusion module
cssa = CSSA()

# Forward pass to fuse features
fused_output = cssa(RGB, IR)

print(fused_output.shape)  # Output: (4, 64, 32, 32)