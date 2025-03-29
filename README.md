# CSSA: Channel Switching and Spatial Attention for Multimodal Object Detection

This repository contains an implementation of the **CVPR 2023 paper**:  
**"Multimodal Object Detection by Channel Switching and Spatial Attention"** by Yue Cao et al.

> 📄 [Paper PDF](https://openaccess.thecvf.com/content/CVPR2023W/PBVS/papers/Cao_Multimodal_Object_Detection_by_Channel_Switching_and_Spatial_Attention_CVPRW_2023_paper.pdf)

---

## Overview
This project implements **CSSA** (Channel Switching and Spatial Attention), a lightweight yet effective multimodal fusion module for object detection using paired RGB and IR images. The method is designed to fuse complementary information from different modalities while maintaining low computational cost, suitable for real-time applications.

---

## Implementation

The paper proposes a two-part fusion strategy:

- **Channel Switching**: Uses Efficient Channel Attention (ECA) to selectively replace uninformative channels of one modality with the corresponding channels from the other modality.
- **Spatial Attention**: Applies average and max pooling to emphasize important spatial locations without introducing additional parameters.

<img width="750" alt="image" src="https://github.com/user-attachments/assets/1b627b60-64c7-4c94-a450-d2c9bcd616ca" />

> The following diagram is taken from the CVPR 2023 paper
“Multimodal Object Detection by Channel Switching and Spatial Attention”

In this implementation, we only provide the CSSA module (as illustrated in the figure abov). The full training code is not included.

---

## Included Models

- ECA: Efficient Channel Attention module that computes channel-wise importance using lightweight 1D convolution on global average pooled features.

- Channel_Switching: Replaces less informative channels from one modality with more informative channels from the other, guided by ECA weights.

- Spatial_Attention: Applies both average and max pooling along the channel dimension to compute spatial weights, and uses them to fuse modality-specific features.

- CSSA: The main fusion module that sequentially applies channel_switching and spatial_attention to combine two modalities into a unified representation.

Note: This repository includes only the core fusion modules and does not include the complete object detection pipeline.

---

## Usage Example
```python
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
```
### Input Format
- `RGB`: Tensor of shape `(B, C, H, W)` representing modality A feature maps (e.g., RGB)
- `IR`: Tensor of shape `(B, C, H, W)` representing modality B feature maps (e.g., IR)

### Output Format
- A fused feature map tensor of shape `(B, C, H, W)`

---

## Dependencies
To install all required packages, use the following command:

```bash
pip install -r requirements.txt
```

---

## Citation
```bibtex
@INPROCEEDINGS{10209020,
  author={Cao, Yue and Bin, Junchi and Hamari, Jozsef and Blasch, Erik and Liu, Zheng},
  booktitle={2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)}, 
  title={Multimodal Object Detection by Channel Switching and Spatial Attention}, 
  year={2023},
  volume={},
  number={},
  pages={403-411},
  keywords={Computer vision;Fuses;Computational modeling;Conferences;Object detection;Switches;Stability analysis},
  doi={10.1109/CVPRW59228.2023.00046}}

```





