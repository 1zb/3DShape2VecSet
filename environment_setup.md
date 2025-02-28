# Environment Setup for 3DShape2VecSet

This guide provides step-by-step instructions to set up the environment required to run the 3DShape2VecSet repository.

## Prerequisites
- Ensure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) installed.
- A compatible GPU with CUDA (12.4 used here) support is recommended.

## Steps to Set Up the Environment

### 1. Create and Activate a Conda Environment
```bash
conda create -n vecset python=3.12 -y
conda activate vecset
```

### 2. Install Required Packages
#### Install PyTorch with CUDA 12.4 Support
```bash
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
```

#### Install Additional Dependencies
```bash
pip install numpy==2.2.3
pip install pymcubes==0.1.6
pip install trimesh==4.6.2
pip install einops==0.8.1
pip install timm==1.0.14
```

#### Install PyTorch Geometric Cluster Library
```bash
curl -O https://data.pyg.org/whl/torch-2.5.0%2Bcu124/torch_cluster-1.6.3%2Bpt25cu124-cp312-cp312-linux_x86_64.whl
pip install torch_cluster-1.6.3+pt25cu124-cp312-cp312-linux_x86_64.whl
```

## Verify Installation
After installing the dependencies, verify the setup by running sampling code:
```
python sample_class_cond.py \
    --ae kl_d512_m512_l8 \
    --ae-pth output/ae/kl_d512_m512_l8/checkpoint-199.pth \
    --dm kl_d512_m512_l8_d24_edm \
    --dm-pth output/class_cond_dm/kl_d512_m512_l8_d24_edm/checkpoint-499.pth
```

Pretrained model can be found [here](https://drive.google.com/drive/folders/1tX4pFulWqtICYgchRXmzscHDRJ5q2iSz?usp=sharing).



