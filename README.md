# 3DShape2VecSet: A 3D Shape Representation for Neural Fields and Generative Diffusion Models (SIGGRAPH 2023)

### [Project Page](https://1zb.github.io/3DShape2VecSet/) | [Paper (arXiv)](https://arxiv.org/abs/2301.11445)

**This repository is the official pytorch implementation of  *3DShape2VecSet (https://arxiv.org/abs/2301.11445)*.**

[Biao Zhang](https://1zb.github.io/)<sup>1</sup>,
[Jiapeng Tang](https://tangjiapeng.github.io/)<sup>2</sup>
[Matthias Niessner](https://www.niessnerlab.org/)<sup>2</sup>
[Peter Wonka](http://peterwonka.net/)<sup>1</sup>,<br>
<sup>1</sup>KAUST, <sup>2</sup>Technical University of Munich

## :bullettrain_front: Training
Download the preprocessed data from [here](https://drive.google.com/drive/folders/1UFPi_UklH5clWKxxeL1IsxfjdUfc7i4x). In case that this link is inaccessable, send [me](mailto:biao.zhang@kaust.edu.sa) an email for the data. Uncompress `occupancies.zip` and `surfaces.zip` to somewhere in your hard disk. They are required in the training phase.

### First stage (autoencoder):
```
torchrun \
    --nproc_per_node=4 main_ae.py \
    --accum_iter=2 \
    --model ae_d512_m512  \
    --output_dir output/ae/ae_d512_m512 \
    --log_dir output/ae/ae_d512_m512 \
    --num_workers 60 \
    --point_cloud_size 2048 \
    --batch_size 64 \
    --epochs 200 \
    --warmup_epochs 5
```
```
python eval.py
```

### Second stage (category-conditioned generation):
```
torchrun \
    --nproc_per_node=4 main_class_cond.py \
    --accum_iter 2 \
    --model kl_d512_m512_l8_d24_edm \
    --ae kl_d512_m512_l8 \
    --ae-pth output/ae/kl_d512_m512_l8/checkpoint-199.pth \
    --output_dir output/dm/kl_d512_m512_l8_d24_edm \
    --log_dir output/dm/kl_d512_m512_l8_d24_edm \
    --num_workers 64 \
    --point_cloud_size 2048 \
    --batch_size 64 \
    --epochs 1000 \
    --data_path ~/data/
```

## :balloon: Sampling
This will generate 1000 samples of the cateogry chair.
```
python sample_class_cond.py \
    --ae kl_d512_m512_l8 \
    --ae-pth output/ae/kl_d512_m512_l8/checkpoint-199.pth \
    --dm kl_d512_m512_l8_d24_edm \
    --dm-pth output/class_cond_dm/kl_d512_m512_l8_d24_edm/checkpoint-499.pth
```

Pretrained model can be found [here](https://drive.google.com/drive/folders/1tX4pFulWqtICYgchRXmzscHDRJ5q2iSz?usp=sharing).

## :scroll: Data Processing
For data processing, please look at this repository:
[https://github.com/1zb/sdf_gen](https://github.com/1zb/sdf_gen)

## :e-mail: Contact

Contact [Biao Zhang](mailto:biao.zhang@kaust.edu.sa) ([@1zb](https://github.com/1zb)) if you have any further questions. This repository is for academic research use only.

## :blue_book: Citation

```bibtex
@article{10.1145/3592442,
author = {Zhang, Biao and Tang, Jiapeng and Nie\ss{}ner, Matthias and Wonka, Peter},
title = {3DShape2VecSet: A 3D Shape Representation for Neural Fields and Generative Diffusion Models},
year = {2023},
issue_date = {August 2023},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {42},
number = {4},
issn = {0730-0301},
url = {https://doi.org/10.1145/3592442},
doi = {10.1145/3592442},
abstract = {We introduce 3DShape2VecSet, a novel shape representation for neural fields designed for generative diffusion models. Our shape representation can encode 3D shapes given as surface models or point clouds, and represents them as neural fields. The concept of neural fields has previously been combined with a global latent vector, a regular grid of latent vectors, or an irregular grid of latent vectors. Our new representation encodes neural fields on top of a set of vectors. We draw from multiple concepts, such as the radial basis function representation, and the cross attention and self-attention function, to design a learnable representation that is especially suitable for processing with transformers. Our results show improved performance in 3D shape encoding and 3D shape generative modeling tasks. We demonstrate a wide variety of generative applications: unconditioned generation, category-conditioned generation, text-conditioned generation, point-cloud completion, and image-conditioned generation. Code: https://1zb.github.io/3DShape2VecSet/.},
journal = {ACM Trans. Graph.},
month = {jul},
articleno = {92},
numpages = {16},
keywords = {3D shape generation, generative models, shape reconstruction, 3D shape representation, diffusion models}
}
```
