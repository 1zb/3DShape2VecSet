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

## :e-mail: Contact

Contact [Biao Zhang](mailto:biao.zhang@kaust.edu.sa) ([@1zb](https://github.com/1zb)) if you have any further questions. This repository is for academic research use only.

## :blue_book: Citation

```bibtex
@article{zhang20233dshape2vecset,
  title={3{DShape2VecSet}: A 3d shape representation for neural fields and generative diffusion models},
  author={Zhang, Biao and Tang, Jiapeng and Niessner, Matthias and Wonka, Peter},
  journal={arXiv preprint arXiv:2301.11445},
  year={2023}
}
```
