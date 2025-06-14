# DualMamba
**TGRS2024 DualMamba: A Lightweight Spectral–Spatial Mamba-Convolution Network for Hyperspectral Image Classification**
[Link](https://ieeexplore.ieee.org/abstract/document/10798573)

## Installation
Follow the installation instructions of the [VMamba](https://github.com/VMamba/VMamba) project to set up the `rs_mamba` environment.

## Dataset Directory Structure
Please organize your dataset as follows:
```
├── datasets
│ ├── ip
│ ├── whulk
│ ├── hu2018
│ ├── pu
```

## Model Training
To train DualMamba, run the following command:
```bash
python main.py --dataset_name ip --model asf_rsm_group

```


## Citation
If this code repository is helpful to you, please cite the paper using the BibTeX entry below:
```bibtex
@ARTICLE{dualmamba,
  author={Sheng, Jiamu and Zhou, Jingyi and Wang, Jiong and Ye, Peng and Fan, Jiayuan},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={DualMamba: A Lightweight Spectral–Spatial Mamba-Convolution Network for Hyperspectral Image Classification}, 
  year={2025},
  volume={63},
  number={},
  pages={1-15},
  doi={10.1109/TGRS.2024.3516817}
}
```
