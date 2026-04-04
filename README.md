<div align="center">
    <h1>MS-Temba: Multi-Scale Temporal Mamba for Efficient Temporal Action Detection</h1>
    <a href="https://arxiv.org/abs/2501.06138" target="_blank">
        <img src="https://img.shields.io/badge/arXiv-2501.06138-B31B1B?style=flat-square" alt="arXiv">
    </a>
</div>

<div align="center">
<img src="assets/Teaser.png" />
</div>

## Abstract
Temporal Action Detection (TAD) in untrimmed videos requires models that can efficiently (1) process long-duration videos, (2) capture temporal variations within action classes, and (3) handle dense, overlapping actions, all while remaining suitable for resource-constrained edge deployment. While Transformer-based methods achieve high accuracy, their quadratic complexity hinders deployment in such scenarios. Given the recent popularity of linear complexity Mamba-based models, leveraging them for TAD is a natural choice. However, naively adapting Mamba from language or vision tasks fails to provide an optimal solution and does not address the challenges of long, untrimmed videos. Therefore, we propose <b>Multi-Scale Temporal Mamba (MS-Temba)</b>, the first Mamba-based architecture specifically designed for densely labeled TAD tasks. MS-Temba features Temporal Mamba Blocks (Temba Blocks), consisting of Temporal Convolutional
Module (TCM) and Dilated SSM (D-SSM). TCM captures short-term dependencies using dilated convolutions, while D-SSM introduces a novel dilated state-space mechanism tomodel long-range temporal relationships effectively at each temporal scale. These multi-scale representations are aggregated by Scale-Aware State Fuser, which learns a unified representation for detecting densely overlapping actions. Experiments show that MS-Temba achieves state-of-the-art performance on long-duration videos, remains competitive on shorter segments, and reduces model complexity by 88%. Its efficiency and effectiveness make MS-Temba well-suited for real-world edge deployment.

## Prepare the environment

- Python 3.10.13

  - `conda create -n your_env_name python=3.10.13`

- torch 2.1.1 + cu118
  - `pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118`

- Requirements: vim_requirements.txt
  - `pip install -r vim/vim_requirements.txt`

- Install ``causal_conv1d`` and ``mamba``
  - `pip install -e causal_conv1d>=1.1.0`
  - `pip install -e mamba-1p1p1`

## Prepare the Input Video Features
Like the previous works (e.g. MS-TCT, PDAN), MS-Temba is built on top of the pre-trained video features. Thus, feature extraction is needed before training the network. We train MS-Temba on features extracted using I3D and CLIP backbones.

- Please download the Charades dataset (24 fps version) from this [link](https://prior.allenai.org/projects/charades).
- Please download the Toyota Smarthome Untrimmed dataset from this [link](https://project.inria.fr/toyotasmarthome/).
- Please download the MultiTHUMOS dataset from this [link](http://ai.stanford.edu/~syyeung/everymoment.html).

### I3D features
Follow this [repository](https://github.com/piergiaj/pytorch-i3d) to extract the snippet-level I3D feature. 

### CLIP features
To extract the CLIP features, first extract frames from each video using ffmpeg and save the frames in a directory. Then, please follow the instructions in `vim/clip_feature_extraction.py` to extract the CLIP features.

## Train MS-Temba
We provide the training scripts for Charades, TSU, and MultiTHUMOS datasets in `vim/scripts/`. Please update the paths in the scripts to match the ones on your machine. Modify the argument `-backbone` to set it to `i3d` or `clip` based on the feature extractor backbone used.

For example to train MS-Temba on TSU dataset, run:

`bash vim/scripts/run_MSTemba_TSU.sh`

# Citation
If you use our approach (code or methods) in your research, please consider citing:
```
@article{sinha2025ms,
  title={MS-Temba: Multi-Scale Temporal Mamba for Efficient Temporal Action Detection},
  author={Sinha, Arkaprava and Raj, Monish Soundar and Wang, Pu and Helmy, Ahmed and Das, Srijan},
  journal={arXiv preprint arXiv:2501.06138},
  year={2025}
}
```

## Acknowledgement 
This project is based on Mamba ([paper](https://arxiv.org/abs/2312.00752), [code](https://github.com/state-spaces/mamba)), Vision-Mamba ([paper](), [code](https://github.com/hustvl/Vim)), MS-TCT ([paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Dai_MS-TCT_Multi-Scale_Temporal_ConvTransformer_for_Action_Detection_CVPR_2022_paper.pdf), [code](https://github.com/dairui01/MS-TCT)). Thanks for their wonderful works.


