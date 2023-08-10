# UMC ![Static Badge](https://img.shields.io/badge/Progress-20%25-yellow) ![Static Badge](https://img.shields.io/badge/ICCV-2023-critical)



**UMC: A Unified Bandwidth-efficient and Multi-resolution based Collaborative Perception Framework**

[Tianhang Wang](https://github.com/TianhangWang), [Guang Chen](https://ispc-group.github.io/)†, [Kai Chen](https://github.com/Flawless1202), [Zhengfa Liu](https://liuzhengfa88.github.io/), Bo Zhang, Alois Knoll, Changjun Jiang

#### **[Paper (arXiv)](https://arxiv.org/abs/2303.12400) | [Paper (ICCV)](https://arxiv.org/abs/2303.12400) | [Project Page](https://tianhangwang.github.io/UMC/)  | [Video](https://arxiv.org/abs/2303.12400) | [Talk](https://arxiv.org/abs/2303.12400) | [Slides](https://arxiv.org/abs/2303.12400) | [Poster](https://arxiv.org/abs/2303.12400)**  

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#changelog">Changelog</a>
    </li>
    <li>
      <a href="#introduction">Introduction</a>
    </li>
    <li>
      <a href="#get-started">Get Started</a>
    </li>
    <li>
      <a href="#dataset">Datasets</a>
    </li>
    <li>
      <a href="#pretrained-model">Pretrained Model</a>
    </li>
    <li>
      <a href="#visualization">Visualization</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
    <li>
      <a href="#acknowledgements">Acknowledgements</a>
    </li>
  </ol>
</details>

## Changelog  
* 2023-8-10: We release the project page.
* 2023-7-14: This paper is accepted by **ICCV 2023** 🎉🎉.

## Introduction

* This repository is the PyTorch implementation ![Static Badge](https://img.shields.io/badge/Pytorch-v1.7.0|v.1.7.1-blue) under ![Static Badge](https://img.shields.io/badge/CUDA-11.0-green) for **UMC**. 

* We aim to propose a **U**nified **C**ollaborative perception framework named
**UMC**, optimizing the communication, collaboration, and reconstruction processes with the **M**ulti-resolution technique.

<p align="center">
<img src="img\intro.png" width=90% >
</p>

- The communication introduces a novel trainable multi-resolution and selective-region (MRSR) mechanism, achieving higher quality and lower bandwidth. Then, a graph-based collaboration is proposed, conducting on each resolution to adapt the MRSR. Finally, the reconstruction integrates the multi-resolution collaborative features for downstream tasks.

<p align="center">
<img src="img\framework.png" width=90% >
</p>

## Get Started

## Datasets

## Pretrained Model

## Visualization

* Detection and communication selection for Agent 1. The green and red boxes represent the ground truth (GT) and predictions, respectively. (a-c) shows the results of no fusion, early fusion, and UMC compared to GT. (d) The coarse-grained collaborative feature of Agent 1. (e) Matrix-valued entropy-based selected communication coarse-grained feature map from Agent 2. (f) The fine-grained collaborative feature of Agent 1. (g) Matrix-valued entropy-based selected communication fine-grained feature map from Agent 2.
<p align="center">
<img src="img\compare_2.png" width=90% >
</p>

* UMC qualitatively outperforms the state-of-the-art methods. The green and red boxes denote ground truth and detection, respectively. (a) Results of When2com. (b) Results of DiscoNet. (c) Results of UMC. (d)-(e) Agent 1's coarse-grained and fine-grained collaborative feature maps, respectively.

<p align="center">
<img src="img\compare_1.png" width=90% >
</p>

* Detection results of UMC, Early Fusion, When2com, V2VNet and DiscoNet on V2X-Sim dataset.

<p align="center">
<img src="img\v2xsim_results.png" width=90% >
</p>

* Detection results of UMC, Early Fusion, Where2comm, V2VNet and DiscoNet on OPV2V dataset.

<p align="center">
<img src="img\opv2v_results.png" width=90% >
</p>

## Citation

If you find our code or paper useful, please cite
```bibtex
@inproceedings{wang2023umc,
  title     = {UMC: A Unified Bandwidth-efficient and Multi-resolution based Collaborative Perception Framework},
  author    = {Tianhang, Wang and Guang, Chen and Kai, Chen and Zhengfa, Liu, Bo, Zhang, Alois, Knoll, Changjun, Jiang},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2023}
  }
```

## Acknowledgements

* We thanks for the following wonderful open source codes: ![Static Badge](https://img.shields.io/badge/DiscoNet|Yiming_Li-NIPS2021-blue) ![Static Badge](https://img.shields.io/badge/OpenC00D|Runsheng_Xu-ECCV2022-blue)

