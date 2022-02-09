<div align="center">
  <h1>Knowledge Transfer Based Hierarchical Few-shot
Learning via Tree-structured Knowledge Graph</h1>
</div>


## :heavy_check_mark: Requirements
* Ubuntu 18.04.5
* Python 3.6.4
* [CUDA 9.1](https://developer.nvidia.com/cuda-toolkit)
* [PyTorch 1.6.0](https://pytorch.org)


## :gear: Conda environmnet installation
```bash
conda env create --name HFKT --file environment.yml
conda activate HFKT
```

## :books: Datasets
```bash
cd datasets
bash download_miniimagenet.sh
bash download_cub.sh
bash download_cifar_fs.sh
bash download_tieredimagenet.sh
```
    
   
## :pushpin: Quick start: testing scripts
To test in the 5-way K-shot setting:
```bash
bash scripts/test/{dataset_name}_5wKs.sh
```
For example, to test HFKT on the CIFAR-FS dataset in the 5-way 1-shot setting:
```bash
bash scripts/test/cifar_fs_5w1s.sh
```

## :fire: Training scripts
To train in the 5-way K-shot setting:
```bash
bash scripts/train/{dataset_name}_5wKs.sh
```
For example, to train HFKT on the CIFAR-FS dataset in the 5-way 1-shot setting:
```bash
bash scripts/train/cifar_fs_5w1s.sh
```

## :mag: Related repos
Our project references the codes in the following repos:

* Kang _et al_., [ReNet](https://github.com/dahyun-kang/renet).
* Zhang _et al_., [DeepEMD](https://github.com/icoz69/DeepEMD).
* Ye _et al_., [FEAT](https://github.com/Sha-Lab/FEAT)
* Wang _et al_., [Non-local neural networks](https://github.com/AlexHex7/Non-local_pytorch)
* Ramachandran _et al_., [Stand-alone self-attention](https://github.com/leaderj1001/Stand-Alone-Self-Attention)
* Huang _et al_., [DCCNet](https://github.com/ShuaiyiHuang/DCCNet)
* Yang _et al_., [VCN](https://github.com/gengshan-y/VCN)

## :love_letter: Acknowledgement
We adopted the main code bases from [ReNet](https://github.com/dahyun-kang/renet), and we really appreciate it :smiley:.

