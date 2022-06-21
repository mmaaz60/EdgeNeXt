# EdgeNeXt
### **EdgeNeXt: Efficiently Amalgamated CNN-Transformer Architecture for Mobile Vision Applications**

[Muhammad Maaz](https://scholar.google.com/citations?user=vTy9Te8AAAAJ&hl=en&authuser=1&oi=sra), 
[Abdelrahman Shaker](https://scholar.google.com/citations?hl=en&user=eEz4Wu4AAAAJ),
[Hisham Cholakkal](https://scholar.google.com/citations?hl=en&user=bZ3YBRcAAAAJ),
[Salman Khan](https://salman-h-khan.github.io),
[Syed Waqas Zamir](https://www.waqaszamir.com),
[Rao Muhammad Anwer](https://scholar.google.com/citations?hl=en&authuser=1&user=_KlvMVoAAAAJ)
and [Fahad Shahbaz Khan](https://scholar.google.es/citations?user=zvaeYnUAAAAJ&hl=en)

### ****Paper**: comming soon**

## :rocket: News
* **(Jun 22, 2022)**
  * Training and evaluation code along with pre-trained models are released.

<hr />

![main figure](images/EdgeNext.png)
> **Abstract:** *In the pursuit of achieving ever-increasing accuracy, large and complex neural networks are usually developed. Such models demand high computational resources and therefore cannot be deployed on edge devices. It is of great interest to build resource-efficient general purpose networks due to their usefulness in several application areas. In this work, we strive to effectively combine the strengths of both CNN and Transformer models and propose a new efficient hybrid architecture EdgeNeXt. Specifically in EdgeNeXt, we introduce split depth-wise transpose attention (SDTA) encoder that splits input tensors into multiple channel groups and utilizes depth-wise convolution along with self-attention across channel dimensions to implicitly increase the receptive field and encode multi-scale features. Our extensive experiments on classification, detection and segmentation tasks, reveal the merits of the proposed approach, outperforming state-of-the-art methods with comparatively lower compute requirements. Our EdgeNeXt model with 1.3M parameters achieves 71.2\% top-1 accuracy on ImageNet-1K, outperforming MobileViT with an absolute gain of 2.2\% with 28\% reduction in FLOPs. Further, our EdgeNeXt model with 5.6M parameters achieves 79.4\% top-1 accuracy on ImageNet-1K.* 
<hr />

## Comparison with SOTA ViTs and Hybrid Architectures
![results](images/madds_vs_top_1.png)

<hr />

## Comparison with Previous SOTA [MobileViT (ICLR-2022)](https://arxiv.org/abs/2110.02178)
![results](images/table_2.png)

<hr />

## Qualitative Results (Segmentation)
![results](images/Segmentation.png)

## Installation
1. Create conda environment
```shell
conda create --name edgenext python=3.8
conda activate edgenext
```
2. Install PyTorch and torchvision
```shell
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
```
3. Install other dependencies
```shell
pip install -r requirements.txt
```

<hr />

## Dataset Preparation
Download the [ImageNet-1K](http://image-net.org/) classification dataset and structure the data as follows:
```
/path/to/imagenet-1k/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class2/
      img4.jpeg
```

<hr />

## Evaluation
Download the pretrained weights and run the following command for evaluation on ImageNet-1K dataset.

```shell
wget https://github.com/mmaaz60/EdgeNeXt/releases/download/v1.0/edgenext_small.pth
python main.py --model edgenext_small --eval True --batch_size 16 --data_path <path to imagenet> --output_dir <results> --resume edgenext_small.pth
```
This should give,
```text
Acc@1 79.412 Acc@5 94.512 loss 0.881
```
<hr />

## Training

On a single machine with 8 GPUs, run the following command to train EdgeNeXt-S model.

```shell
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model edgenext_small --drop_path 0.1 \
--batch_size 256 --lr 6e-3 --update_freq 2 \
--model_ema true --model_ema_eval true \
--data_path </path/to/imagenet-1k> \
--output_dir </path/to/save_results> \
--use_amp True --multi_scale_sampler
```
<hr />

## Model Zoo

| Name |Acc@1 | #Params | MAdds | Model |
|:---:|:---:|:---:| :---:|:---:|
| edgenext_small | 79.41 | 5.59M | 1.26G | [model](https://github.com/mmaaz60/EdgeNeXt/releases/download/v1.0/edgenext_small.pth)
| edgenext_x_small | 74.96 | 2.34M | 538M | [model](https://github.com/mmaaz60/EdgeNeXt/releases/download/v1.0/edgenext_x_small.pth)
| edgenext_xx_small | 71.23 | 1.33M | 261M | [model](https://github.com/mmaaz60/EdgeNeXt/releases/download/v1.0/edgenext_xx_small.pth)
| edgenext_small_bn_hs | 78.39 | 5.58M | 1.25G | [model](https://github.com/mmaaz60/EdgeNeXt/releases/download/v1.0/edgenext_small_bn_hs.pth)
| edgenext_x_small_bn_hs | 74.87 | 2.34M | 536M | [model](https://github.com/mmaaz60/EdgeNeXt/releases/download/v1.0/edgenext_x_small_bn_hs.pth)
| edgenext_xx_small_bn_hs | 70.33 | 1.33M | 260M | [model](https://github.com/mmaaz60/EdgeNeXt/releases/download/v1.0/edgenext_xx_small_bn_hs.pth)

<hr />

## Citation
If you use our work, please consider citing:
```bibtex
    @article{Maaz2022EdgeNeXt,
        title={EdgeNeXt: Efficiently Amalgamated CNN-Transformer Architecture for Mobile Vision Applications},
        author={Muhammad Maaz and Abdelrahman Shaker and Hisham Cholakkal and Salman Khan and Syed Waqas Zamir and Rao Muhammad Anwer and Fahad Shahbaz Khan},
        journal={coming soon},
        year={2022}
    }
```

<hr />

## Contact
Should you have any question, please create an issue on this repository or contact at muhammad.maaz@mbzuai.ac.ae & abdelrahman.youssief@mbzuai.ac.ae

<hr />

## References
Our code is based on [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) repository. 
We thank them for releasing their code.