# EdgeNeXt: 
Light-weight and Efficient Networks for Mobile Vision Applications

## :rocket: News
* Training and evaluation code along with pre-trained models is released.

<hr />

![main figure](images/EdgeNext.jpeg)
> **Abstract:** *Designing lightweight general purpose networks for edge devices is a challenging task due to the compute constraints. In this domain, CNN-based light-weight architectures are considered the de-facto choice due to their efficiency in terms of parameters and complexity. However, they are based on spatially local operations and exhibit a limited receptive field. While vision transformers alleviate these issues and can learn global representations, they are typically compute intensive and difficult to optimize. Here, we investigate how to effectively encode both local and global information, while being efficient in terms of both parameters and MAdds on vision tasks. To this end, we propose EdgeNeXt, a hybrid CNN-Transformer architecture that strives to jointly optimize parameters and MAdds for efficient inference on edge devices. Within our EdgeNeXt, we introduce split depthwise transpose attention (SDTA) encoder that splits input tensors into multiple channel groups and utilizes depthwise convolution along with self-attention across channel dimensions to implicitly increase the receptive field and encode multi-scale features. Our extensive experiments on classification, detection and segmentation settings, reveal the merits of the proposed approach, outperforming state-of-the-art methods with comparatively lower compute requirements. Our EdgeNeXt model with 1.3M parameters achieves 71.2\% top-1 accuracy on ImageNet-1K, outperforming MobileViT with an absolute gain of 2.2\% with similar parameters and 28\% reduction in MAdds. Further, our EdgeNeXt model with 5.6M parameters achieves 79.4\% top-1 accuracy on ImageNet-1K.* 
<hr />

## Comparison with SOTA ViTs and Hybrid Architectures
![results](images/Figure_1.png)

<hr />

## Comparison with Previous SOTA [MobileViT (ICLR-2022)](https://arxiv.org/abs/2110.02178)
![results](images/table_2.png)

<hr />

## Installation
1. Create conda environment
```shell
conda create --name mobilenext python=3.8
conda activate mobilenext
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
python main.py --model edgenext_small --eval True --batch_size 16 --data_path <path to imagenet> --output_dir <results> --resume <path to downloaded checkpoints>
```

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
