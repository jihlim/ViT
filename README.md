# ViT
The unofficial Pytorch implementation of ViT

![patchify](./docs/patchify.png)

## Updates


## Installation
```
$ git clone https://github.com/jihlim/ViT.git
$ cd ViT
$ pip install -r requirement.txt
```

## Models
- ViT-Base
- ViT-Large
- ViT-Huge

## Dataset
### ImageNet-1K
- Add `ImageNet-1k` dataset and `ILSVRC2012_devkit_t12.tar` file under `data` folder
```
CNN
├── data
│   └── imagenet1k
│       ├── train
│       ├── val
│       └── ILSVRC2012_devkit_t12.tar

```

## Train
```
$ cd ViT
$ python src/train.py --model <model_name> --dataset <dataset_name>
```

Flags:
- `--model` Set model
- `--dataset` Set dataset 
- `--image_size` Set input image size
- `--epochs` Set training epochs
- `--batch_size` Set batch size
- `--lr` Set learning rate
- `--lr_decay` Set learning rate decay
- `--momentum` Set momentum
- `--wd` Set weight decay 
- `--betas` Set betas for optimizer
- `--gamma` Set gamma for scheduler
- `--milestones`Set milestones for scheduler
- `--optimizer` Set optimizer
- `--scheduler` Set Scheduler
- `--resume` Continue training

## Evaluation
```
$ cd ViT
$ python src/test.py --model <model_name> --dataset <dataset_name>
```

Flags:
- `--model` Set model 
- `--dataset` Set dataset 
- `--image_size` Set input image size
- `--batch_size` Set batch size


## Wandb (Weights & Biases)
- Type `Wandb API key` in `wandb.login()` in `src/uitls/train_utils.py`