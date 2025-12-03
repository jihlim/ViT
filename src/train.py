import argparse
import os
from datetime import datetime
from pytz import timezone

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import wandb
from torch.utils.tensorboard import SummaryWriter

from models.model import get_model, load_model
from utils.basic_utils import set_seeds, get_cfg, merge_args_cfg
from utils.data_utils import *
from utils.optimizer import get_optimizer
from utils.scheduler import get_scheduler
from utils.train_utils import *

def train(model, trainloader, criterion, optimizer, epoch, log_interval=1000):
    model.train()
    
    train_loss = 0
    correct = 0

    for batch_idx, (image, label) in enumerate(trainloader):
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        prediction = output.max(1, keepdim=True)[1]
        correct += prediction.eq(label.view_as(prediction)).sum().item()

        if batch_idx % log_interval == 0:
            print(
                "[INFO] \t EPOCH {} [{}/{}({:.2f}%)] Train Loss: {:.8f} Train Accuracy: {:.2f}%".format(
                    epoch,
                    batch_idx * len(image),
                    int(len(trainloader.dataset)),
                    batch_idx / len(trainloader) * 100,
                    loss.item(),
                    correct / len(trainloader.dataset) * 100,
                )
            )
    
    train_loss = train_loss / len(trainloader.dataset) 
    train_accuracy = (correct / len(trainloader.dataset)) * 100
    return train_loss, train_accuracy
            
def evaluate(model, testloader, criterion):
    model.eval()
    
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for image, label in testloader: 
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            output = model(image)
            loss = criterion(output, label)

            test_loss += loss.item()
            prediction = output.max(1, keepdim=True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()
    
    test_loss = test_loss / len(testloader.dataset)
    test_accuracy = (correct / (len(testloader.dataset) )) * 100
    return test_loss, test_accuracy          

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, type=str, help="Model to Train")
    parser.add_argument("--dataset", default=None, type=str, help="Training Dataset")
    parser.add_argument("--image_size", default=None, type=int, help="Input Image Size")
    parser.add_argument("--epochs", default=None, type=int, help="Train Epochs")
    parser.add_argument("--batch_size", default=None, type=int, help="Batch Size")
    parser.add_argument("--lr", default=None, type=float, help="Learning Rate")
    parser.add_argument("--lr_decay", default=None, type=float, help="Learning Rate Decay")
    parser.add_argument("--momentum", default=None, type=float, help="Momentum")
    parser.add_argument("--wd", default=None, type=float, help="Weight Decay")
    parser.add_argument("--betas", default=None, type=tuple, help="Betas for Optimizer")
    parser.add_argument("--gamma", default=None, type=float, help="Gamma for Scheduler")
    parser.add_argument("--milestones", default=None, type=list, help="Milestones for Scheduler")
    parser.add_argument("--optimizer", default=None, type=str, help="Optimizer")
    parser.add_argument("--scheduler", default=None, type=str, help="Scheduler")
    parser.add_argument("--resume", default=None, type=bool, help="Resume training")

    args = parser.parse_args()

    # Config
    src_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(src_dir)
    cfg = get_cfg(root_dir)
    cfg["src_dir"] = src_dir
    cfg["root_dir"] = root_dir

    # Merge Config and and Args 
    cfg = merge_args_cfg(args, cfg)

    # Device Setting
    global DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] \t DEVICE IS SET TO: ", DEVICE)
    
    # Fix Seed
    set_seeds()

    # Experiment Setting
    global EXP_NAME
    model_name = cfg["model"]["model_name"]
    dataset_name = cfg["data"]["dataset_name"]
    EXP_NAME = f"{model_name}_{dataset_name}"
    data_root = os.path.join(root_dir,"data")
    os.makedirs(data_root, exist_ok=True)
    train_continue = cfg["train"]["resume"]

    # Output Directory
    output_path = os.path.join(cfg['root_dir'], f"workdir/{EXP_NAME}")
    os.makedirs(output_path, exist_ok=True)

    # Set WandB
    set_wandb(output_path, EXP_NAME)
    
    # Parameters
    i_size = get_input_image_size(dataset_name)
    if args.image_size != None:
        i_size = args.image_size
    
    # Datasets
    trainset, testset, num_classes = get_dataset(data_root, dataset_name, i_size)
    
    trainloader = data.DataLoader(trainset, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=0)
    testloader = data.DataLoader(testset, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=0)
    
    # Model
    model = get_model(model_name, num_classes, DEVICE)

    # Load Model
    if train_continue:
        model, start_epoch = load_model(cfg, model, model_name, output_path)
        print("[INFO] \t Continue Training the Model!")
    else:
        start_epoch = 1
        print("[INFO] \t Training the Model from Scratch!")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["train"]["label_smoothing"])
    optimizer = get_optimizer(cfg, model)
    scheduler = get_scheduler(cfg, optimizer)
    
    x_epoch = []
    y_trainloss = []
    y_trainaccuracy = []
    y_testloss = []
    y_testaccuracy = []
    best_accuracy = 0

    # Tensorboard
    writer = SummaryWriter(os.path.join(root_dir, f"log/{EXP_NAME}/"))

    # Start training
    start_time = get_time()
    print("[INFO] \t Start!")
    print("[INFO] \t Start Time : ", start_time)

    for epoch in range(start_epoch, cfg["train"]["epochs"] + 1):
        train_loss, train_accuracy = train(
            model, trainloader, criterion, optimizer, epoch, log_interval=1000
        )
        eval_loss, eval_accuracy = evaluate(model, testloader, criterion)
        if scheduler != None:
            scheduler.step()

        print(
            "[INFO] \t [EPOCH {}] Train Loss : {:.8f}, Train Accuracy : {:.2f}%, Test Loss : {:.8f}, Test Accuracy : {:.2f}%".format(
                epoch, train_loss, train_accuracy, eval_loss, eval_accuracy
            )
        )

        # Log metrics inside the training loop
        current_lr = get_lr(optimizer)
        log_wandb(train_loss, eval_loss, train_accuracy, eval_accuracy, current_lr)

        x_epoch.append(epoch)
        y_trainloss.append(train_loss)
        y_trainaccuracy.append(train_accuracy)
        y_testloss.append(eval_loss)
        y_testaccuracy.append(eval_accuracy)

        # Tensorboard
        writer.add_scalar("Loss [Train]", np.array(train_loss), epoch)
        writer.add_scalar("Loss [Test]", np.array(eval_loss), epoch)
        writer.add_scalar("accuracy [Train]", np.array(train_accuracy), epoch)
        writer.add_scalar("accuracy [Test]", np.array(eval_accuracy), epoch)

        # Matplotlib
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.ion()
        plt.clf()
        plt.figure(1)
        plt.subplot(1, 2, 1)
        plt.plot(x_epoch, y_trainloss, "-", label="Train Loss")
        plt.plot(x_epoch, y_testloss, "-", label="Test Loss")
        plt.xlabel("Epoch"), plt.ylabel("Loss"), plt.title("Train/Test Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(x_epoch, y_trainaccuracy, "-", label="Train Accuracy")
        plt.plot(x_epoch, y_testaccuracy, "-", label="Test Accuracy")
        plt.xlabel("Epoch"), plt.ylabel("Accuracy"), plt.title("Train/Test Accuracy")
        plt.legend()

        # Save the Best Model
        if eval_accuracy > best_accuracy:
            torch.save({
                "epoch": epoch,
                "model_state_dict":model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler" : scheduler.state_dict(),
                },
                os.path.join(output_path, f"{model_name}_best_model.pth"))
            best_accuracy = eval_accuracy
            print("[INFO] \t [EPOCH {}] Saving the Best Model".format(epoch))
        
        torch.save({
            "epoch":epoch,
            "model_state_dict":model.state_dict(),
            "optimizer_state_dict":optimizer.state_dict(),
            "scheduler" : scheduler.state_dict(),
            },
            os.path.join(output_path, "snapshot.pth"))
        print("[INFO] \t [EPOCH {}] Saving Model Snapshot".format(epoch))
    
    # Complete training
    finish_wandb()
    end_time = get_time()
    print("[INFO] \t End Time : ", end_time)
    print("[INFO] \t Best Accuracy: {:.2f}%".format(best_accuracy))
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "loss_accuracy_plot.png"))
    print("[INFO] \t Saving Plot...")
    print("[INFO] \t DONE!")

if __name__ == "__main__":
    main()
