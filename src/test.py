import os
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as datasets
from pytz import timezone

from models.model import get_model, load_model
from utils.basic_utils import set_seeds, get_cfg, merge_test_args_cfg
from utils.data_utils import *
from utils.train_utils import get_time

def test(model, testloader, criterion):
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
    
    test_loss /= len(testloader.dataset)
    test_accuracy = (correct / (len(testloader.dataset) )) * 100
    return test_loss, test_accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, type=str, help="Model to Train")
    parser.add_argument("--dataset", default=None, type=str, help="Training Dataset")
    parser.add_argument("--image_size", default=None, type=int, help="Input Image Size")
    parser.add_argument("--batch_size", default=None, type=int, help="Batch Size")

    args = parser.parse_args()

    # Device Setting
    global DEVICE
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("[INFO] \t DEVICE IS SET TO: ", DEVICE)
    
    # Experiments Settings
    src_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(src_dir)
    cfg = get_cfg(root_dir)
    cfg["src_dir"] = src_dir
    cfg["root_dir"] = root_dir

    # Merge Config and and Args 
    cfg = merge_test_args_cfg(args, cfg)

    global EXP_NAME
    model_name = cfg["model"]["model_name"]
    dataset_name = cfg["data"]["dataset_name"]
    EXP_NAME = f"{model_name}_{dataset_name}"
    data_root = os.path.join(root_dir,"data")
    workdir_path = os.path.join(root_dir, f"workdir/{EXP_NAME}/")
    
    # Parameters
    i_size = get_input_image_size(dataset_name)
    if args.image_size != None:
        i_size = args.image_size

    # Datasets
    _, testset, num_classes = get_dataset(data_root, dataset_name, i_size)
    testloader = data.DataLoader(testset, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=0)

    model = get_model(model_name, num_classes, DEVICE)
    model = model.to(DEVICE)

    # Load Checkpoint
    model, _ = load_model(cfg, model, model_name, workdir_path)
    
    # Criterion
    criterion = nn.CrossEntropyLoss()

    start_time =  get_time()
    # Start test
    print("[INFO] \t Start!")
    print("[INFO] \t Start Time : ", start_time)

    test_loss, test_accuracy = test(model, testloader, criterion)

    print(
        "[INFO] \t Test Loss : {:.8f}, Test Accuracy : {:.2f}%".format(
            test_loss, test_accuracy
        )
    )

    end_time = get_time()
    print("[INFO] \t End Time : ", end_time)
    print("[INFO] \t DONE!")

if __name__ == "__main__":
    main()