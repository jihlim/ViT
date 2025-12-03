import os

from models.vit import *

def get_model(model_name, num_classes, device):
    if model_name.lower() == "vit_base":
        model = vit_base(num_classes)
    
    elif model_name.lower() == "vit_large":
        model = vit_large(num_classes)

    elif model_name.lower() == "vit_huge":
        model = vit_huge(num_classes)
    
    return model.to(device)

def load_model(cfg, model, model_name, output_path):
    model_type = cfg["model"]["pretrained_type"]
    checkpoint_path = os.path.join(output_path, f"{model_name}_{model_type}")
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = int(checkpoint['epoch']) + 1
        print("[INFO] \t Load the Checkpoint...")
    else:
        print("[INFO] \t PRETRAINED MODEL DOES NOT EXIST! Please Train a Model from Scratch!")
    
    return model, start_epoch