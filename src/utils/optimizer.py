import torch

def get_optimizer(cfg, model):
    lr = cfg["train"]["lr"]
    lr_decay = cfg["train"]["lr_decay"]
    momentum = cfg["train"]["momentum"]
    weight_decay = cfg["train"]["weight_decay"]
    betas = eval(cfg["train"]["betas"])
    
    if isinstance(cfg["train"]["optimizer"], str):
        if cfg["train"]["optimizer"].lower() == "sgd":
            optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=momentum,  weight_decay=weight_decay)
        elif cfg["train"]["optimizer"].lower() == "adam":
            optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        elif cfg["train"]["optimizer"].lower() == "adamw":
            optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    
    return optimizer