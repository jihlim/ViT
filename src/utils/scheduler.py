import torch

def get_scheduler(cfg, optimizer):
    gamma = cfg["train"]["gamma"]
    milestones = list(cfg["train"]["milestones"])
    lambda_decay = cfg["train"]["lambda_decay"]

    if isinstance(cfg["train"]["scheduler"], str):
        if cfg["train"]["scheduler"].lower() == "multisteplr":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        elif cfg["train"]["scheduler"].lower() == "lambdalr":
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda= lambda epoch: lambda_decay ** epoch)
        elif cfg["train"]["scheduler"].lower() == "none":
            scheduler = None
    return scheduler