#!/usr/bin/env python3

import glob
import os
import os.path as osp
import shutil

import torch
import yaml

import lcnn
from lcnn.datasets import WireframeDataset, collate
from lcnn.models.line_vectorizer import LineVectorizer
from lcnn.models.multitask_learner import MultitaskHead, MultitaskLearner

from xfeat.xfeatmodel import XFeatModel
from xfeat.utils import draw_keypoints_on_image, draw_scores_heatmap, visualize_descriptors


def main(config):

    config_path = "config/wireframe.yaml"
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ###############################dataset####################################
    datadir = config["io"]["datadir"]
    kwargs = {
        "collate_fn": collate,
        "num_workers": config["io"]["num_workers"],
        "pin_memory": True,
    }
    
    train_loader = torch.utils.data.DataLoader(
        WireframeDataset(datadir, split="train"),
        shuffle=True,
        batch_size=config["model"]["batch_size"],
        **kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        WireframeDataset(datadir, split="valid"),
        shuffle=False,
        batch_size=config["model"]["batch_size_eval"],
        **kwargs,
    )
    epoch_size = len(train_loader)
    print("epoch_size (train):", epoch_size)
    print("epoch_size (valid):", len(val_loader))

    ###############################model####################################
    
    # xfeat as model
    model = XFeatModel()
    
    # hourglass as model
    # model = lcnn.models.hg(
    #         depth=config["model"]["depth"],
    #         head=MultitaskHead,
    #         num_stacks=config["model"]["num_stacks"],
    #         num_blocks=config["model"]["num_blocks"],
    #         num_classes=sum(sum(config["model"]["head_size"], [])),)
    
    model = MultitaskLearner(model)
    model = LineVectorizer(model)

    ###############################optimizer####################################
    optim = optim = torch.optim.Adam(
            model.parameters(),
            lr=config["optim"]["lr"],
            weight_decay=config["optim"]["weight_decay"],
            amsgrad=config["optim"]["amsgrad"],
        )
    
    if config["io"]["pretrain"]:
        model.load_state_dict(torch.load(config["io"]["pretrain"])["model_state_dict"])
        optim.load_state_dict(torch.load(config["io"]["pretrain"])["optim_state_dict"])
    model = model.to(device)
    outdir = config["io"]["output"]
    print("outdir:", outdir)

    try:
        trainer = lcnn.trainer.Trainer(
            device=device,
            model=model,
            optimizer=optim,
            train_loader=train_loader,
            val_loader=val_loader,
            out=outdir,
        )
        trainer.train()
    except BaseException:
        if len(glob.glob(f"{outdir}/viz/*")) <= 1:
            shutil.rmtree(outdir)
        raise


if __name__ == "__main__":
    config_path = "config/wireframe.yaml"
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)
    main(config)
    pass
