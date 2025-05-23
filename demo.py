#!/usr/bin/env python3

import os
import os.path as osp
import pprint
import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import skimage.transform
import torch
import yaml

import lcnn
from lcnn.models.line_vectorizer import LineVectorizer


from lcnn.postprocess import postprocess
from lcnn.utils import recursive_to

from xfeat.xfeatmodel import XFeatModel

PLTOPTS = {"color": "#33FFFF", "s": 15, "edgecolors": "none", "zorder": 5}
cmap = plt.get_cmap("jet")
norm = mpl.colors.Normalize(vmin=0.9, vmax=1.0)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])


def c(x):
    return sm.to_rgba(x)


def main():
    config_path = "config/wireframe.yaml"
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # print(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model_flag = 'lcnn'
    
    if model_flag == 'lcnn':
        from lcnn.models.multitask_learner_orig import MultitaskHead, MultitaskLearner
        checkpoint_path = "/home/wenhuanyao/lcnn/pretrained/lcnn_pretrianed.pth"
        model = lcnn.models.hg(
            depth=config["model"]["depth"],
            head=MultitaskHead,
            num_stacks=config["model"]["num_stacks"],
            num_blocks=config["model"]["num_blocks"],
            num_classes=sum(sum(config["model"]["head_size"], [])),)
        model = MultitaskLearner(model)
        model = LineVectorizer(model)
    
    if model_flag == 'xfeat':
        from lcnn.models.multitask_learner_new import MultitaskLearner
        checkpoint_path = "/home/wenhuanyao/lcnn/pretrained/checkpoint_best.pth"
        model = XFeatModel()
        model = MultitaskLearner(model)
        model = LineVectorizer(model)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    model = model.to(device)
    model.eval()

    img_paths = ['/home/wenhuanyao/lcnn/demo_figs/berlin_000003_000019_leftImg8bit.png']
    
    for imname in img_paths:
        print(f"Processing {imname}")
        im = skimage.io.imread(imname)
        if im.ndim == 2:
            im = np.repeat(im[:, :, None], 3, 2)
        im = im[:, :, :3]
        im_resized = skimage.transform.resize(im, (512, 512)) * 255
        image = (im_resized - config["model"]["image"]["mean"])/config["model"]["image"]["stddev"]   #M.image.mean) / M.image.stddev
        image = torch.from_numpy(np.rollaxis(image, 2)[None].copy()).float()
        with torch.no_grad():
            input_dict = {
                "image": image.to(device),
                "meta": [
                    {
                        "junc": torch.zeros(1, 2).to(device),
                        "jtyp": torch.zeros(1, dtype=torch.uint8).to(device),
                        "Lpos": torch.zeros(2, 2, dtype=torch.uint8).to(device),
                        "Lneg": torch.zeros(2, 2, dtype=torch.uint8).to(device),
                    }
                ],
                "target": {
                    "jmap": torch.zeros([1, 1, 128, 128]).to(device),
                    "joff": torch.zeros([1, 1, 2, 128, 128]).to(device),
                },
                "mode": "testing",
            }
            H = model(input_dict)["preds"]

        lines = H["lines"][0].cpu().numpy() / 128 * im.shape[:2]
        scores = H["score"][0].cpu().numpy()
        for i in range(1, len(lines)):
            if (lines[i] == lines[0]).all():
                lines = lines[:i]
                scores = scores[:i]
                break

        # postprocess lines to remove overlapped lines
        diag = (im.shape[0] ** 2 + im.shape[1] ** 2) ** 0.5
        nlines, nscores = postprocess(lines, scores, diag * 0.01, 0, False)

        for i, t in enumerate([0.94, 0.95, 0.96, 0.97, 0.98, 0.99]):
            plt.gca().set_axis_off()
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            for (a, b), s in zip(nlines, nscores):
                if s < t:
                    continue
                plt.plot([a[1], b[1]], [a[0], b[0]], c=c(s), linewidth=2, zorder=s)
                plt.scatter(a[1], a[0], **PLTOPTS)
                plt.scatter(b[1], b[0], **PLTOPTS)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.imshow(im)
            plt.savefig(imname.replace(".png", f"-{model_flag}-{t:.02f}.png"), bbox_inches="tight")
            # plt.show()
            plt.close()


if __name__ == "__main__":
    main()
