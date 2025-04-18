from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# from lcnn.config import M

config_path = "config/wireframe.yaml"
with open(config_path, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
print(config)


class FeatureMapNet(nn.Module):
    def __init__(self):
        super(FeatureMapNet, self).__init__()
        self.conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.upsample(x)
        return x

class DualMapNet(nn.Module):
    def __init__(self):
        super(DualMapNet, self).__init__()
        self.conv1 = nn.Conv2d(65, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_1 = nn.Conv2d(256, 5, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 5, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        out1 = self.conv3_1(x)
        out2 = self.conv3_2(x)
        out1 = self.upsample(out1)
        out2 = self.upsample(out2)
        return [out1, out2]

class MultitaskHead(nn.Module):
    def __init__(self, input_channels, num_class):
        super(MultitaskHead, self).__init__()

        m = int(input_channels / 4)
        heads = []
        for output_channels in sum([[2], [1], [2]], []):
            heads.append(
                nn.Sequential(
                    nn.Conv2d(input_channels, m, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(m, output_channels, kernel_size=1),
                )
            )
        self.heads = nn.ModuleList(heads)
        assert num_class == sum(sum([[2], [1], [2]], []))

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=1)


class MultitaskLearner(nn.Module):
    def __init__(self, backbone):
        super(MultitaskLearner, self).__init__()
        self.backbone = backbone
        self.ftmapnet = FeatureMapNet()
        self.dualmapnet = DualMapNet()
        head_size = [[2], [1], [2]]  # M.head_size
        self.num_class = sum(sum(head_size, []))
        self.head_off = np.cumsum([sum(h) for h in head_size])

    def forward(self, input_dict):
        image = input_dict["image"]
        xf_weight = config["xfeat"]["pretrain"]
        self.backbone.load_state_dict(torch.load(xf_weight))
        self.backbone.eval()
        xf_output = self.backbone(image)
        feature = self.ftmapnet(xf_output[0])
        outputs = self.dualmapnet(xf_output[1])
        
        # outputs, feature = self.backbone(image)
        result = {"feature": feature}
        batch, channel, row, col = outputs[0].shape

        T = input_dict["target"].copy()
        n_jtyp = T["jmap"].shape[1]

        # switch to CNHW
        for task in ["jmap"]:
            T[task] = T[task].permute(1, 0, 2, 3)
        for task in ["joff"]:
            T[task] = T[task].permute(1, 2, 0, 3, 4)

        offset = self.head_off
        # loss_weight = M.loss_weight
        loss_weight = {"jmap": 8.0, "lmap": 0.5, "joff": 0.25, "lpos": 1, "lneg": 1}
        losses = []
        for stack, output in enumerate(outputs):
            output = output.transpose(0, 1).reshape([-1, batch, row, col]).contiguous()
            jmap = output[0 : offset[0]].reshape(n_jtyp, 2, batch, row, col)
            lmap = output[offset[0] : offset[1]].squeeze(0)
            joff = output[offset[1] : offset[2]].reshape(n_jtyp, 2, batch, row, col)
            if stack == 0:
                result["preds"] = {
                    "jmap": jmap.permute(2, 0, 1, 3, 4).softmax(2)[:, :, 1],
                    "lmap": lmap.sigmoid(),
                    "joff": joff.permute(2, 0, 1, 3, 4).sigmoid() - 0.5,
                }
                if input_dict["mode"] == "testing":
                    return result

            L = OrderedDict()
            L["jmap"] = sum(
                cross_entropy_loss(jmap[i], T["jmap"][i]) for i in range(n_jtyp)
            )
            L["lmap"] = (
                F.binary_cross_entropy_with_logits(lmap, T["lmap"], reduction="none")
                .mean(2)
                .mean(1)
            )
            L["joff"] = sum(
                sigmoid_l1_loss(joff[i, j], T["joff"][i, j], -0.5, T["jmap"][i])
                for i in range(n_jtyp)
                for j in range(2)
            )
            for loss_name in L:
                L[loss_name].mul_(loss_weight[loss_name])
            losses.append(L)
        result["losses"] = losses
        return result


def l2loss(input, target):
    return ((target - input) ** 2).mean(2).mean(1)


def cross_entropy_loss(logits, positive):
    nlogp = -F.log_softmax(logits, dim=0)
    return (positive * nlogp[1] + (1 - positive) * nlogp[0]).mean(2).mean(1)


def sigmoid_l1_loss(logits, target, offset=0.0, mask=None):
    logp = torch.sigmoid(logits) + offset
    loss = torch.abs(logp - target)
    if mask is not None:
        w = mask.mean(2, True).mean(1, True)
        w[w == 0] = 1
        loss = loss * (mask / w)

    return loss.mean(2).mean(1)
