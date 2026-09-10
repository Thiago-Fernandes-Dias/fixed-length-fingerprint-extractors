import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from flx.models.flare.units import FastCartoonTexture, NormalizeModule, FingerprintCompose, DecoderSkip2, ConvBnPRelu
from flx.models.flare.cbam import CBAM
from flx.models.flare import resnet
from flx.models.flare.resnext import ResNextBlock



def dense_hough_voting4(
    center_prob,
    grid_prob,
    center_att,
    theta_att,
    img_H,
    img_W,
    img_ppi=500,
    middle_shape=(640, 640),
    bin_type="arcsin",
    activate="sigmoid",
    calc_confidence=False,
):
    def interval_location(x, bin_type="x1"):
        x = x.clamp(-1, 1)
        if bin_type == "x1":
            return x
        elif bin_type == "x2":
            return x.abs() ** 2
        elif bin_type == "invprop":
            return x / (2 - x.abs())
        elif bin_type == "arcsin":
            return torch.arcsin(x) / (np.pi / 2)
        else:
            raise ValueError(f"Unsupported bin_type:{bin_type}")

    def custom_linspace(num, bin_type=None, delta=False):
        x = torch.linspace(-1, 1, num + 1)
        if bin_type is not None:
            x = interval_location(x, bin_type)
        if delta:
            return x[1:] - x[:-1]
        else:
            return (x[:-1] + x[1:]) / 2

    c_prob_x, c_prob_y = center_prob
    g_prob_x, g_prob_y = grid_prob
    prob_H, prob_W = c_prob_x.shape[2:]
    max_range = img_ppi / 500 * np.array(middle_shape) / 2

    if c_prob_x.size(1) > 1:
        if activate == "sigmoid":
            c_prob_x = torch.sigmoid(c_prob_x)
            c_prob_y = torch.sigmoid(c_prob_y)
            g_prob_x = torch.sigmoid(g_prob_x)
            g_prob_y = torch.sigmoid(g_prob_y)
            c_prob_x = c_prob_x / c_prob_x.sum(dim=1, keepdim=True)
            c_prob_y = c_prob_y / c_prob_y.sum(dim=1, keepdim=True)
            g_prob_x = g_prob_x / g_prob_x.sum(dim=1, keepdim=True)
            g_prob_y = g_prob_y / g_prob_y.sum(dim=1, keepdim=True)
        elif activate == "softmax":
            c_prob_x = torch.softmax(c_prob_x, dim=1)
            c_prob_y = torch.softmax(c_prob_y, dim=1)
            g_prob_x = torch.softmax(g_prob_x, dim=1)
            g_prob_y = torch.softmax(g_prob_y, dim=1)

        x_vec = custom_linspace(c_prob_x.size(1), bin_type).view(1, -1, 1, 1).type_as(c_prob_x) * max_range[1]
        y_vec = custom_linspace(c_prob_y.size(1), bin_type).view(1, -1, 1, 1).type_as(c_prob_y) * max_range[0]

        c_exp_x = (c_prob_x * x_vec).sum(dim=1, keepdim=True)
        c_exp_y = (c_prob_y * y_vec).sum(dim=1, keepdim=True)
        g_exp_x = (g_prob_x * x_vec).sum(dim=1, keepdim=True)
        g_exp_y = (g_prob_y * y_vec).sum(dim=1, keepdim=True)
    else:
        c_exp_x = c_prob_x * max_range[1]
        c_exp_y = c_prob_y * max_range[0]
        g_exp_x = g_prob_x * max_range[1]
        g_exp_y = g_prob_y * max_range[0]

    norm_g = torch.sqrt(g_exp_x ** 2 + g_exp_y ** 2).clamp_min(1e-6)
    norm_c = torch.sqrt(c_exp_x ** 2 + c_exp_y ** 2).clamp_min(1e-6)
    norm_project = norm_g * norm_c

    sin_theta = g_exp_x * c_exp_y.detach() - g_exp_y * c_exp_x.detach()
    cos_theta = g_exp_x * c_exp_x.detach() + g_exp_y * c_exp_y.detach()

    sin_theta = sin_theta / norm_project
    cos_theta = cos_theta / norm_project

    out_theta = torch.rad2deg(torch.atan2((sin_theta * theta_att).mean((1, 2, 3)), (cos_theta * theta_att).mean((1, 2, 3))))

    grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, prob_H), torch.linspace(-1, 1, prob_W), indexing="ij")
    grid_x = (grid_x.type_as(c_prob_x) + 1) / 2 * (img_W - 1)
    grid_y = (grid_y.type_as(c_prob_y) + 1) / 2 * (img_H - 1)

    if c_prob_x.size(1) > 1:
        x_bin = x_vec + grid_x[None, None]
        y_bin = y_vec + grid_y[None, None]
        out_x = (x_bin * c_prob_x * center_att).sum((1, 2, 3)) / (c_prob_x * center_att).sum((1, 2, 3)).clamp_min(1e-6)
        out_y = (y_bin * c_prob_y * center_att).sum((1, 2, 3)) / (c_prob_y * center_att).sum((1, 2, 3)).clamp_min(1e-6)
    else:
        x_bin = c_exp_x + grid_x[None, None]
        y_bin = c_exp_y + grid_y[None, None]
        out_x = (x_bin * center_att).sum((1, 2, 3)) / center_att.sum((1, 2, 3)).clamp_min(1e-6)
        out_y = (y_bin * center_att).sum((1, 2, 3)) / center_att.sum((1, 2, 3)).clamp_min(1e-6)

    return (out_x[:, None], out_y[:, None]), out_theta[:, None], (g_exp_x, g_exp_y, c_exp_x, c_exp_y)


class GRIDNET4(nn.Module):
    def __init__(
        self,
        num_pose_2d=[1, 1, 1],
        num_layers=[64, 128, 256, 512],
        img_ppi=500,
        middle_shape=[512, 512],
        with_tv=False,
        with_enh=False,
        bin_type="arcsin",
        activate="sigmoid",
        pretrained=False,
    ) -> None:
        super().__init__()
        self.num_center = num_pose_2d[:2]
        self.img_ppi = img_ppi
        self.with_tv = with_tv
        self.middle_shape = middle_shape
        self.with_enh = with_enh
        self.bin_type = bin_type
        self.activate = activate

        self.preprocess_tv = FastCartoonTexture(sigma=2.5 * img_ppi / 500) if self.with_tv else nn.Identity()
        self.input_layer = nn.Sequential(
            NormalizeModule(m0=0, var0=1),
            FingerprintCompose(win_size=np.rint(8 * img_ppi / 500).astype(int)),
        )

        block = resnet.BasicBlock
        base_model = resnet._resnet("resnet18", block, [2, 2, 2, 2], pretrained, True, num_layers=num_layers, num_in=3)
        base_layers = list(base_model.children())
        self.layer0 = nn.Sequential(*base_layers[:3])
        self.layer1 = nn.Sequential(*base_layers[3:5])
        self.layer2 = base_layers[5]
        self.layer3 = base_layers[6]
        self.layer4 = base_layers[7]
        num_up = 3
        self.decoder = DecoderSkip2(num_layers[-1], num_layers=num_layers[-2 : -2 - num_up : -1], expansion=block.expansion)
        self.pixels_out = nn.Conv2d(num_layers[-1 - num_up] * block.expansion, sum(self.num_center) * 2 + 2, 1)

    def forward(self, input, seg=None):
        processed_tv = self.preprocess_tv(input)
        processed = self.input_layer(processed_tv)

        layer0 = self.layer0(processed)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        decoder = self.decoder((layer4, layer3, layer2, layer1))
        pixels_out = torch.split(self.pixels_out(decoder), (1, *self.num_center, *self.num_center, 1), dim=1)
        out_seg = torch.sigmoid(pixels_out[0])
        out_center = pixels_out[1:3]
        out_grid = pixels_out[3:5]
        out_att = torch.sigmoid(pixels_out[-1]) * out_seg.detach()

        att_c = seg if seg is not None else out_seg
        att_d = torch.sigmoid(pixels_out[-1]) * seg if seg is not None else out_att

        out_center_2d, out_theta_2d, out_exp = dense_hough_voting4(
            out_center,
            out_grid,
            att_c,
            att_d,
            input.size(-2),
            input.size(-1),
            self.img_ppi,
            self.middle_shape,
            bin_type=self.bin_type,
            activate=self.activate,
        )
        out_pose_2d = torch.cat([*out_center_2d, out_theta_2d], dim=1)

        return {
            "center": out_center,
            "grid": out_grid,
            "pose_2d": out_pose_2d,
            'theta': out_theta_2d,
            "seg": out_seg,
            "img_sup": [processed_tv, *out_exp[:2]],
            "seg_sup": [out_att, *out_exp[2:4]],
        }


class FingerPose_2D_Single(nn.Module):
    def __init__(self,
                 inp_mode="fp",
                 trans_out_form="reg",
                 trans_num_classes=120,
                 rot_out_form="claSum",
                 rot_num_classes=120,
                 channel_lst=[64, 128, 256, 512, 1024],
                 layer_lst=[3, 4, 6, 3]):
        super(FingerPose_2D_Single, self).__init__()
        self.trans_out_form = trans_out_form
        self.rot_out_form = rot_out_form
        self.norm_layer = NormalizeModule(m0=0, var0=1)

        if inp_mode == "cap":
            self.layer1 = nn.Sequential(
                ConvBnPRelu(1, channel_lst[0], 3),
                ConvBnPRelu(channel_lst[0], channel_lst[0], 3))
        else:
            self.layer1 = nn.Sequential(
                ConvBnPRelu(1, channel_lst[0], 7, stride=2, padding=3),
                ConvBnPRelu(channel_lst[0], channel_lst[0], 3),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.layer2 = self._make_layers(ResNextBlock,
                                        in_channels=channel_lst[0],
                                        out_channels=channel_lst[1],
                                        groups=32,
                                        stride=1 if inp_mode == "cap" else 2,
                                        num_layers=layer_lst[0])
        self.layer3 = self._make_layers(ResNextBlock,
                                        in_channels=channel_lst[1],
                                        out_channels=channel_lst[2],
                                        groups=32,
                                        stride=1 if inp_mode == "cap" else 2,
                                        num_layers=layer_lst[1])

        self.att3 = CBAM(channel_lst[2])

        self.layer4 = self._make_layers(ResNextBlock,
                                        in_channels=channel_lst[2],
                                        out_channels=channel_lst[3],
                                        groups=32,
                                        stride=1 if inp_mode == "cap" else 2,
                                        num_layers=layer_lst[2])
        self.att4 = CBAM(channel_lst[3])

        self.layer5 = self._make_layers(ResNextBlock,
                                        in_channels=channel_lst[3],
                                        out_channels=channel_lst[4],
                                        groups=32,
                                        stride=1 if inp_mode == "cap" else 2,
                                        num_layers=layer_lst[3])
        self.att5 = CBAM(channel_lst[4])

        self.avgpool_flatten_layer = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                                   nn.Flatten(start_dim=1))
        if trans_out_form in ["claSum", "claMax"]:
            self.trans_fc_theta = nn.Linear(channel_lst[4], trans_num_classes)
        elif trans_out_form == "reg":
            self.trans_fc_theta = nn.Linear(channel_lst[4], 2)

        if rot_out_form in ["claSum", "claMax"]:
            self.rot_fc_theta = nn.Linear(channel_lst[4], rot_num_classes)
        elif rot_out_form == "reg_ang":
            self.rot_fc_theta = nn.Linear(channel_lst[4], 1)
        elif rot_out_form == "reg_tan":
            self.rot_fc_theta = nn.Linear(channel_lst[4], 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, Block, in_channels, out_channels, groups, stride, num_layers):
        layers = [Block(in_channels=in_channels, out_channels=out_channels, groups=groups, stride=stride)]
        for _ in range(num_layers - 1):
            layers.append(Block(in_channels=out_channels, out_channels=out_channels, groups=groups, stride=1))
        return nn.Sequential(*layers)

    def forward(self, inp):
        inp = self.norm_layer(inp)
        feat = self.layer1(inp)
        feat = self.layer2(feat)
        feat = self.layer3(feat)
        feat, _, _ = self.att3(feat)
        feat = self.layer4(feat)
        feat, _, _ = self.att4(feat)
        feat = self.layer5(feat)
        feat, _, _ = self.att5(feat)

        feat = self.avgpool_flatten_layer(feat)
        if self.trans_out_form in ["claSum", "claMax"]:
            pred_xy = self.trans_fc_theta(feat)
            _, c = pred_xy.shape
            pred_x = F.softmax(pred_xy[:, :c // 2], dim=1)
            pred_y = F.softmax(pred_xy[:, c // 2:], dim=1)
            pred_xy = torch.cat([pred_x, pred_y], dim=1)
        elif self.trans_out_form == "reg":
            pred_xy = self.trans_fc_theta(feat) * 64

        if self.rot_out_form in ["claSum", "claMax"]:
            pred_theta = self.rot_fc_theta(feat)
            pred_theta = F.softmax(pred_theta, dim=1)
        elif self.rot_out_form == "reg_ang":
            pred_theta = self.rot_fc_theta(feat) * 90
        elif self.rot_out_form == "reg_tan":
            pred_theta = self.rot_fc_theta(feat)

        return [pred_xy, pred_theta]
