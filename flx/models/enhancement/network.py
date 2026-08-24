import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flx.models.enhancement.blocks import Encoder, Decoder, VectorQuantizer
from flx.models.enhancement.gatedpixelcnn import GatedPixelCNN, Fuse_sft_block


class Clahe(torch.nn.Module):
    def __init__(self, clip_limit=40, grid_size=(8, 8)) -> None:
        super().__init__()
        self.clip_limit, self.grid_size = float(clip_limit), grid_size

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        try:
            import kornia
            return kornia.enhance.equalize_clahe(img, self.clip_limit, self.grid_size)
        except ImportError:
            device = img.device
            img_np = (img.squeeze().detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.grid_size)
            enh_np = clahe.apply(img_np)
            enh_tensor = torch.from_numpy((enh_np.astype(np.float32) / 255.0)).to(device)
            if img.ndim == 4:
                return enh_tensor[None, None]
            return enh_tensor


class FireModule(nn.Module):
    def __init__(self, in_channels, squeeze=16, expand=64):
        super(FireModule, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(squeeze)
        self.expand1x1 = nn.Conv2d(squeeze, expand, kernel_size=1, padding=0)
        self.expand3x3 = nn.Conv2d(squeeze, expand, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.bn(self.squeeze(x)))
        left = F.relu(self.expand1x1(x))
        right = F.relu(self.expand3x3(x))
        return torch.cat([left, right], 1)


class SqueezeUNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=2, deconv_ksize=3, dropout=0.5, activation="sigmoid", pre_enh=False):
        super(SqueezeUNet, self).__init__()
        self.activation = activation
        self.pre_enh = pre_enh
        if self.pre_enh:
            self.enhance = Clahe(clip_limit=20)
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.fire2 = FireModule(in_channels=64, squeeze=16, expand=64)
        self.fire3 = FireModule(in_channels=128, squeeze=16, expand=64)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.fire4 = FireModule(in_channels=128, squeeze=32, expand=128)
        self.fire5 = FireModule(in_channels=256, squeeze=32, expand=128)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.fire6 = FireModule(in_channels=256, squeeze=48, expand=192)
        self.fire7 = FireModule(in_channels=384, squeeze=48, expand=192)
        self.fire8 = FireModule(in_channels=384, squeeze=64, expand=256)
        self.fire9 = FireModule(in_channels=512, squeeze=64, expand=256)

        self.dropout = nn.Dropout(dropout)

        self.upconv1 = nn.ConvTranspose2d(512, 192, kernel_size=deconv_ksize, stride=1, padding=1)
        self.fire10 = FireModule(in_channels=384 + 192, squeeze=48, expand=192)

        self.upconv2 = nn.ConvTranspose2d(384, 128, kernel_size=deconv_ksize, stride=1, padding=1)
        self.fire11 = FireModule(in_channels=256 + 128, squeeze=32, expand=128)

        self.upconv3 = nn.ConvTranspose2d(256, 64, kernel_size=deconv_ksize, stride=2, padding=1, output_padding=1)
        self.fire12 = FireModule(in_channels=128 + 64, squeeze=16, expand=64)

        self.upconv4 = nn.ConvTranspose2d(128, 32, kernel_size=deconv_ksize, stride=2, padding=1, output_padding=1)
        self.fire13 = FireModule(in_channels=64 + 32, squeeze=16, expand=32)

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.final_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.final_conv2 = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        if self.pre_enh:
            x = (x + 1) / 2
            x = self.enhance(x)
            x = x * 2 - 1
        x1 = F.relu(self.conv1(x))
        x2 = self.pool1(x1)

        x3 = self.fire2(x2)
        x3 = self.fire3(x3)
        x4 = self.pool3(x3)

        x5 = self.fire4(x4)
        x5 = self.fire5(x5)
        x6 = self.pool5(x5)

        x7 = self.fire6(x6)
        x7 = self.fire7(x7)
        x8 = self.fire8(x7)
        x9 = self.fire9(x8)

        if self.training:
            x9 = self.dropout(x9)

        up1 = torch.cat([self.upconv1(x9), x7], 1)
        up1 = self.fire10(up1)

        up2 = torch.cat([self.upconv2(up1), x6], 1)
        up2 = self.fire11(up2)

        up3 = torch.cat([self.upconv3(up2), x4], 1)
        up3 = self.fire12(up3)

        up4 = torch.cat([self.upconv4(up3), x2], 1)
        up4 = self.fire13(up4)
        up4 = self.upsample(up4)

        x_final = torch.cat([up4, x1], 1)
        x_final = F.relu(self.final_conv1(x_final))
        x_final = self.upsample(x_final)

        output = self.final_conv2(x_final)
        if self.activation == "sigmoid":
            output = torch.sigmoid(output)

        return output


class VQFPEnhancer_PCNN(nn.Module):
    def __init__(
        self,
        hdconfig,
        ldconfig,
        n_embed=16384,
        embed_dim=16,
        pcn_embed=64,
        remap=None,
        sane_index_shape=False,
        ckpt_path=None,
        pre_enh=False,
    ):
        super(VQFPEnhancer_PCNN, self).__init__()
        self.embed_dim = embed_dim
        self.pcn_embed = pcn_embed
        self.pre_enh = pre_enh
        if self.pre_enh:
            self.enhancer = Clahe(clip_limit=20)
        self.hr_encoder = Encoder(**hdconfig)
        self.hr_decoder = Decoder(**hdconfig)

        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25, remap=remap, sane_index_shape=sane_index_shape, legacy=False)
        self.quant_conv = torch.nn.Conv2d(hdconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, hdconfig["z_channels"], 1)
        self.Prior = nn.ModuleList([self.hr_decoder, self.hr_decoder, self.quantize, self.quant_conv, self.post_quant_conv])

        if ckpt_path is not None and os.path.exists(ckpt_path):
            dict_test = torch.load(ckpt_path, map_location="cpu")["state_dict"]
            self.save_dict_to_prior(dict_test)

        self.lr_encoder = Encoder(**ldconfig)
        self.lr_quant_conv = torch.nn.Conv2d(ldconfig["z_channels"], embed_dim, 1)
        ratio = 2 ** (len(hdconfig["ch_mult"]) - 1)
        self.feat_res = ldconfig["resolution"] // ratio
        self.pixelcnn = GatedPixelCNN(embed_dim, pcn_embed, n_embed)
        self.n_embed = n_embed

        self.channels = {}
        base_ch = hdconfig["ch"]
        ch_mul = hdconfig["ch_mult"]
        reso = hdconfig["resolution"]
        for i in range(len(ch_mul)):
            self.channels[str(reso)] = base_ch * ch_mul[i]
            reso = reso // 2

        self.fuse_feature = self.lr_encoder.fuse_feature
        if self.lr_encoder.fuse_feature:
            self.fuse_convs_dict = nn.ModuleList()
            for res, ch in self.channels.items():
                self.fuse_convs_dict.append(Fuse_sft_block(ch, ch))

    def save_dict_to_prior(self, dict_test):
        encoder_process_dict = {}
        decoder_process_dict = {}
        quant_dict = {}
        quant_conv_dict = {}
        post_quant_conv_dict = {}
        for name, param in dict_test.items():
            if name.startswith("encoder"):
                name = name.replace("encoder.", "")
                encoder_process_dict[name] = param
            elif name.startswith("decoder"):
                name = name.replace("decoder.", "")
                decoder_process_dict[name] = param
            elif name.startswith("quantize"):
                name = name.replace("quantize.", "")
                quant_dict[name] = param
            elif name.startswith("quant_conv"):
                name = name.replace("quant_conv.", "")
                quant_conv_dict[name] = param
            elif name.startswith("post_quant_conv"):
                name = name.replace("post_quant_conv.", "")
                post_quant_conv_dict[name] = param
        self.hr_encoder.load_state_dict(encoder_process_dict)
        self.hr_decoder.load_state_dict(decoder_process_dict)
        self.quantize.load_state_dict(quant_dict)
        self.quant_conv.load_state_dict(quant_conv_dict)
        self.post_quant_conv.load_state_dict(post_quant_conv_dict)

    @torch.no_grad()
    def enhance(self, x, w=0):
        if self.pre_enh:
            x = (x + 1) / 2.0
            x = self.enhancer(x)
            x = x * 2.0 - 1
        if self.fuse_feature:
            z_l, fuse_features = self.lr_encoder(x)
        else:
            z_l = self.lr_encoder(x)
        z_l = self.lr_quant_conv(z_l)
        logits = self.pixelcnn(z_l)
        logits = logits.permute(0, 2, 3, 1).reshape(-1, self.feat_res * self.feat_res, self.n_embed)
        soft_one_hot = F.softmax(logits, dim=2)
        _, top_index = torch.topk(soft_one_hot, 1, dim=2)
        z_lq = self.quantize.get_codebook_entry(top_index, shape=[x.shape[0], self.feat_res, self.feat_res, self.embed_dim])
        z_lq = self.post_quant_conv(z_lq)
        if self.fuse_feature:
            dec = self.hr_decoder(z_lq, self.fuse_convs_dict, fuse_features, w)
        else:
            dec = self.hr_decoder(z_lq)
        return dec
