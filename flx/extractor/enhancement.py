import os
import cv2
import numpy as np
import torch
import yaml


class AttrDict(dict):
    def __getattr__(self, key):
        val = self[key]
        if isinstance(val, dict):
            return AttrDict(val)
        return val


from flx.models.enhancement.network import SqueezeUNet, VQFPEnhancer_PCNN


def load_model_weights(model: torch.nn.Module, model_path: str) -> None:
    if not os.path.exists(model_path):
        print(f"Warning: Enhancement model path {model_path} does not exist.")
        return
    try:
        state = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(model_path, map_location="cpu")
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state

    example_key = list(state_dict.keys())[0]
    if example_key.startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded enhancement model weights from {model_path}")


class FLAREEnhancer:
    """
    High-level fingerprint enhancement wrapper supporting:
    1. UNetEnh (SqueezeUNet)
    2. PriorEnh (VQFPEnhancer_PCNN)
    """
    def __init__(
        self,
        method: str = "UNetEnh",
        model_path: str = "../FLARE_ENH/pretrained_model/unetenh/unetenh.pth",
        ckpt_dir: str = "../FLARE_ENH/pretrained_model/priorenh",
        device: str = "cuda",
        w: float = 0.0,
        pre_enh: bool = False,
    ):
        self.method = method
        self.device_str = device if torch.cuda.is_available() and "cuda" in device else "cpu"
        self.device = torch.device(self.device_str)
        self.w = w

        if method.lower() in ["unetenh", "squeezeunet"]:
            self.model = SqueezeUNet(input_channels=1, num_classes=2, pre_enh=pre_enh).to(self.device)
            if os.path.exists(model_path):
                load_model_weights(self.model, model_path)
            else:
                alt_path = "pretrained_model/unetenh/unetenh.pth"
                if os.path.exists(alt_path):
                    load_model_weights(self.model, alt_path)
        else:
            if not os.path.exists(ckpt_dir):
                ckpt_dir = "pretrained_model/priorenh"
            model_file = os.path.join(ckpt_dir, "priorenh.pth")
            cfg_file = os.path.join(ckpt_dir, "vq.yaml")
            prior_ckpt = os.path.join(ckpt_dir, "Prior.ckpt")

            config = AttrDict(yaml.safe_load(open(cfg_file, "r")))
            config.ckpt_path = prior_ckpt

            self.model = VQFPEnhancer_PCNN(
                config.hdconfig,
                config.ldconfig,
                n_embed=config.n_codebook,
                embed_dim=config.embed_dim,
                pcn_embed=config.pcn_embed,
                ckpt_path=config.ckpt_path,
                pre_enh=pre_enh,
            ).to(self.device)

            if os.path.exists(model_file):
                load_model_weights(self.model, model_file)

        self.model.eval()

    def enhance_tensor(self, img_tensor: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            img_tensor = img_tensor.to(self.device)
            if self.method.lower() in ["unetenh", "squeezeunet"]:
                pred = self.model(img_tensor)
                enh, _ = torch.split(pred, [1, 1], dim=1)
                enh = (enh * 2.0) - 1.0
            else:
                enh = self.model.enhance(img_tensor, w=self.w)
                enh = torch.clamp(enh, -1.0, 1.0)
            return enh

    def enhance_numpy(self, img_np: np.ndarray) -> np.ndarray:
        h_org, w_org = img_np.shape
        h = int(np.ceil(h_org / 16) * 16)
        w = int(np.ceil(w_org / 16) * 16)
        img_resized = cv2.resize(img_np, (w, h))
        img_norm = (img_resized.astype(np.float32) - 127.5) / 127.5
        img_tensor = torch.from_numpy(img_norm[None, None]).float()

        enh_tensor = self.enhance_tensor(img_tensor)
        enh_np = enh_tensor.squeeze().cpu().numpy()

        enh_uint = ((enh_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        return cv2.resize(enh_uint, (w_org, h_org))


def get_unetenh_enhancer(
    model_path: str = "../FLARE_ENH/pretrained_model/unetenh/unetenh.pth",
    device: str = "cuda",
) -> FLAREEnhancer:
    return FLAREEnhancer(method="UNetEnh", model_path=model_path, device=device)


def get_priorenh_enhancer(
    ckpt_dir: str = "../FLARE_ENH/pretrained_model/priorenh",
    device: str = "cuda",
) -> FLAREEnhancer:
    return FLAREEnhancer(method="PriorEnh", ckpt_dir=ckpt_dir, device=device)
