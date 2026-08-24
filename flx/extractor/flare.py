import os
import torch
import numpy as np
import tqdm

from flx.data.dataset import Dataset
from flx.data.embedding_loader import FLAREEmbeddingLoader
from flx.data.image_helpers import flare_image_transform
from flx.models.flare.fdd import FDD
from flx.models.flare.pose import GRIDNET4, FingerPose_2D_Single
from flx.extractor.enhancement import FLAREEnhancer


def remove_module_string(k: str) -> str:
    items = k.split(".")
    items = items[0:1] + items[2:]
    return ".".join(items)


def load_flare_model(model: torch.nn.Module, ckp_path: str) -> None:
    if not os.path.exists(ckp_path):
        print(f"Warning: FLARE model checkpoint {ckp_path} does not exist.")
        return

    try:
        ckp = torch.load(ckp_path, map_location=lambda storage, loc: storage, weights_only=False)
    except TypeError:
        ckp = torch.load(ckp_path, map_location=lambda storage, loc: storage)
    ckp_model_dict = ckp["model"] if isinstance(ckp, dict) and "model" in ckp else ckp

    example_key = list(ckp_model_dict.keys())[0]
    if "module" in example_key:
        ckp_model_dict = {remove_module_string(k): v for k, v in ckp_model_dict.items()}

    if hasattr(model, "module"):
        model.module.load_state_dict(ckp_model_dict)
    else:
        model.load_state_dict(ckp_model_dict)
    print(f"Loaded FLARE model parameters from {ckp_path}")


def classify2vector_trans(pred_xy: torch.Tensor, out_form: str = "claSum", trans_num_classes: int = 512, eps: float = 1e-6) -> torch.Tensor:
    trans_const = 256
    if out_form == "claSum":
        trans_tensor = np.linspace(-trans_const, trans_const, trans_num_classes // 2)
        trans_tensor = torch.FloatTensor(trans_tensor).to(pred_xy.device)
        _, c = pred_xy.shape
        x_pred = pred_xy[:, : c // 2]
        y_pred = pred_xy[:, c // 2 :]
        x_pred = torch.sum(x_pred * trans_tensor, dim=-1) / (torch.sum(x_pred, dim=-1) + eps)
        y_pred = torch.sum(y_pred * trans_tensor, dim=-1) / (torch.sum(y_pred, dim=-1) + eps)
    else:
        x_pred = pred_xy[:, 0]
        y_pred = pred_xy[:, 1]

    if len(x_pred.shape) > 1:
        x_pred = x_pred.squeeze(1)
        y_pred = y_pred.squeeze(1)
    return torch.stack([x_pred, y_pred]).transpose(1, 0)


def classify2vector_rot(pred_theta: torch.Tensor, out_form: str = "claSum", rot_num_classes: int = 180, eps: float = 1e-6) -> torch.Tensor:
    if out_form == "claSum":
        rot_tensor = np.linspace(-np.pi, np.pi, rot_num_classes)
        rot_tensor = torch.FloatTensor(rot_tensor).to(pred_theta.device)
        cos_pred = torch.sum(pred_theta * torch.cos(rot_tensor), dim=-1) / (torch.sum(pred_theta, dim=-1) + eps)
        sin_pred = torch.sum(pred_theta * torch.sin(rot_tensor), dim=-1) / (torch.sum(pred_theta, dim=-1) + eps)
    else:
        rad_pred = torch.deg2rad(pred_theta)
        cos_pred = torch.cos(rad_pred)
        sin_pred = torch.sin(rad_pred)

    if len(cos_pred.shape) > 1:
        cos_pred = cos_pred.squeeze(1)
        sin_pred = sin_pred.squeeze(1)

    ang_pred = torch.rad2deg(torch.arctan2(sin_pred, cos_pred))
    if len(ang_pred.shape) > 1:
        ang_pred = ang_pred.squeeze(1)

    return torch.stack([cos_pred, sin_pred, ang_pred]).transpose(1, 0)


class FLAREPoseEstimator:
    """
    Estimates fingerprint pose (center x, y and orientation theta) using
    VotingPose (GRIDNET4) or RegressionPose (FingerPose_2D_Single) trained weights.
    """
    def __init__(
        self,
        pose_type: str = "VotingPose",
        model_path: str = "../FLARE/model_weights/VotingPose.pth",
        device: str = "cuda",
    ):
        self.pose_type = pose_type
        self.device_str = device if torch.cuda.is_available() and "cuda" in device else "cpu"
        self.device = torch.device(self.device_str)

        if pose_type == "VotingPose":
            self.model = GRIDNET4(
                num_pose_2d=(33, 33, 1),
                num_layers=(64, 128, 256, 512),
                img_ppi=500,
                middle_shape=np.array([512, 512]),
                activate="sigmoid",
                bin_type="invprop",
                with_tv=True,
            ).to(self.device)
        else:
            self.model = FingerPose_2D_Single(
                inp_mode="fp",
                trans_out_form="claSum",
                trans_num_classes=512,
                rot_out_form="claSum",
                rot_num_classes=180,
            ).to(self.device)

        if os.path.exists(model_path):
            load_flare_model(self.model, model_path)

    def predict_pose(self, img_tensor: torch.Tensor) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            img_tensor = img_tensor.to(self.device)
            if img_tensor.ndim == 3:
                img_tensor = img_tensor[None]
            output = self.model(img_tensor)
            if isinstance(output, dict) and "pose_2d" in output:
                pose_2d = output["pose_2d"][0].detach().cpu().numpy()
            else:
                pred_xy, pred_theta = output[0], output[1]
                vec_xy = classify2vector_trans(pred_xy, out_form="claSum", trans_num_classes=512)
                vec_xy = vec_xy + 256.0
                vec_theta = classify2vector_rot(pred_theta, out_form="claSum", rot_num_classes=180)
                pose_2d = np.array([vec_xy[0, 0].item(), vec_xy[0, 1].item(), vec_theta[0, 2].item()])
            return pose_2d


class FLAREExtractor:
    def __init__(
        self,
        model_path: str = "../FLARE/model_weights/desc_model.pth.tar",
        ndim_feat: int = 6,
        input_norm: bool = False,
        tar_shape: tuple[int, int] = (256, 256),
        batch_size: int = 16,
        device: str = "cuda",
        pose_estimator: FLAREPoseEstimator | None = None,
        enhancer: FLAREEnhancer | None = None,
    ):
        self.model_path = model_path
        self.batch_size = batch_size
        self.device_str = device if torch.cuda.is_available() and "cuda" in device else "cpu"
        self.device = torch.device(self.device_str)
        self.pose_estimator = pose_estimator
        self.enhancer = enhancer

        self.model = FDD(
            ndim_feat=ndim_feat,
            input_norm=input_norm,
            tar_shape=tar_shape,
        ).to(self.device)

        if os.path.exists(model_path):
            load_flare_model(self.model, model_path)

    def load_weights(self, path: str) -> None:
        load_flare_model(self.model, path)

    def extract(self, dataset: Dataset) -> FLAREEmbeddingLoader:
        self.model.eval()
        features_list = []
        masks_list = []

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False
        )

        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader, desc="Extracting FLARE Descriptors"):
                if isinstance(batch, dict) and "img" in batch:
                    img = batch["img"].to(self.device)
                else:
                    img = batch.to(self.device)

                if self.enhancer is not None:
                    img = self.enhancer.enhance_tensor(img)

                if self.pose_estimator is not None:
                    aligned_imgs = []
                    for i in range(img.shape[0]):
                        pose = self.pose_estimator.predict_pose(img[i])
                        aligned = flare_image_transform(img[i], pose_2d=pose)
                        aligned_imgs.append(aligned)
                    img = torch.stack(aligned_imgs).to(self.device)

                outputs = self.model.get_embedding(img)
                feat = outputs["feature"].cpu().numpy()
                mask = outputs["mask"].cpu().numpy()

                features_list.append(feat)
                masks_list.append(mask)

        features = np.concatenate(features_list, axis=0)
        masks = np.concatenate(masks_list, axis=0)

        return FLAREEmbeddingLoader(dataset.ids, features, masks)


class FLAREFullPipeline:
    """
    Complete Official FLARE Matching Pipeline (IEEE TIFS 2026):
    1. Align raw image using 2 pose estimators (VotingPose & RegressionPose) -> 2 aligned images.
    2. Enhance each aligned image using 2 enhancers (PriorEnh & UNetEnh) -> 4 enhanced images.
    3. Extract dense descriptors & foreground masks for each of the 4 images using FDRN (FDD).
    4. Store 4 representations per fingerprint for max similarity score matching across combinations (Eq. 7 & 8).
    """
    def __init__(
        self,
        desc_model_path: str = "../FLARE/model_weights/desc_model.pth.tar",
        voting_pose_path: str = "../FLARE/model_weights/VotingPose.pth",
        regression_pose_path: str = "../FLARE/model_weights/RegressionPose.pth",
        priorenh_dir: str = "../FLARE_ENH/pretrained_model/priorenh",
        unetenh_path: str = "../FLARE_ENH/pretrained_model/unetenh/unetenh.pth",
        device: str = "cuda",
        batch_size: int = 16,
    ):
        self.device_str = device if torch.cuda.is_available() and "cuda" in device else "cpu"
        self.device = torch.device(self.device_str)
        self.batch_size = batch_size

        # 2 Pose Estimators
        self.pose_voting = FLAREPoseEstimator(pose_type="VotingPose", model_path=voting_pose_path, device=device)
        self.pose_regression = FLAREPoseEstimator(pose_type="RegressionPose", model_path=regression_pose_path, device=device)

        # 2 Enhancers
        self.enh_prior = FLAREEnhancer(method="PriorEnh", ckpt_dir=priorenh_dir, device=device)
        self.enh_unet = FLAREEnhancer(method="UNetEnh", model_path=unetenh_path, device=device)

        # 1 FDRN Descriptor Extractor
        self.extractor = FLAREExtractor(model_path=desc_model_path, device=device)

    def extract(self, dataset: Dataset) -> FLAREEmbeddingLoader:
        self.extractor.model.eval()
        features_all = []
        masks_all = []

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False
        )

        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader, desc="Extracting Official FLARE 4-Combination Descriptors"):
                if isinstance(batch, dict) and "img" in batch:
                    raw_imgs = batch["img"].to(self.device)
                else:
                    raw_imgs = batch.to(self.device)

                batch_size = raw_imgs.shape[0]
                batch_features = []
                batch_masks = []

                for idx in range(batch_size):
                    img = raw_imgs[idx]

                    # 1. Two Pose Alignments (VotingPose & RegressionPose)
                    pose0 = self.pose_voting.predict_pose(img)
                    pose1 = self.pose_regression.predict_pose(img)

                    aligned0 = flare_image_transform(img, pose_2d=pose0).to(self.device)
                    aligned1 = flare_image_transform(img, pose_2d=pose1).to(self.device)

                    # 2. Four Enhanced Images (2 Poses x 2 Enhancers)
                    # Comb 0: Pose 0 + PriorEnh
                    # Comb 1: Pose 0 + UNetEnh
                    # Comb 2: Pose 1 + PriorEnh
                    # Comb 3: Pose 1 + UNetEnh
                    comb0 = self.enh_prior.enhance_tensor(aligned0[None])
                    comb1 = self.enh_unet.enhance_tensor(aligned0[None])
                    comb2 = self.enh_prior.enhance_tensor(aligned1[None])
                    comb3 = self.enh_unet.enhance_tensor(aligned1[None])

                    four_imgs = torch.cat([comb0, comb1, comb2, comb3], dim=0).to(self.device)

                    # 3. FDRN Descriptor & Mask Extraction for all 4 images
                    outputs = self.extractor.model.get_embedding(four_imgs)
                    feats = outputs["feature"].cpu().numpy()  # [4, 3072]
                    masks = outputs["mask"].cpu().numpy()     # [4, 256]

                    batch_features.append(feats)
                    batch_masks.append(masks)

                features_all.append(np.stack(batch_features, axis=0))  # [B, 4, 3072]
                masks_all.append(np.stack(batch_masks, axis=0))        # [B, 4, 256]

        features_array = np.concatenate(features_all, axis=0)  # [N, 4, 3072]
        masks_array = np.concatenate(masks_all, axis=0)        # [N, 4, 256]

        return FLAREEmbeddingLoader(dataset.ids, features_array, masks_array)


# Alias FLAREPipeline to FLAREFullPipeline for standard usage
FLAREPipeline = FLAREFullPipeline


def get_FLARE_FDD_extractor(
    model_path: str = "../FLARE/model_weights/desc_model.pth.tar",
    device: str = "cuda",
    pose_estimator: FLAREPoseEstimator | None = None,
    enhancer: FLAREEnhancer | None = None,
) -> FLAREExtractor:
    return FLAREExtractor(model_path=model_path, device=device, pose_estimator=pose_estimator, enhancer=enhancer)
