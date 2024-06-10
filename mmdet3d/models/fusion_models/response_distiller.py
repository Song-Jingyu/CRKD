from typing import Any, Dict

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
    build_loss
)
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import FUSIONMODELS
from mmdet3d.models.fusion_models.bevfusion import BEVFusion
from mmcv.runner import  load_checkpoint, _load_checkpoint, load_state_dict, _load_checkpoint_with_prefix

__all__ = ["ResponseDistiller"]

@FUSIONMODELS.register_module()
class ResponseDistiller(BEVFusion):
    def __init__(
        self,
        encoders,
        fuser,
        decoder,
        heads,
        encoders_student: Dict[str, Any],
        fuser_student: Dict[str, Any],
        decoder_student: Dict[str, Any],
        heads_student: Dict[str, Any],
        teacher_ckpt_path,
        student_ckpt_path,
        **kwargs,
    ) -> None:
        super().__init__(encoders, fuser, decoder, heads)

        # Initialize encoders of the student model
        self.encoders_student = nn.ModuleDict()
        if encoders_student.get("camera") is not None:
            self.encoders_student["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders_student["camera"]["backbone"]),
                    "neck": build_neck(encoders_student["camera"]["neck"]),
                    "vtransform": build_vtransform(encoders_student["camera"]["vtransform"]),
                }
            )
        if encoders_student.get("lidar") is not None:
            if encoders_student["lidar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders_student["lidar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders_student["lidar"]["voxelize"])
            self.encoders_student["lidar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders_student["lidar"]["backbone"]),
                }
            )
            self.voxelize_reduce_student = encoders_student["lidar"].get("voxelize_reduce", True)

        if encoders_student.get("radar") is not None:
            if encoders_student["radar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders_student["radar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders_student["radar"]["voxelize"])
            self.encoders_student["radar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders_student["radar"]["backbone"]),
                }
            )
            self.voxelize_reduce_student = encoders_student["radar"].get("voxelize_reduce", True)

        # Initialize the fuser of the student model
        if fuser_student is not None:
            self.fuser_student = build_fuser(fuser_student)
        else:
            self.fuser_student = None

        # Initialize decoder of the student model
        self.decoder_student = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder_student["backbone"]),
                "neck": build_neck(decoder_student["neck"]),
            }
        )

        # Initialize heads of the student model
        self.heads_student = nn.ModuleDict()
        for name in heads_student:
            if heads_student[name] is not None:
                self.heads_student[name] = build_head(heads_student[name])

        # This loss scale is for the original det loss of the student model
        if "loss_scale" in kwargs:
            self.loss_scale_student = kwargs["loss_scale"]
        else:
            self.loss_scale_student = dict()
            for name in heads_student:
                if heads_student[name] is not None:
                    self.loss_scale_student[name] = 1.0

        # If the camera's vtransform is a BEVDepth version, then we're using depth loss. 
        self.use_depth_loss_student = ((encoders_student.get('camera', {}) or {}).get('vtransform', {}) or {}).get('type', '') in ['BEVDepth', 'AwareBEVDepth', 'DBEVDepth', 'AwareDBEVDepth']

        self.init_weights_distill(teacher_ckpt_path, student_ckpt_path)

    
    """
    Initialize the pre-trained weights of the teacher model and freeze it
    Initialize the pre-trained weights of the student model
    """
    def init_weights_distill(self, teacher_ckpt_path, student_ckpt_path) -> None:
        # Step 1: Initialize the camera part of the student model
        # encoders_student
        student_encoder_ckpt = _load_checkpoint_with_prefix(prefix='encoders',filename=student_ckpt_path, map_location='cpu')
        load_state_dict(self.encoders_student, student_encoder_ckpt)
        # fuser_student
        student_fuser_ckpt = _load_checkpoint_with_prefix(prefix='fuser',filename=student_ckpt_path, map_location='cpu')
        load_state_dict(self.fuser_student, student_fuser_ckpt)
        # decoder_student
        student_decoder_ckpt = _load_checkpoint_with_prefix(prefix='decoder',filename=student_ckpt_path, map_location='cpu')
        load_state_dict(self.decoder_student, student_decoder_ckpt)
        # heads_student
        student_heads_ckpt = _load_checkpoint_with_prefix(prefix='heads',filename=student_ckpt_path, map_location='cpu')
        load_state_dict(self.heads_student, student_heads_ckpt)

        # Step 2: Initialize the trained part of the teacher model and freeze it
        # encoders
        teacher_encoder_ckpt = _load_checkpoint_with_prefix(prefix='encoders',filename=teacher_ckpt_path, map_location='cpu')
        load_state_dict(self.encoders, teacher_encoder_ckpt)
        for param in self.encoders.parameters():
            param.requires_grad = False
        self.encoders.eval()
        # fuser
        teacher_fuser_ckpt = _load_checkpoint_with_prefix(prefix='fuser',filename=teacher_ckpt_path, map_location='cpu')
        load_state_dict(self.fuser, teacher_fuser_ckpt)
        for param in self.fuser.parameters():
            param.requires_grad = False
        self.fuser.eval()
        # decoder
        teacher_decoder_ckpt = _load_checkpoint_with_prefix(prefix='decoder',filename=teacher_ckpt_path, map_location='cpu')
        load_state_dict(self.decoder, teacher_decoder_ckpt)
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.decoder.eval()
        # heads
        teacher_heads_ckpt = _load_checkpoint_with_prefix(prefix='heads',filename=teacher_ckpt_path, map_location='cpu')
        load_state_dict(self.heads, teacher_heads_ckpt)
        for param in self.heads.parameters():
            param.requires_grad = False
        self.heads.eval()
    

    """
    Extract the camera features of the student model
    """
    def extract_camera_features_student(
        self,
        x,
        points,
        radar_points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
        gt_depths=None,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)

        x = self.encoders_student["camera"]["backbone"](x)
        x = self.encoders_student["camera"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        x = self.encoders_student["camera"]["vtransform"](
            x,
            points,
            radar_points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
            depth_loss=self.use_depth_loss_student, 
            gt_depths=gt_depths,
        )
        return x
    

    """
    Extract the lidar/radar features of the student model
    """
    def extract_features_student(self, x, sensor) -> torch.Tensor:
        feats, coords, sizes = self.voxelize_student(x, sensor)
        batch_size = coords[-1, 0] + 1
        x = self.encoders_student[sensor]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x
    
    """
    Voxelization for student model
    """
    @torch.no_grad()
    @force_fp32()
    def voxelize_student(self, points, sensor):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders_student[sensor]["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce_student:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes

    
    """
    Forward the student model
    """
    @auto_fp16(apply_to=("img", "points"))
    def forward(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        depths,
        radar=None,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        if isinstance(img, list):
            raise NotImplementedError
        else:
            outputs = self.forward_single(
                img,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                depths,
                radar,
                gt_masks_bev,
                gt_bboxes_3d,
                gt_labels_3d,
                **kwargs,
            )
            return outputs

    @auto_fp16(apply_to=("img", "points"))
    def forward_single(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        depths=None,
        radar=None,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        # Forward the student model: Backbone, Neck
        features_student = []
        auxiliary_losses_student = {}
        for sensor in (
            self.encoders_student if self.training else list(self.encoders_student.keys())[::-1]
        ):
            if sensor == "camera":
                feature_student = self.extract_camera_features_student(
                    img,
                    points,
                    radar,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas,
                    gt_depths=depths,
                )
                if self.use_depth_loss_student:
                    feature_student, auxiliary_losses_student['depth'] = feature_student[0], feature_student[-1]
            elif sensor == "lidar":
                feature_student = self.extract_features_student(points, sensor)
            elif sensor == "radar":
                feature_student = self.extract_features_student(radar, sensor)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")

            features_student.append(feature_student)

        if not self.training:
            # avoid OOM
            features_student = features_student[::-1]

        if self.fuser_student is not None:
            _, x_student, _, _ = self.fuser_student(features_student)
        else:
            assert len(features_student) == 1, features_student
            x_student = features_student[0]

        batch_size = x_student.shape[0]

        x_student = self.decoder_student["backbone"](x_student)
        x_student = self.decoder_student["neck"](x_student)

        # Get the fused features of the teacher model
        if self.training:
            x_teacher = self.get_fused_feature_teacher( img,
                                                        points,
                                                        camera2ego,
                                                        lidar2ego,
                                                        lidar2camera,
                                                        lidar2image,
                                                        camera_intrinsics,
                                                        camera2lidar,
                                                        img_aug_matrix,
                                                        lidar_aug_matrix,
                                                        metas,
                                                        depths )
            x_teacher = self.decoder["backbone"](x_teacher)
            x_teacher = self.decoder["neck"](x_teacher)

        # Forward the student model: Decoder, Heads
        # Compute the loss of the student model
        if self.training:
            outputs = {}
            for type, head in self.heads_student.items():
                if type == "object": # for 3d object detection
                    # Student Model Forward
                    pred_dict = head(x_student, metas)
                    pred_dict_det = pred_dict
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict_det)
                    
                    # Teacher Model Forward
                    pred_dict_kd = pred_dict
                    pred_dict_teacher = self.heads[type](x_teacher, metas)
                    losses_response_kd = head.loss_soft(pred_dict_teacher, pred_dict_kd)

                elif type == "map": # for 3d semantic segmentation
                    losses = head(x_student, gt_masks_bev)
                else:
                    raise ValueError(f"unsupported head: {type}")
                
                # Det Loss scale
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{type}/{name}"] = val * self.loss_scale_student[type]
                    else:
                        outputs[f"stats/{type}/{name}"] = val
                
                # KD Loss scale
                for name, val in losses_response_kd.items():
                    if val.requires_grad:
                        outputs[f"loss/{type}/{name}"] = val * self.loss_scale_student[type]
                    else:
                        outputs[f"stats/{type}/{name}"] = val
            
            # Depth loss
            if self.use_depth_loss_student:
                if 'depth' in auxiliary_losses_student:
                    outputs["loss/depth"] = auxiliary_losses_student['depth']
                else:
                    raise ValueError('Use depth loss is true, but depth loss not found')

            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for type, head in self.heads_student.items():
                if type == "object":
                    pred_dict = head(x_student, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                elif type == "map":
                    logits = head(x_student)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")
            return outputs

        
