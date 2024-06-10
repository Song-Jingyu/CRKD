from typing import Any, Dict

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F
import numpy as np
from mmdet3d.core.points import LiDARPoints
from mmdet3d.core.bbox import LiDARInstance3DBoxes, box_np_ops
from copy import deepcopy

from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
    build_da,
    build_loss
)
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import FUSIONMODELS
from mmdet3d.models.fusion_models.bevfusion import BEVFusion
from mmcv.runner import  load_checkpoint, _load_checkpoint, load_state_dict, _load_checkpoint_with_prefix

__all__ = ["RelationResponseDistiller", "BEVResponseDistillerFusedLRC2CMask", "BEVResponseDistillerFusedLRC2CMaskScale", "BEVResponseDistillerFusedLRC2CMaskScaleRelation", "BEVResponseDistillerFusedLRC2CMaskScaleGaussian", "BEVResponseDistillerFusedLRC2CMaskScaleRelationGaussian"]

def sigmoid_normalized(x: torch.Tensor) -> torch.Tensor:
    # x shape 2, 1, 180, 180
    # out = torch.clamp(x.sigmoid_(), min=0.0, max=1.0)
    out = x.sigmoid_()
    # normalized to 0-1
    # find the max in dim 2 and dim 3
    max_x = torch.max(out, dim=2, keepdim=True)[0]
    max_x = torch.max(max_x, dim=3, keepdim=True)[0]
    
    min_x = torch.min(out, dim=2, keepdim=True)[0]
    min_x = torch.min(min_x, dim=3, keepdim=True)[0]

    out = (out - min_x) / (max_x - min_x) # 2, 1, 180, 180

    return out


def clip_sigmoid(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    return torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)




@FUSIONMODELS.register_module()
class RelationResponseDistiller(BEVFusion):
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
        # For fused kd
        fused_affinity_loss = dict(type="Affinity_Loss", reduction="mean", loss_weight=1.0, downsample_size=[32, 16, 8], input_channels=256, use_adapt=True),
        fused_affinity_loss_scale_kd: float = 1.0,
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

        # Initialize the relation kd loss
        self.fused_affinity_loss = build_loss(fused_affinity_loss)

        # This loss scale is for the original det loss of the student model
        # loss_scale is default for student model in BEVDistiller
        if "loss_scale" in kwargs:
            self.loss_scale_student = kwargs["loss_scale"]
        else:
            self.loss_scale_student = dict()
            for name in heads_student:
                if heads_student[name] is not None:
                    self.loss_scale_student[name] = 1.0

        # Fused KD loss scale
        self.fused_affinity_loss_scale_kd = fused_affinity_loss_scale_kd

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
        B, N, C, H, W = x.size() # batchsize, num_view, channel, height, width
        x = x.view(B * N, C, H, W) # batchsize * num_view, channel, height, width

        x = self.encoders_student["camera"]["backbone"](x)
        x = self.encoders_student["camera"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size() # batchsize * num_view, channel, height, width
        x = x.view(B, int(BN / B), C, H, W) # batchsize, num_view, channel, height, width

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
        # Forward the teacher model: Backbone, Neck
        # Get the fused features of the teacher model
        if self.training:
            x_teacher, lidar_teacher, cam_teacher = self.get_feature_teacher( img,
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
            fused_teacher_features = x_teacher
            # print("teacher feature shape: ", fused_teacher_features.shape) # (3, 256, 180, 180)
            x_teacher = self.decoder["backbone"](x_teacher)
            x_teacher = self.decoder["neck"](x_teacher)

        # Forward the student model: Backbone, Neck
        features_student = []
        radar_student = None
        cam_student = None
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
            fused_student_features = x_student
            # print("fused student feature shape: ", fused_student_features.shape) # (3, 64, 128, 128)
        else:
            assert len(features_student) == 1, features_student
            x_student = features_student[0]

        batch_size = x_student.shape[0]

        x_student = self.decoder_student["backbone"](x_student)
        x_student = self.decoder_student["neck"](x_student)
        # print("student feature shape: ", x_student.shape) # (3, 256, 128, 128)


        # Forward the student model: Decoder, Heads
        # Compute the loss of the student model
        if self.training:
            outputs = {}
            for type, head in self.heads_student.items():
                if type == "object": # for 3d object detection
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
            
            if self.use_depth_loss_student:
                if 'depth' in auxiliary_losses_student:
                    outputs["loss/depth"] = auxiliary_losses_student['depth']
                else:
                    raise ValueError('Use depth loss is true, but depth loss not found')
            
            # Fusion Relation KD loss
            outputs["loss/object/fused_affinity_kd"] = self.fused_affinity_loss_scale_kd * self.fused_affinity_loss(fused_student_features, fused_teacher_features)

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



@FUSIONMODELS.register_module()
class BEVResponseDistillerFusedLRC2CMask(BEVFusion):
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
        lira_domain_adaptation: Dict[str, Any] = None,
        lira_conv_channel: Dict[str, Any] = None,
        # lira_loss = dict(type="MSELoss", reduction="mean", loss_weight=1.0),
        lira_loss: Dict[str, Any] = None,
        lira_feat_loss_scale_kd: float = 1.0,
        cam_domain_adaptation: Dict[str, Any] = None,
        cam_conv_channel: Dict[str, Any] = None,
        # cam_loss = dict(type="MSELoss", reduction="mean", loss_weight=1.0),
        cam_loss: Dict[str, Any] = None,
        cam_feat_loss_scale_kd: float = 1.0,
        fused_domain_adaptation: Dict[str, Any] = None,
        fused_conv_channel: Dict[str, Any] = None,
        fused_loss: Dict[str, Any] = None,
        fused_feat_loss_scale_kd: float = 1.0,
        lr_kd_loss: Dict[str, Any] = None,
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
        
        # Initialize the domain adaptation module
        if lira_domain_adaptation is not None:
            self.lira_da = build_da(lira_domain_adaptation)
        else:
            self.lira_da = None
        if lira_conv_channel is not None:
            self.lira_conv_channel = build_da(lira_conv_channel)
        else:
            self.lira_conv_channel = None
        if cam_domain_adaptation is not None:
            self.cam_da = build_da(cam_domain_adaptation)
        else:
            self.cam_da = None
        if cam_conv_channel is not None:
            self.cam_conv_channel = build_da(cam_conv_channel)
        else:
            self.cam_conv_channel = None

        if fused_domain_adaptation is not None:
            self.fused_da = build_da(fused_domain_adaptation)
        else:
            self.fused_da = None
        if fused_conv_channel is not None:
            self.fused_conv_channel = build_da(fused_conv_channel)
        else:
            self.fused_conv_channel = None

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

        self.fused_loss = fused_loss

        # Initialize the feature kd loss
        if fused_loss is not None:
            self.fused_feat_loss = build_loss(fused_loss)
        if lira_loss is not None:   
            self.lira_feat_loss = build_loss(lira_loss)
        if cam_loss is not None:
            self.cam_feat_loss = build_loss(cam_loss)
        
        if lr_kd_loss:
            self.lr_kd_loss = build_loss(lr_kd_loss)

        # This loss scale is for the original det loss of the student model
        # loss_scale is default for student model in BEVDistiller
        if "loss_scale" in kwargs:
            self.loss_scale_student = kwargs["loss_scale"]
        else:
            self.loss_scale_student = dict()
            for name in heads_student:
                if heads_student[name] is not None:
                    self.loss_scale_student[name] = 1.0

        # Fused KD loss scale
        self.fused_feat_loss_scale_kd = fused_feat_loss_scale_kd
        
        # LiRa KD loss scale
        self.lira_feat_loss_scale_kd = lira_feat_loss_scale_kd

        # Cam2Cam KD loss scale
        self.cam_feat_loss_scale_kd = cam_feat_loss_scale_kd

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
        B, N, C, H, W = x.size() # batchsize, num_view, channel, height, width
        x = x.view(B * N, C, H, W) # batchsize * num_view, channel, height, width

        x = self.encoders_student["camera"]["backbone"](x)
        x = self.encoders_student["camera"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size() # batchsize * num_view, channel, height, width
        x = x.view(B, int(BN / B), C, H, W) # batchsize, num_view, channel, height, width

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
        if sensor == "radar" and self.lr_kd_loss is not None:
            feats, coords, sizes = self.voxelize_student(x, sensor)
            batch_size = coords[-1, 0] + 1
            x, x_activation = self.encoders_student[sensor]["backbone"](feats, coords, batch_size, sizes=sizes)
            return x, x_activation
        else:
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
        # Forward the teacher model: Backbone, Neck
        # Get the fused features of the teacher model
        if self.training:
            x_teacher, lidar_teacher, cam_teacher = self.get_gated_feature_teacher( img,
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
            fused_teacher_features = x_teacher
            # print("teacher feature shape: ", fused_teacher_features.shape) # (3, 256, 180, 180)
            x_teacher = self.decoder["backbone"](x_teacher)
            x_teacher = self.decoder["neck"](x_teacher)

        # Forward the student model: Backbone, Neck
        features_student = []
        radar_student = None
        cam_student = None
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
                # Cam2Cam KD
                cam_student = feature_student
                if self.cam_conv_channel is not None:
                    cam_student = self.cam_conv_channel(cam_student)
                if self.cam_da is not None:
                    cam_student = self.cam_da(cam_student)
            elif sensor == "lidar":
                feature_student = self.extract_features_student(points, sensor)
            elif sensor == "radar":
                if self.lr_kd_loss is not None:
                    feature_student, radar_activation = self.extract_features_student(radar, sensor)
                else:
                    feature_student = self.extract_features_student(radar, sensor)
                # Li2Ra KD
                radar_student = feature_student
                if self.lira_conv_channel is not None:
                    radar_student = self.lira_conv_channel(radar_student)
                if self.lira_da is not None:
                    radar_student = self.lira_da(radar_student)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")

            features_student.append(feature_student)

        if not self.training:
            # avoid OOM
            features_student = features_student[::-1]

        if self.fuser_student is not None:
            _, x_student, cam_student, radar_student = self.fuser_student(features_student)
            fused_student_features = x_student
            # print("fused student feature shape: ", fused_student_features.shape) # (3, 64, 128, 128)

            # Fusion KD
            if self.fused_da is not None:
                fused_student_features = self.fused_da(fused_student_features)
                x_student = fused_student_features
            if self.fused_conv_channel is not None:
                fused_student_features = self.fused_conv_channel(fused_student_features)
        else:
            assert len(features_student) == 1, features_student
            x_student = features_student[0]

        batch_size = x_student.shape[0]

        x_student = self.decoder_student["backbone"](x_student)
        x_student = self.decoder_student["neck"](x_student)
        # print("student feature shape: ", x_student.shape) # (3, 256, 128, 128)


        # Forward the student model: Decoder, Heads
        # Compute the loss of the student model
        if self.training:
            outputs = {}
            for type, head in self.heads_student.items():
                if type == "object": # for 3d object detection
                    pred_dict = head(x_student, metas)
                    pred_dict_det = pred_dict
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict_det)

                    # Teacher Model Forward
                    pred_dict_kd = pred_dict
                    pred_dict_teacher = self.heads[type](x_teacher, metas)

                    if self.lr_kd_loss is not None:
                        lidar_heatmap_dynamic = [] # 2, 1, 180, 180
                        for pred in pred_dict_teacher:
                            # print max and min of the heatmap
                            # print(torch.max(pred[0]["heatmap"]))
                            # print(torch.min(pred[0]["heatmap"]))
                            # print('------------')
                            # print(pred[0]["heatmap"].shape)
                            
                            # first version
                            # pred_mean = torch.mean(pred[0]["heatmap"], dim=1, keepdim=True)
                            # pred_mean = clip_sigmoid(pred_mean.to(torch.float32))
                            # lidar_heatmap_dynamic += pred_mean # TODO: try dynamic object only

                            # second version
                            # pred_max = torch.max(pred[0]["heatmap"], dim=1, keepdim=True)[0]
                            # pred_max = sigmoid_normalized(pred_max.to(torch.float32))
                            # lidar_heatmap_dynamic.append(pred_max)
                            pred_mean = torch.mean(pred[0]["heatmap"], dim=1, keepdim=True)
                            pred_mean = clip_sigmoid(pred_mean.to(torch.float32))
                            lidar_heatmap_dynamic.append(pred_mean)

                            
                        
                        lidar_heatmap_dynamic = torch.cat(lidar_heatmap_dynamic, dim=1) # 2, 6, 180, 180
                        # get the max of the heatmap
                        # lidar_heatmap_dynamic = torch.max(lidar_heatmap_dynamic, dim=1, keepdim=True)[0] # 2, 1, 180, 180
                        # get the mean of the heatmap
                        lidar_heatmap_dynamic = torch.mean(lidar_heatmap_dynamic, dim=1, keepdim=True) # 2, 1, 180, 180
                    # print('----------------------')
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
            
            if self.use_depth_loss_student:
                if 'depth' in auxiliary_losses_student:
                    outputs["loss/depth"] = auxiliary_losses_student['depth']
                else:
                    raise ValueError('Use depth loss is true, but depth loss not found')
            
            # Fusion KD loss
            if self.fused_loss is not None:
                outputs["loss/object/fused_feat_kd"] = self.fused_feat_loss_scale_kd * self.fused_feat_loss(fused_student_features, fused_teacher_features)
            # Cam2Cam KD loss
            if self.cam_feat_loss_scale_kd is not None:
                gt_mask_feat = self.heads_student["object"].get_gt_masks(gt_bboxes_3d) # should be list of tensor
                outputs["loss/object/cam_mask_feat_kd"] = self.cam_feat_loss_scale_kd * self.cam_feat_loss(cam_student, cam_teacher, gt_mask_feat)
                # outputs["loss/object/cam_feat_kd"] = self.cam_feat_loss_scale_kd * self.cam_feat_loss(cam_student, cam_teacher)

            # LR KD loss
            if self.lr_kd_loss is not None:
                # print(radar_activateion.shape)
                # print max and min of radar_activation and lidar_heatmap_dynamic
                # print(torch.max(radar_activation))
                # print(torch.min(radar_activation))
                # print(torch.max(lidar_heatmap_dynamic))
                # print(torch.min(lidar_heatmap_dynamic))

                outputs["loss/object/lr_kd"] = self.lr_kd_loss(radar_activation, lidar_heatmap_dynamic)

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



@FUSIONMODELS.register_module()
class BEVResponseDistillerFusedLRC2CMaskScale(BEVFusion):
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
        lira_domain_adaptation: Dict[str, Any] = None,
        lira_conv_channel: Dict[str, Any] = None,
        # lira_loss = dict(type="MSELoss", reduction="mean", loss_weight=1.0),
        lira_loss: Dict[str, Any] = None,
        lira_feat_loss_scale_kd: float = 1.0,
        cam_domain_adaptation: Dict[str, Any] = None,
        cam_conv_channel: Dict[str, Any] = None,
        # cam_loss = dict(type="MSELoss", reduction="mean", loss_weight=1.0),
        cam_loss: Dict[str, Any] = None,
        cam_feat_loss_scale_kd: float = 1.0,
        fused_domain_adaptation: Dict[str, Any] = None,
        fused_conv_channel: Dict[str, Any] = None,
        fused_loss: Dict[str, Any] = None,
        fused_feat_loss_scale_kd: float = 1.0,
        lr_kd_loss: Dict[str, Any] = None,
        scale_bbox: bool = False,
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
        
        # Initialize the domain adaptation module
        if lira_domain_adaptation is not None:
            self.lira_da = build_da(lira_domain_adaptation)
        else:
            self.lira_da = None
        if lira_conv_channel is not None:
            self.lira_conv_channel = build_da(lira_conv_channel)
        else:
            self.lira_conv_channel = None
        if cam_domain_adaptation is not None:
            self.cam_da = build_da(cam_domain_adaptation)
        else:
            self.cam_da = None
        if cam_conv_channel is not None:
            self.cam_conv_channel = build_da(cam_conv_channel)
        else:
            self.cam_conv_channel = None

        if fused_domain_adaptation is not None:
            self.fused_da = build_da(fused_domain_adaptation)
        else:
            self.fused_da = None
        if fused_conv_channel is not None:
            self.fused_conv_channel = build_da(fused_conv_channel)
        else:
            self.fused_conv_channel = None

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

        self.fused_loss = fused_loss

        # Initialize the feature kd loss
        if fused_loss is not None:
            self.fused_feat_loss = build_loss(fused_loss)
        if lira_loss is not None:   
            self.lira_feat_loss = build_loss(lira_loss)
        if cam_loss is not None:
            self.cam_feat_loss = build_loss(cam_loss)
        
        if lr_kd_loss:
            self.lr_kd_loss = build_loss(lr_kd_loss)

        # This loss scale is for the original det loss of the student model
        # loss_scale is default for student model in BEVDistiller
        if "loss_scale" in kwargs:
            self.loss_scale_student = kwargs["loss_scale"]
        else:
            self.loss_scale_student = dict()
            for name in heads_student:
                if heads_student[name] is not None:
                    self.loss_scale_student[name] = 1.0

        # Fused KD loss scale
        self.fused_feat_loss_scale_kd = fused_feat_loss_scale_kd
        
        # LiRa KD loss scale
        self.lira_feat_loss_scale_kd = lira_feat_loss_scale_kd

        # Cam2Cam KD loss scale
        self.cam_feat_loss_scale_kd = cam_feat_loss_scale_kd

        # If the camera's vtransform is a BEVDepth version, then we're using depth loss. 
        self.use_depth_loss_student = ((encoders_student.get('camera', {}) or {}).get('vtransform', {}) or {}).get('type', '') in ['BEVDepth', 'AwareBEVDepth', 'DBEVDepth', 'AwareDBEVDepth']

        self.scale_bbox = scale_bbox

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
        B, N, C, H, W = x.size() # batchsize, num_view, channel, height, width
        x = x.view(B * N, C, H, W) # batchsize * num_view, channel, height, width

        x = self.encoders_student["camera"]["backbone"](x)
        x = self.encoders_student["camera"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size() # batchsize * num_view, channel, height, width
        x = x.view(B, int(BN / B), C, H, W) # batchsize, num_view, channel, height, width

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
        if sensor == "radar" and self.lr_kd_loss is not None:
            feats, coords, sizes = self.voxelize_student(x, sensor)
            batch_size = coords[-1, 0] + 1
            x, x_activation = self.encoders_student[sensor]["backbone"](feats, coords, batch_size, sizes=sizes)
            return x, x_activation
        else:
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

    def scale_mask(self, gt_bboxes_3d: list):
        # create a copy of gt_bboxes_3d
        output = []
        for i in range(len(gt_bboxes_3d)):
            gt_bboxes_3d_copy = deepcopy(gt_bboxes_3d[i])
            gt_bboxes_3d_copy_tensor = gt_bboxes_3d_copy.tensor # Nx9
            for j in range(gt_bboxes_3d_copy_tensor.shape[0]):
                # get the range of the bbox
                bbox = gt_bboxes_3d_copy_tensor[j] # box: x, y, z, x_size, y_size, z_size, yaw, vx, vy
                x = bbox[0]
                y = bbox[1]
                bbox_range = np.sqrt(x**2 + y**2)

                d_x_size = 0
                d_y_size = 0

                if bbox_range > 20 and bbox_range < 30:
                    # scale up the size of the bbox
                    # d_x_size = max([bbox[3] * 0.25, 1.0])
                    # d_y_size = max([bbox[4] * 0.25, 1.0])
                    d_x_size += bbox[3] * 0.25
                    d_y_size += bbox[4] * 0.25
                
                if bbox_range >= 30:
                    # scale up the size of the bbox
                    # d_x_size = max([bbox[3] * 0.5, 1.0])
                    # d_y_size = max([bbox[4] * 0.5, 1.0])
                    d_x_size += bbox[3] * 0.5
                    d_y_size += bbox[4] * 0.5
                
                # if the bbox_range < 10 do nothing
               
                # consider the velocity of the bbox
                vx = bbox[7]
                vy = bbox[8]

                if abs(vx) > 0.3 and abs(vx) < 0.8:
                    d_x_size += bbox[3] * 0.25
                if abs(vx) >= 0.8:
                    d_x_size += bbox[3] * 0.5

                if abs(vy) > 0.3 and abs(vy) < 0.8:
                    d_y_size += bbox[4] * 0.25
                if abs(vy) >= 0.8:
                    d_y_size += bbox[4] * 0.5
                
                # min for d_size is 0.5 and max is 4.0
                d_x_size = min([max([d_x_size, 0.5]), 4.0])
                d_y_size = min([max([d_y_size, 0.5]), 4.0])

                # update the bbox
                gt_bboxes_3d_copy_tensor[j][3] += d_x_size
                gt_bboxes_3d_copy_tensor[j][4] += d_y_size
            
            output.append(gt_bboxes_3d_copy)
        return output




        # gt_bboxes_3d_copy = deepcopy(gt_bboxes_3d)
        # gt_bboxes_3d_copy_tensor = gt_bboxes_3d_copy.tensor # Nx9
        # assert gt_bboxes_3d_copy.tensor.shape[1] == 9 # x, y, z, x_size, y_size, z_size, yaw, vx, vy (?)

        # print("gt boxes: ", gt_bboxes_3d_copy_tensor)
        # print('------------------')

        # # go through each bbox
        # for i in range(len(gt_bboxes_3d_scaled)):
        #     # get the bbox
        #     bbox = gt_bboxes_3d_scaled[i]

        #     # get the center of the bbox
        #     box_x = 
            


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
        # Forward the teacher model: Backbone, Neck
        # Get the fused features of the teacher model
        if self.training:
            x_teacher, lidar_teacher, cam_teacher = self.get_gated_feature_teacher( img,
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
            fused_teacher_features = x_teacher
            # print("teacher feature shape: ", fused_teacher_features.shape) # (3, 256, 180, 180)
            x_teacher = self.decoder["backbone"](x_teacher)
            x_teacher = self.decoder["neck"](x_teacher)

        # Forward the student model: Backbone, Neck
        features_student = []
        radar_student = None
        cam_student = None
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
                # Cam2Cam KD
                cam_student = feature_student
                if self.cam_conv_channel is not None:
                    cam_student = self.cam_conv_channel(cam_student)
                if self.cam_da is not None:
                    cam_student = self.cam_da(cam_student)
            elif sensor == "lidar":
                feature_student = self.extract_features_student(points, sensor)
            elif sensor == "radar":
                if self.lr_kd_loss is not None:
                    feature_student, radar_activation = self.extract_features_student(radar, sensor)
                else:
                    feature_student = self.extract_features_student(radar, sensor)
                # Li2Ra KD
                radar_student = feature_student
                if self.lira_conv_channel is not None:
                    radar_student = self.lira_conv_channel(radar_student)
                if self.lira_da is not None:
                    radar_student = self.lira_da(radar_student)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")

            features_student.append(feature_student)

        if not self.training:
            # avoid OOM
            features_student = features_student[::-1]

        if self.fuser_student is not None:
            _, x_student, cam_student, radar_student = self.fuser_student(features_student)
            fused_student_features = x_student
            # print("fused student feature shape: ", fused_student_features.shape) # (3, 64, 128, 128)

            # Fusion KD
            if self.fused_da is not None:
                fused_student_features = self.fused_da(fused_student_features)
                x_student = fused_student_features
            if self.fused_conv_channel is not None:
                fused_student_features = self.fused_conv_channel(fused_student_features)
        else:
            assert len(features_student) == 1, features_student
            x_student = features_student[0]

        batch_size = x_student.shape[0]

        x_student = self.decoder_student["backbone"](x_student)
        x_student = self.decoder_student["neck"](x_student)
        # print("student feature shape: ", x_student.shape) # (3, 256, 128, 128)


        # Forward the student model: Decoder, Heads
        # Compute the loss of the student model
        if self.training:
            outputs = {}
            for type, head in self.heads_student.items():
                if type == "object": # for 3d object detection
                    pred_dict = head(x_student, metas)
                    pred_dict_det = pred_dict
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict_det)

                    # Teacher Model Forward
                    pred_dict_kd = pred_dict
                    pred_dict_teacher = self.heads[type](x_teacher, metas)

                    if self.lr_kd_loss is not None:
                        lidar_heatmap_dynamic = [] # 2, 1, 180, 180
                        for pred in pred_dict_teacher:
                            # print max and min of the heatmap
                            # print(torch.max(pred[0]["heatmap"]))
                            # print(torch.min(pred[0]["heatmap"]))
                            # print('------------')
                            # print(pred[0]["heatmap"].shape)
                            
                            # first version
                            # pred_mean = torch.mean(pred[0]["heatmap"], dim=1, keepdim=True)
                            # pred_mean = clip_sigmoid(pred_mean.to(torch.float32))
                            # lidar_heatmap_dynamic += pred_mean # TODO: try dynamic object only

                            # second version
                            # pred_max = torch.max(pred[0]["heatmap"], dim=1, keepdim=True)[0]
                            # pred_max = sigmoid_normalized(pred_max.to(torch.float32))
                            # lidar_heatmap_dynamic.append(pred_max)
                            pred_mean = torch.mean(pred[0]["heatmap"], dim=1, keepdim=True)
                            pred_mean = clip_sigmoid(pred_mean.to(torch.float32))
                            lidar_heatmap_dynamic.append(pred_mean)

                            
                        
                        lidar_heatmap_dynamic = torch.cat(lidar_heatmap_dynamic, dim=1) # 2, 6, 180, 180
                        # get the max of the heatmap
                        # lidar_heatmap_dynamic = torch.max(lidar_heatmap_dynamic, dim=1, keepdim=True)[0] # 2, 1, 180, 180
                        # get the mean of the heatmap
                        lidar_heatmap_dynamic = torch.mean(lidar_heatmap_dynamic, dim=1, keepdim=True) # 2, 1, 180, 180
                    # print('----------------------')
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
            
            if self.use_depth_loss_student:
                if 'depth' in auxiliary_losses_student:
                    outputs["loss/depth"] = auxiliary_losses_student['depth']
                else:
                    raise ValueError('Use depth loss is true, but depth loss not found')
            
            if self.scale_bbox:
                # print("scale bbox")
                gt_bboxes_scaled = self.scale_mask(gt_bboxes_3d)
                gt_mask_feat = self.heads_student["object"].get_gt_masks(gt_bboxes_scaled) # should be list of tensor
                # gt_mask_feat_compare = self.heads_student["object"].get_gt_masks(gt_bboxes_3d) # should be list of tensor
                # print("mask feat: ", gt_mask_feat)
                # print('----------------------------------')
                # print("mask feat compare: ", gt_mask_feat_compare)
                # print('----------------------------------')
            else:
                gt_mask_feat = self.heads_student["object"].get_gt_masks(gt_bboxes_3d) # should be list of tensor
            
            # Fusion KD loss
            if self.fused_loss is not None:
                outputs["loss/object/fused_mask_scale_feat_kd"] = self.fused_feat_loss_scale_kd * self.fused_feat_loss(fused_student_features, fused_teacher_features, gt_mask_feat)
            # Cam2Cam KD loss
            if self.cam_feat_loss_scale_kd is not None:
                # print("cam student: ", cam_student)
                # print("cam teacher: ", cam_teacher)
                outputs["loss/object/cam_mask_scale_feat_kd"] = self.cam_feat_loss_scale_kd * self.cam_feat_loss(cam_student, cam_teacher, gt_mask_feat)
                # print("cam_mask_feat_kd: ", outputs["loss/object/cam_mask_feat_kd"])
                # outputs["loss/object/cam_feat_kd"] = self.cam_feat_loss_scale_kd * self.cam_feat_loss(cam_student, cam_teacher)


            # LR KD loss
            if self.lr_kd_loss is not None:
                # print(radar_activateion.shape)
                # print max and min of radar_activation and lidar_heatmap_dynamic
                # print(torch.max(radar_activation))
                # print(torch.min(radar_activation))
                # print(torch.max(lidar_heatmap_dynamic))
                # print(torch.min(lidar_heatmap_dynamic))
                # radar_activation_np = radar_activation.detach().cpu().numpy()
                # lidar_heatmap_dynamic_np = lidar_heatmap_dynamic.detach().cpu().numpy()
                # save
                # print("saving files")
                # np.save('/mnt/ws-frb/users/jingyuso/docker_data/docker/home/crkd/KD-CR/debug_files/radar_activation.npy', radar_activation_np)
                # np.save('/mnt/ws-frb/users/jingyuso/docker_data/docker/home/crkd/KD-CR/debug_files/lidar_heatmap_dynamic.npy', lidar_heatmap_dynamic_np)
                outputs["loss/object/lr_kd"] = self.lr_kd_loss(radar_activation, lidar_heatmap_dynamic)

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



@FUSIONMODELS.register_module()
class BEVResponseDistillerFusedLRC2CMaskScaleGaussian(BEVFusion):
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
        cam_domain_adaptation: Dict[str, Any] = None,
        cam_conv_channel: Dict[str, Any] = None,
        # cam_loss = dict(type="MSELoss", reduction="mean", loss_weight=1.0),
        cam_loss: Dict[str, Any] = None,
        cam_feat_loss_scale_kd: float = 1.0,
        fused_domain_adaptation: Dict[str, Any] = None,
        fused_conv_channel: Dict[str, Any] = None,
        fused_loss: Dict[str, Any] = None,
        fused_feat_loss_scale_kd: float = 1.0,
        lr_kd_loss: Dict[str, Any] = None,
        scale_bbox: bool = False,
        use_gaussian: bool = False,
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
        
        # Initialize the domain adaptation module
        if cam_domain_adaptation is not None:
            self.cam_da = build_da(cam_domain_adaptation)
        else:
            self.cam_da = None
        if cam_conv_channel is not None:
            self.cam_conv_channel = build_da(cam_conv_channel)
        else:
            self.cam_conv_channel = None

        if fused_domain_adaptation is not None:
            self.fused_da = build_da(fused_domain_adaptation)
        else:
            self.fused_da = None
        if fused_conv_channel is not None:
            self.fused_conv_channel = build_da(fused_conv_channel)
        else:
            self.fused_conv_channel = None

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

        self.fused_loss = fused_loss
        self.cam_loss = cam_loss

        # Initialize the feature kd loss
        if fused_loss is not None:
            self.fused_feat_loss = build_loss(fused_loss)
        if cam_loss is not None:
            self.cam_feat_loss = build_loss(cam_loss)
        if lr_kd_loss is not None:
            self.lr_kd_loss = build_loss(lr_kd_loss)

        # This loss scale is for the original det loss of the student model
        # loss_scale is default for student model in BEVDistiller
        if "loss_scale" in kwargs:
            self.loss_scale_student = kwargs["loss_scale"]
        else:
            self.loss_scale_student = dict()
            for name in heads_student:
                if heads_student[name] is not None:
                    self.loss_scale_student[name] = 1.0

        # Fused KD loss scale
        self.fused_feat_loss_scale_kd = fused_feat_loss_scale_kd

        # Cam2Cam KD loss scale
        self.cam_feat_loss_scale_kd = cam_feat_loss_scale_kd

        # If the camera's vtransform is a BEVDepth version, then we're using depth loss. 
        self.use_depth_loss_student = ((encoders_student.get('camera', {}) or {}).get('vtransform', {}) or {}).get('type', '') in ['BEVDepth', 'AwareBEVDepth', 'DBEVDepth', 'AwareDBEVDepth']

        self.scale_bbox = scale_bbox

        self.use_gaussian = use_gaussian

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
        B, N, C, H, W = x.size() # batchsize, num_view, channel, height, width
        x = x.view(B * N, C, H, W) # batchsize * num_view, channel, height, width

        x = self.encoders_student["camera"]["backbone"](x)
        x = self.encoders_student["camera"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size() # batchsize * num_view, channel, height, width
        x = x.view(B, int(BN / B), C, H, W) # batchsize, num_view, channel, height, width

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
        if sensor == "radar" and self.lr_kd_loss is not None:
            feats, coords, sizes = self.voxelize_student(x, sensor)
            batch_size = coords[-1, 0] + 1
            x, x_activation = self.encoders_student[sensor]["backbone"](feats, coords, batch_size, sizes=sizes)
            return x, x_activation
        else:
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

    def scale_mask(self, gt_bboxes_3d: list):
        # create a copy of gt_bboxes_3d
        output = []
        for i in range(len(gt_bboxes_3d)):
            gt_bboxes_3d_copy = deepcopy(gt_bboxes_3d[i])
            gt_bboxes_3d_copy_tensor = gt_bboxes_3d_copy.tensor # Nx9
            gt_bboxes_3d_copy_tensor.to(gt_bboxes_3d[0].device)
            for j in range(gt_bboxes_3d_copy_tensor.shape[0]):
                # get the range of the bbox
                bbox = gt_bboxes_3d_copy_tensor[j] # box: x, y, z, x_size, y_size, z_size, yaw, vx, vy
                x = bbox[0]
                y = bbox[1]
                bbox_range = np.sqrt(x**2 + y**2)

                d_x_size = 0
                d_y_size = 0

                if bbox_range > 20 and bbox_range < 30:
                    # scale up the size of the bbox
                    # d_x_size = max([bbox[3] * 0.25, 1.0])
                    # d_y_size = max([bbox[4] * 0.25, 1.0])
                    d_x_size += bbox[3] * 0.25
                    d_y_size += bbox[4] * 0.25
                
                if bbox_range >= 30:
                    # scale up the size of the bbox
                    # d_x_size = max([bbox[3] * 0.5, 1.0])
                    # d_y_size = max([bbox[4] * 0.5, 1.0])
                    d_x_size += bbox[3] * 0.5
                    d_y_size += bbox[4] * 0.5
                
                # if the bbox_range < 10 do nothing
               
                # consider the velocity of the bbox
                vx = bbox[7]
                vy = bbox[8]

                if abs(vx) > 0.3 and abs(vx) < 0.8:
                    d_x_size += bbox[3] * 0.25
                if abs(vx) >= 0.8:
                    d_x_size += bbox[3] * 0.5

                if abs(vy) > 0.3 and abs(vy) < 0.8:
                    d_y_size += bbox[4] * 0.25
                if abs(vy) >= 0.8:
                    d_y_size += bbox[4] * 0.5
                
                # min for d_size is 0.5 and max is 4.0
                d_x_size = min([max([d_x_size, 0.5]), 4.0])
                d_y_size = min([max([d_y_size, 0.5]), 4.0])

                # update the bbox
                gt_bboxes_3d_copy_tensor[j][3] += d_x_size
                gt_bboxes_3d_copy_tensor[j][4] += d_y_size
            
            output.append(gt_bboxes_3d_copy)
        return output




        # gt_bboxes_3d_copy = deepcopy(gt_bboxes_3d)
        # gt_bboxes_3d_copy_tensor = gt_bboxes_3d_copy.tensor # Nx9
        # assert gt_bboxes_3d_copy.tensor.shape[1] == 9 # x, y, z, x_size, y_size, z_size, yaw, vx, vy (?)

        # print("gt boxes: ", gt_bboxes_3d_copy_tensor)
        # print('------------------')

        # # go through each bbox
        # for i in range(len(gt_bboxes_3d_scaled)):
        #     # get the bbox
        #     bbox = gt_bboxes_3d_scaled[i]

        #     # get the center of the bbox
        #     box_x = 
            


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
        # Forward the teacher model: Backbone, Neck
        # Get the fused features of the teacher model
        if self.training:
            x_teacher, lidar_teacher, cam_teacher = self.get_gated_feature_teacher( img,
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
            fused_teacher_features = x_teacher
            # print("teacher feature shape: ", fused_teacher_features.shape) # (3, 256, 180, 180)
            x_teacher = self.decoder["backbone"](x_teacher)
            x_teacher = self.decoder["neck"](x_teacher)

        # Forward the student model: Backbone, Neck
        features_student = []
        radar_student = None
        cam_student = None
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
                # Cam2Cam KD
                cam_student = feature_student
                if self.cam_conv_channel is not None:
                    cam_student = self.cam_conv_channel(cam_student)
                if self.cam_da is not None:
                    cam_student = self.cam_da(cam_student)
            elif sensor == "lidar":
                feature_student = self.extract_features_student(points, sensor)
            elif sensor == "radar":
                if self.lr_kd_loss is not None:
                    feature_student, radar_activation = self.extract_features_student(radar, sensor)
                else:
                    feature_student = self.extract_features_student(radar, sensor)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")

            features_student.append(feature_student)

        if not self.training:
            # avoid OOM
            features_student = features_student[::-1]

        if self.fuser_student is not None:
            _, x_student, cam_student, radar_student = self.fuser_student(features_student)
            fused_student_features = x_student
            # print("fused student feature shape: ", fused_student_features.shape) # (3, 64, 128, 128)

            # Fusion KD
            if self.fused_da is not None:
                fused_student_features = self.fused_da(fused_student_features)
                x_student = fused_student_features
            if self.fused_conv_channel is not None:
                fused_student_features = self.fused_conv_channel(fused_student_features)
        else:
            assert len(features_student) == 1, features_student
            x_student = features_student[0]

        batch_size = x_student.shape[0]

        x_student = self.decoder_student["backbone"](x_student)
        x_student = self.decoder_student["neck"](x_student)
        # print("student feature shape: ", x_student.shape) # (3, 256, 128, 128)


        # Forward the student model: Decoder, Heads
        # Compute the loss of the student model
        if self.training:
            outputs = {}
            for type, head in self.heads_student.items():
                if type == "object": # for 3d object detection
                    pred_dict = head(x_student, metas)
                    pred_dict_det = pred_dict
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict_det)

                    # Teacher Model Forward
                    pred_dict_kd = pred_dict
                    pred_dict_teacher = self.heads[type](x_teacher, metas)

                    if self.lr_kd_loss is not None:
                        lidar_heatmap_dynamic = [] # 2, 1, 180, 180
                        for pred in pred_dict_teacher:
                            # print max and min of the heatmap
                            # print(torch.max(pred[0]["heatmap"]))
                            # print(torch.min(pred[0]["heatmap"]))
                            # print('------------')
                            # print(pred[0]["heatmap"].shape)
                            
                            # first version
                            # pred_mean = torch.mean(pred[0]["heatmap"], dim=1, keepdim=True)
                            # pred_mean = clip_sigmoid(pred_mean.to(torch.float32))
                            # lidar_heatmap_dynamic += pred_mean # TODO: try dynamic object only

                            # second version
                            # pred_max = torch.max(pred[0]["heatmap"], dim=1, keepdim=True)[0]
                            # pred_max = sigmoid_normalized(pred_max.to(torch.float32))
                            # lidar_heatmap_dynamic.append(pred_max)
                            pred_mean = torch.mean(pred[0]["heatmap"], dim=1, keepdim=True)
                            pred_mean = clip_sigmoid(pred_mean.to(torch.float32))
                            lidar_heatmap_dynamic.append(pred_mean)

                            
                        
                        lidar_heatmap_dynamic = torch.cat(lidar_heatmap_dynamic, dim=1) # 2, 6, 180, 180
                        # get the max of the heatmap
                        # lidar_heatmap_dynamic = torch.max(lidar_heatmap_dynamic, dim=1, keepdim=True)[0] # 2, 1, 180, 180
                        # get the mean of the heatmap
                        lidar_heatmap_dynamic = torch.mean(lidar_heatmap_dynamic, dim=1, keepdim=True) # 2, 1, 180, 180
                    # print('----------------------')
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
            
            if self.use_depth_loss_student:
                if 'depth' in auxiliary_losses_student:
                    outputs["loss/depth"] = auxiliary_losses_student['depth']
                else:
                    raise ValueError('Use depth loss is true, but depth loss not found')
            
            if self.scale_bbox:
                # print("scale bbox")
                gt_bboxes_scaled = self.scale_mask(gt_bboxes_3d)
                gt_mask_feat = self.heads_student["object"].get_gt_masks(gt_bboxes_scaled) # should be list of tensor
                # gt_mask_feat_compare = self.heads_student["object"].get_gt_masks(gt_bboxes_3d) # should be list of tensor
                # print("mask feat: ", gt_mask_feat)
                # print('----------------------------------')
                # print("mask feat compare: ", gt_mask_feat_compare)
                # print('----------------------------------')
            else:
                gt_bboxes_scaled = gt_bboxes_3d
                gt_mask_feat = self.heads_student["object"].get_gt_masks(gt_bboxes_3d) # should be list of tensor

            if self.use_gaussian:
                gaussian_fg_map = self.heads_student["object"].get_gaussian_gt_masks(gt_bboxes_scaled) # (B, H, W)
                gaussian_fg_map = gaussian_fg_map.to(fused_student_features.device)
                # TODO: Please uncomment these lines to enable debug on gaussian map
                # Test start
                # gaussian_fg_map_np = gaussian_fg_map.detach().cpu().numpy()
                # np.save('/mnt/ws-frb/users/jingyuso/docker_data/docker/home/crkd/KD-CR/debug_files/gaussian_fg_map_np.npy', gaussian_fg_map_np)
                # Test end

                if self.fused_loss is not None:
                    fused_feat_loss_gaussian = self.fused_feat_loss_scale_kd * self.fused_feat_loss(fused_student_features, fused_teacher_features) # (B, H, W)
                    # mean by batchsize and map size
                    fused_feat_loss_gaussian = torch.sum(fused_feat_loss_gaussian * gaussian_fg_map) / torch.sum(gaussian_fg_map)
                    outputs["loss/object/fused_gaussian_mask_scale_feat_kd"] = fused_feat_loss_gaussian
                # Cam2Cam KD loss
                if self.cam_feat_loss_scale_kd is not None:
                    cam_feat_loss_gaussian = self.cam_feat_loss_scale_kd * self.cam_feat_loss(cam_student, cam_teacher) # (B, H, W)
                    # mean by batchsize and map size
                    cam_feat_loss_gaussian = torch.sum(cam_feat_loss_gaussian * gaussian_fg_map) / torch.sum(gaussian_fg_map)
                    outputs["loss/object/cam_gaussian_mask_scale_feat_kd"] = cam_feat_loss_gaussian
            else:
                # Fusion KD loss
                if self.fused_loss is not None:
                    outputs["loss/object/fused_mask_scale_feat_kd"] = self.fused_feat_loss_scale_kd * self.fused_feat_loss(fused_student_features, fused_teacher_features, gt_mask_feat)
                # Cam2Cam KD loss
                if self.cam_feat_loss_scale_kd is not None:
                    outputs["loss/object/cam_mask_scale_feat_kd"] = self.cam_feat_loss_scale_kd * self.cam_feat_loss(cam_student, cam_teacher, gt_mask_feat)


            # LR KD loss
            if self.lr_kd_loss is not None:
                # print(radar_activateion.shape)
                # print max and min of radar_activation and lidar_heatmap_dynamic
                # print(torch.max(radar_activation))
                # print(torch.min(radar_activation))
                # print(torch.max(lidar_heatmap_dynamic))
                # print(torch.min(lidar_heatmap_dynamic))
                # radar_activation_np = radar_activation.detach().cpu().numpy()
                # lidar_heatmap_dynamic_np = lidar_heatmap_dynamic.detach().cpu().numpy()
                # save
                # print("saving files")
                # np.save('/mnt/ws-frb/users/jingyuso/docker_data/docker/home/crkd/KD-CR/debug_files/radar_activation.npy', radar_activation_np)
                # np.save('/mnt/ws-frb/users/jingyuso/docker_data/docker/home/crkd/KD-CR/debug_files/lidar_heatmap_dynamic.npy', lidar_heatmap_dynamic_np)
                outputs["loss/object/lr_kd"] = self.lr_kd_loss(radar_activation, lidar_heatmap_dynamic)

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



@FUSIONMODELS.register_module()
class BEVResponseDistillerFusedLRC2CMaskScaleRelation(BEVFusion):
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
        cam_domain_adaptation: Dict[str, Any] = None,
        cam_conv_channel: Dict[str, Any] = None,
        # cam_loss = dict(type="MSELoss", reduction="mean", loss_weight=1.0),
        cam_loss: Dict[str, Any] = None,
        cam_feat_loss_scale_kd: float = 1.0,
        fused_domain_adaptation: Dict[str, Any] = None,
        fused_conv_channel: Dict[str, Any] = None,
        fused_loss: Dict[str, Any] = None,
        fused_feat_loss_scale_kd: float = 1.0,
        lr_kd_loss: Dict[str, Any] = None,
        scale_bbox: bool = False,
        fused_affinity_loss: Dict[str, Any] = None,
        # fused_affinity_loss = dict(type="Affinity_Loss", reduction="mean", loss_weight=1.0, downsample_size=[32, 16, 8], input_channels=256, use_adapt=True),
        fused_affinity_loss_scale_kd: float = 1.0,
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
        
        # Initialize the domain adaptation module
        if cam_domain_adaptation is not None:
            self.cam_da = build_da(cam_domain_adaptation)
        else:
            self.cam_da = None
        if cam_conv_channel is not None:
            self.cam_conv_channel = build_da(cam_conv_channel)
        else:
            self.cam_conv_channel = None

        if fused_domain_adaptation is not None:
            self.fused_da = build_da(fused_domain_adaptation)
        else:
            self.fused_da = None
        if fused_conv_channel is not None:
            self.fused_conv_channel = build_da(fused_conv_channel)
        else:
            self.fused_conv_channel = None

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

        self.fused_loss = fused_loss
        self.cam_loss = cam_loss
        self.fused_affinity_loss = fused_affinity_loss

        # Initialize the feature kd loss
        if fused_loss is not None:
            self.fused_feat_loss = build_loss(fused_loss)
        if cam_loss is not None:
            self.cam_feat_loss = build_loss(cam_loss)
        if fused_affinity_loss is not None:
            self.fused_feat_affinity_loss = build_loss(fused_affinity_loss)
        if lr_kd_loss:
            self.lr_kd_loss = build_loss(lr_kd_loss)
        else:
            self.lr_kd_loss = None

        # This loss scale is for the original det loss of the student model
        # loss_scale is default for student model in BEVDistiller
        if "loss_scale" in kwargs:
            self.loss_scale_student = kwargs["loss_scale"]
        else:
            self.loss_scale_student = dict()
            for name in heads_student:
                if heads_student[name] is not None:
                    self.loss_scale_student[name] = 1.0

        # Fused KD loss scale
        self.fused_feat_loss_scale_kd = fused_feat_loss_scale_kd

        # Cam2Cam KD loss scale
        self.cam_feat_loss_scale_kd = cam_feat_loss_scale_kd

        # Fused KD loss scale
        self.fused_affinity_loss_scale_kd = fused_affinity_loss_scale_kd

        # If the camera's vtransform is a BEVDepth version, then we're using depth loss. 
        self.use_depth_loss_student = ((encoders_student.get('camera', {}) or {}).get('vtransform', {}) or {}).get('type', '') in ['BEVDepth', 'AwareBEVDepth', 'DBEVDepth', 'AwareDBEVDepth']

        self.scale_bbox = scale_bbox

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
        B, N, C, H, W = x.size() # batchsize, num_view, channel, height, width
        x = x.view(B * N, C, H, W) # batchsize * num_view, channel, height, width

        x = self.encoders_student["camera"]["backbone"](x)
        x = self.encoders_student["camera"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size() # batchsize * num_view, channel, height, width
        x = x.view(B, int(BN / B), C, H, W) # batchsize, num_view, channel, height, width

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
        if sensor == "radar" and self.lr_kd_loss is not None:
            feats, coords, sizes = self.voxelize_student(x, sensor)
            batch_size = coords[-1, 0] + 1
            x, x_activation = self.encoders_student[sensor]["backbone"](feats, coords, batch_size, sizes=sizes)
            return x, x_activation
        else:
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

    def scale_mask(self, gt_bboxes_3d: list):
        # create a copy of gt_bboxes_3d
        output = []
        for i in range(len(gt_bboxes_3d)):
            gt_bboxes_3d_copy = deepcopy(gt_bboxes_3d[i])
            gt_bboxes_3d_copy_tensor = gt_bboxes_3d_copy.tensor # Nx9
            for j in range(gt_bboxes_3d_copy_tensor.shape[0]):
                # get the range of the bbox
                bbox = gt_bboxes_3d_copy_tensor[j] # box: x, y, z, x_size, y_size, z_size, yaw, vx, vy
                x = bbox[0]
                y = bbox[1]
                bbox_range = np.sqrt(x**2 + y**2)

                d_x_size = 0
                d_y_size = 0

                if bbox_range > 20 and bbox_range < 30:
                    # scale up the size of the bbox
                    # d_x_size = max([bbox[3] * 0.25, 1.0])
                    # d_y_size = max([bbox[4] * 0.25, 1.0])
                    d_x_size += bbox[3] * 0.25
                    d_y_size += bbox[4] * 0.25
                
                if bbox_range >= 30:
                    # scale up the size of the bbox
                    # d_x_size = max([bbox[3] * 0.5, 1.0])
                    # d_y_size = max([bbox[4] * 0.5, 1.0])
                    d_x_size += bbox[3] * 0.5
                    d_y_size += bbox[4] * 0.5
                
                # if the bbox_range < 10 do nothing
               
                # consider the velocity of the bbox
                vx = bbox[7]
                vy = bbox[8]

                if abs(vx) > 0.3 and abs(vx) < 0.8:
                    d_x_size += bbox[3] * 0.25
                if abs(vx) >= 0.8:
                    d_x_size += bbox[3] * 0.5

                if abs(vy) > 0.3 and abs(vy) < 0.8:
                    d_y_size += bbox[4] * 0.25
                if abs(vy) >= 0.8:
                    d_y_size += bbox[4] * 0.5
                
                # min for d_size is 0.5 and max is 4.0
                d_x_size = min([max([d_x_size, 0.5]), 4.0])
                d_y_size = min([max([d_y_size, 0.5]), 4.0])

                # update the bbox
                gt_bboxes_3d_copy_tensor[j][3] += d_x_size
                gt_bboxes_3d_copy_tensor[j][4] += d_y_size
            
            output.append(gt_bboxes_3d_copy)
        return output




        # gt_bboxes_3d_copy = deepcopy(gt_bboxes_3d)
        # gt_bboxes_3d_copy_tensor = gt_bboxes_3d_copy.tensor # Nx9
        # assert gt_bboxes_3d_copy.tensor.shape[1] == 9 # x, y, z, x_size, y_size, z_size, yaw, vx, vy (?)

        # print("gt boxes: ", gt_bboxes_3d_copy_tensor)
        # print('------------------')

        # # go through each bbox
        # for i in range(len(gt_bboxes_3d_scaled)):
        #     # get the bbox
        #     bbox = gt_bboxes_3d_scaled[i]

        #     # get the center of the bbox
        #     box_x = 
            


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
        # Forward the teacher model: Backbone, Neck
        # Get the fused features of the teacher model
        if self.training:
            x_teacher, lidar_teacher, cam_teacher = self.get_gated_feature_teacher( img,
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
            fused_teacher_features = x_teacher
            # print("teacher feature shape: ", fused_teacher_features.shape) # (3, 256, 180, 180)
            x_teacher = self.decoder["backbone"](x_teacher)
            x_teacher = self.decoder["neck"](x_teacher)

        # Forward the student model: Backbone, Neck
        features_student = []
        radar_student = None
        cam_student = None
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
                # Cam2Cam KD
                cam_student = feature_student
                if self.cam_conv_channel is not None:
                    cam_student = self.cam_conv_channel(cam_student)
                if self.cam_da is not None:
                    cam_student = self.cam_da(cam_student)
            elif sensor == "lidar":
                feature_student = self.extract_features_student(points, sensor)
            elif sensor == "radar":
                if self.lr_kd_loss is not None:
                    feature_student, radar_activation = self.extract_features_student(radar, sensor)
                else:
                    feature_student = self.extract_features_student(radar, sensor)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")

            features_student.append(feature_student)

        if not self.training:
            # avoid OOM
            features_student = features_student[::-1]

        if self.fuser_student is not None:
            _, x_student, cam_student, radar_student = self.fuser_student(features_student)
            fused_student_features = x_student
            # print("fused student feature shape: ", fused_student_features.shape) # (3, 64, 128, 128)

            # Fusion KD
            if self.fused_da is not None:
                fused_student_features = self.fused_da(fused_student_features)
                x_student = fused_student_features
            if self.fused_conv_channel is not None:
                fused_student_features = self.fused_conv_channel(fused_student_features)
        else:
            assert len(features_student) == 1, features_student
            x_student = features_student[0]

        batch_size = x_student.shape[0]

        x_student = self.decoder_student["backbone"](x_student)
        x_student = self.decoder_student["neck"](x_student)
        # print("student feature shape: ", x_student.shape) # (3, 256, 128, 128)


        # Forward the student model: Decoder, Heads
        # Compute the loss of the student model
        if self.training:
            outputs = {}
            for type, head in self.heads_student.items():
                if type == "object": # for 3d object detection
                    pred_dict = head(x_student, metas)
                    pred_dict_det = pred_dict
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict_det)

                    # Teacher Model Forward
                    pred_dict_kd = pred_dict
                    pred_dict_teacher = self.heads[type](x_teacher, metas)

                    if self.lr_kd_loss is not None:
                        lidar_heatmap_dynamic = [] # 2, 1, 180, 180
                        for pred in pred_dict_teacher:
                            # print max and min of the heatmap
                            # print(torch.max(pred[0]["heatmap"]))
                            # print(torch.min(pred[0]["heatmap"]))
                            # print('------------')
                            # print(pred[0]["heatmap"].shape)
                            
                            # first version
                            # pred_mean = torch.mean(pred[0]["heatmap"], dim=1, keepdim=True)
                            # pred_mean = clip_sigmoid(pred_mean.to(torch.float32))
                            # lidar_heatmap_dynamic += pred_mean # TODO: try dynamic object only

                            # second version
                            # pred_max = torch.max(pred[0]["heatmap"], dim=1, keepdim=True)[0]
                            # pred_max = sigmoid_normalized(pred_max.to(torch.float32))
                            # lidar_heatmap_dynamic.append(pred_max)
                            pred_mean = torch.mean(pred[0]["heatmap"], dim=1, keepdim=True)
                            pred_mean = clip_sigmoid(pred_mean.to(torch.float32))
                            lidar_heatmap_dynamic.append(pred_mean)

                            
                        
                        lidar_heatmap_dynamic = torch.cat(lidar_heatmap_dynamic, dim=1) # 2, 6, 180, 180
                        # get the max of the heatmap
                        # lidar_heatmap_dynamic = torch.max(lidar_heatmap_dynamic, dim=1, keepdim=True)[0] # 2, 1, 180, 180
                        # get the mean of the heatmap
                        lidar_heatmap_dynamic = torch.mean(lidar_heatmap_dynamic, dim=1, keepdim=True) # 2, 1, 180, 180
                    # print('----------------------')
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
            
            if self.use_depth_loss_student:
                if 'depth' in auxiliary_losses_student:
                    outputs["loss/depth"] = auxiliary_losses_student['depth']
                else:
                    raise ValueError('Use depth loss is true, but depth loss not found')
            
            if self.scale_bbox:
                # print("scale bbox")
                gt_bboxes_scaled = self.scale_mask(gt_bboxes_3d)
                gt_mask_feat = self.heads_student["object"].get_gt_masks(gt_bboxes_scaled) # should be list of tensor
                # gt_mask_feat_compare = self.heads_student["object"].get_gt_masks(gt_bboxes_3d) # should be list of tensor
                # print("mask feat: ", gt_mask_feat)
                # print('----------------------------------')
                # print("mask feat compare: ", gt_mask_feat_compare)
                # print('----------------------------------')
            else:
                gt_mask_feat = self.heads_student["object"].get_gt_masks(gt_bboxes_3d) # should be list of tensor
            
            # Fusion KD loss
            if self.fused_loss is not None:
                # For mask loss
                outputs["loss/object/fused_mask_scale_feat_kd"] = self.fused_feat_loss_scale_kd * self.fused_feat_loss(fused_student_features, fused_teacher_features, gt_mask_feat)
                # For dense loss
                # outputs["loss/object/fused_dense_feat_kd"] = self.fused_feat_loss_scale_kd * self.fused_feat_loss(fused_student_features, fused_teacher_features)
            
            if self.fused_affinity_loss is not None:
                # Fusion Relation KD loss
                outputs["loss/object/fused_affinity_kd"] = self.fused_affinity_loss_scale_kd * self.fused_feat_affinity_loss(fused_student_features, fused_teacher_features)
            
            # Cam2Cam KD loss
            if self.cam_loss is not None:
                # For mask loss
                outputs["loss/object/cam_mask_scale_feat_kd"] = self.cam_feat_loss_scale_kd * self.cam_feat_loss(cam_student, cam_teacher, gt_mask_feat)
                # For dense loss
                # outputs["loss/object/cam_dense_feat_kd"] = self.cam_feat_loss_scale_kd * self.cam_feat_loss(cam_student, cam_teacher)


            # LR KD loss
            if self.lr_kd_loss is not None:
                # print(radar_activateion.shape)
                # print max and min of radar_activation and lidar_heatmap_dynamic
                # print(torch.max(radar_activation))
                # print(torch.min(radar_activation))
                # print(torch.max(lidar_heatmap_dynamic))
                # print(torch.min(lidar_heatmap_dynamic))
                # radar_activation_np = radar_activation.detach().cpu().numpy()
                # lidar_heatmap_dynamic_np = lidar_heatmap_dynamic.detach().cpu().numpy()
                # save
                # print("saving files")
                # np.save('/mnt/ws-frb/users/jingyuso/docker_data/docker/home/crkd/KD-CR/debug_files/radar_activation.npy', radar_activation_np)
                # np.save('/mnt/ws-frb/users/jingyuso/docker_data/docker/home/crkd/KD-CR/debug_files/lidar_heatmap_dynamic.npy', lidar_heatmap_dynamic_np)
                outputs["loss/object/lr_kd"] = self.lr_kd_loss(radar_activation, lidar_heatmap_dynamic)

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



@FUSIONMODELS.register_module()
class BEVResponseDistillerFusedLRC2CMaskScaleRelationGaussian(BEVFusion):
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
        cam_domain_adaptation: Dict[str, Any] = None,
        cam_conv_channel: Dict[str, Any] = None,
        # cam_loss = dict(type="MSELoss", reduction="mean", loss_weight=1.0),
        cam_loss: Dict[str, Any] = None,
        cam_feat_loss_scale_kd: float = 1.0,
        fused_domain_adaptation: Dict[str, Any] = None,
        fused_conv_channel: Dict[str, Any] = None,
        fused_loss: Dict[str, Any] = None,
        fused_feat_loss_scale_kd: float = 1.0,
        lr_kd_loss: Dict[str, Any] = None,
        scale_bbox: bool = False,
        fused_affinity_loss: Dict[str, Any] = None,
        # fused_affinity_loss = dict(type="Affinity_Loss", reduction="mean", loss_weight=1.0, downsample_size=[32, 16, 8], input_channels=256, use_adapt=True),
        fused_affinity_loss_scale_kd: float = 1.0,
        use_gaussian: bool = False,
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
        
        # Initialize the domain adaptation module
        if cam_domain_adaptation is not None:
            self.cam_da = build_da(cam_domain_adaptation)
        else:
            self.cam_da = None
        if cam_conv_channel is not None:
            self.cam_conv_channel = build_da(cam_conv_channel)
        else:
            self.cam_conv_channel = None

        if fused_domain_adaptation is not None:
            self.fused_da = build_da(fused_domain_adaptation)
        else:
            self.fused_da = None
        if fused_conv_channel is not None:
            self.fused_conv_channel = build_da(fused_conv_channel)
        else:
            self.fused_conv_channel = None

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

        self.fused_loss = fused_loss
        self.cam_loss = cam_loss
        self.fused_affinity_loss = fused_affinity_loss

        # Initialize the feature kd loss
        if fused_loss is not None:
            self.fused_feat_loss = build_loss(fused_loss)
        if cam_loss is not None:
            self.cam_feat_loss = build_loss(cam_loss)
        if fused_affinity_loss is not None:
            self.fused_feat_affinity_loss = build_loss(fused_affinity_loss)
        if lr_kd_loss:
            self.lr_kd_loss = build_loss(lr_kd_loss)
        else:
            self.lr_kd_loss = None

        # This loss scale is for the original det loss of the student model
        # loss_scale is default for student model in BEVDistiller
        if "loss_scale" in kwargs:
            self.loss_scale_student = kwargs["loss_scale"]
        else:
            self.loss_scale_student = dict()
            for name in heads_student:
                if heads_student[name] is not None:
                    self.loss_scale_student[name] = 1.0

        # Fused KD loss scale
        self.fused_feat_loss_scale_kd = fused_feat_loss_scale_kd

        # Cam2Cam KD loss scale
        self.cam_feat_loss_scale_kd = cam_feat_loss_scale_kd

        # Fused KD loss scale
        self.fused_affinity_loss_scale_kd = fused_affinity_loss_scale_kd

        # If the camera's vtransform is a BEVDepth version, then we're using depth loss. 
        self.use_depth_loss_student = ((encoders_student.get('camera', {}) or {}).get('vtransform', {}) or {}).get('type', '') in ['BEVDepth', 'AwareBEVDepth', 'DBEVDepth', 'AwareDBEVDepth']

        self.scale_bbox = scale_bbox

        self.use_gaussian = use_gaussian

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
        B, N, C, H, W = x.size() # batchsize, num_view, channel, height, width
        x = x.view(B * N, C, H, W) # batchsize * num_view, channel, height, width

        x = self.encoders_student["camera"]["backbone"](x)
        x = self.encoders_student["camera"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size() # batchsize * num_view, channel, height, width
        x = x.view(B, int(BN / B), C, H, W) # batchsize, num_view, channel, height, width

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
        if sensor == "radar" and self.lr_kd_loss is not None:
            feats, coords, sizes = self.voxelize_student(x, sensor)
            batch_size = coords[-1, 0] + 1
            x, x_activation = self.encoders_student[sensor]["backbone"](feats, coords, batch_size, sizes=sizes)
            return x, x_activation
        else:
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

    def scale_mask(self, gt_bboxes_3d: list):
        # create a copy of gt_bboxes_3d
        output = []
        for i in range(len(gt_bboxes_3d)):
            gt_bboxes_3d_copy = deepcopy(gt_bboxes_3d[i])
            gt_bboxes_3d_copy_tensor = gt_bboxes_3d_copy.tensor # Nx9
            gt_bboxes_3d_copy_tensor.to(gt_bboxes_3d[0].device)
            for j in range(gt_bboxes_3d_copy_tensor.shape[0]):
                # get the range of the bbox
                bbox = gt_bboxes_3d_copy_tensor[j] # box: x, y, z, x_size, y_size, z_size, yaw, vx, vy
                x = bbox[0]
                y = bbox[1]
                bbox_range = np.sqrt(x**2 + y**2)

                d_x_size = 0
                d_y_size = 0

                if bbox_range > 20 and bbox_range < 30:
                    # scale up the size of the bbox
                    # d_x_size = max([bbox[3] * 0.25, 1.0])
                    # d_y_size = max([bbox[4] * 0.25, 1.0])
                    d_x_size += bbox[3] * 0.25
                    d_y_size += bbox[4] * 0.25
                
                if bbox_range >= 30:
                    # scale up the size of the bbox
                    # d_x_size = max([bbox[3] * 0.5, 1.0])
                    # d_y_size = max([bbox[4] * 0.5, 1.0])
                    d_x_size += bbox[3] * 0.5
                    d_y_size += bbox[4] * 0.5
                
                # if the bbox_range < 10 do nothing
               
                # consider the velocity of the bbox
                vx = bbox[7]
                vy = bbox[8]

                if abs(vx) > 0.3 and abs(vx) < 0.8:
                    d_x_size += bbox[3] * 0.25
                if abs(vx) >= 0.8:
                    d_x_size += bbox[3] * 0.5

                if abs(vy) > 0.3 and abs(vy) < 0.8:
                    d_y_size += bbox[4] * 0.25
                if abs(vy) >= 0.8:
                    d_y_size += bbox[4] * 0.5
                
                # min for d_size is 0.5 and max is 4.0
                d_x_size = min([max([d_x_size, 0.5]), 4.0])
                d_y_size = min([max([d_y_size, 0.5]), 4.0])

                # update the bbox
                gt_bboxes_3d_copy_tensor[j][3] += d_x_size
                gt_bboxes_3d_copy_tensor[j][4] += d_y_size
            
            output.append(gt_bboxes_3d_copy)
        return output




        # gt_bboxes_3d_copy = deepcopy(gt_bboxes_3d)
        # gt_bboxes_3d_copy_tensor = gt_bboxes_3d_copy.tensor # Nx9
        # assert gt_bboxes_3d_copy.tensor.shape[1] == 9 # x, y, z, x_size, y_size, z_size, yaw, vx, vy (?)

        # print("gt boxes: ", gt_bboxes_3d_copy_tensor)
        # print('------------------')

        # # go through each bbox
        # for i in range(len(gt_bboxes_3d_scaled)):
        #     # get the bbox
        #     bbox = gt_bboxes_3d_scaled[i]

        #     # get the center of the bbox
        #     box_x = 
            


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
        # Forward the teacher model: Backbone, Neck
        # Get the fused features of the teacher model
        if self.training:
            x_teacher, lidar_teacher, cam_teacher = self.get_gated_feature_teacher( img,
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
            fused_teacher_features = x_teacher
            # print("teacher feature shape: ", fused_teacher_features.shape) # (3, 256, 180, 180)
            x_teacher = self.decoder["backbone"](x_teacher)
            x_teacher = self.decoder["neck"](x_teacher)

        # Forward the student model: Backbone, Neck
        features_student = []
        radar_student = None
        cam_student = None
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
                # Cam2Cam KD
                cam_student = feature_student
                if self.cam_conv_channel is not None:
                    cam_student = self.cam_conv_channel(cam_student)
                if self.cam_da is not None:
                    cam_student = self.cam_da(cam_student)
            elif sensor == "lidar":
                feature_student = self.extract_features_student(points, sensor)
            elif sensor == "radar":
                if self.lr_kd_loss is not None:
                    feature_student, radar_activation = self.extract_features_student(radar, sensor)
                else:
                    feature_student = self.extract_features_student(radar, sensor)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")

            features_student.append(feature_student)

        if not self.training:
            # avoid OOM
            features_student = features_student[::-1]

        if self.fuser_student is not None:
            _, x_student, cam_student, radar_student = self.fuser_student(features_student)
            fused_student_features = x_student
            # print("fused student feature shape: ", fused_student_features.shape) # (3, 64, 128, 128)

            # Fusion KD
            if self.fused_da is not None:
                fused_student_features = self.fused_da(fused_student_features)
                x_student = fused_student_features
            if self.fused_conv_channel is not None:
                fused_student_features = self.fused_conv_channel(fused_student_features)
        else:
            assert len(features_student) == 1, features_student
            x_student = features_student[0]

        batch_size = x_student.shape[0]

        x_student = self.decoder_student["backbone"](x_student)
        x_student = self.decoder_student["neck"](x_student)
        # print("student feature shape: ", x_student.shape) # (3, 256, 128, 128)


        # Forward the student model: Decoder, Heads
        # Compute the loss of the student model
        if self.training:
            outputs = {}
            for type, head in self.heads_student.items():
                if type == "object": # for 3d object detection
                    pred_dict = head(x_student, metas)
                    pred_dict_det = pred_dict
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict_det)

                    # Teacher Model Forward
                    pred_dict_kd = pred_dict
                    pred_dict_teacher = self.heads[type](x_teacher, metas)

                    if self.lr_kd_loss is not None:
                        lidar_heatmap_dynamic = [] # 2, 1, 180, 180
                        for pred in pred_dict_teacher:
                            # print max and min of the heatmap
                            # print(torch.max(pred[0]["heatmap"]))
                            # print(torch.min(pred[0]["heatmap"]))
                            # print('------------')
                            # print(pred[0]["heatmap"].shape)
                            
                            # first version
                            # pred_mean = torch.mean(pred[0]["heatmap"], dim=1, keepdim=True)
                            # pred_mean = clip_sigmoid(pred_mean.to(torch.float32))
                            # lidar_heatmap_dynamic += pred_mean # TODO: try dynamic object only

                            # second version
                            # pred_max = torch.max(pred[0]["heatmap"], dim=1, keepdim=True)[0]
                            # pred_max = sigmoid_normalized(pred_max.to(torch.float32))
                            # lidar_heatmap_dynamic.append(pred_max)
                            pred_mean = torch.mean(pred[0]["heatmap"], dim=1, keepdim=True)
                            pred_mean = clip_sigmoid(pred_mean.to(torch.float32))
                            lidar_heatmap_dynamic.append(pred_mean)

                            
                        
                        lidar_heatmap_dynamic = torch.cat(lidar_heatmap_dynamic, dim=1) # 2, 6, 180, 180
                        # get the max of the heatmap
                        # lidar_heatmap_dynamic = torch.max(lidar_heatmap_dynamic, dim=1, keepdim=True)[0] # 2, 1, 180, 180
                        # get the mean of the heatmap
                        lidar_heatmap_dynamic = torch.mean(lidar_heatmap_dynamic, dim=1, keepdim=True) # 2, 1, 180, 180
                    # print('----------------------')
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
            
            if self.use_depth_loss_student:
                if 'depth' in auxiliary_losses_student:
                    outputs["loss/depth"] = auxiliary_losses_student['depth']
                else:
                    raise ValueError('Use depth loss is true, but depth loss not found')
            
            if self.scale_bbox:
                # print("scale bbox")
                gt_bboxes_scaled = self.scale_mask(gt_bboxes_3d)
                gt_mask_feat = self.heads_student["object"].get_gt_masks(gt_bboxes_scaled) # should be list of tensor
                # gt_mask_feat_compare = self.heads_student["object"].get_gt_masks(gt_bboxes_3d) # should be list of tensor
                # print("mask feat: ", gt_mask_feat)
                # print('----------------------------------')
                # print("mask feat compare: ", gt_mask_feat_compare)
                # print('----------------------------------')
            else:
                gt_bboxes_scaled = gt_bboxes_3d
                gt_mask_feat = self.heads_student["object"].get_gt_masks(gt_bboxes_3d) # should be list of tensor
            
            if self.use_gaussian:
                gaussian_fg_map = self.heads_student["object"].get_gaussian_gt_masks(gt_bboxes_scaled) # (B, H, W)
                gaussian_fg_map = gaussian_fg_map.to(fused_student_features.device)
                # TODO: Please uncomment these lines to enable debug on gaussian map
                # Test start
                # gaussian_fg_map_np = gaussian_fg_map.detach().cpu().numpy()
                # np.save('/mnt/ws-frb/users/jingyuso/docker_data/docker/home/crkd/KD-CR/debug_files/gaussian_fg_map_np.npy', gaussian_fg_map_np)
                # Test end

                if self.fused_loss is not None:
                    fused_feat_loss_gaussian = self.fused_feat_loss_scale_kd * self.fused_feat_loss(fused_student_features, fused_teacher_features) # (B, H, W)
                    # mean by batchsize and map size
                    fused_feat_loss_gaussian = torch.sum(fused_feat_loss_gaussian * gaussian_fg_map) / torch.sum(gaussian_fg_map)
                    outputs["loss/object/fused_gaussian_mask_scale_feat_kd"] = fused_feat_loss_gaussian
                # Cam2Cam KD loss
                if self.cam_feat_loss_scale_kd is not None:
                    cam_feat_loss_gaussian = self.cam_feat_loss_scale_kd * self.cam_feat_loss(cam_student, cam_teacher) # (B, H, W)
                    # mean by batchsize and map size
                    cam_feat_loss_gaussian = torch.sum(cam_feat_loss_gaussian * gaussian_fg_map) / torch.sum(gaussian_fg_map)
                    outputs["loss/object/cam_gaussian_mask_scale_feat_kd"] = cam_feat_loss_gaussian
            else:
                # Fusion KD loss
                if self.fused_loss is not None:
                    outputs["loss/object/fused_mask_scale_feat_kd"] = self.fused_feat_loss_scale_kd * self.fused_feat_loss(fused_student_features, fused_teacher_features, gt_mask_feat)
                # Cam2Cam KD loss
                if self.cam_feat_loss_scale_kd is not None:
                    outputs["loss/object/cam_mask_scale_feat_kd"] = self.cam_feat_loss_scale_kd * self.cam_feat_loss(cam_student, cam_teacher, gt_mask_feat)


            # Fusion Relation KD loss
            if self.fused_affinity_loss is not None:
                outputs["loss/object/fused_affinity_kd"] = self.fused_affinity_loss_scale_kd * self.fused_feat_affinity_loss(fused_student_features, fused_teacher_features)

            # LR KD loss
            if self.lr_kd_loss is not None:
                # print(radar_activateion.shape)
                # print max and min of radar_activation and lidar_heatmap_dynamic
                # print(torch.max(radar_activation))
                # print(torch.min(radar_activation))
                # print(torch.max(lidar_heatmap_dynamic))
                # print(torch.min(lidar_heatmap_dynamic))
                # radar_activation_np = radar_activation.detach().cpu().numpy()
                # lidar_heatmap_dynamic_np = lidar_heatmap_dynamic.detach().cpu().numpy()
                # save
                # print("saving files")
                # np.save('/mnt/ws-frb/users/jingyuso/docker_data/docker/home/crkd/KD-CR/debug_files/radar_activation.npy', radar_activation_np)
                # np.save('/mnt/ws-frb/users/jingyuso/docker_data/docker/home/crkd/KD-CR/debug_files/lidar_heatmap_dynamic.npy', lidar_heatmap_dynamic_np)
                outputs["loss/object/lr_kd"] = self.lr_kd_loss(radar_activation, lidar_heatmap_dynamic)

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



