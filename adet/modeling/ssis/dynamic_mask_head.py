import torch
from torch.nn import functional as F
from torch import nn

from adet.utils.comm import compute_locations, aligned_bilinear
from detectron2.layers import Conv2d, ModulatedDeformConv, DeformConv
from fvcore.nn import smooth_l1_loss
import kornia


def dice_coefficient(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss



def parse_dynamic_params(params, channels, weight_nums, bias_nums, with_thick_boundary=False):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)

    num_insts = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(torch.split_with_sizes(
        params, weight_nums + bias_nums, dim=1
    ))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        if l < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
        else:
            # out_channels x in_channels x 1 x 1
            if with_thick_boundary:
                weight_splits[l] = weight_splits[l].reshape(num_insts * 2, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * 2)
            else:
                weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * 1)


    return weight_splits, bias_splits


def build_dynamic_mask_head(cfg):
    return DynamicMaskHead(cfg)


class MaskIoUHead(nn.Module):
    def __init__(self):
        super(MaskIoUHead, self).__init__()

        num_classes = 2
        input_channels = 9
        conv_dims = 2
        num_conv = 2

        self.conv_relus = []
        self.conv_masks = []
        stride = 1

        def _single_conv( nc, kh, kw, dd, dg):
            conv = nn.Conv2d(
                nc,
                dg * 3 * kh * kw,
                kernel_size=(3, 3),
                stride=(1, 1),
                dilation=(dd, dd),
                padding=(1*dd, 1*dd),
                bias=False)
            return conv


        def _deform_conv( nc, kh, kw, dd, dg):
            conv_offset2d = ModulatedDeformConv(
                nc,
                nc, (kh, kw),
                stride=1,
                padding=int(kh/2)*dd,
                dilation=dd,
                deformable_groups=dg)
            return conv_offset2d

        self.conv1x1_1 = nn.Conv2d(
                input_channels,
                2,
                kernel_size=3,
                stride=1,
                padding=1
            )
        self.conv_offset = _single_conv(2,3,3,3,1)
        self.conv = _deform_conv(2,3,3,3,1)
        self.conv1x1_2 = Conv2d(
            2, 
            2 ,
            kernel_size=1,
            stride=1,
            padding=0,
            activation=F.relu
        )
        self.add_module("maskiou_deformfcn{}".format(1), self.conv1x1_1)
        self.add_module("maskiou_deformfcn{}".format(2), self.conv)
        self.add_module("maskiou_deformfcn{}".format(3), self.conv1x1_2)
        self.add_module("maskiou_deformmask{}".format(1),self.conv_offset)

        
        self.adaptive_maxpool = nn.AdaptiveMaxPool2d((64,64))
        self.maskiou_fc1 = nn.Linear(2*64*64, 2048)
        self.maskiou_fc2 = nn.Linear(2048, 1024)
        self.maskiou = nn.Linear(1024, 1)
        
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv1x1_1.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.conv1x1_1.bias, 0)
        nn.init.kaiming_normal_(self.conv1x1_2.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.conv1x1_2.bias, 0)
        nn.init.constant_(self.conv_offset.weight, 0)
        for l in [self.maskiou_fc1, self.maskiou_fc2]:
            nn.init.kaiming_normal_(l.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(l.bias, 0)


        nn.init.normal_(self.maskiou.weight, mean=0, std=0.01)
        nn.init.constant_(self.maskiou.bias, 0)

    def forward(self, mask, x):
        x = torch.cat((x,mask),1)
        x = self.conv1x1_1(x)
        offset_mask = self.conv_offset(x)
        offset_x, offset_y, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((offset_x, offset_y), dim=1)
        mask = mask.sigmoid()
        x = self.conv(x, offset, mask)
        x = self.conv1x1_2(x)

        x = self.adaptive_maxpool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.maskiou_fc1(x))
        x = F.relu(self.maskiou_fc2(x))
        x = self.maskiou(x).sigmoid()
        return x




class DynamicMaskHead(nn.Module):
    def __init__(self, cfg):
        super(DynamicMaskHead, self).__init__()
        self.num_layers = cfg.MODEL.CONDINST.MASK_HEAD.NUM_LAYERS
        self.channels = cfg.MODEL.CONDINST.MASK_HEAD.CHANNELS
        self.in_channels = cfg.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS
        self.mask_out_stride = cfg.MODEL.CONDINST.MASK_OUT_STRIDE
        self.disable_rel_coords = cfg.MODEL.CONDINST.MASK_HEAD.DISABLE_REL_COORDS
        self.maskiou = cfg.MODEL.CONDINST.MASK_HEAD.DEFORM_MASKIOU
        self.boundary_loss = cfg.MODEL.CONDINST.MASK_HEAD.BOUNDARY_LOSS
        soi = cfg.MODEL.FCOS.SIZES_OF_INTEREST
        stride = cfg.MODEL.FCOS.FPN_STRIDES

        self.register_buffer("sizes_of_interest", torch.tensor(soi + [soi[-1] * 2]))
        self.register_buffer("strides", torch.tensor(stride,dtype=torch.float32))
        self.iter = 0

        weight_nums, bias_nums = [], []
        for l in range(self.num_layers):
            if l == 0:
                if not self.disable_rel_coords:
                    weight_nums.append((self.in_channels + 2) * self.channels)
                else:
                    weight_nums.append(self.in_channels * self.channels)
                bias_nums.append(self.channels)
            elif l == self.num_layers - 1:
                if self.boundary_loss:
                    weight_nums.append(self.channels * 2)
                    bias_nums.append(2)
                else:
                    weight_nums.append(self.channels )
                    bias_nums.append(1)
            else:
                weight_nums.append(self.channels * self.channels)
                bias_nums.append(self.channels)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)

        if self.maskiou:
            self.maskiou_head = MaskIoUHead()

   
    def mask_heads_forward(self, features, weights, biases, num_insts):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def mask_heads_forward_with_coords(
            self, mask_feats, mask_feat_stride, instances
    ):
        def convert_locations(relative_coords,H,W):
            coords = relative_coords.permute(0,2,1).reshape(-1,H,W,2)
            x = torch.abs(coords).sum(-1).min(1)[1][:,0:1]
            y = torch.abs(coords).sum(-1).min(2)[1][:,0:1]
            h = torch.ones_like(x)*H
            w = torch.ones_like(y)*W
            return torch.cat([x,y,h,w],dim=1)
        locations = compute_locations(
            mask_feats.size(2), mask_feats.size(3),
            stride=mask_feat_stride, device=mask_feats.device
        )
        n_inst = len(instances)

        im_inds = instances.im_inds
        mask_head_params = instances.mask_head_params
        asso_mask_head_params = instances.mask_head_params2

        N, _, H, W = mask_feats.size()

        if not self.disable_rel_coords:
            instance_locations = instances.locations 
            offset = instances.offset_pred # N,2 
            
            asso_instance_locations = instances.locations  +  offset * 128

            asso_relative_coords = asso_instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
            relative_coords = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
            
            asso_relative_coords = asso_relative_coords.permute(0,2,1).float()
            relative_coords = relative_coords.permute(0, 2, 1).float()
            
            soi = self.sizes_of_interest.float()[instances.fpn_levels]

            asso_relative_coords = asso_relative_coords / soi.reshape(-1,1,1) #shape: N,2,HW
            relative_coords = relative_coords / soi.reshape(-1, 1, 1)

            asso_relative_coords = asso_relative_coords.to(dtype=mask_feats.dtype)
            relative_coords = relative_coords.to(dtype=mask_feats.dtype)

            mask_head_inputs = torch.cat([
                relative_coords, mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)
            ], dim=1)
            asso_mask_head_inputs = torch.cat([
                asso_relative_coords, mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)
            ], dim=1)
        else:
            mask_head_inputs = mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)

        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)
        asso_mask_head_inputs = asso_mask_head_inputs.reshape(1, -1, H, W)
        
        weights, biases = parse_dynamic_params(
            mask_head_params, self.channels,
            self.weight_nums, self.bias_nums, self.boundary_loss
        )

        weights2, biases2 = parse_dynamic_params(
            asso_mask_head_params, self.channels,
            self.weight_nums, self.bias_nums, self.boundary_loss
        )

        mask_logits = self.mask_heads_forward(mask_head_inputs, weights, biases, n_inst)
        asso_mask_logits = self.mask_heads_forward(asso_mask_head_inputs, weights2, biases2, n_inst)
        
        if self.boundary_loss:
            mask_logits = mask_logits.reshape(-1, 2, H, W)
            asso_mask_logits = asso_mask_logits.reshape(-1, 2, H, W)
            boundary_logits = mask_logits[:,1:2] 
            asso_boundary_logits = asso_mask_logits[:,1:2] 
        else:
            mask_logits = mask_logits.reshape(-1, 1, H, W)
            asso_mask_logits = asso_mask_logits.reshape(-1, 1, H, W)

        if self.maskiou:
            mask_iou = self.maskiou_head((mask_logits.sigmoid()>0.5).float(),mask_feats[im_inds].reshape(n_inst, self.in_channels, H , W))
            asso_mask_iou = self.maskiou_head((asso_mask_logits.sigmoid()>0.5).float(),mask_feats[im_inds].reshape(n_inst, self.in_channels, H , W))
        else:
            mask_iou = None
            asso_mask_iou = None

        assert mask_feat_stride >= self.mask_out_stride
        assert mask_feat_stride % self.mask_out_stride == 0
        mask_logits = aligned_bilinear(mask_logits, int(mask_feat_stride / self.mask_out_stride))
        asso_mask_logits = aligned_bilinear(asso_mask_logits, int(mask_feat_stride / self.mask_out_stride))
        
        if self.boundary_loss:
            boundary_logits = aligned_bilinear(boundary_logits, int(mask_feat_stride / self.mask_out_stride)).sigmoid()
            asso_boundary_logits = aligned_bilinear(asso_boundary_logits, int(mask_feat_stride / self.mask_out_stride)).sigmoid()
        else:
            boundary_logits = None
            asso_boundary_logits = None

        return mask_logits.sigmoid(), asso_mask_logits.sigmoid(), mask_iou, asso_mask_iou, boundary_logits, asso_boundary_logits

    
    def find_gt_relation_masks(self, gt_instances):
        gt_relations_masks = []
        gt_relations_dismaps = []
        for per_im in gt_instances:
            half_length = int(len(per_im.gt_bitmasks) / 2)
            gt_relations_masks.append(per_im.gt_bitmasks[half_length:])
            gt_relations_masks.append(per_im.gt_bitmasks[:half_length])
            gt_relations_dismaps.append(per_im.gt_dismaps[half_length:])
            gt_relations_dismaps.append(per_im.gt_dismaps[:half_length])
        return torch.cat(gt_relations_masks),  torch.cat(gt_relations_dismaps)
    
    def maskiou_loss(self, mask_iou, gt_iou):
        return 1 * ((torch.abs(mask_iou - gt_iou)**2).sum()/mask_iou.shape[0])

    def get_target_iou(self, mask,gt):
        mask = (mask>0.5).float()
        mask_area = mask.sum((-1,-2))
        overlap = (mask*gt).sum((-1,-2))
        gt_area = gt.sum((-1,-2))
        return overlap / (mask_area + gt_area - overlap + 1e-7)
    
    def __call__(self, mask_feats, mask_feat_stride, pred_instances, gt_instances=None):
        if self.training:
            
            gt_inds = pred_instances.gt_inds
            gt_bitmasks = torch.cat([per_im.gt_bitmasks for per_im in gt_instances])
            gt_bitmasks = gt_bitmasks[gt_inds].unsqueeze(dim=1).to(dtype=mask_feats.dtype)
            gt_asso_masks, gt_asso_dismaps = self.find_gt_relation_masks(gt_instances)
        
            gt_asso_dismaps = gt_asso_dismaps[gt_inds].unsqueeze(dim=1).to(dtype=mask_feats.dtype)
            gt_asso_masks = gt_asso_masks[gt_inds].unsqueeze(dim=1).to(dtype=mask_feats.dtype)
            gt_offset = torch.cat([per_im.gt_relations for per_im in gt_instances])
            gt_offset = (gt_offset[gt_inds]  - pred_instances.locations).to(dtype=mask_feats.dtype)

            gt_dismaps =  torch.cat([per_im.gt_dismaps for per_im in gt_instances])
            gt_dismaps = gt_dismaps[gt_inds].unsqueeze(dim=1).to(dtype=mask_feats.dtype)

            if len(pred_instances) == 0:
                asso_offset_losses = pred_instances.offset_pred.sum() * 0
                loss_maskiou = mask_feats.sum() * 0.0
                association_masks_loss = mask_feats.sum() * 0.0
                boundary_loss = mask_feats.sum() * 0.0  + pred_instances.mask_head_params.sum() * 0 + pred_instances.mask_head_params2.sum() * 0 
            else:
                if self.maskiou & self.boundary_loss: # SSISv2
                    mask_scores, asso_mask_scores, mask_iou, asso_mask_iou,boun_score,asso_boun_score= self.mask_heads_forward_with_coords(
                        mask_feats, mask_feat_stride, pred_instances
                    )
                    self.iter += 1
                    kernel_size = 5
                    # thick boudary loss
                    boundary_loss = dice_coefficient(boun_score, (gt_dismaps>0.5).float()).mean() + dice_coefficient(asso_boun_score, (gt_asso_dismaps>0.5).float()).mean()
                    
                    if self.iter > 10000:
                        # thin boudary loss
                        boundary_loss += (torch.abs(torch.abs(kornia.filters.laplacian((mask_scores>0.5).float(), kernel_size)) - torch.abs(kornia.filters.laplacian(gt_bitmasks,kernel_size))).sum([2,3]) / ((torch.abs(kornia.filters.laplacian(gt_bitmasks,kernel_size)) > 0).float().sum([2,3])).mean()+ + 1e-7) * 5

                        boundary_loss += (torch.abs(torch.abs(kornia.filters.laplacian((asso_mask_scores>0.5).float(),kernel_size)) - torch.abs(kornia.filters.laplacian(gt_asso_masks,kernel_size))).sum([2,3]) / ((torch.abs(kornia.filters.laplacian(gt_asso_masks,kernel_size)) > 0).float().sum([2,3])).mean()+ + 1e-7) * 5

                    mask_losses = dice_coefficient(mask_scores, gt_bitmasks).mean() 
                    
                    asso_masks_losses =  dice_coefficient(asso_mask_scores, gt_asso_masks).mean() 

                    asso_offset_losses = smooth_l1_loss(pred_instances.offset_pred,gt_offset/128,0.5,reduction='mean')
                    
                    if self.iter > 5000:
                        loss_maskiou = 1 * (self.maskiou_loss(mask_iou, self.get_target_iou(mask_scores.detach(),gt_bitmasks)) + \
                                    self.maskiou_loss(asso_mask_iou, self.get_target_iou(asso_mask_scores.detach(),gt_asso_masks)))
                    else:
                        loss_maskiou = None
                else: # SSISv1
                    self.iter +=1
                    mask_scores, asso_mask_scores,_,_,boun_score,asso_boun_score = self.mask_heads_forward_with_coords(
                        mask_feats, mask_feat_stride, pred_instances
                    )
                    if self.boundary_loss:
                        # thick boudary loss
                        boundary_loss = dice_coefficient(boun_score, (gt_dismaps>0.5).float()).mean() + dice_coefficient(asso_boun_score, (gt_asso_dismaps>0.5).float()).mean()
                        
                        if self.iter > 10000:
                            # thin boudary loss
                            boundary_loss += (torch.abs(torch.abs(kornia.filters.laplacian((mask_scores>0.5).float(), kernel_size)) - torch.abs(kornia.filters.laplacian(gt_bitmasks,kernel_size))).sum([2,3]) / (torch.abs(kornia.filters.laplacian(gt_bitmasks,kernel_size)) > 0).float().sum([2,3])).mean() * 5

                            boundary_loss += (torch.abs(torch.abs(kornia.filters.laplacian((asso_mask_scores>0.5).float(),kernel_size)) - torch.abs(kornia.filters.laplacian(gt_asso_masks,kernel_size))).sum([2,3]) / (torch.abs(kornia.filters.laplacian(gt_asso_masks,kernel_size)) > 0).float().sum([2,3])).mean() * 5
                    else:
                        boundary_loss = None

                    mask_losses = dice_coefficient(mask_scores, gt_bitmasks).mean()
                    
                    asso_masks_losses =  dice_coefficient(asso_mask_scores,gt_asso_masks ).mean() 
                    
                    asso_offset_losses =   smooth_l1_loss(pred_instances.offset_pred,gt_offset/128,0.5,reduction='mean')

                    loss_maskiou = None

            return mask_losses.float(),asso_masks_losses.float(), asso_offset_losses, loss_maskiou, boundary_loss
        else:
            if len(pred_instances) > 0:
                if self.maskiou:
                    mask_scores,asso_mask_scores,  mask_iou, asso_mask_iou,_,_= self.mask_heads_forward_with_coords(
                        mask_feats, mask_feat_stride, pred_instances
                    )
                    
                    pred_instances.pred_global_masks = mask_scores.float()

                    pred_instances.pred_asso_global_masks = asso_mask_scores.float()

                    pred_instances.scores = mask_iou[:,0] * pred_instances.scores
                else:
                    mask_scores,asso_mask_scores,_,_,_,_ = self.mask_heads_forward_with_coords(
                        mask_feats, mask_feat_stride, pred_instances
                    )
                    pred_instances.pred_global_masks = mask_scores.float()
                    pred_instances.pred_asso_global_masks = asso_mask_scores.float()
            return pred_instances
