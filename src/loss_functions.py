from __future__ import division
import torch
from torch import nn
import torch.nn.functional as F
#from inverse_warp import inverse_warp2, inverse_warp
from inverse_warp_ucm import inverse_warp_ucm as inverse_warp2
import math


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

 def total_variation_sq_loss(image):
        # Calculate differences along both dimensions
        x_deltas = image[:, :, 1:, :] - image[:, :, :-1, :]
        y_deltas = image[:, 1:, :, :] - image[:, :-1, :, :]

        # Calculate the total variation loss with L2 (square) loss
        tv_loss = torch.mean(torch.abs(x_deltas)) + torch.mean(torch.abs(y_deltas))

        return tv_loss


def clip_loss_fn(x):
        # penalize values outside of 0-1
        x = (x*0.225)+0.45
        return 1 * torch.mean((torch.clamp(x, min=0, max=1) - x)**2)
    

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


compute_ssim_loss = SSIM().to(device)

def get_compute_pairwise_loss_fn(with_ssim, with_mask, with_auto_mask, padding_mode):
    def compute_pairwise_loss(img_a, depth_a, img_b, depth_b, pose, intrinsic_a, intrinsic_b):

        img_b_warped, valid_mask, projected_depth, computed_depth = inverse_warp2(img_b, depth_a, depth_b, pose, intrinsic_a, intrinsic_b, padding_mode)
        valid_mask_ = valid_mask


        
        depth_mask = 1# (depth_a < computed_depth).float().detach()

        diff_img_ = (img_a - img_b_warped) * depth_mask

        diff_img = (diff_img_).abs().clamp(0, 1)
        diff_depth_ = (computed_depth - projected_depth) 
        diff_depth = ((diff_depth_).abs() / (computed_depth + projected_depth)).clamp(0, 1)

        if with_auto_mask == True:
            auto_mask = torch.clamp((diff_img.mean(dim=1, keepdim=True) < (img_a - img_b).abs().mean(dim=1, keepdim=True)).float() * valid_mask, 0.01, 1)
            valid_mask = auto_mask

        if with_ssim == True:
            ssim_map = compute_ssim_loss(img_a, img_b_warped) * depth_mask
            diff_img = (0.15 * diff_img + 0.85 * ssim_map)

        if with_mask == True:
            weight_mask = (1 - diff_depth)
            diff_img = diff_img * weight_mask

        # compute all loss
        reconstruction_loss = mean_on_mask(diff_img, valid_mask)
        geometry_consistency_loss = mean_on_mask(diff_depth, valid_mask)

        return reconstruction_loss, geometry_consistency_loss, valid_mask_, diff_img_, diff_depth_, (depth_a < computed_depth).float()
    return compute_pairwise_loss


# compute mean value given a binary mask
def mean_on_mask(diff, valid_mask):
    mask = valid_mask.expand_as(diff)
    return (diff * mask).mean()

def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """

    # normalize
    mean_disp = disp.mean(2, True).mean(3, True)
    norm_disp = disp / (mean_disp + 1e-7)
    disp = norm_disp

    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()

def get_smooth_loss_on_mask(disp, img, mask):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    mask = torch.clamp(mask + 0.01, 1)
    
    # normalize
    mean_disp = disp.mean(2, True).mean(3, True)
    norm_disp = disp / (mean_disp + 1e-7)
    disp = norm_disp

    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:]) * mask[:, :, :, :-1]*mask[:, :, :, 1:]
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :]) * mask[:, :, :-1, :]*mask[:, :, 1:, :]

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()



@torch.no_grad()
def compute_errors(gt, pred, dataset):
    abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0, 0, 0, 0, 0, 0
    batch_size, h, w = gt.size()

    '''
    crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    construct a mask of False values, with the same size as target
    and then set to True values inside the crop
    '''
    if dataset == 'kitti':
        crop_mask = gt[0] != gt[0]
        y1, y2 = int(0.40810811 * gt.size(1)), int(0.99189189 * gt.size(1))
        x1, x2 = int(0.03594771 * gt.size(2)), int(0.96405229 * gt.size(2))
        crop_mask[y1:y2, x1:x2] = 1
        max_depth = 80

    if dataset == 'nyu':
        crop_mask = gt[0] != gt[0]
        y1, y2 = int(0.09375 * gt.size(1)), int(0.98125 * gt.size(1))
        x1, x2 = int(0.0640625 * gt.size(2)), int(0.9390625 * gt.size(2))
        crop_mask[y1:y2, x1:x2] = 1
        max_depth = 10

    for current_gt, current_pred in zip(gt, pred):
        valid = (current_gt > 0.1) & (current_gt < max_depth)
        valid = valid & crop_mask

        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid].clamp(1e-3, max_depth)

        valid_pred = valid_pred * torch.median(valid_gt)/torch.median(valid_pred)

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

        sq_rel += torch.mean(((valid_gt - valid_pred)**2) / valid_gt)

    return [metric.item() / batch_size for metric in [abs_diff, abs_rel, sq_rel, a1, a2, a3]]


def get_all_loss_fn(neighbor_range, subsampled_sequence_length, photometric_loss_weight, geometric_consistency_loss_weight, smoothness_loss_weight, with_ssim, with_mask, with_auto_mask, padding_mode):
    pairwise_loss = get_compute_pairwise_loss_fn(with_ssim, with_mask, with_auto_mask, padding_mode)
    single_weight = 2 / subsampled_sequence_length
    pairwise_count = 0
    for i in range(subsampled_sequence_length):
        for j in range(subsampled_sequence_length):
            if i != j and abs(i - j) <= neighbor_range:
                pairwise_count += 1
    pairwise_weight = 2 / pairwise_count
    #print(single_weight, pairwise_weight)
                
    def compute_all_losses(images, depths, poses, intrinsics):
        total_photometric_loss = 0
        total_geometric_consistency_loss = 0
        total_smoothness_loss = 0
        for i in range(subsampled_sequence_length):
            total_smoothness_loss += get_smooth_loss(depths[i], images[i]) * single_weight
            for j in range(subsampled_sequence_length):
                if i != j and abs(i - j) <= neighbor_range:
                    photometric_loss, geometric_consistency_loss = pairwise_loss(images[i], depths[i], images[j], depths[j], poses[i][j], intrinsics)

                    total_photometric_loss += photometric_loss * pairwise_weight
                    total_geometric_consistency_loss += geometric_consistency_loss * pairwise_weight
        return total_photometric_loss * photometric_loss_weight, total_geometric_consistency_loss * geometric_consistency_loss_weight, total_smoothness_loss * smoothness_loss_weight
    return compute_all_losses

def get_all_loss_fn(neighbor_range, subsampled_sequence_length, photometric_loss_weight, geometric_consistency_loss_weight, smoothness_loss_weight, with_ssim, with_mask, with_auto_mask, padding_mode, return_reprojections):
    pairwise_loss = get_compute_pairwise_loss_fn(with_ssim, with_mask, with_auto_mask, padding_mode)
    single_weight = 2 / subsampled_sequence_length
    pairwise_count = 0
    for i in range(subsampled_sequence_length):
        for j in range(subsampled_sequence_length):
            if i != j and abs(i - j) <= neighbor_range:
                pairwise_count += 1
    pairwise_weight = 2 / pairwise_count
    #print(single_weight, pairwise_weight)
                
    def compute_all_losses(images, depths, poses, intrinsics):
        total_photometric_loss = 0
        total_geometric_consistency_loss = 0
        total_smoothness_loss = 0

        masks = [torch.zeros_like(depths[0]) for i in range(subsampled_sequence_length)]

        if type(intrinsics) != list:
            intrinsics = [intrinsics for i in range(subsampled_sequence_length)]

        if return_reprojections:
            projected_imgs = [[[] for i in range(subsampled_sequence_length)] for i in range(subsampled_sequence_length)]
            projected_depths = [[[] for i in range(subsampled_sequence_length)] for i in range(subsampled_sequence_length)]
            depth_a_minus_computed_depth = [[[] for i in range(subsampled_sequence_length)] for i in range(subsampled_sequence_length)]


        for i in range(subsampled_sequence_length):
            for j in range(subsampled_sequence_length):
                if i != j and abs(i - j) <= neighbor_range:
                    
                    photometric_loss, geometric_consistency_loss, valid_mask_i, projected_img, projected_depth, _1 = pairwise_loss(images[i], depths[i], images[j], depths[j], poses[i][j], intrinsics[i], intrinsics[j])

                    if return_reprojections:
                        projected_imgs[i][j] = projected_img.detach() * valid_mask_i.detach()
                        projected_depths[i][j] = projected_depth.detach() * valid_mask_i.detach()

                        depth_a_minus_computed_depth[i][j] = _1.detach() * valid_mask_i.detach()

                    masks[i] += valid_mask_i.detach()

                    total_photometric_loss += photometric_loss * pairwise_weight
                    total_geometric_consistency_loss += geometric_consistency_loss * pairwise_weight
        for i in range(subsampled_sequence_length):
            total_smoothness_loss += get_smooth_loss_on_mask(depths[i], images[i], torch.clamp(masks[i], 0, 1)) * single_weight
        
        if not return_reprojections:
            return total_photometric_loss * photometric_loss_weight, total_geometric_consistency_loss * geometric_consistency_loss_weight, total_smoothness_loss * smoothness_loss_weight
        else:
            return projected_imgs, projected_depths, masks, depth_a_minus_computed_depth, \
                    total_photometric_loss * photometric_loss_weight, total_geometric_consistency_loss * geometric_consistency_loss_weight, total_smoothness_loss * smoothness_loss_weight
    return compute_all_losses


def get_all_loss_fn_clean_and_other(neighbor_range, subsampled_sequence_length, photometric_loss_weight, geometric_consistency_loss_weight, smoothness_loss_weight, with_ssim, with_mask, with_auto_mask, padding_mode, return_reprojections):
    pairwise_loss = get_compute_pairwise_loss_fn(with_ssim, with_mask, with_auto_mask, padding_mode)
    single_weight = 2 / subsampled_sequence_length
    pairwise_count = 0
    for i in range(subsampled_sequence_length):
        for j in range(subsampled_sequence_length):
            if i != j and abs(i - j) <= neighbor_range:
                pairwise_count += 1
    pairwise_weight = 2 / pairwise_count
    #print(single_weight, pairwise_weight)
                
    def compute_all_losses(clean_images, images, depths, poses, intrinsics):
        total_photometric_loss = 0
        total_geometric_consistency_loss = 0
        total_smoothness_loss = 0

        masks = [torch.zeros_like(depths[0]) for i in range(subsampled_sequence_length)]

        if type(intrinsics) != list:
            intrinsics = [intrinsics for i in range(subsampled_sequence_length)]

        if return_reprojections:
            projected_imgs = [[[] for i in range(subsampled_sequence_length)] for i in range(subsampled_sequence_length)]
            projected_depths = [[[] for i in range(subsampled_sequence_length)] for i in range(subsampled_sequence_length)]
            depth_a_minus_computed_depth = [[[] for i in range(subsampled_sequence_length)] for i in range(subsampled_sequence_length)]


        for i in range(subsampled_sequence_length):
            for j in range(subsampled_sequence_length):
                if i != j and abs(i - j) <= neighbor_range:
                    
                    photometric_loss, geometric_consistency_loss, valid_mask_i, projected_img, projected_depth, _1 = pairwise_loss(images[i], depths[i], clean_images[j], depths[j], poses[i][j], intrinsics[i], intrinsics[j])

                    if return_reprojections:
                        projected_imgs[i][j] = projected_img.detach() * valid_mask_i.detach()
                        projected_depths[i][j] = projected_depth.detach() * valid_mask_i.detach()

                        depth_a_minus_computed_depth[i][j] = _1.detach() * valid_mask_i.detach()

                    masks[i] += valid_mask_i.detach()

                    total_photometric_loss += photometric_loss * pairwise_weight
                    total_geometric_consistency_loss += geometric_consistency_loss * pairwise_weight
        for i in range(subsampled_sequence_length):
            total_smoothness_loss += get_smooth_loss_on_mask(depths[i], images[i], torch.clamp(masks[i], 0, 1)) * single_weight
        
        if not return_reprojections:
            return total_photometric_loss * photometric_loss_weight, total_geometric_consistency_loss * geometric_consistency_loss_weight, total_smoothness_loss * smoothness_loss_weight
        else:
            return projected_imgs, projected_depths, masks, depth_a_minus_computed_depth, \
                    total_photometric_loss * photometric_loss_weight, total_geometric_consistency_loss * geometric_consistency_loss_weight, total_smoothness_loss * smoothness_loss_weight
    return compute_all_losses
