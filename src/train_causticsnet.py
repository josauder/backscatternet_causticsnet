import torch
from dataset import CausticsDataset
import custom_transforms
from tqdm import tqdm
from model import SfMModel
from loss_functions import get_all_loss_fn, total_variation_sq_loss, clip_loss
import argparse
import wandb
import numpy as np
from utils import tensor2array, change_bn_momentum
import torchvision
from inverse_warp_ucm import pose_vec2mat
import torch.nn as nn
from loss_functions import SSIM, get_smooth_loss
import segmentation_models_pytorch as smp

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
parser = argparse.ArgumentParser(description='Train CausticsNet using checkpoints from Monocular SLAM Networks',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', type=str, default='data/sequences', help='path to dataset')
parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers to load data')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--max_frame_distance', type=int, default=2, help='length of long sequences')
parser.add_argument('--with_replacement', action='store_true', default=False, help='use replacement when subsampling')
parser.add_argument('--with_ssim', action='store_true', help='use SSIM loss')
parser.add_argument('--with_mask', action='store_true', help='use mask loss')
parser.add_argument('--with_auto_mask', action='store_true', help='use auto mask loss')
parser.add_argument('--padding_mode', type=str, default='zeros', help='padding mode for inverse warp')
parser.add_argument('--photometric_loss_weight', type=float, default=1.0, help='weight for photometric loss')
parser.add_argument('--geometric_consistency_loss_weight', type=float, default=0.5, help='weight for geometric consistency loss')
parser.add_argument('--smoothness_loss_weight', type=float, default=0.1, help='weight for smoothness loss')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--name', type=str, default='default', help='name of the experiment')

def main():
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_transform = custom_transforms.Compose([
        custom_transforms.ColorJitter(0.10, 0.10, 0.10, 0.05),
        custom_transforms.RandomHorizontalFlip(),
    ])
    run_wandb = wandb.init(
        project="causticsnet",
        name=args.name,
        config=vars(args)
    )

    compute_loss = get_all_loss_fn(
        1,
        2,
        args.photometric_loss_weight,
        args.geometric_consistency_loss_weight,
        args.smoothness_loss_weight,
        args.with_ssim, 
        args.with_mask, 
        args.with_auto_mask, 
        args.padding_mode,
        return_reprojections=True,
    )
    compute_ssim_loss = SSIM().to(device)

    train_dataset = CausticsDataset(args.data, train=True, transform=train_transform, individual_transform=None, 
                                                seed=args.seed, max_frame_distance=args.max_frame_distance, sequence_length=2, with_replacement=args.with_replacement)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    val_dataset = CausticsDataset(args.data, train=False, transform=None, seed=args.seed, max_frame_distance=2, sequence_length=2, with_replacement=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    sfm_model = SfMModel().to(device)
    #TODO: Exchange with args.sfm_model
    sfm_model.load_state_dict(torch.load("JointExperimentFixedIntrinsicsCenterLargeDatasetkeepalpha_best.pth"))

    sfm_model.intrinsics = torch.stack([torch.tensor(317.),torch.tensor(327.),torch.tensor(304.),torch.tensor(176.),sfm_model.intrinsics_[2]]).to(device)
    sfm_model.eval()
    
    #TODO!
    intrinsics = intrinsics.to(device)
    intrinsics = (sfm_model.intrinsics.unsqueeze(0)*sfm_model.const_mul + sfm_model.const_add).repeat(args.batch_size, 1)                
    
    caustics_model = smp.DeepLabV3Plus(encoder_name="resnext50_32x4d", encoder_weights='swsl', activation=None, classes=3).to(device)#smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=3, activation=None).to(device)
    change_bn_momentum(caustics_model,  0.01)
    optimizer = torch.optim.Adam(caustics_model.parameters(), lr=args.lr)
    best_loss = 100000
    
    for epoch in range(args.epochs):
        caustics_model.train()

        for batch_idx, images in tqdm(enumerate(train_loader)):

            images = [img.to(device) for img in images]

            with torch.no_grad():
                sfm_model = sfm_model.cuda()
                depths, poses, intermediate_losses = sfm_model(images)
                sfm_model = sfm_model.cpu()
            
            predicted_caustics = [(caustics_model(images[i]))/5 for i in range(len(images))]
            projected_imgs, projected_depths, masks, _1, photometric_loss, geometric_consistency_loss, smoothness_loss = compute_loss(images, depths, poses, intrinsics)
            
            loss = photometric_loss + geometric_consistency_loss + smoothness_loss
            
            def clamp(x):
                return torch.clamp(x, min=-0.1, max=5)

            caustics_loss = (masks[0][1].detach() * torch.abs((clamp(projected_imgs[0][1].detach())) + predicted_caustics[0]) + masks[1][0].detach() * torch.abs((clamp(projected_imgs[1][0])).detach() + predicted_caustics[1])).mean()
            images_without_caustics = [images[i] + predicted_caustics[i] for i in range(len(images))]
            _,_,_,_, photometric_loss_wo, geometric_consistency_loss_wo, smoothness_loss_wo = compute_loss(images_without_caustics, depths, poses, updated_intrinsics)
            loss_caustics_removed = photometric_loss_wo + geometric_consistency_loss_wo + smoothness_loss_wo
            
            sq_loss = 0.1 * sum([total_variation_sq_loss(predicted_caustics[i]) for i in range(len(predicted_caustics))])
            l2_caustics_loss = 0.01 * sum([torch.mean((predicted_caustics[i])**2) for i in range(len(predicted_caustics))])
            loss += l2_caustics_loss
            loss += caustics_loss
            loss += sq_loss
        
            clip_loss = sum([clip_loss_fn(images_without_caustics[i]) for i in range(len(images))])
    
            
            loss += clip_loss
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            run_wandb.log({
                    "train/photometric_loss": photometric_loss.item(),
                    "train/geometric_consistency_loss": geometric_consistency_loss.item(),
                    "train/smoothness_loss": smoothness_loss.item(),
                    "train/total_loss": loss.item(),
                    "train/sq_loss": sq_loss.item(),
                    "train/loss_caustics_removed": loss_caustics_removed.item(),
                    "train/photometric_loss_wo": photometric_loss_wo.item(),
                    "train/l2_caustics_loss": l2_caustics_loss.item(),
                    "train/caustics_loss": caustics_loss.item(),
                    "train/clip_loss": clip_loss.item(),
                })
           

            if batch_idx % 100 == 0:
                log_images = []

                for i in range(2):
                    for j in range(2):
                        log_images.append(wandb.Image(
                            np.swapaxes(tensor2array(images[i][j]).T, 0, 1),
                            caption="val Input ["+str(i)+"]["+str(j)+"]"))
                        log_images.append(wandb.Image(
                                np.swapaxes(tensor2array(images_without_caustics[i][j]).T, 0, 1),
                                caption="val Image wo Caustics ["+str(i)+"]["+str(j)+"]"))
                        log_images.append(wandb.Image(
                            np.swapaxes(tensor2array(projected_imgs[i][(i+1)%2][j]).T, 0, 1),
                            caption="val Projected Image ["+str(i)+"]["+str(j)+"]"))
                        log_images.append(wandb.Image(
                                np.swapaxes(tensor2array(predicted_caustics[i][j]).T, 0, 1),
                                caption="val Caustics ["+str(i)+"]["+str(j)+"]"))
                        log_images.append(wandb.Image(
                            np.swapaxes(tensor2array(depths[i][j], max_value=None, colormap='magma').T, 0, 1),
                            caption="val Dispnet Output Normalized ["+str(i)+"]["+str(j)+"]"))
                        log_images.append(wandb.Image(
                            np.swapaxes(tensor2array(1/ (depths[i][j]), max_value=10, colormap='magma').T, 0, 1),
                            caption="val Depth Output ["+str(i)+"]["+str(j)+"]"))

                run_wandb.log({f"Train {batch_idx}": log_images, "epoch": epoch})
                import matplotlib.pyplot as plt
                plt.imsave(f"train_{batch_idx}.png", np.swapaxes(tensor2array(1/depths[0][0], max_value=10, colormap='magma').T, 0, 1))
            if batch_idx == 50000:
                break

        caustics_model.eval()

        with torch.no_grad():
            val_photometric_loss = []
            val_photometric_loss_wo = []
            val_caustics_loss = []
            for batch_idx, images in tqdm(enumerate(val_loader)):

                images = [img.to(device) for img in images]

                with torch.no_grad():
                    sfm_model = sfm_model.cuda()
                    intrinsics = (sfm_model.intrinsics.unsqueeze(0)*sfm_model.const_mul + sfm_model.const_add).repeat(images[0].shape[0], 1)
                    updated_intrinsics = intrinsics
                    intrinsics = intrinsics.to(device)
        
                    depths, poses, intermediate_losses = sfm_model(images, intrinsics)
                    sfm_model = sfm_model.cpu()

                
                projected_imgs, projected_depths, masks, _, photometric_loss, geometric_consistency_loss, smoothness_loss = compute_loss(images, depths, poses, updated_intrinsics)

                predicted_caustics = [(caustics_model(images[i]))/5 for i in range(len(images))]
            
                caustics_loss = (masks[0][1] * torch.abs((clamp(projected_imgs[0][1])) + predicted_caustics[0]) + masks[1][0] * torch.abs((clamp(projected_imgs[1][0])) + predicted_caustics[1])).mean()
                val_caustics_loss.append(caustics_loss.item())

                images_without_caustics = [images[i] + predicted_caustics[i] for i in range(len(images))]

                _,_,_,_, photometric_loss_wo, geometric_consistency_loss_wo, smoothness_loss_wo = compute_loss(images_without_caustics, depths, poses, updated_intrinsics)

                val_photometric_loss_wo.append(photometric_loss_wo.item())
                val_photometric_loss.append(photometric_loss.item())                                

            run_wandb.log({
                "val/photometric_loss": np.mean(val_photometric_loss),
                "val/photometric_loss_wo": np.mean(val_photometric_loss_wo),
                "val/caustics_loss": np.mean(val_caustics_loss),
            })
            torch.save(caustics_model.state_dict(), args.name + "_caustics_last.pth")
            if np.mean(val_photometric_loss_wo) < best_loss:
                best_loss = np.mean(val_photometric_loss_wo)
                torch.save(caustics_model.state_dict(), args.name + "_caustics_best.pth")
            if epoch % 10 == 0 and epoch > 0:
                torch.save(caustics_model.state_dict(), args.name + "_caustics_" + str(epoch) + ".pth")
            
if __name__ == '__main__':
    main()