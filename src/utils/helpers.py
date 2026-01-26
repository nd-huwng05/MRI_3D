import os

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm


def visualization(writer, images, healed_images, anomaly_maps, masks, epoch):
    limit = min(4, images.shape[0])
    img_show = (images[:limit, 1:2] * 0.5 + 0.5).clamp(0, 1).repeat(1, 3, 1, 1)
    heal_show = (healed_images[:limit, 1:2] * 0.5 + 0.5).clamp(0, 1).repeat(1, 3, 1, 1)
    map_show = anomaly_maps[:limit].repeat(1, 3, 1, 1)
    map_show = (map_show - map_show.min()) / (map_show.max() - map_show.min() + 1e-8)
    mask_show = masks[:limit].repeat(1, 3, 1, 1)

    writer.add_images("Vis/Input_Tumor", img_show, epoch)
    writer.add_images("Vis/Healed_Healthy", heal_show, epoch)
    writer.add_images("Vis/Anomaly_Map", map_show, epoch)
    writer.add_images("Vis/Ground_Truth", mask_show, epoch)

def save_visualization(images, healed_images, anomaly_maps, masks, save_dir, batch_idx):
    os.makedirs(save_dir, exist_ok=True)
    img = images[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5
    heal = healed_images[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5
    amap = anomaly_maps[0, 0].cpu().numpy()
    mask = masks[0, 0].cpu().numpy()

    amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
    heatmap = cv2.applyColorMap(np.uint8(amap * 255), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    overlay = 0.6 * img + 0.4 * heatmap

    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    axs[0].imshow(np.clip(img, 0, 1));
    axs[0].set_title("Input")
    axs[1].imshow(np.clip(heal, 0, 1));
    axs[1].set_title("Healed")
    axs[2].imshow(np.abs(img - heal).mean(axis=2), cmap='jet');
    axs[2].set_title("Diff")
    axs[3].imshow(np.clip(overlay, 0, 1));
    axs[3].set_title("Anomaly Map")
    axs[4].imshow(mask, cmap='gray');
    axs[4].set_title("GT")
    for ax in axs: ax.axis('off')
    plt.savefig(os.path.join(save_dir, f"vis_batch_{batch_idx}.png"), bbox_inches='tight')
    plt.close()

def compute_best_dice(anomaly_maps, ground_truth_masks):
    print("Computing best DICE.....")
    device = anomaly_maps.device
    best_dice = 0.0
    best_threshold = 0.0
    thresholds = torch.linspace(0, 1, 101, device=device)

    for thr in tqdm(thresholds, desc="Searching Threshold"):
        pred_binary = (anomaly_maps > thr).float()

        # Dice: 2TP / (2TP + FP + FN)
        intersection = (pred_binary * ground_truth_masks).sum()
        union = pred_binary.sum() + ground_truth_masks.sum()

        if union == 0:
            dice = 1.0
        else:
            dice = (2.0 * intersection) / union

        if dice > best_dice:
            best_dice = dice
            best_threshold = thr

    return best_dice.item(), best_threshold.item()