import os
import sys
import argparse
import torch
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAUROC
from tqdm import tqdm

from src.utils.helpers import save_visualization

sys.path.append(os.getcwd())

from src.data.dataset import BraTSPoissonDataset
from src.models.LADN import LatentDiffusion

def inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)

    print("Loading test dataset...")
    test_dataset = BraTSPoissonDataset(root_path=args.data_dir, image_size=args.image_size, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print("Test dataset have: ", len(test_dataset), " images")

    print("Loading model...")
    model = LatentDiffusion().to(device)

    if args.checkpoint_best is None or not os.path.exists(args.checkpoint_best):
        raise FileNotFoundError("Checkpoint best not found")

    checkpoint = torch.load(args.checkpoint_best, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded weights.")

    model.eval()

    pixel_auroc = BinaryAUROC(thresholds=2000).to(device)
    image_auroc = BinaryAUROC().to(device)

    thresholds = torch.linspace(0, 1, 100).to(device).view(100, 1)
    stats_intersection = torch.zeros(100).to(device)
    stats_union = torch.zeros(100).to(device)

    print("Start inference...")
    save_dir = os.path.join(args.log_dir, "results")

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="[Inference] ")):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device).long()
            labels = batch['label'].to(device).long()
            B = images.shape[0]

            _, _, _, latents = model.vae(images)

            t_val = torch.full((B,), args.healing_t, device=device).long()
            noise = torch.randn_like(latents)
            alpha = 1 - (args.healing_t / 1000.0)
            sigma = args.healing_t / 1000.0

            noisy_latents = latents * alpha + noise * sigma

            noise_pred = model.unet(noisy_latents, t_val)
            healed_latents = (noisy_latents - noise_pred * sigma) / alpha
            healed_images = model.vae.decoder(healed_latents)

            diff = torch.abs(images - healed_images)
            anomaly_map = torch.mean(diff, dim=1, keepdim=True)
            anomaly_map = torch.nn.functional.avg_pool2d(anomaly_map, 5, stride=1, padding=2)

            for b in range(B):
                mi, ma = anomaly_map[b].min(), anomaly_map[b].max()
                if ma - mi > 1e-6:
                    anomaly_map[b] = (anomaly_map[b] - mi) / (ma - mi)

            pixel_auroc.update(anomaly_map.view(-1), masks.view(-1))
            max_scores = anomaly_map.view(B, -1).max(dim=1)[0]
            image_auroc.update(max_scores, labels)

            flat_preds = anomaly_map.view(-1)
            flat_masks = masks.view(-1)
            pred_binary = (flat_preds.unsqueeze(0) > thresholds).float()

            intersection = (pred_binary * flat_masks.unsqueeze(0)).sum(dim=1)
            union = pred_binary.sum(dim=1) + flat_masks.sum()

            stats_intersection += intersection
            stats_union += union

            if i < 5:
                save_visualization(images, healed_images, anomaly_map, masks, save_dir, i)

        print("Computing Final Metrics.")
        final_p_auroc = pixel_auroc.compute().item()
        final_i_auroc = image_auroc.compute().item()

        dices = torch.zeros_like(stats_union)
        valid_mask = stats_union > 0
        dices[valid_mask] = (2.0 * stats_intersection[valid_mask]) / stats_union[valid_mask]

        best_dice, best_idx = torch.max(dices, dim=0)
        best_thr = thresholds[best_idx].item()
        best_dice = best_dice.item()

        print(f"\n{'=' * 40}")
        print(f"PIXEL-AUROC: {final_p_auroc:.4f}")
        print(f"IMAGE-AUROC: {final_i_auroc:.4f}")
        print(f"BEST DICE:   {best_dice:.4f} (at threshold {best_thr:.4f})")
        print(f"{'=' * 40}")

        with open(os.path.join(save_dir, "results.txt"), "w") as f:
            f.write(f"PIXEL-AUROC: {final_p_auroc:.4f}\n")
            f.write(f"IMAGE-AUROC: {final_i_auroc:.4f}\n")
            f.write(f"BEST DICE:   {best_dice:.4f} (Threshold: {best_thr:.4f})\n")

        print("DONE!!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../config/brats_config.yaml')
    parser.add_argument('--checkpoint', type=str, help='Path file .pth')
    parser.add_argument('--healing_t', type=int, default=300)
    cli_args = parser.parse_args()

    if not os.path.exists(cli_args.config):
        raise FileNotFoundError(cli_args.config)

    print("Loading config...")
    with open(cli_args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print(yaml.dump(config, sort_keys=False, allow_unicode=True, default_flow_style=False))
    args = argparse.Namespace(**config["data"], **config["train"])

    if cli_args.checkpoint:
        args.checkpoint_best = cli_args.checkpoint
    elif args.checkpoint_best is None:
        args.checkpoint_best = os.path.join(args.checkpoint_dir, "best_auroc.pth")

    args.healing_t = cli_args.healing_t

    inference(args)