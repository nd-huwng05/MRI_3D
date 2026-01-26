import os
import argparse
import torch
import yaml
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import BinaryAUROC, BinaryF1Score, BinaryJaccardIndex
from tqdm import tqdm

from data.dataset import BraTSPoissonDataset
from models.LADN import LatentDiffusion
from utils.helpers import visualization


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device used: ", device)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=args.log_dir)

    print("Loading data.....")
    dataset = BraTSPoissonDataset(root_path=args.data_dir, image_size=args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    print("Dataset train have: ", len(dataset), "images")
    val_dataset = BraTSPoissonDataset(root_path=args.data_dir, image_size=args.image_size, mode="val")
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print("Dataset val have: ", len(val_dataset), "images")

    model = LatentDiffusion().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    # Pixel-level metrics
    pixel_auroc = BinaryAUROC(thresholds=2000).to(device)
    pixel_dice = BinaryF1Score().to(device)
    pixel_iou = BinaryJaccardIndex().to(device)

    # Image-level metrics
    image_auroc = BinaryAUROC().to(device)

    start_epoch = 0
    best_pixel_auroc = 0.0
    global_step = 0

    if hasattr(args, "resume_path") and args.resume_path and os.path.exists(args.resume_path):
        if os.path.isfile(args.resume_path):
            print("Loading checkpoint to resume training from:", args.resume_path)
            checkpoint = torch.load(args.resume_path, map_location=device)

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_pixel_auroc = checkpoint.get('best_pixel_auroc', 0.0)
                print("Model will continue training from: ", args.resume_path, "Start epoch: ", start_epoch)
            else:
                model.load_state_dict(checkpoint)
                print("Loaded checkpoint from: ", args.resume_path, "but will train with new optimizer",
                      "Start epoch: ", start_epoch)
        else:
            print("[Warning] No checkpoint found at: ", args.resume_path)

    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()

    strategy = getattr(args, 'training_strategy', 'joint')
    healing_t = getattr(args, 'healing_t', 400)

    if strategy == 'sequential':
        total_epochs = args.vae_epochs + args.diff_epochs
        print(f"Strategy: SEQUENTIAL (VAE: {args.vae_epochs} -> Diff: {args.diff_epochs})")
    else:
        total_epochs = args.epochs
        print(f"Strategy: JOINT ({total_epochs} epochs)")

    print(f"Starting training form {start_epoch + 1} to {total_epochs} epochs...")
    for epoch in range(start_epoch, total_epochs):
        model.train()

        train_vae = True
        train_diff = True

        if strategy == 'sequential':
            if epoch < args.vae_epochs:
                train_diff = False
            else:
                train_vae = False

        for p in model.vae.parameters(): p.requires_grad = train_vae
        for p in model.unet.parameters(): p.requires_grad = train_diff

        if not train_vae: model.vae.eval()
        if not train_diff: model.unet.eval()

        pbar = tqdm(dataloader, desc=f"[Train] Epoch {epoch + 1}/{total_epochs}")

        total_vae_loss = 0
        total_diff_loss = 0

        for batch in pbar:
            images = batch['target'].to(device)
            B = images.shape[0]

            optimizer.zero_grad()

            # Training VAE
            if train_vae:
                rec_imgs, mu, log_var, latents = model.vae(images)
                rec_loss = l1_loss(rec_imgs, images)
                kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / (
                        B * 256 * 256)
                vae_loss = rec_loss + 0.00025 * kld_loss
            else:
                with torch.no_grad():
                    _, _, _, latents = model.vae(images)
                vae_loss = torch.tensor(0.0, device=device)

            # Training diffusion
            if train_diff:
                latents_input = latents.detach()
                t = torch.randint(0, 1000, (B,), device=device).long()
                noise = torch.randn_like(latents_input)
                noise_level = t.view(-1, 1, 1, 1) / 1000.0
                noisy_latents = latents_input * (1 - noise_level) + noise * noise_level
                noise_pred = model.unet(noisy_latents, t)
                diff_loss = mse_loss(noise_pred, noise)
            else:
                diff_loss = torch.tensor(0.0, device=device)

            # Update
            loss = vae_loss + diff_loss
            if loss > 0:
                loss.backward()
                optimizer.step()

            writer.add_scalar('Train/Loss_vae', vae_loss.item(), global_step)
            writer.add_scalar('Train/Loss_diff', diff_loss.item(), global_step)
            writer.add_scalar('Train/Loss_total', loss.item(), global_step)
            global_step += 1

            total_vae_loss += vae_loss.item()
            total_diff_loss += diff_loss.item()

            pbar.set_postfix({
                "vae": f"{vae_loss.item():.4f}",
                "diff": f"{diff_loss.item():.4f}"
            })

        avg_vae_loss = total_vae_loss / len(dataloader)
        avg_diff_loss = total_diff_loss / len(dataloader)

        model.eval()
        pixel_auroc.reset()
        pixel_dice.reset()
        pixel_iou.reset()
        image_auroc.reset()

        sample_imgs, sample_healed, sample_maps, sample_masks = None, None, None, None
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_dataloader, desc=f"[Val] Epoch {epoch + 1}/{total_epochs}")):
                images = batch['image'].to(device)
                masks = batch['mask'].to(device).long()
                labels = batch['label'].to(device).long()
                B = images.shape[0]

                _, _, _, latents = model.vae(images)
                t_val = torch.full((B,), healing_t, device=device).long()
                noise = torch.randn_like(latents)
                alpha = 1 - (healing_t / 1000.0)
                sigma = healing_t / 1000.0
                noisy_latents = latents * alpha + noise * sigma

                # Denoise
                noise_pred = model.unet(noisy_latents, t_val)

                healed_latents = (noisy_latents - noise_pred * sigma) / alpha
                healed_images = model.vae.decoder(healed_latents)

                # --- FIX: Convert to [0,1] range for Diff Calculation ---
                # Dataset & Model use [-1, 1]. Convert to [0, 1] to calculate meaningful difference
                images_01 = (images + 1) / 2
                healed_images_01 = (healed_images + 1) / 2

                # Diff calculation
                diff = torch.abs(images_01 - healed_images_01)
                anomaly_map = torch.mean(diff, dim=1, keepdim=True)
                anomaly_map = torch.nn.functional.avg_pool2d(anomaly_map, 5, stride=1, padding=2)

                # --- CRITICAL FIX: REMOVED MIN-MAX SCALING LOOP HERE ---
                # Do NOT normalize per image. Healthy images should have low scores.

                preds_flat = anomaly_map.view(-1)
                targets_flat = masks.view(-1)

                pixel_auroc.update(preds_flat, targets_flat)
                pixel_dice.update((preds_flat > 0.1).long(), targets_flat)
                pixel_iou.update((preds_flat > 0.1).long(), targets_flat)

                max_scores = anomaly_map.view(B, -1).max(dim=1)[0]
                labels_cpu = labels
                image_auroc.update(max_scores, labels_cpu)

                if i == 0:
                    sample_imgs, sample_healed, sample_maps, sample_masks = images, healed_images, anomaly_map, masks

        val_p_auroc = pixel_auroc.compute().item()
        val_dice = pixel_dice.compute().item()
        val_iou = pixel_iou.compute().item()
        val_img_auroc = image_auroc.compute().item()

        print(
            f"Epoch {epoch + 1}: Pixel-AUROC: {val_p_auroc:.4f} DICE: {val_dice:.4f} Img-AUROC: {val_img_auroc:.4f} Val_IOU: {val_iou:.4f} vae-loss: {avg_vae_loss:.4f}, diff-loss: {avg_diff_loss:.4f}")

        writer.add_scalar('Val/Pixel_AUROC', val_p_auroc, epoch)
        writer.add_scalar('Val/Pixel_Dice', val_dice, epoch)
        writer.add_scalar('Val/Pixel_IoU', val_iou, epoch)
        writer.add_scalar('Val/Image_AUROC', val_img_auroc, epoch)

        visualization(writer, sample_imgs, sample_healed, sample_maps, sample_masks, epoch)
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_pixel_auroc': best_pixel_auroc,
            'config': vars(args)
        }

        torch.save(state, os.path.join(args.checkpoint_dir, "last.pth"))
        if val_p_auroc > best_pixel_auroc:
            best_pixel_auroc = val_p_auroc
            torch.save(state, os.path.join(args.checkpoint_dir, "best_auroc.pth"))

    writer.close()
    print("DONE!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training model Latent Anomaly Diffusion Network')
    parser.add_argument('--config', type=str, default='../config/brats_config.yaml', help='Path to config file')
    parser.add_argument('--healing_t', type=int, default=None, help='Override healing_t')
    cli_args = parser.parse_args()

    if not os.path.exists(cli_args.config):
        if os.path.exists(f"../{cli_args.config}"):
            cli_args.config = f"../{cli_args.config}"
        else:
            raise FileNotFoundError(cli_args.config)

    print("Loading config...")
    with open(cli_args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print(yaml.dump(config, sort_keys=False, allow_unicode=True, default_flow_style=False))
    args = argparse.Namespace(**config["data"], **config["train"])

    if cli_args.healing_t is not None:
        args.healing_t = cli_args.healing_t

    train(args)