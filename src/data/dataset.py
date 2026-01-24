import os
import random
import argparse
import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms


class BraTSPoissonDataset(Dataset):
    def __init__(self, root_path, mode='train', image_size=256, anomaly_prob=0.5):
        self.root_path = root_path
        self.mode = mode
        self.image_size = image_size
        self.anomaly_prob = anomaly_prob
        self.image_paths = os.path.join(root_path, mode, 'images')
        self.masks_paths = os.path.join(root_path, mode, 'masks')

        if os.path.exists(self.image_paths):
            self.image_files = sorted([f for f in os.listdir(self.image_paths) if f.endswith('.png')])
        else:
            self.image_files = []
            print(f"Warning: Path not found {self.image_paths}")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def generate_fractal_noise(self, h, w, scales=[4, 8, 16, 32]):
        # Generate multi-scale fractal noise for realistic tumor texture
        noise = np.zeros((h, w), dtype=np.float32)
        for scale in scales:
            grid_h, grid_w = h // scale, w // scale
            if grid_h == 0 or grid_w == 0: continue

            random_noise = np.random.rand(grid_h, grid_w).astype(np.float32)
            upscaled = cv2.resize(random_noise, (w, h), interpolation=cv2.INTER_CUBIC)
            noise += upscaled * (scale / max(scales))

        noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise) + 1e-5)
        return noise

    def generate_synomaly_mask(self, img_numpy, h, w):
        if len(img_numpy.shape) == 3:
            img_gray = cv2.cvtColor(img_numpy, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img_numpy

        _, thresh = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY)
        brain_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        brain_mask_filled = np.zeros((h, w), dtype=np.uint8)
        if brain_contours: cv2.drawContours(brain_mask_filled, brain_contours, -1, 255, thickness=-1)

        # Create safety zone: Erode brain mask to prevent tumor from touching borders
        kernel_safety = np.ones((9, 9), np.uint8)
        safe_zone_mask = cv2.erode(brain_mask_filled, kernel_safety, iterations=3)
        brain_pixels = np.where(safe_zone_mask > 0)

        # Fallback if erosion removes too much (e.g., small brain slices)
        if len(brain_pixels[0]) < 100: brain_pixels = np.where(brain_mask_filled > 0)

        total_brain_pixels = np.sum(brain_mask_filled) / 255
        if total_brain_pixels < 500: return np.zeros((h, w), dtype=np.uint8), (w // 2, h // 2)

        # Determine tumor size based on brain area
        tumor_type = random.choices(['small', 'medium', 'large', 'huge'], weights=[0.4, 0.3, 0.2, 0.1])[0]
        if tumor_type == 'small':
            target_ratio = random.uniform(0.05, 0.10); noise_scale = random.randint(24, 32)
        elif tumor_type == 'medium':
            target_ratio = random.uniform(0.10, 0.25); noise_scale = random.randint(16, 24)
        elif tumor_type == 'large':
            target_ratio = random.uniform(0.25, 0.40); noise_scale = random.randint(12, 16)
        else:
            target_ratio = random.uniform(0.40, 0.55); noise_scale = random.randint(8, 12)

        target_area = total_brain_pixels * target_ratio

        # Generate organic shape
        noise_small = np.random.rand(noise_scale, noise_scale).astype(np.float32)
        noise_blurred = cv2.GaussianBlur(noise_small, (3, 3), 0)
        noise_large = cv2.resize(noise_blurred, (w, h), interpolation=cv2.INTER_CUBIC)

        best_thresh = 0.7
        min_diff = float('inf')
        for t in np.linspace(0.4, 0.9, 10):
            temp_mask = (noise_large > t).astype(np.uint8)
            temp_area = np.sum(temp_mask)
            estimated_blob_area = temp_area * 0.4
            diff = abs(estimated_blob_area - target_area)
            if diff < min_diff: min_diff = diff; best_thresh = t

        raw_mask = (noise_large > best_thresh).astype(np.uint8) * 255

        # Select best blob
        contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)

        if contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for cnt in contours[:3]:
                area = cv2.contourArea(cnt)
                blob_ratio = area / total_brain_pixels
                if blob_ratio < 0.02 or blob_ratio > 0.65: continue

                blob_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(blob_mask, [cnt], -1, 255, thickness=-1)
                M = cv2.moments(cnt)
                if M["m00"] == 0: continue
                blob_cX, blob_cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

                for _ in range(10):
                    if len(brain_pixels[0]) == 0: break
                    idx = random.randint(0, len(brain_pixels[0]) - 1)
                    target_cY, target_cX = brain_pixels[0][idx], brain_pixels[1][idx]

                    M_shift = np.float32([[1, 0, target_cX - blob_cX], [0, 1, target_cY - blob_cY]])
                    shifted_mask = cv2.warpAffine(blob_mask, M_shift, (w, h))
                    final_mask_candidate = cv2.bitwise_and(shifted_mask, brain_mask_filled)

                    original_area = np.sum(shifted_mask) / 255
                    final_area = np.sum(final_mask_candidate) / 255
                    if original_area == 0: continue

                    # Retention check: Ensure tumor shape isn't cut off by image boundary
                    if (final_area / original_area) > 0.95:
                        final_mask = final_mask_candidate
                        M_final = cv2.moments(final_mask)
                        if M_final["m00"] != 0:
                            center = (int(M_final["m10"] / M_final["m00"]), int(M_final["m01"] / M_final["m00"]))
                        return final_mask, center

        return final_mask, center

    def poisson_blending(self, img):
        if len(img.shape) == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_rgb = img
        h, w, _ = img_rgb.shape

        mask_core, center = self.generate_synomaly_mask(img_rgb, h, w)
        if np.sum(mask_core) < 50: return img_rgb, mask_core

        # Generate realistic texture using fractal noise
        fractal = self.generate_fractal_noise(h, w, scales=[4, 8, 16, 32])
        fractal_img = (fractal * 255).astype(np.uint8)
        fractal_rgb = cv2.merge([fractal_img, fractal_img, fractal_img])

        # Synthesize Active Tumor (Bright) and Necrotic Core (Dark)
        active_tumor = cv2.addWeighted(fractal_rgb, 0.7, np.full_like(fractal_rgb, 200), 0.3, 0)
        necrotic_core = cv2.addWeighted(fractal_rgb, 0.8, np.zeros_like(fractal_rgb), 0.2, 0)

        tumor_texture = np.where(fractal_rgb < 100, necrotic_core, active_tumor)
        tumor_texture = np.clip(tumor_texture.astype(np.float32) * 0.8 + 40, 0, 255).astype(np.uint8)

        # Blend tumor core using gradient blending
        mask_blurred = cv2.GaussianBlur(mask_core, (11, 11), 5)
        alpha = mask_blurred.astype(np.float32) / 255.0
        alpha = cv2.merge([alpha, alpha, alpha])

        img_float = img_rgb.astype(np.float32)
        tumor_float = tumor_texture.astype(np.float32)

        # Preserve high-frequency brain details for realistic integration
        brain_detail = img_float - cv2.GaussianBlur(img_float, (5, 5), 0)
        tumor_with_detail = tumor_float + (brain_detail * 0.3)

        final_blended = (tumor_with_detail * alpha) + (img_float * (1.0 - alpha))

        # Add peritumoral edema
        edema_mask = cv2.dilate(mask_core, np.ones((15, 15), np.uint8), iterations=1)
        edema_mask = cv2.GaussianBlur(edema_mask, (41, 41), 15)
        edema_alpha = (edema_mask.astype(np.float32) / 255.0) * 0.4
        edema_alpha = cv2.merge([edema_alpha, edema_alpha, edema_alpha])

        edema_region = img_float * 1.5
        final_blended = (edema_region * edema_alpha * (1 - alpha)) + (final_blended * (1 - (edema_alpha * (1 - alpha))))

        final_blended = np.clip(final_blended, 0, 255).astype(np.uint8)

        return final_blended, mask_core

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_paths, img_name)
        image_pil = Image.open(img_path).convert('RGB')
        image_np = np.array(image_pil)

        input_image = image_np.copy()
        target_image = image_np.copy()
        anomaly_mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        has_anomaly = 0.0

        if self.mode == 'train' and random.random() < self.anomaly_prob:
            try:
                blended, mask = self.poisson_blending(image_np)
                if np.max(mask) > 0:
                    input_image = blended
                    anomaly_mask = mask.astype(np.float32) / 255.0
                    has_anomaly = 1.0
            except:
                pass
        elif self.mode in ['val', 'test']:
            mask_path = os.path.join(self.masks_paths, img_name)
            if os.path.exists(mask_path):
                mask_pil = Image.open(mask_path).convert('L')
                anomaly_mask = np.array(mask_pil).astype(np.float32) / 255.0
                if np.max(anomaly_mask) > 0: has_anomaly = 1.0

        input_tensor = self.transform(Image.fromarray(input_image))
        target_tensor = self.transform(Image.fromarray(target_image))
        mask_tensor = torch.tensor(anomaly_mask).unsqueeze(0)
        mask_tensor = (mask_tensor > 0.5).float()

        return {'image': input_tensor, 'target': target_tensor, 'mask': mask_tensor,
                'label': torch.tensor(has_anomaly).float(), 'name': img_name}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../../config/brats_config.yaml')
    args = parser.parse_args()

    if os.path.exists(args.config):
        cfg = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
        dataset = BraTSPoissonDataset(root_path=cfg['data']['output'], mode='train', anomaly_prob=1.0)
        print(f"Dataset Loaded: {len(dataset)} images.")

        if len(dataset) > 0:
            found = False
            for _ in range(50):
                idx = random.randint(0, len(dataset) - 1)
                sample = dataset[idx]
                if torch.max(sample['mask']) > 0: found = True; break

            if not found:
                print("[ERROR] Could not generate tumor.")
            else:
                print(f"Sample: {sample['name']}")


                def to_img(tensor):
                    img = tensor.permute(1, 2, 0).numpy();
                    img = (img * 0.5 + 0.5);
                    return np.clip(img, 0, 1)


                input_img = to_img(sample['image'])
                target_img = to_img(sample['target'])
                mask_img = sample['mask'].squeeze().numpy()

                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                ax[0].imshow(target_img);
                ax[0].set_title("Target (Healthy)");
                ax[0].axis('off')
                ax[1].imshow(input_img);
                ax[1].set_title("Input (Fractal Tumor)");
                ax[1].axis('off')
                ax[2].imshow(mask_img, cmap='gray', vmin=0, vmax=1, interpolation='nearest');
                ax[2].set_title("Mask");
                ax[2].axis('off')

                plt.savefig("test_realistic_fractal.png")
                print("\n[SUCCESS] Image saved to 'test_realistic_fractal.png'.")
        else:
            print("[ERROR] Dataset empty.")
    else:
        print(f"Config not found: {args.config}")