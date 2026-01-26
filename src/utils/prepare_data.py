import os
import glob
import numpy as np
import nibabel as nib
from PIL import Image
import yaml
import argparse
import random
from tqdm import tqdm


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def normalize_image(img_data):
    min_val = np.min(img_data)
    max_val = np.max(img_data)
    if (max_val - min_val) == 0:
        return img_data.astype(np.uint8)
    img_norm = (img_data - min_val) / (max_val - min_val) * 255
    return img_norm.astype(np.uint8)


def save_slice(img_slice, save_path, resize_to=None):
    im = Image.fromarray(img_slice)
    if resize_to:
        im = im.resize((resize_to, resize_to), resample=Image.BILINEAR)
    im.save(save_path)


def process_dataset(config):
    data_cfg = config['data']
    raw_dir = data_cfg['raw_dir']
    output_dir = data_cfg['output']
    modality = data_cfg['modality']
    slice_threshold = data_cfg['slice_threshold']
    image_size = data_cfg.get('image_size', 256)
    seed = data_cfg.get('seed', 42)

    train_ratio = data_cfg.get('train_ratio', 0.7)
    val_ratio = data_cfg.get('validate_ratio', 0.1)

    random.seed(seed)
    np.random.seed(seed)

    patient_folders = sorted(glob.glob(os.path.join(raw_dir, '*')))
    patient_ids = [p for p in patient_folders if os.path.isdir(p)]

    if not patient_ids:
        print(f"Error: No data found in {raw_dir}")
        return

    random.shuffle(patient_ids)

    total_patients = len(patient_ids)
    n_train = int(total_patients * train_ratio)
    n_val = int(total_patients * val_ratio)

    splits = {
        'train': patient_ids[:n_train],
        'val': patient_ids[n_train: n_train + n_val],
        'test': patient_ids[n_train + n_val:]
    }

    print(f"Modality: {modality} | Size: {image_size}x{image_size}")
    print(f"Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")

    for split_name, patients in splits.items():
        img_save_dir = os.path.join(output_dir, split_name, 'images')
        mask_save_dir = os.path.join(output_dir, split_name, 'masks')

        os.makedirs(img_save_dir, exist_ok=True)
        if split_name != 'train':
            os.makedirs(mask_save_dir, exist_ok=True)

        print(f"Processing {split_name}...")

        for patient_path in tqdm(patients):
            pid = os.path.basename(patient_path)

            img_files = glob.glob(os.path.join(patient_path, f"*{modality}.nii.gz"))
            seg_files = glob.glob(os.path.join(patient_path, "*seg.nii.gz"))

            if not img_files or not seg_files:
                continue

            img_vol = nib.load(img_files[0]).get_fdata()
            seg_vol = nib.load(seg_files[0]).get_fdata()

            num_slices = img_vol.shape[2]

            for i in range(num_slices):
                img_slice = img_vol[:, :, i]
                seg_slice = seg_vol[:, :, i]

                if np.count_nonzero(img_slice) < (img_slice.size * slice_threshold):
                    continue

                img_slice_norm = normalize_image(img_slice)

                seg_slice[seg_slice > 0] = 255
                seg_slice = seg_slice.astype(np.uint8)

                has_tumor = np.max(seg_slice) > 0
                filename = f"{pid}_slice_{i:03d}.png"

                if split_name == 'train':
                    # Only save healthy slices for training
                    if not has_tumor:
                        save_slice(img_slice_norm, os.path.join(img_save_dir, filename), resize_to=image_size)
                else:
                    save_slice(img_slice_norm, os.path.join(img_save_dir, filename), resize_to=image_size)
                    mask_im = Image.fromarray(seg_slice).resize((image_size, image_size), resample=Image.NEAREST)
                    mask_im.save(os.path.join(mask_save_dir, filename))

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../config/brats_config.yaml')
    args = parser.parse_args()

    if os.path.exists(args.config):
        cfg = load_config(args.config)
        process_dataset(cfg)
    else:
        print(f"Config file not found: {args.config}")