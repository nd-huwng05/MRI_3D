
import numpy as np
import torch.nn.functional as F
import torch

def crop_3d_volume_to_size(data,resize_size=None, target_size=None,output_type=np.ndarray):
    """ Resize the volume to resize_size, and then center-crop to target_size
    Args:
        data [H,W,D] (any of np.ndarray, list, torch.tensor): 3D volume data

    """
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    elif isinstance(data, list):
        data = torch.tensor(data)
    if resize_size:
        resized_volume = F.interpolate(data.unsqueeze(0).unsqueeze(0),
                                   size = resize_size,
                                   mode = 'trilinear',
                                   align_corners=False)
    else:
        resized_volume = data
    h, w, d = resized_volume.shape
    if target_size:
        crop_h, crop_w, crop_d= target_size

        crop_d_start = max(0, (d - crop_d) // 2)
        crop_h_start = max(0, (h - crop_h) // 2)
        crop_w_start = max(0, (w - crop_w) // 2)

        crop_d_end = min(d, crop_d_start + crop_d)
        crop_h_end = min(h, crop_h_start + crop_h)
        crop_w_end = min(w, crop_w_start + crop_w)

        # Perform center crop
        cropped_volume = resized_volume[crop_h_start:crop_h_end, crop_w_start:crop_w_end,crop_d_start:crop_d_end]
    else:
        cropped_volume = resized_volume
    if output_type == np.ndarray:
        cropped_volume = cropped_volume.numpy()
    elif output_type == list:
        cropped_volume = cropped_volume.tolist()
    return cropped_volume