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