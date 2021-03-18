import numpy as np


def create_circular_mask(h, w, center=None, radius=None):
    if center is None:
        center = (int(w / 2), int(h / 2))
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def get_img_spector(img, w=64):
    img = img - 100.4

    spector = np.zeros((40, 3))
    for r in range(0, 40):
        mask = create_circular_mask(w, w, radius=r) * (~create_circular_mask(w, w, radius=r - 1))
        spector[r, 0] = np.sum(mask * img) / np.sum(mask)
        spector[r, 1] = np.std(mask * img) / np.sum(mask) * 100
        spector[r, 2] = np.sum(mask * img > 0) / np.sum(mask) * 20
    return spector
