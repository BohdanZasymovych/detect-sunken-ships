import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, label


def filter_detections(ship_mask, min_size, max_size):
    """
    Remove small detections from the ship mask.
    :param ship_mask: Binary mask of detected ship regions.
    :param min_size: Minimum size of a ship to keep.
    :return: Cleaned ship mask.
    """
    labeled, num_features = label(ship_mask)
    cleaned_mask = np.zeros_like(ship_mask, dtype=bool)
    for i in range(1, num_features + 1):
        component = (labeled == i)
        if min_size <= np.sum(component) <= max_size:
            cleaned_mask |= component
    return cleaned_mask


def detect_ships(depth_map, ship_length, ship_width, ship_height):
    """
    Detect sunken ships in the depth map.
    :param depth_map: Depth map to analyze.
    :param ship_length: Length of the ship.
    :param ship_width: Width of the ship.
    :param ship_height: Height of the ship (negative for a sunken ship).
    :return: Binary mask of detected ship regions.
    """
    expected_area = ship_length * ship_width
    min_region_size = expected_area * 0.05
    max_region_size = expected_area * 0.9

    smoothed = gaussian_filter(depth_map, sigma=max(1, int((ship_length + ship_width) / 10)))
    anomaly = smoothed - depth_map
    anomaly_threshold = (ship_height*0.7) * 0.35
    ship_mask = anomaly > anomaly_threshold
    cleaned_mask = filter_detections(ship_mask, min_region_size, max_region_size)

    return cleaned_mask


def visualize_detection(depth_map, ship_mask, size):
    """
    Visualize the depth map and detected ship regions.
    :param depth_map: Depth map to visualize.
    :param ship_mask: Binary mask of detected ship regions.
    :param size: Size of the map in pixels.
    """
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    im0 = axs[0].imshow(depth_map, cmap='viridis')
    axs[0].set_title(f"Depth Map ({size}x{size}m)")
    axs[0].legend()

    cbar = fig.colorbar(im0, ax=axs[0])
    cbar.set_label('Depth (m)')

    axs[1].imshow(ship_mask, cmap='gray')
    axs[1].set_title("Detected Ship Regions")
    plt.tight_layout()
    plt.show()
