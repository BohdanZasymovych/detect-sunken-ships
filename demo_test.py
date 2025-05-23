import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from detect_sunken_ships import detect_ships

# Map parameters
SIZE = 200
BASE_DEPTH = 200


def generate_synthetic_seabed(size, base_depth):
    """
    Generate a synthetic seabed with Gaussian noise.
    :param size: Size of the seabed (size x size).
    :param base_depth: Base depth of the seabed.
    :return: Synthetic seabed depth map.
    """
    noise = gaussian_filter(np.random.randn(size, size), sigma=10)
    seabed = base_depth + (noise - noise.min()) / (noise.max() - noise.min()) * 80
    return seabed


def add_sensor_noise(depth_map, noise_level=0.5):
    """
    Add sensor noise to the depth map.
    :param depth_map: Depth map to modify.
    :param noise_level: Standard deviation of the Gaussian noise.
    :return: Noisy depth map.
    """
    noise = np.random.normal(loc=0.0, scale=noise_level, size=depth_map.shape)
    noisy_map = depth_map + noise
    return noisy_map


def add_ship(depth_map, center, length, width, height):
    """
    Add a ship to the depth map at the specified center.
    :param depth_map: Depth map to modify.
    :param center: Center coordinates (x, y) of the ship.
    :param length: Length of the ship.
    :param width: Width of the ship.
    :param height: Height of the ship.
    """
    for i in range(-length//2, length//2):
        for j in range(-width//2, width//2):
            x, y = center[0] + i, center[1] + j
            if 0 <= x < depth_map.shape[0] and 0 <= y < depth_map.shape[1]:
                dx = i / (length / 2)
                dy = j / (width / 2)
                falloff = np.exp(-4 * (dx**2 + dy**2))
                depth_map[x, y] += -height*0.7 * falloff


def generate_syntetic_echolot_data(ship_length: float,
                                   ship_width: float,
                                   ship_height: float,
                                   ship_locations: list[tuple[int, int]],
                                   filename: str=None):
    """
    Generate synthetic echolot data with ships.
    :param ship_length: Length of the ship.
    :param ship_width: Width of the ship.
    :param ship_height: Height of the ship.
    :param ship_locations: List of tuples representing the center coordinates of the ships.
    :param filename: Optional filename with csv extention to save the depth map.
    :return: Synthetic depth map with ships.
    """
    depth_map = generate_synthetic_seabed(SIZE, BASE_DEPTH)
    for ship_center in ship_locations:
        add_ship(depth_map, ship_center, ship_length, ship_width, ship_height)

    depth_map_noisy = add_sensor_noise(depth_map, noise_level=0.5)

    if filename:
        np.savetxt(filename, depth_map_noisy, delimiter=',')

    return depth_map_noisy


def visualize_detection(depth_map, ship_mask, ship_centers):
    """
    Visualize the depth map and detected ship regions.
    :param depth_map: Depth map to visualize.
    :param ship_mask: Binary mask of detected ship regions.
    :param ship_centers: List of ship center coordinates.
    """
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    im0 = axs[0].imshow(depth_map, cmap='viridis')
    axs[0].set_title(f"Depth Map ({SIZE}x{SIZE}m)")
    axs[0].scatter([y for _, y in ship_centers],
                    [x for x, _ in ship_centers],
                    color='red',
                    label='True Ship Locations',
                    marker='x')
    axs[0].legend()

    cbar = fig.colorbar(im0, ax=axs[0])
    cbar.set_label('Depth (m)')

    axs[1].imshow(ship_mask, cmap='gray')
    axs[1].set_title("Detected Ship Regions")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Ship size (approximate size of the Island-class patrol boat)
    SHIP_LENGTH = 34
    SHIP_WIDTH = 7
    SHIP_HEIGHT = 16

    # Generate synthetic echolot data with ships
    true_ship_locations = [(120, 80)]
    syntetic_depth_map = generate_syntetic_echolot_data(SHIP_LENGTH, SHIP_WIDTH, SHIP_HEIGHT, true_ship_locations)

    # Detect ships and visualize results
    detected_ship_regions = detect_ships(syntetic_depth_map, SHIP_LENGTH, SHIP_WIDTH, SHIP_HEIGHT)
    visualize_detection(syntetic_depth_map, detected_ship_regions, true_ship_locations)
