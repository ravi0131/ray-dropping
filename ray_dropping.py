import pandas as pd
import numpy as np
from typing import Tuple
def _random_beam_drop(points: np.ndarray) -> np.ndarray:
    """
    Randomly drop beams from the point cloud data
    
    Args:
        points (nx4 np.ndarray): 3D points in Cartesian coordinates (x, y, z) along with the beam index
    
    Returns:
        nx4 np.ndarray: Points after dropping beams
    
    Example:
        >>> points = np.array([[1, 1, 1, 0], [0, 0, 0, 1], [3, 4, 5, 2]])
        >>> points_after_beam_drop = random_beam_drop(points)
    """
    # Randomly select a beam drop ratio from [1, 2, 3]
    beam_drop_ratio = np.random.choice([1, 2, 3])
    # Randomly select a starting beam index
    start_index = np.random.randint(0, beam_drop_ratio)

    # Apply the ray-dropping condition
    mask = (points[:, 3] - start_index) % beam_drop_ratio == 0
    return points[mask]



def _spherical_coordinates_conversion(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """"
    Converts 3D points from Euclidean (x, y, z) to spherical coordinates (theta, phi, radial_dist).

    Args:
        points (np.ndarray): A numpy array of shape (n, 3) or (n, 4), where each row represents a 3D point.
                             If there are 4 columns, the last column will be ignored.

    Returns:
        np.ndarray: A (m, 3) array of spherical coordinates (theta, phi, radial_dist) for valid points.
        np.ndarray: A boolean mask (n,) indicating which input points have radial distance > 0.1.

    Example:
        >>> points = np.array([[1, 1, 1], [0, 0, 0], [3, 4, 5]])
        >>> spherical_coords, valid_mask = spherical_coordinates_conversion(points)
    """
    # Convert to spherical coordinates
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    radial_dist = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)  # azimuth
    phi = np.arcsin(z / radial_dist)  # elevation

    # Filter out points with radial distance < 0.1
    valid_mask = radial_dist > 0.1

    return np.vstack((theta[valid_mask], phi[valid_mask], radial_dist[valid_mask])).T, valid_mask

def _random_spherical_drop(points: np.ndarray) -> np.ndarray:
    """
    Randomly drop rays in the spherical coordinates
    Args:
        points (nx4 np.ndarray): 3D points in Cartesian coordinates (x, y, z) along with the beam index
    
    Returns:
        nx4 np.ndarray: Points after dropping rays

    Example:
        >>> points = np.array([[1, 1, 1, 0], [0, 0, 0, 1], [3, 4, 5, 2]])
        >>> points_after_spherical_drop = random_spherical_drop(points)
    """
    # Sample spherical resolutions for theta and phi
    spherical_resolutions = np.random.choice([600, 900, 1200, 1500])
    
    # Convert theta and phi to grid cells
    spherical_coords, valid_mask = _spherical_coordinates_conversion(points)
    theta_grid = (spherical_coords[:, 0] * spherical_resolutions).astype(int)
    phi_grid = (spherical_coords[:, 1] * spherical_resolutions).astype(int)
    
    # Randomly sample spherical drop ratio
    spherical_drop_ratio = np.random.choice([1, 2])
    # Apply the ray-dropping condition in the spherical coordinates
    theta_mask = (theta_grid % spherical_drop_ratio == 0)
    phi_mask = (phi_grid % spherical_drop_ratio == 0)
    
    # Combine the valid_mask from earlier with spherical mask
    combined_mask = valid_mask & theta_mask & phi_mask
    return points[combined_mask]


def drop_rays(points: np.ndarray) -> np.ndarray:
    """
    Drop rays from the point cloud data
    Args:
        points (nx4 np.ndarray): 3D points in Cartesian coordinates (x, y, z) along with the beam index
    
    Returns:
        nx4 np.ndarray: Points after dropping rays
    
    Example:
        >>> points = np.array([[1, 1, 1, 0], [0, 0, 0, 1], [3, 4, 5, 2]])
        >>> points_after_drop = drop_rays(points)
    """
    # Step 1: Random beam drop
    points_after_beam_drop = _random_beam_drop(points)
    
    # Step 2: Spherical drop
    points_after_spherical_drop = _random_spherical_drop(points_after_beam_drop)
    
    return points_after_spherical_drop

from typing import Tuple
import numpy as np
def filter_points_in_ROI(points: np.ndarray, beam_ids: np.ndarray, x_range: Tuple[int, int], y_range: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter out points and corresponding beam IDs that are not in the region of interest (ROI).
    
    Parameters:
        points: np.ndarray
        beam_ids: np.ndarray
        x_range: Tuple[int, int]  # Range of x values (longitudinal axis)
        y_range: Tuple[int, int]  # Range of y values (lateral axis)
        
    Returns:
        Tuple of filtered points and corresponding beam IDs in the region of interest.
    """
    # Define the ROI boundaries
    x_min, x_max = x_range[0], x_range[1]   # Longitudinal (x-axis)
    y_min, y_max = y_range[0], y_range[1]   # Lateral (y-axis)

    # Apply the conditions to filter the points within the ROI
    in_roi = (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & \
             (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
        
    # Use the conditions to index the original points and beam IDs arrays
    filtered_points = points[in_roi]
    filtered_beam_ids = beam_ids[in_roi]

    return filtered_points, filtered_beam_ids