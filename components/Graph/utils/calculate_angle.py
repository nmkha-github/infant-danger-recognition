import numpy as np


def calculate_angle(a, b, c):
    """
    Calculates the angle between three points.

    Args:
      a: The first point.
      b: The root point.
      c: The third point.

    Returns:
      The angle between the three points in degrees.
    """

    # Convert the points to NumPy arrays.
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # Calculate the arctangent of the slopes between the points.
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )

    # Convert the radians to degrees and take the absolute value.
    angle = np.abs(radians * 180.0 / np.pi)

    # If the angle is greater than 180 degrees, subtract it from 360 degrees.
    if angle > 180.0:
        angle = 360 - angle

    return angle
