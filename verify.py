import cv2
print(cv2.__version__)  # Should be ≥ 4.7.0
has_aruco = hasattr(cv2, 'aruco')
print(f"ArUco available: {has_aruco}")