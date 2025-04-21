import cv2
import numpy as np

def generate_aruco_marker(marker_id, marker_size=500, save_path=None):
    """Generate an ArUco marker image"""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size)
    
    if save_path:
        cv2.imwrite(save_path, marker_img)
        print(f"DEBUG: Marker saved to {save_path}")
    
    return marker_img

def overlay_image_on_marker(marker_img, overlay_img_path, output_size=(800, 800)):
    """Overlay an image on the ArUco marker"""
    # Read the overlay image
    overlay_img = cv2.imread(overlay_img_path, cv2.IMREAD_UNCHANGED)
    
    if overlay_img is None:
        raise ValueError(f"Could not read overlay image at {overlay_img_path}")
    
    # Convert marker to 3 channels if grayscale
    if len(marker_img.shape) == 2:
        marker_img = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
    
    # Resize overlay to match marker
    overlay_img = cv2.resize(overlay_img, (marker_img.shape[1], marker_img.shape[0]))
    
    # If overlay has alpha channel, blend it
    if overlay_img.shape[2] == 4:
        alpha = overlay_img[:, :, 3] / 255.0
        for c in range(3):
            marker_img[:, :, c] = (1.0 - alpha) * marker_img[:, :, c] + alpha * overlay_img[:, :, c]
    else:
        marker_img = overlay_img
    
    # Resize final output
    return cv2.resize(marker_img, output_size)

def verify_overlay_image(overlay_img_path):
    """Debug function to verify overlay image loads correctly"""
    print("\nDEBUG: Verifying overlay image...")
    img = cv2.imread(overlay_img_path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        print(f"ERROR: Could not load image from {overlay_img_path}")
        return None
    
    print(f"DEBUG: Image loaded successfully. Dimensions: {img.shape}")
    print(f"DEBUG: Image type: {img.dtype}")
    
    if len(img.shape) == 3:
        if img.shape[2] == 4:
            print("DEBUG: Image has alpha channel (BGRA format)")
        else:
            print("DEBUG: Image has 3 channels (BGR format)")
    
    cv2.imshow("DEBUG: Overlay Image Preview", img)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    return img

def order_points(pts):
    """Arrange corner points in consistent order (TL, TR, BR, BL)"""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    
    return rect

def detect_and_overlay_in_realtime(overlay_img_path, marker_id):
    """Main detection function with enhanced debugging"""
    print("\nDEBUG: Initializing camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open video capture")
        return
    
    # Verify overlay image
    overlay_img = verify_overlay_image(overlay_img_path)
    if overlay_img is None:
        return
    
    # Convert to BGRA if needed
    if overlay_img.shape[2] == 3:
        overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2BGRA)
    
    # Initialize detector
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    detector = cv2.aruco.ArucoDetector(dictionary)
    
    print("\nDEBUG: Starting detection loop...")
    print("DEBUG: Show the marker to your webcam")
    print("DEBUG: Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("DEBUG: Could not read frame from camera")
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect markers
        corners, ids, rejected = detector.detectMarkers(gray)
        
        # Debug output
        if ids is not None:
            print(f"DEBUG: Detected markers - IDs: {ids.flatten()}")
            print(f"DEBUG: Corner shapes: {[c.shape for c in corners]}")
            
            # Draw all detected markers
            frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            if marker_id in ids:
                idx = np.where(ids == marker_id)[0][0]
                marker_corners = corners[idx][0]
                
                # Draw debug info
                for i, corner in enumerate(marker_corners):
                    cv2.circle(frame, tuple(corner.astype(int)), 10, (0, 255, 0), -1)
                    cv2.putText(frame, str(i), tuple(corner.astype(int)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                # Order corners and apply overlay
                marker_corners = order_points(marker_corners)
                h, w = overlay_img.shape[0], overlay_img.shape[1]
                pts_src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
                
                try:
                    M = cv2.getPerspectiveTransform(pts_src, marker_corners)
                    warped = cv2.warpPerspective(overlay_img, M, (frame.shape[1], frame.shape[0]))
                    mask = warped[:, :, 3] > 0
                    frame[mask] = warped[mask][:, :3]
                except Exception as e:
                    print(f"DEBUG: Overlay error: {str(e)}")
        else:
            print("DEBUG: No markers detected")
        
        # Display FPS for performance monitoring
        cv2.putText(frame, "Press 'q' to quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('AR Detection Debug View', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("DEBUG: Camera released")

def main():
    print("=== ARUCO MARKER DEBUG VERSION ===")
    print("ArUco Marker AR Demonstration")
    
    # Get user input
    marker_id = int(input("Enter ArUco Marker ID (0-249): "))
    overlay_path = input("Enter path to overlay image: ")
    
    # Generate marker
    print(f"\nGenerating ArUco marker with ID {marker_id}...")
    marker = generate_aruco_marker(marker_id, save_path=f"marker_{marker_id}.png")
    cv2.imshow("Generated Marker", marker)
    cv2.waitKey(2000)  # Display for 2 seconds
    cv2.destroyAllWindows()
    
    # Overlay image on marker
    print(f"\nOverlaying image on marker {marker_id}...")
    try:
        overlaid = overlay_image_on_marker(marker, overlay_path)
        cv2.imshow("Marker with Overlay", overlaid)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error overlaying image: {e}")
    
    # Real-time AR detection
    print("\nStarting real-time AR detection...")
    print("Press 'q' to quit the live view.")
    try:
        detect_and_overlay_in_realtime(overlay_path, marker_id)
    except Exception as e:
        print(f"Error in real-time detection: {e}")

if __name__ == "__main__":
    main()