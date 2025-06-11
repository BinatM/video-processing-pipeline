import cv2
import numpy as np
import os

def moving_average(curve, radius):
    """Apply moving average smoothing with better boundary handling"""
    window_size = 2 * radius + 1
    
    # If radius is too large for the curve length, reduce it
    if radius >= len(curve) // 2:
        radius = max(1, len(curve) // 4)
        window_size = 2 * radius + 1
    
    # Define the filter
    f = np.ones(window_size) / window_size
    # Add padding to the boundaries (updated for newer numpy)
    curve_pad = np.pad(curve, (radius, radius), 'edge')  
    
    # Apply convolution
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    # Remove padding
    curve_smoothed = curve_smoothed[radius:-radius]
    return curve_smoothed

def smooth_trajectory(trajectory, smooth_radius):
    """Smooth trajectory using moving average"""
    smoothed_trajectory = np.copy(trajectory)
    for i in range(smoothed_trajectory.shape[1]):
        smoothed_trajectory[:, i] = moving_average(trajectory[:, i], radius=smooth_radius)
    return smoothed_trajectory

def fix_border(frame):
    """Fix black borders by scaling image slightly"""
    h, w = frame.shape[:2]
    # Scale the image 4% without moving the center
    T = cv2.getRotationMatrix2D((w / 2, h / 2), 0, 1.04)
    frame = cv2.warpAffine(frame, T, (w, h))
    return frame

def load_entire_video(cap):
    """Load all frames into memory"""
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    
    print(f"Loading {n_frames} frames into memory...")
    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)    
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return frames

def stabilize_video_improved(input_path, output_path):
    """
    Improved video stabilization using trajectory smoothing
    Based on your friend's approach
    """
    
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Parameters 
    MAX_CORNERS = 500
    QUALITY_LEVEL = 0.01
    MIN_DISTANCE = 30
    BLOCK_SIZE = 3
    SMOOTH_RADIUS = 5
    
    # Load entire video into memory
    frames = load_entire_video(cap)
    n_frames = len(frames)
    
    # Initialize arrays for transformations
    transforms = np.zeros((n_frames - 1, 9), np.float32)  # Flattened 3x3 matrices
    
    print("Computing transformations between consecutive frames...")
    
    # Compute transformations between consecutive frames
    for i in range(n_frames - 1):
        print(f"Processing frame {i+1}/{n_frames-1}")
        
        # Get current and next frame
        prev_frame = frames[i]
        curr_frame = frames[i + 1]
        
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect features in previous frame
        prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                         maxCorners=MAX_CORNERS,
                                         qualityLevel=QUALITY_LEVEL,
                                         minDistance=MIN_DISTANCE,
                                         blockSize=BLOCK_SIZE)
        
        if prev_pts is not None and len(prev_pts) > 10:
            # Track features to current frame
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
            
            # Filter only valid points
            idx = np.where(status == 1)[0]
            if len(idx) > 8:  # Need at least 8 points for homography
                prev_pts_good = prev_pts[idx]
                curr_pts_good = curr_pts[idx]
                
                # Find homography transformation
                try:
                    transform_matrix, _ = cv2.findHomography(prev_pts_good, curr_pts_good, 
                                                           cv2.RANSAC, 5.0)
                    if transform_matrix is not None:
                        transforms[i] = transform_matrix.flatten()
                    else:
                        # If homography fails, use identity
                        transforms[i] = np.eye(3).flatten()
                except:
                    # If homography fails, use identity
                    transforms[i] = np.eye(3).flatten()
            else:
                # Not enough points, use identity
                transforms[i] = np.eye(3).flatten()
        else:
            # No features detected, use identity
            transforms[i] = np.eye(3).flatten()
    
    print("Computing trajectory and applying smoothing...")
    
    # Compute trajectory using cumulative sum of transformations
    trajectory = np.cumsum(transforms, axis=0)
    
    # Apply smoothing to trajectory
    smoothed_trajectory = smooth_trajectory(trajectory, SMOOTH_RADIUS)
    
    # Calculate difference and smooth transformations
    diff = smoothed_trajectory - trajectory
    transforms_smooth = transforms + diff
    
    print("Applying transformations to create stabilized video...")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write first frame (no transformation)
    stabilized_frame = fix_border(frames[0])
    out.write(stabilized_frame)
    
    # Apply smooth transformations to create stabilized frames
    for i in range(n_frames - 1):
        if (i + 1) % 30 == 0:
            print(f"Stabilizing frame {i+1}/{n_frames-1}")
        
        # Skip stabilization for last few frames to avoid end artifacts
        if i >= n_frames - 5:  # Last 5 frames use original
            frame_stabilized = fix_border(frames[i])
        else:
            # Reshape transformation matrix
            transform_matrix = transforms_smooth[i].reshape((3, 3))
            
            # Apply transformation
            frame_stabilized = cv2.warpPerspective(frames[i], transform_matrix, (width, height))
            
            # Fix borders
            frame_stabilized = fix_border(frame_stabilized)
        
        # Write frame
        out.write(frame_stabilized)
    
    # Cleanup
    cap.release()
    out.release()
    print(f"Improved stabilization complete! Output saved to: {output_path}")

def main():
    """Main function for improved video stabilization"""
    
    # File paths
    input_video = "../Inputs/INPUT.avi"
    output_video = "../Outputs/stabilize_208484097_318931573.avi"  
    
    # Create output directory if it doesn't exist
    os.makedirs("Outputs", exist_ok=True)
    
    # Check if input file exists
    if not os.path.exists(input_video):
        print(f"Input video not found at: {input_video}")
        return
    
    try:
        stabilize_video_improved(input_video, output_video)
    except Exception as e:
        print(f"Error during stabilization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
