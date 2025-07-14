import time
import os
import cv2
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

def load_previous_outputs(ids, input_dir, output_dir):
    """
    Load the outputs from previous stages
    """
    id1, id2 = ids
    
    # Define paths
    extracted_path = os.path.join(output_dir, f"extracted_{id1}_{id2}.avi")
    binary_path = os.path.join(output_dir, f"binary_{id1}_{id2}.avi")
    background_path =os.path.join(input_dir, "background.jpg")
    
    # Load background image
    background = cv2.imread(background_path)
    if background is None:
        raise FileNotFoundError(f"Background image not found: {background_path}")
    
    # Open video files
    extracted_cap = cv2.VideoCapture(extracted_path)
    binary_cap = cv2.VideoCapture(binary_path)
    
    if not extracted_cap.isOpened():
        raise FileNotFoundError(f"Cannot open extracted video: {extracted_path}")
    if not binary_cap.isOpened():
        raise FileNotFoundError(f"Cannot open binary video: {binary_path}")
    
    # Get video properties
    fps = int(extracted_cap.get(cv2.CAP_PROP_FPS))
    width = int(extracted_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(extracted_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(extracted_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    return extracted_cap, binary_cap, background, (width, height, fps, total_frames)


def read_frame_pair(extracted_cap, binary_cap):
    """
    Read corresponding frames from both videos
    """
    ret1, extracted_frame = extracted_cap.read()
    ret2, binary_frame = binary_cap.read()
    
    if not (ret1 and ret2):
        return None, None, False
    
    # Convert binary frame to single channel (if it's 3-channel)
    if len(binary_frame.shape) == 3:
        binary_frame = cv2.cvtColor(binary_frame, cv2.COLOR_BGR2GRAY)
    
    # Normalize binary frame to 0-1
    binary_mask = (binary_frame > 128).astype(np.float32)
    
    return extracted_frame, binary_mask, True


def create_trimap(binary_mask, erode_size=3, dilate_size=10):
    """
    Creates trimap: 0=background, 128=unknown, 255=foreground
    """
    # Convert to uint8 for morphological operations
    mask_uint8 = (binary_mask * 255).astype(np.uint8)
    
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
    
    # Definite foreground (eroded mask)
    fg_sure = cv2.erode(mask_uint8, kernel_erode, iterations=1)
    
    # Definite background (dilated inverted mask)
    bg_sure = cv2.dilate(mask_uint8, kernel_dilate, iterations=1)
    bg_sure = 255 - bg_sure
    
    # Unknown region
    trimap = np.full_like(mask_uint8, 128, dtype=np.uint8)
    trimap[fg_sure == 255] = 255  # Foreground
    trimap[bg_sure == 255] = 0    # Background
    
    return trimap

def build_matting_laplacian(img, win_size=1):
    """
    Build the matting Laplacian matrix
    """
    h, w, c = img.shape
    n_pixels = h * w
    
    # Window radius
    win_rad = win_size
    win_area = (2 * win_rad + 1) ** 2
    
    # Number of windows
    n_windows = (h - 2 * win_rad) * (w - 2 * win_rad)
    
    # Pre-allocate for sparse matrix construction
    row_inds = np.zeros((n_windows * win_area * win_area,), dtype=np.int32)
    col_inds = np.zeros((n_windows * win_area * win_area,), dtype=np.int32)
    vals = np.zeros((n_windows * win_area * win_area,), dtype=np.float64)
    
    print(f"Processing {n_windows} windows...")
    
    counter = 0
    for y in range(win_rad, h - win_rad):
        if y % 50 == 0:
            print(f"Processing row {y}/{h}")
            
        for x in range(win_rad, w - win_rad):
            # Extract window
            win_img = img[y-win_rad:y+win_rad+1, x-win_rad:x+win_rad+1, :]
            win_img = win_img.reshape(win_area, c)
            
            # Get pixel indices in the window
            win_indices = []
            for wy in range(y-win_rad, y+win_rad+1):
                for wx in range(x-win_rad, x+win_rad+1):
                    win_indices.append(wy * w + wx)
            win_indices = np.array(win_indices)
            
            # Compute covariance matrix
            win_mu = np.mean(win_img, axis=0)
            win_var = np.cov(win_img.T) + np.eye(c) * 1e-6
            
            # Inverse covariance
            win_var_inv = np.linalg.inv(win_var)
            
            # Compute matting weights
            for i in range(win_area):
                for j in range(win_area):
                    if i == j:
                        val = 1.0
                    else:
                        val = 0.0
                    
                    val -= (1.0 / win_area) * (1.0 + 
                           (win_img[i] - win_mu).T @ win_var_inv @ (win_img[j] - win_mu))
                    
                    row_inds[counter] = win_indices[i]
                    col_inds[counter] = win_indices[j]
                    vals[counter] = val
                    counter += 1
    
    # Build sparse matrix
    L = sp.csr_matrix((vals[:counter], (row_inds[:counter], col_inds[:counter])), 
                      shape=(n_pixels, n_pixels))
    
    return L


def optimized_closed_form_matting(image, trimap, lambda_val=100):
    """
    More efficient closed-form matting implementation
    Uses smaller windows and optimized operations
    """
    h, w, c = image.shape
    img = image.astype(np.float64) / 255.0
    
    # Convert trimap to alpha constraints
    alpha = np.zeros((h, w), dtype=np.float64)
    alpha[trimap == 255] = 1.0
    alpha[trimap == 0] = 0.0
    
    unknown_mask = (trimap == 128)
    if not np.any(unknown_mask):
        return alpha
    
    # Use guided filter approach for efficiency
    epsilon = 1e-6
    radius = 1  # Small radius for efficiency
    
    # Guided filter implementation
    mean_I = cv2.boxFilter(img, cv2.CV_64F, (2*radius+1, 2*radius+1))
    mean_alpha = cv2.boxFilter(alpha, cv2.CV_64F, (2*radius+1, 2*radius+1))
    
    # Correlation and covariance
    corr_Ia = cv2.boxFilter(img * alpha[:,:,np.newaxis], cv2.CV_64F, (2*radius+1, 2*radius+1))
    cov_Ia = corr_Ia - mean_I * mean_alpha[:,:,np.newaxis]
    
    # Variance of I in each channel
    mean_II = cv2.boxFilter(img * img, cv2.CV_64F, (2*radius+1, 2*radius+1))
    var_I = mean_II - mean_I * mean_I
    
    # Coefficients
    a = cov_Ia / (var_I + epsilon)
    b = mean_alpha[:,:,np.newaxis] - a * mean_I
    
    # Average coefficients
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (2*radius+1, 2*radius+1))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (2*radius+1, 2*radius+1))
    
    # Output
    result = np.sum(mean_a * img, axis=2) + np.sum(mean_b, axis=2)
    
    # Apply constraints for known pixels
    result[trimap == 255] = 1.0
    result[trimap == 0] = 0.0
    
    return np.clip(result, 0, 1)


def perform_matting_closed_form(extracted_frame, binary_mask, background):
    """
    Complete matting pipeline using closed-form matting
    """
    # Resize background to match frame size
    h, w = extracted_frame.shape[:2]
    background_resized = cv2.resize(background, (w, h))
    
    # Create trimap with larger unknown region for better matting
    trimap = create_trimap(binary_mask, erode_size=5, dilate_size=15)
    
    # Generate alpha matte using closed-form matting
    alpha = optimized_closed_form_matting(extracted_frame, trimap, lambda_val=100)
    
    
    # Additional smoothing
    alpha = cv2.bilateralFilter(alpha.astype(np.float32), 5, 0.1, 5)
    alpha = np.clip(alpha, 0, 1)
    
    # Create 3-channel alpha
    alpha_3ch = np.stack([alpha, alpha, alpha], axis=2)
    
    # Extract person pixels (non-zero regions from extracted video)
    person_mask = np.any(extracted_frame > 0, axis=2, keepdims=True).astype(np.float32)
    person_pixels = extracted_frame.astype(np.float32) * person_mask
    
    # Final composition
    matted_frame = (person_pixels * alpha_3ch + 
                   background_resized.astype(np.float32) * (1 - alpha_3ch))
    
    return matted_frame.astype(np.uint8), alpha


# Update the main matting function
def run_matting_stage_closed_form(ids, main_start_time, input_dir, output_dir):
    """
    Main function to run the matting stage with closed-form matting
    """
    print(f"[MATTING | {time.strftime('%H:%M:%S')}] Starting closed-form matting stage")
    id1, id2 = ids
    # Load previous outputs
    extracted_cap, binary_cap, background, (width, height, fps, total_frames) = load_previous_outputs(ids, input_dir, output_dir)
    time_alpha_created = 0
    time_matted_created = 0
    
    # Setup output videos
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    matted_path = os.path.join(output_dir, f"matted_{id1}_{id2}.avi")
    alpha_path = os.path.join(output_dir, f"alpha_{id1}_{id2}.avi")
    
    matted_writer = cv2.VideoWriter(matted_path, fourcc, fps, (width, height))
    alpha_writer = cv2.VideoWriter(alpha_path, fourcc, fps, (width, height))
    
    if not matted_writer.isOpened() or not alpha_writer.isOpened():
        raise RuntimeError("Failed to open video writers")
    
    frame_count = 0
    
    print(f"[MATTING | {time.strftime('%H:%M:%S')}] Processing {total_frames} frames")
    
    try:
        while True:
            # Read frame pair
            extracted_frame, binary_mask, success = read_frame_pair(extracted_cap, binary_cap)
            
            if not success:
                break
            
            # Perform closed-form matting
            matted_frame, alpha = perform_matting_closed_form(
                extracted_frame, binary_mask, background)
            
            # Convert alpha to uint8 for saving (scale from [0,1] to [0,255])
            alpha_uint8 = (alpha * 255).astype(np.uint8)
            alpha_3ch = cv2.cvtColor(alpha_uint8, cv2.COLOR_GRAY2BGR)
            
            # Write frames
            matted_writer.write(matted_frame)
            alpha_writer.write(alpha_3ch)
            
            frame_count += 1
    
    finally:
        # Cleanup and capture timing
        extracted_cap.release()
        binary_cap.release()

        # [MODIFICATION] Release writers and capture the precise time of file completion
        matted_writer.release()
        time_matted_created = time.time() - main_start_time

        alpha_writer.release()
        time_alpha_created = time.time() - main_start_time
    
    print(f"[MATTING | {time.strftime('%H:%M:%S')}] Closed-form matting completed, processed {frame_count} frames")
    print(f"[MATTING | {time.strftime('%H:%M:%S')}] Closed-form output saved: {matted_path}")
    print(f"[MATTING | {time.strftime('%H:%M:%S')}] Closed-form alpha saved: {alpha_path}")
    return {"time_to_alpha": time_alpha_created, "time_to_matted": time_matted_created}
