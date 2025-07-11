import cv2
import numpy as np
import os
import time
from scipy.stats import gaussian_kde

# Constants for bandwidth in KDE estimation
BW_MEDIUM = 1
BW_NARROW = 0.2

# Height thresholds for different body parts in the frame
LEGS_HEIGHT = 805
SHOES_HEIGHT = 870
SHOULDERS_HEIGHT = 405

# Threshold for blue color masking
BLUE_MASK_THR = 130

# Window dimensions for processing regions of interest
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 1000
FACE_WINDOW_HEIGHT = 250
FACE_WINDOW_WIDTH = 300


def check_in_dict(cache_dict, element, function):
    """Check if an element exists in a dictionary; if not, compute and store it using the provided function."""
    if element in cache_dict:
        return cache_dict[element]
    else:
        cache_dict[element] = function(np.asarray(element))[0]
        return cache_dict[element]


def get_video_files(video_path):
    """Open a video file and retrieve its properties like width, height, and FPS."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, width, height, fps


def write_video(output_path, frames, fps, out_size, is_color):
    """Write a sequence of frames to a video file."""
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_out = cv2.VideoWriter(output_path, fourcc, fps, out_size, isColor=is_color)
    for frame in frames:
        video_out.write(frame)
    video_out.release()


def scale_matrix_0_to_255(input_matrix):
    """Scale a matrix to the range 0-255 for visualization or saving as video."""
    if input_matrix.dtype == np.bool:
        input_matrix = np.uint8(input_matrix)
    input_matrix = input_matrix.astype(np.uint8)
    scaled = 255 * (input_matrix - np.min(input_matrix)) / np.ptp(input_matrix)
    return np.uint8(scaled)


def load_entire_video(cap, color_space='bgr'):
    """Load all frames of a video into memory in the specified color space."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(n_frames):
        success, curr = cap.read()
        if not success:
            break
        if color_space == 'bgr':
            frames.append(curr)
        elif color_space == 'yuv':
            frames.append(cv2.cvtColor(curr, cv2.COLOR_BGR2YUV))
        elif color_space == 'bw':
            frames.append(cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY))
        else:
            frames.append(cv2.cvtColor(curr, cv2.COLOR_BGR2HSV))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return np.asarray(frames)


def apply_mask_on_color_frame(frame, mask):
    """Apply a binary mask to a color frame, setting non-mask areas to black."""
    frame_after_mask = np.copy(frame)
    frame_after_mask[:, :, 0] = frame_after_mask[:, :, 0] * mask
    frame_after_mask[:, :, 1] = frame_after_mask[:, :, 1] * mask
    frame_after_mask[:, :, 2] = frame_after_mask[:, :, 2] * mask
    return frame_after_mask


def choose_indices_for_foreground(mask, number_of_choices):
    """Randomly select indices from the foreground (mask=1) area."""
    indices = np.where(mask == 1)
    if len(indices[0]) == 0:
        return np.column_stack((indices[0], indices[1]))
    indices_choices = np.random.choice(len(indices[0]), number_of_choices)
    return np.column_stack((indices[0][indices_choices], indices[1][indices_choices]))


def choose_indices_for_background(mask, number_of_choices):
    """Randomly select indices from the background (mask=0) area."""
    indices = np.where(mask == 0)
    if len(indices[0]) == 0:
        return np.column_stack((indices[0], indices[1]))
    indices_choices = np.random.choice(len(indices[0]), number_of_choices)
    return np.column_stack((indices[0][indices_choices], indices[1][indices_choices]))


def new_estimate_pdf(omega_values, bw_method):
    """Create a Kernel Density Estimation (KDE) function for the given data."""
    pdf = gaussian_kde(omega_values.T, bw_method=bw_method)
    return lambda x: pdf(x.T)


def disk_kernel(size):
    """Create a disk-shaped kernel for morphological operations."""
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def initialize_knn_background_subtraction(frames_hsv, n_frames, height, width):
    """Apply KNN background subtraction to generate initial foreground masks."""
    backSub = cv2.createBackgroundSubtractorKNN()
    mask_list = np.zeros((n_frames, height, width)).astype(np.uint8)
    print(f"[BG_SUB | {time.strftime('%H:%M:%S')}] Starting BackgroundSubtractorKNN process...")
    for j in range(8):  # Multiple passes to refine the model
        for index_frame, frame in enumerate(frames_hsv):
            frame_sv = frame[:, :, 1:]  # Use saturation and value channels
            fgMask = backSub.apply(frame_sv)
            fgMask = (fgMask > 200).astype(np.uint8)
            mask_list[index_frame] = fgMask
    print(f"[BG_SUB | {time.strftime('%H:%M:%S')}] BackgroundSubtractorKNN process completed.")
    return mask_list


def collect_body_and_shoes_colors(frames_bgr, mask_list, n_frames, height, width):
    """Collect color samples for body and shoes from foreground and background."""
    body_foreground_colors = None
    body_background_colors = None
    shoes_foreground_colors = None
    shoes_background_colors = None
    person_and_blue_mask_list = np.zeros((n_frames, height, width))
    
    print(f"[BG_SUB | {time.strftime('%H:%M:%S')}] Starting color collection for body and shoes KDEs...")
    for frame_index, frame in enumerate(frames_bgr):
        blue_channel, _, _ = cv2.split(frame)
        frame_mask = mask_list[frame_index].astype(np.uint8)
        frame_mask = cv2.morphologyEx(frame_mask, cv2.MORPH_CLOSE, disk_kernel(6))
        frame_mask = cv2.medianBlur(frame_mask, 7)
        contours, _ = cv2.findContours(frame_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            person_mask = np.zeros(frame_mask.shape)
            cv2.fillPoly(person_mask, pts=[contours[0]], color=1)
        else:
            person_mask = np.zeros(frame_mask.shape)
        blue_mask = (blue_channel < BLUE_MASK_THR).astype(np.uint8)
        person_and_blue_mask = (person_mask * blue_mask).astype(np.uint8)
        
        # Collect indices for body color sampling
        body_foreground_indices = choose_indices_for_foreground(person_and_blue_mask, 20)
        body_background_indices = choose_indices_for_background(person_and_blue_mask, 20)
        
        # Collect indices for shoes color sampling
        shoes_mask = np.copy(person_and_blue_mask)
        shoes_mask[:SHOES_HEIGHT, :] = 0
        shoes_foreground_indices = choose_indices_for_foreground(shoes_mask, 100)
        shoes_mask = np.copy(person_and_blue_mask)
        shoes_mask[:SHOES_HEIGHT - 120, :] = 1
        shoes_background_indices = choose_indices_for_background(shoes_mask, 100)
        
        person_and_blue_mask_list[frame_index] = person_and_blue_mask
        
        # Accumulate color samples for body
        if body_foreground_colors is None:
            body_foreground_colors = frame[body_foreground_indices[:, 0], body_foreground_indices[:, 1], :]
            body_background_colors = frame[body_background_indices[:, 0], body_background_indices[:, 1], :]
            shoes_foreground_colors = frame[shoes_foreground_indices[:, 0], shoes_foreground_indices[:, 1], :]
            shoes_background_colors = frame[shoes_background_indices[:, 0], shoes_background_indices[:, 1], :]
        else:
            body_foreground_colors = np.concatenate((body_foreground_colors, frame[body_foreground_indices[:, 0], body_foreground_indices[:, 1], :]))
            body_background_colors = np.concatenate((body_background_colors, frame[body_background_indices[:, 0], body_background_indices[:, 1], :]))
            shoes_foreground_colors = np.concatenate((shoes_foreground_colors, frame[shoes_foreground_indices[:, 0], shoes_foreground_indices[:, 1], :]))
            shoes_background_colors = np.concatenate((shoes_background_colors, frame[shoes_background_indices[:, 0], shoes_background_indices[:, 1], :]))
    
    print(f"[BG_SUB | {time.strftime('%H:%M:%S')}] Color collection for body and shoes KDEs completed.")
    return body_foreground_colors, body_background_colors, shoes_foreground_colors, shoes_background_colors, person_and_blue_mask_list


def apply_kde_filtering_for_body_and_shoes(frames_bgr, person_and_blue_mask_list, body_foreground_pdf, body_background_pdf, shoes_foreground_pdf, shoes_background_pdf, n_frames, height, width):
    """Apply KDE-based filtering to refine masks for body and shoes."""
    body_foreground_pdf_cache = dict()
    body_background_pdf_cache = dict()
    shoes_foreground_pdf_cache = dict()
    shoes_background_pdf_cache = dict()
    combined_mask_list = np.zeros((n_frames, height, width))
    
    print(f"[BG_SUB | {time.strftime('%H:%M:%S')}] Starting KDE filtering for body and shoes...")
    for frame_index, frame in enumerate(frames_bgr):
        person_and_blue_mask = person_and_blue_mask_list[frame_index]
        mask_indices = np.where(person_and_blue_mask == 1)
        y_center, x_center = int(np.mean(mask_indices[0])), int(np.mean(mask_indices[1]))
        
        # Define a window around the person for processing
        window_frame_bgr = frame[max(0, y_center - WINDOW_HEIGHT // 2):min(height, y_center + WINDOW_HEIGHT // 2),
                                 max(0, x_center - WINDOW_WIDTH // 2):min(width, x_center + WINDOW_WIDTH // 2), :]
        window_mask = person_and_blue_mask[max(0, y_center - WINDOW_HEIGHT // 2):min(height, y_center + WINDOW_HEIGHT // 2),
                                           max(0, x_center - WINDOW_WIDTH // 2):min(width, x_center + WINDOW_WIDTH // 2)]
        
        window_mask_indices = np.where(window_mask == 1)
        window_foreground_mask = np.zeros(window_mask.shape)
        
        # Compute probabilities for body foreground vs background
        fg_probs = np.fromiter(map(lambda elem: check_in_dict(body_foreground_pdf_cache, elem, body_foreground_pdf),
                                    map(tuple, window_frame_bgr[window_mask_indices])), dtype=float)
        bg_probs = np.fromiter(map(lambda elem: check_in_dict(body_background_pdf_cache, elem, body_background_pdf),
                                    map(tuple, window_frame_bgr[window_mask_indices])), dtype=float)
        window_foreground_mask[window_mask_indices] = (fg_probs > bg_probs).astype(np.uint8)
        
        # Handle shoes restoration separately
        shoes_area_mask = np.copy(window_foreground_mask)
        shoes_area_mask[:-270, :] = 1  # Keep upper part as is
        shoes_area_indices = np.where(shoes_area_mask == 0)
        
        shoes_fg_probs = np.fromiter(map(lambda elem: check_in_dict(shoes_foreground_pdf_cache, elem, shoes_foreground_pdf),
                                          map(tuple, window_frame_bgr[shoes_area_indices])), dtype=float)
        shoes_bg_probs = np.fromiter(map(lambda elem: check_in_dict(shoes_background_pdf_cache, elem, shoes_background_pdf),
                                          map(tuple, window_frame_bgr[shoes_area_indices])), dtype=float)
        
        shoes_ratio = shoes_fg_probs / (shoes_fg_probs + shoes_bg_probs)
        shoes_foreground_mask = np.zeros(window_mask.shape)
        shoes_foreground_mask[shoes_area_indices] = (shoes_ratio > 0.7).astype(np.uint8)
        shoes_foreground_mask = cv2.morphologyEx(shoes_foreground_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        
        shoes_indices = np.where(shoes_foreground_mask == 1)
        y_shoes_center, x_shoes_center = int(np.mean(shoes_indices[0])), int(np.mean(shoes_indices[1]))
        
        # Combine body and shoes masks
        combined_window_mask = np.zeros(window_foreground_mask.shape)
        combined_window_mask[:y_shoes_center, :] = window_foreground_mask[:y_shoes_center, :]
        combined_window_mask[y_shoes_center:, :] = np.maximum(window_foreground_mask[y_shoes_center:, :],
                                                              shoes_foreground_mask[y_shoes_center:, :]).astype(np.uint8)
        
        DELTA_Y = 30
        combined_window_mask[y_shoes_center - DELTA_Y:, :] = cv2.morphologyEx(combined_window_mask[y_shoes_center - DELTA_Y:, :],
                                                                              cv2.MORPH_CLOSE, np.ones((1, 20)))
        combined_window_mask[y_shoes_center - DELTA_Y:, :] = cv2.morphologyEx(combined_window_mask[y_shoes_center - DELTA_Y:, :],
                                                                              cv2.MORPH_CLOSE, disk_kernel(20))
        
        combined_mask = np.zeros(person_and_blue_mask.shape)
        combined_mask[max(0, y_center - WINDOW_HEIGHT // 2):min(height, y_center + WINDOW_HEIGHT // 2),
                      max(0, x_center - WINDOW_WIDTH // 2):min(width, x_center + WINDOW_WIDTH // 2)] = combined_window_mask
        combined_mask_list[frame_index] = combined_mask
    
    return combined_mask_list


def collect_face_colors(frames_bgr, combined_mask_list, n_frames):
    """Collect color samples for face region to build face-specific KDEs."""
    face_foreground_colors = None
    face_background_colors = None
    
    print(f"[BG_SUB | {time.strftime('%H:%M:%S')}] Starting color collection for face KDE...")
    for frame_index, frame in enumerate(frames_bgr):
        current_mask = combined_mask_list[frame_index]
        face_mask = np.copy(current_mask)
        face_mask[SHOULDERS_HEIGHT:, :] = 0  # Focus on upper body for face
        face_mask_indices = np.where(face_mask == 1)
        y_center, x_center = int(np.mean(face_mask_indices[0])), int(np.mean(face_mask_indices[1]))
        
        window_frame_bgr = frame[max(0, y_center - FACE_WINDOW_HEIGHT // 2):min(frame.shape[0], y_center + FACE_WINDOW_HEIGHT // 2),
                                 max(0, x_center - FACE_WINDOW_WIDTH // 2):min(frame.shape[1], x_center + FACE_WINDOW_WIDTH // 2), :]
        face_mask_window = face_mask[max(0, y_center - FACE_WINDOW_HEIGHT // 2):min(frame.shape[0], y_center + FACE_WINDOW_HEIGHT // 2),
                                     max(0, x_center - FACE_WINDOW_WIDTH // 2):min(frame.shape[1], x_center + FACE_WINDOW_WIDTH // 2)]
        
        face_mask_window = cv2.morphologyEx(face_mask_window, cv2.MORPH_OPEN, np.ones((20, 1), np.uint8))
        face_mask_window = cv2.morphologyEx(face_mask_window, cv2.MORPH_OPEN, np.ones((1, 20), np.uint8))
        
        face_foreground_indices = choose_indices_for_foreground(face_mask_window, 20)
        face_background_indices = choose_indices_for_background(face_mask_window, 20)
        
        if face_foreground_colors is None:
            face_foreground_colors = window_frame_bgr[face_foreground_indices[:, 0], face_foreground_indices[:, 1], :]
            face_background_colors = window_frame_bgr[face_background_indices[:, 0], face_background_indices[:, 1], :]
        else:
            face_foreground_colors = np.concatenate((face_foreground_colors, window_frame_bgr[face_foreground_indices[:, 0], face_foreground_indices[:, 1], :]))
            face_background_colors = np.concatenate((face_background_colors, window_frame_bgr[face_background_indices[:, 0], face_background_indices[:, 1], :]))
    
    print(f"[BG_SUB | {time.strftime('%H:%M:%S')}] Color collection for face KDE completed.")
    return face_foreground_colors, face_background_colors


def apply_face_kde_and_finalize_masks(frames_bgr, combined_mask_list, face_foreground_pdf, face_background_pdf, n_frames, height, width):
    """Apply face-specific KDE filtering and finalize the foreground masks, ensuring the upper part is preserved."""
    face_foreground_pdf_cache = dict()
    face_background_pdf_cache = dict()
    final_masks_list = []
    final_frames_list = []
    
    print(f"[BG_SUB | {time.strftime('%H:%M:%S')}] Starting final processing with face KDE...")
    for frame_index, frame in enumerate(frames_bgr):
        current_mask = combined_mask_list[frame_index]
        face_mask = np.copy(current_mask)
        face_mask[SHOULDERS_HEIGHT:, :] = 0
        face_mask_indices = np.where(face_mask == 1)
        if len(face_mask_indices[0]) > 0:
            y_center, x_center = int(np.mean(face_mask_indices[0])), int(np.mean(face_mask_indices[1]))
        else:
            y_center, x_center = height // 2, width // 2  # Fallback if no mask pixels are found
        
        window_frame_bgr = frame[max(0, y_center - FACE_WINDOW_HEIGHT // 2):min(height, y_center + FACE_WINDOW_HEIGHT // 2),
                                 max(0, x_center - FACE_WINDOW_WIDTH // 2):min(width, x_center + FACE_WINDOW_WIDTH // 2), :]
        face_mask_window = face_mask[max(0, y_center - FACE_WINDOW_HEIGHT // 2):min(height, y_center + FACE_WINDOW_HEIGHT // 2),
                                     max(0, x_center - FACE_WINDOW_WIDTH // 2):min(width, x_center + FACE_WINDOW_WIDTH // 2)]
        
        window_frame_bgr_stacked = window_frame_bgr.reshape((-1, 3))
        fg_face_probs = np.fromiter(map(lambda elem: check_in_dict(face_foreground_pdf_cache, elem, face_foreground_pdf),
                                         map(tuple, window_frame_bgr_stacked)), dtype=float)
        bg_face_probs = np.fromiter(map(lambda elem: check_in_dict(face_background_pdf_cache, elem, face_background_pdf),
                                         map(tuple, window_frame_bgr_stacked)), dtype=float)
        
        fg_face_probs_reshaped = fg_face_probs.reshape(face_mask_window.shape)
        bg_face_probs_reshaped = bg_face_probs.reshape(face_mask_window.shape)
        face_foreground_mask = (fg_face_probs_reshaped > bg_face_probs_reshaped).astype(np.uint8)
        
        laplacian = cv2.Laplacian(face_foreground_mask, cv2.CV_32F)
        laplacian = np.abs(laplacian)
        face_foreground_mask = np.maximum(face_foreground_mask - laplacian, 0)
        face_foreground_mask[np.where(face_foreground_mask > 1)] = 0
        face_foreground_mask = face_foreground_mask.astype(np.uint8)
        
        contours, _ = cv2.findContours(face_foreground_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            contour_mask = np.zeros(face_foreground_mask.shape, dtype=np.uint8)
            cv2.fillPoly(contour_mask, pts=[contours[0]], color=1)
        else:
            contour_mask = np.zeros(face_foreground_mask.shape, dtype=np.uint8)
        
        contour_mask = cv2.morphologyEx(contour_mask, cv2.MORPH_CLOSE, disk_kernel(12))
        contour_mask = cv2.dilate(contour_mask, disk_kernel(3), iterations=1).astype(np.uint8)
        contour_mask[-50:, :] = face_mask_window[-50:, :]
        
        # Combine the face contour mask with the original mask by directly overwriting the windowed region
        final_mask = np.copy(current_mask).astype(np.uint8)
        window_y_start = max(0, y_center - FACE_WINDOW_HEIGHT // 2)
        window_y_end = min(height, y_center + FACE_WINDOW_HEIGHT // 2)
        window_x_start = max(0, x_center - FACE_WINDOW_WIDTH // 2)
        window_x_end = min(width, x_center + FACE_WINDOW_WIDTH // 2)
        # Directly overwrite the windowed region with the face contour mask to ensure completeness in upper part
        window_region = final_mask[window_y_start:window_y_end, window_x_start:window_x_end]
        contour_region = contour_mask
        final_mask[window_y_start:window_y_end, window_x_start:window_x_end] = np.where(
            contour_region == 1, 1, window_region
        ).astype(np.uint8)
        
        # Apply morphological operations to clean up noise in the relevant region
        morph_region_y_start = max(0, y_center - FACE_WINDOW_HEIGHT // 2)
        morph_region_y_end = min(height, LEGS_HEIGHT)
        if morph_region_y_start < morph_region_y_end:
            final_mask[morph_region_y_start:morph_region_y_end, :] = cv2.morphologyEx(
                final_mask[morph_region_y_start:morph_region_y_end, :], cv2.MORPH_OPEN, np.ones((6, 1), np.uint8))
            final_mask[morph_region_y_start:morph_region_y_end, :] = cv2.morphologyEx(
                final_mask[morph_region_y_start:morph_region_y_end, :], cv2.MORPH_OPEN, np.ones((1, 6), np.uint8))
        
        # Select the largest contour to ensure only the main foreground object is kept, reducing background noise
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            final_contour_mask = np.zeros(final_mask.shape, dtype=np.uint8)
            cv2.fillPoly(final_contour_mask, pts=[contours[0]], color=1)
            final_mask = (final_contour_mask * final_mask).astype(np.uint8)
        else:
            final_mask = np.zeros(final_mask.shape, dtype=np.uint8)
        
        final_masks_list.append(scale_matrix_0_to_255(final_mask))
        final_frames_list.append(apply_mask_on_color_frame(frame=frame, mask=final_mask))
    
    return final_masks_list, final_frames_list


def background_subtraction(input_video_path, output_path):
    """Perform background subtraction on a video to separate foreground (person) from background."""
    # Step 1: Load video and initialize properties
    cap, width, height, fps = get_video_files(video_path=input_video_path)
    frames_bgr = load_entire_video(cap, color_space='bgr')
    frames_hsv = load_entire_video(cap, color_space='hsv')
    n_frames = len(frames_bgr)
    
    # Step 2: Initial background subtraction using KNN
    initial_masks = initialize_knn_background_subtraction(frames_hsv, n_frames, height, width)
    
    # Step 3: Collect color samples for body and shoes KDEs
    body_fg_colors, body_bg_colors, shoes_fg_colors, shoes_bg_colors, person_blue_masks = collect_body_and_shoes_colors(
        frames_bgr, initial_masks, n_frames, height, width)
    
    # Step 4: Build KDE models for body and shoes
    body_foreground_pdf = new_estimate_pdf(omega_values=body_fg_colors, bw_method=BW_MEDIUM)
    body_background_pdf = new_estimate_pdf(omega_values=body_bg_colors, bw_method=BW_MEDIUM)
    shoes_foreground_pdf = new_estimate_pdf(omega_values=shoes_fg_colors, bw_method=BW_MEDIUM)
    shoes_background_pdf = new_estimate_pdf(omega_values=shoes_bg_colors, bw_method=BW_MEDIUM)
    
    # Step 5: Apply KDE filtering for body and shoes
    combined_masks = apply_kde_filtering_for_body_and_shoes(
        frames_bgr, person_blue_masks, body_foreground_pdf, body_background_pdf, 
        shoes_foreground_pdf, shoes_background_pdf, n_frames, height, width)
    
    # Step 6: Collect color samples for face KDE
    face_fg_colors, face_bg_colors = collect_face_colors(frames_bgr, combined_masks, n_frames)
    
    # Step 7: Build KDE models for face
    face_foreground_pdf = new_estimate_pdf(omega_values=face_fg_colors, bw_method=BW_NARROW)
    face_background_pdf = new_estimate_pdf(omega_values=face_bg_colors, bw_method=BW_NARROW)
    
    # Step 8: Finalize masks with face KDE and prepare output
    final_masks, final_frames = apply_face_kde_and_finalize_masks(
        frames_bgr, combined_masks, face_foreground_pdf, face_background_pdf, n_frames, height, width)
    
    # Step 9: Save the results as videos
    extracted_output_path = os.path.join("Outputs", "extracted_208484097_318931573.avi")
    binary_output_path = os.path.join("Outputs", "binary_208484097_318931573.avi")
    write_video(extracted_output_path, frames=final_frames, fps=fps, out_size=(width, height), is_color=True)
    write_video(binary_output_path, frames=final_masks, fps=fps, out_size=(width, height), is_color=False)
    print(f"[BG_SUB | {time.strftime('%H:%M:%S')}] Output videos saved: extracted and binary masks.")
    
    cap.release()
    cv2.destroyAllWindows()