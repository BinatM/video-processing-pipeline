import numpy as np
import cv2
import json
import os
import time

def get_video_files(path):
    """Get video properties"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    return cap, video_width, video_height, fps


def load_entire_video(cap, color_space='bgr'):
    """Load all video frames"""
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    return frames


def write_video(output_path, frames, fps, size, is_color=True):
    """Write video"""
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, size, is_color)
    
    for frame in frames:
        out.write(frame)
    
    out.release()


def automatic_roi_selection(frame):
    """
    Automatic ROI selection with coordinates to cover the full person including legs.
    """
    height, width = frame.shape[:2]
    
    # Adjusted coordinates to cover full person including legs
    person_left_percent = 0.03    # Move slightly more left to capture full width
    person_top_percent = 0.12     # Start higher to capture head better
    person_width_percent = 0.20   # Much wider to capture full body width including arms
    person_height_percent = 0.75  # Much taller to capture full legs down to feet
    
    # Convert percentages to pixel coordinates
    x = int(width * person_left_percent)
    y = int(height * person_top_percent)
    w = int(width * person_width_percent)
    h = int(height * person_height_percent)
    
    print(f"[TRACKING | {time.strftime('%H:%M:%S')}] Full-body ROI: ({x}, {y}, {w}, {h})")
    print(f"[TRACKING | {time.strftime('%H:%M:%S')}] This should cover the person from head to feet")
    
    return (x, y, w, h)

def track_video_auto(input_video_path, output_video_path, tracking_json_path):
    """
    Your friend's tracking with automatic ROI selection - no user interaction.
    """
    print(f"[TRACKING | {time.strftime('%H:%M:%S')}] Starting automatic tracking")

    cap_stabilize, video_width, video_height, fps = get_video_files(path=input_video_path)
    frames_bgr = load_entire_video(cap_stabilize, color_space='bgr')
    
    # Use automatic ROI selection
    initBB = automatic_roi_selection(frames_bgr[0])
    
    x, y, w, h = initBB
    print(f"[TRACKING | {time.strftime('%H:%M:%S')}] Using automatic ROI: ({x}, {y}, {w}, {h})")
    
    track_window = (x, y, w, h)

    # Set up the ROI for tracking (exactly like your friend's code)
    roi = frames_bgr[0][y:y + h, x:x + w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    reducing_light_mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0, 1], reducing_light_mask, [256, 256], [0, 256, 0, 256])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    
    # Setup the termination criteria
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0)
    
    # Process frames
    tracking_frames_list = []
    tracking_data = {}
    
    # First frame
    first_frame = cv2.rectangle(frames_bgr[0].copy(), (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(first_frame, f'Frame: 0', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    tracking_frames_list.append(first_frame)
    tracking_data["0"] = [y, x, h, w]
    
    # Track remaining frames
    for frame_index, frame in enumerate(frames_bgr[1:], 1):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0, 1], roi_hist, [0, 256, 0, 256], 1)
        
        # Apply meanshift
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        
        # Draw rectangle
        x, y, w, h = track_window
        tracked_img = cv2.rectangle(frame.copy(), (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(tracked_img, f'Frame: {frame_index}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        tracking_frames_list.append(tracked_img)
        tracking_data[str(frame_index)] = [y, x, h, w]

    # Write outputs
    write_video(output_video_path, tracking_frames_list, fps, (video_width, video_height), is_color=True)
    
    with open(tracking_json_path, 'w') as f:
        json.dump(tracking_data, f, indent=2)
    
    print(f"[TRACKING | {time.strftime('%H:%M:%S')}] Automatic tracking completed")
    print(f"[TRACKING | {time.strftime('%H:%M:%S')}] Output video saved: {output_video_path}")
    print(f"[TRACKING | {time.strftime('%H:%M:%S')}] Tracking JSON saved: {tracking_json_path}")
    
    cap_stabilize.release()
    
    return tracking_data


def run_auto_tracking(student_id1, student_id2, output_dir):
    """
    Fully automatic tracking - no user interaction required.
    """
    matted_video = os.path.join(output_dir, f"matted_{student_id1}_{student_id2}.avi")
    output_video = os.path.join(output_dir, f"OUTPUT_{student_id1}_{student_id2}.avi")
    tracking_json = os.path.join(output_dir, f"tracking.json")
    
    if not os.path.exists(matted_video):
        print(f"Error: Input video not found: {matted_video}")
        return False
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        tracking_data = track_video_auto(matted_video, output_video, tracking_json)
        print(f"[TRACKING | {time.strftime('%H:%M:%S')}] Automatic tracking completed successfully")
        print(f"[TRACKING | {time.strftime('%H:%M:%S')}] Tracked {len(tracking_data)} frames")
        return True
    except Exception as e:
        print(f"[TRACKING | {time.strftime('%H:%M:%S')}] Tracking failed: {e}")
        import traceback
        traceback.print_exc()
        return False
