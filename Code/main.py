import os
import time
import traceback
import json
from video_stab import stabilize_video_improved
from background_sub import background_subtraction
from matting import run_matting_stage_closed_form
from tracking import run_auto_tracking

def main():
    """Main function to run the entire video processing pipeline."""
    start_time = time.time()
    print(f"[MAIN | {time.strftime('%H:%M:%S')}] Starting video processing pipeline...")
    
    # Define student IDs for consistent file naming
    student_id1 = "208484097"
    student_id2 = "318931573"
    
    # Ensure Outputs directory exists
    os.makedirs("../Outputs", exist_ok=True)
    
    # Input and output paths
    input_video = "../Inputs/INPUT.avi"
    stabilized_video = "../Outputs/stabilize_208484097_318931573.avi"
    output_dir = "../Outputs"
    
    try:
        # Step 1: Video Stabilization
        print(f"[MAIN | {time.strftime('%H:%M:%S')}] Initiating video stabilization...")
        stabilize_start = time.time()
        stabilize_video_improved(input_video, stabilized_video)
        print(f"[MAIN | {time.strftime('%H:%M:%S')}] Video stabilization completed in {time.time() - stabilize_start:.2f} seconds.")
        duration_stabilization = time.time() - stabilize_start
        
        # Step 2: Background Subtraction
        print(f"[MAIN | {time.strftime('%H:%M:%S')}] Initiating background subtraction...")
        bg_sub_start = time.time()
        background_subtraction(stabilized_video, output_dir)
        print(f"[MAIN | {time.strftime('%H:%M:%S')}] Background subtraction completed in {time.time() - bg_sub_start:.2f} seconds.")
        duration_bg_sub = time.time() - bg_sub_start
        
        # Step 3: Matting
        print(f"[MAIN | {time.strftime('%H:%M:%S')}] Initiating matting stage...")
        matting_start = time.time()
        run_matting_stage_closed_form((student_id1, student_id2), method='optimized')
        print(f"[MAIN | {time.strftime('%H:%M:%S')}] Matting stage completed in {time.time() - matting_start:.2f} seconds.")
        duration_matting = time.time() - matting_start
        
        # Step 4: Tracking
        print(f"[MAIN | {time.strftime('%H:%M:%S')}] Initiating tracking stage...")
        tracking_start = time.time()
        success = run_auto_tracking(student_id1, student_id2, output_dir)
        if success:
            duration_tracking = time.time() - tracking_start
            print(f"[MAIN | {time.strftime('%H:%M:%S')}] Tracking stage completed in {time.time() - tracking_start:.2f} seconds.")
        else:
            print(f"[MAIN | {time.strftime('%H:%M:%S')}] Tracking stage failed.")
            raise Exception("Tracking stage failed.")
        
        total_time = time.time() - start_time
        print(f"[MAIN | {time.strftime('%H:%M:%S')}] Video processing pipeline completed successfully in {total_time:.2f} seconds.")
        # also display total time in minutes and seconds
        mins, secs = divmod(total_time, 60)
        print(f"[MAIN | {time.strftime('%H:%M:%S')}] Total pipeline time: {int(mins)}m{secs:.2f}s")
        
        # Format durations as minutes and seconds strings
        timings = {
            "stabilization": f"{int(duration_stabilization//60)}m{duration_stabilization%60:.2f}s",
            "background_subtraction": f"{int(duration_bg_sub//60)}m{duration_bg_sub%60:.2f}s",
            "matting": f"{int(duration_matting//60)}m{duration_matting%60:.2f}s",
            "tracking": f"{int(duration_tracking//60)}m{duration_tracking%60:.2f}s"
        }
        with open(os.path.join(output_dir, "timing.json"), "w") as f:
            json.dump(timings, f, indent=2)
    
    except Exception as e:
        print(f"[MAIN | {time.strftime('%H:%M:%S')}] Error in pipeline: {e}")
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()
