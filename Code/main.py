import os
import time
import traceback
import json
# compute base, input and output directories relative to this script
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
INPUT_DIR = os.path.join(BASE_DIR, "Inputs")
OUTPUT_DIR = os.path.join(BASE_DIR, "Outputs")
from video_stab import stabilize_video_improved
from background_sub import background_subtraction
from matting import run_matting_stage_closed_form
from tracking import run_auto_tracking

def main():
    """Main function to run the entire video processing pipeline."""
    main_start_time = time.time()
    print(f"[MAIN | {time.strftime('%H:%M:%S')}] Starting video processing pipeline...")
    
    student_id1 = "208484097"
    student_id2 = "318931573"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    input_video = os.path.join(INPUT_DIR, "INPUT.avi")
    stabilized_video = os.path.join(OUTPUT_DIR, f"stabilize_{student_id1}_{student_id2}.avi")
    
    timing_data = {}
    
    try:
        # Step 1: Video Stabilization
        print(f"[MAIN | {time.strftime('%H:%M:%S')}] Initiating video stabilization...")
        stabilize_video_improved(input_video, stabilized_video)
        timing_data["time_to_stabilize"] = time.time() - main_start_time
        print(f"[MAIN | {time.strftime('%H:%M:%S')}] Stabilization finished. Cumulative time: {timing_data['time_to_stabilize']:.2f}s")
        
        # Step 2: Background Subtraction
        print(f"[MAIN | {time.strftime('%H:%M:%S')}] Initiating background subtraction...")
        background_subtraction(stabilized_video, OUTPUT_DIR)
        timing_data["time_to_binary"] = time.time() - main_start_time
        print(f"[MAIN | {time.strftime('%H:%M:%S')}] Background subtraction finished. Cumulative time: {timing_data['time_to_binary']:.2f}s")
        
        # Step 3: Matting
        print(f"[MAIN | {time.strftime('%H:%M:%S')}] Initiating matting stage...")
        matting_times = run_matting_stage_closed_form((student_id1, student_id2), main_start_time, INPUT_DIR, OUTPUT_DIR, method='optimized')
        timing_data.update(matting_times)
        print(f"[MAIN | {time.strftime('%H:%M:%S')}] Matting stage finished.")

        # Step 4: Tracking
        print(f"[MAIN | {time.strftime('%H:%M:%S')}] Initiating tracking stage...")
        success = run_auto_tracking(student_id1, student_id2, OUTPUT_DIR)
        if success:
            timing_data["time_to_output"] = time.time() - main_start_time
            print(f"[MAIN | {time.strftime('%H:%M:%S')}] Tracking stage finished. Cumulative time: {timing_data['time_to_output']:.2f}s")
        else:
            raise Exception("Tracking stage failed.")
        
        total_time = time.time() - main_start_time
        print(f"\n[MAIN | {time.strftime('%H:%M:%S')}] Video processing pipeline completed successfully in {total_time:.2f} seconds.")
        mins, secs = divmod(total_time, 60)
        print(f"[MAIN | {time.strftime('%H:%M:%S')}] Total pipeline time: {int(mins)}m{secs:.2f}s")
        
        with open(os.path.join(OUTPUT_DIR, "timing.json"), "w") as f:
            json.dump(timing_data, f, indent=4)
        print(f"[MAIN | {time.strftime('%H:%M:%S')}] timing.json file created successfully.")
            
    except Exception as e:
        print(f"[MAIN | {time.strftime('%H:%M:%S')}] Error in pipeline: {e}")
        traceback.print_exc()
        return False
    
    return True

if __name__ == '__main__':
    main()