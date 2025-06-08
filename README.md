# video-processing-pipeline
This repository implements a full video-processing pipeline that takes a shaky input clip of a walking person and produces a set of polished outputs through four key stages:
	1.	Video stabilization — smooths out camera shake to generate a stabilized AVI.
	2.	Background subtraction & mask extraction — isolates the subject into both a full-color “extracted” video and a binary mask (0 = background, 1 = foreground).
	3.	Image matting & alpha generation — composites the isolated person onto a new static background and produces an alpha-channel video for soft edges.
	4.	Object tracking — locates and tracks the subject frame-by-frame, drawing a bounding box in the final output video.

All steps run from a single Python entry point (main.py) and rely on OpenCV-based algorithms. The script outputs six processed videos (stabilized.avi, extracted.avi, binary.avi, alpha.avi, matted.avi, output.avi) plus two JSON files capturing per-video timing and per-frame tracking data .
