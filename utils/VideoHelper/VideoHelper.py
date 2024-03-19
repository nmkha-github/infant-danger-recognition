import cv2
import os
from multiprocessing import Pool


class VideoHelper:
    @staticmethod
    def get_num_processes():
        try:
            # Get the number of CPU cores
            num_cores = os.cpu_count()
            # Limit the number of processes to the number of CPU cores
            num_processes = min(num_cores, 4)  # Limit to a maximum of 4 processes
            return num_processes
        except Exception as e:
            print(f"Error determining the number of CPU cores: {e}")
            return 1  # Default to a single process if an error occurs

    @staticmethod
    def process_frame(frame_num, video_path, frame_interval):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num * frame_interval)
        ret, frame = cap.read()
        cap.release()
        if ret:
            return frame
        else:
            return None

    @staticmethod
    def extract_frames(video_path, frames_per_second):
        num_processes = VideoHelper.get_num_processes()
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / frames_per_second)

        pool = Pool(processes=num_processes)
        frame_indices = list(range(total_frames // frame_interval))
        frames_parallel = pool.starmap(
            VideoHelper.process_frame,
            [(idx, video_path, frame_interval) for idx in frame_indices],
        )
        pool.close()
        pool.join()

        frames = [frame for frame in frames_parallel if frame is not None]

        return frames

    @staticmethod
    def show_video_by_frames(frames):
        if not frames:
            print("No frames to display.")
            return

        for frame in frames:
            cv2.imshow(frame)
