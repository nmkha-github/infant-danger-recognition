import cv2


class VideoHelper:
    @staticmethod
    def extract_frames(video_path, num_frames, fps=4):
        """
        extract frames by fps and drop frames to fix num_frames
        Note:
        you should keep length video (seconds) equal num_frames // fps to get full video context
        """
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_interval = max(total_frames // fps, 1)

            frames = []
            for i in range(num_frames):
                frame_num = i * frame_interval
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                else:
                    # If there are fewer frames than expected, add the final frame repeatedly
                    frames.extend(
                        [frames[len(frames) - 1]] * (num_frames - len(frames))
                    )
                    break

            cap.release()
            return frames
        except Exception as e:
            print(f"Error extracting frames: {e} , {video_path}")
            return []

    @staticmethod
    def show_video_by_frames(frames):
        if not frames:
            print("No frames to display.")
            return

        for frame in frames:
            cv2.imshow(frame)
