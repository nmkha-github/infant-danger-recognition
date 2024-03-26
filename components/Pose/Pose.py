import cv2
import mediapipe as mp

from components.Pose.interface.PoseLandmark import PoseLandmark


class Pose:
    connections = list(mp.solutions.pose.POSE_CONNECTIONS)
    connections_matrix = [
        [
            (
                1
                if (i, j) in list(mp.solutions.pose.POSE_CONNECTIONS)
                or (j, i) in list(mp.solutions.pose.POSE_CONNECTIONS)
                else 0
            )
            for j in range(33)
        ]
        for i in range(33)
    ]

    length_landmarks = len(connections_matrix)
    connections_list = []
    for u in range(length_landmarks):
        connections_list.append([])
        for v in range(length_landmarks):
            if u == v:
                continue
            if connections_matrix[u][v]:
                connections_list[u].append(v)

    def __init__(self, frame, stagery="mediapipe"):
        self.frame = frame
        self.landmarks = Pose.extract_landmarks_mediapipe(frame)

    @staticmethod
    def extract_landmarks_mediapipe(image):
        with mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0,
            min_tracking_confidence=0.5,
        ) as pose:
            results = pose.process(image)
            landmarks = results.pose_landmarks
            # Check if landmarks are detected
            if landmarks is None:
                return [PoseLandmark() for _ in range(33)]

            return [
                PoseLandmark(
                    x=landmark.x,
                    y=landmark.y,
                    z=landmark.z,
                    visibility=landmark.visibility,
                )
                for landmark in landmarks.landmark
            ]

    @staticmethod
    def visualize_skeleton(image):
        pose = Pose(image)
        landmarks = pose.landmarks

        # Draw landmarks and skeleton on the image
        for landmark in landmarks:
            height, width, _ = image.shape
            cx, cy = int(landmark.x * width), int(landmark.y * height)
            cv2.circle(
                image, (cx, cy), 5, (0, 255, 0), -1
            )  # Draw a green circle for each landmark

        for connection in Pose.connections:
            start_landmark = connection[0]
            end_landmark = connection[1]
            start_point = (
                int(landmarks[start_landmark].x * width),
                int(landmarks[start_landmark].y * height),
            )
            end_point = (
                int(landmarks[end_landmark].x * width),
                int(landmarks[end_landmark].y * height),
            )
            cv2.line(
                image, start_point, end_point, (0, 255, 0), 2
            )  # Draw a green line for each connection
        # Display the image
        cv2.imshow(image)
