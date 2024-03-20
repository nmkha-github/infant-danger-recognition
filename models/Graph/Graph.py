import numpy as np

from models.Pose.Pose import Pose
from models.Graph.utils.calculate_angle import calculate_angle


class Graph:
    def __init__(self, frames=[], max_frames=20):
        self.edges = Graph.edges_generate(len(frames))
        self.nodes = Graph.nodes_generate(frames)
        self.max_frames = max_frames

    def append(self, frame):
        number_of_joints = 33
        coordinates, velocities, angles = self.extract_node_feature(frame)
        coordinates = np.array(coordinates)
        velocities = np.array(velocities)
        angles = np.array([angle + [0] * (6 - len(angle)) for angle in angles])

        self.nodes.extend(np.concatenate((coordinates, velocities, angles), axis=1))

        frame_index = len(self.nodes) // number_of_joints - 1
        if frame_index >= self.max_frames:
            np.delete(self.nodes, np.arange(33))
            return

        self.edges = Graph.edges_generate(frame_index + 1)

    def extract_node_feature(self, frame):
        # Extract pose landmarks from the frame
        pose = Pose(frame)
        landmarks, connections_list = pose.landmarks, Pose.connections_list

        # Extract features from the pose landmarks
        coordinates = []  # List to store joint coordinates for this frame
        velocities = []  # List to store joint velocities for this frame
        angles = []  # List to store joint angles for this frame

        previous_coordinates = None
        if len(self.nodes):
            previous_coordinates = [
                [node[0], node[1], node[2]] for node in self.nodes[-len(landmarks) :]
            ]

        # Extract features for each landmark
        for landmark in landmarks:
            # Extract joint coordinates (x, y, z)
            coordinates.append([landmark.x, landmark.y, landmark.z])
        length_landmarks = len(coordinates)

        # Calculate joint velocities (Placeholder: differences between consecutive frames)
        if len(self.nodes):
            if previous_coordinates[0][0] != -1:
                velocities = np.array(coordinates) - np.array(previous_coordinates)
            else:
                velocities = np.array(
                    [np.array([0, 0, 0]).astype(float)] * length_landmarks
                )
        else:
            velocities = np.array(
                [np.array([0, 0, 0]).astype(float)] * length_landmarks
            )

        # Calculate angles
        for root_coord_index in range(length_landmarks):
            num_connections = len(connections_list[root_coord_index])
            root_coord = coordinates[root_coord_index]
            root_angles = []

            if num_connections >= 2:
                for u in range(num_connections):
                    for v in range(u + 1, num_connections):
                        joint_u = coordinates[connections_list[root_coord_index][u]]
                        joint_v = coordinates[connections_list[root_coord_index][v]]
                        angle = calculate_angle(joint_u, root_coord, joint_v)
                        # root_angles.append((u, v, angle))
                        root_angles.append(angle)

            angles.append(root_angles)

        # Update previous coordinates
        previous_coordinates = coordinates

        return coordinates, velocities, angles

    @staticmethod
    def edges_generate(frames_length):
        number_of_joints = 33
        edges = []
        for i in range(frames_length):
            for connect in Pose.connections:
                edges.append(
                    [
                        connect[0] + i * number_of_joints,
                        connect[1] + i * number_of_joints,
                    ]
                )
                edges.append(
                    [
                        connect[1] + i * number_of_joints,
                        connect[0] + i * number_of_joints,
                    ]
                )
            if i > 0:
                for j in range(33):
                    edges.append(
                        [i * number_of_joints + j, (i - 1) * number_of_joints + j]
                    )
                    edges.append(
                        [(i - 1) * number_of_joints + j, i * number_of_joints + j]
                    )

        return edges

    @staticmethod
    def nodes_generate(frames):
        max_feature = 12
        nodes = []
        joint_coordinates, joint_velocities, joint_angles = (
            Graph.extract_nodes_features(frames)
        )

        for frame_index in range(len(frames)):
            coordinates = np.array(joint_coordinates[frame_index])
            velocities = np.array(joint_velocities[frame_index])
            angles = np.array(
                [
                    angles + [0] * (6 - len(angles))
                    for angles in joint_angles[frame_index]
                ]
            )

            # Concatenate coordinates, velocities and angles
            frame_nodes = np.concatenate((coordinates, velocities, angles), axis=1)

            # Append padded frame nodes to nodes list
            nodes.extend(frame_nodes)

        return nodes

    @staticmethod
    def extract_nodes_features(frames):
        # Initialize lists to store features
        joint_coordinates = []
        joint_angles = []
        joint_velocities = []

        previous_coordinates = None

        # Iterate through each frame
        for frame_index, frame in enumerate(frames):
            # Extract pose landmarks from the frame
            pose = Pose(frame)
            landmarks, connections_list = pose.landmarks, Pose.connections_list

            # Extract features from the pose landmarks
            coordinates = []  # List to store joint coordinates for this frame
            velocities = []  # List to store joint velocities for this frame
            angles = []  # List to store joint angles for this frame

            # Extract features for each landmark
            for landmark in landmarks:
                # Extract joint coordinates (x, y, z)
                coordinates.append([landmark.x, landmark.y, landmark.z])
            length_landmarks = len(coordinates)

            # Calculate joint velocities (Placeholder: differences between consecutive frames)
            if frame_index > 0:
                if previous_coordinates[0][0] != -1:
                    velocities = np.array(coordinates) - np.array(previous_coordinates)
                else:
                    velocities = np.array(
                        [np.array([0, 0, 0]).astype(float)] * length_landmarks
                    )
            else:
                velocities = np.array(
                    [np.array([0, 0, 0]).astype(float)] * length_landmarks
                )

            # Calculate angles
            for root_coord_index in range(length_landmarks):
                num_connections = len(connections_list[root_coord_index])
                root_coord = coordinates[root_coord_index]
                root_angles = []

                if num_connections >= 2:
                    for u in range(num_connections):
                        for v in range(u + 1, num_connections):
                            joint_u = coordinates[connections_list[root_coord_index][u]]
                            joint_v = coordinates[connections_list[root_coord_index][v]]
                            angle = calculate_angle(joint_u, root_coord, joint_v)
                            # root_angles.append((u, v, angle))
                            root_angles.append(angle)

                angles.append(root_angles)

            # Append features for this frame to the respective lists
            joint_coordinates.append(coordinates)
            joint_velocities.append(velocities)
            joint_angles.append(angles)

            # Update previous coordinates
            previous_coordinates = coordinates

        # Return the extracted features
        return joint_coordinates, joint_velocities, joint_angles
