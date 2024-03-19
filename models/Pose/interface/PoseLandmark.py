class PoseLandmark:
    def __init__(self, x=-1, y=-1, z=-1, visibility=0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

    def __repr__(self) -> str:
        return f"PoseLandmark(x={self.x}, y={self.y}, z={self.z}, visibility={self.visibility})"

    def __str__(self) -> str:
        return f"PoseLandmark(x={self.x}, y={self.y}, z={self.z}, visibility={self.visibility})"
