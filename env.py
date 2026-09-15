import numpy as np


TURN_ANGLE = np.deg2rad(5)

ACTIONS = {
    0: -TURN_ANGLE,   # turn left
    1: 0.0,           # stay still
    2: TURN_ANGLE,    # turn right
}

ARENA_RADIUS = 1.0


def wrap_angle(angle):
    """
    Keep an angle inside [-pi, pi).
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


LANDMARKS = [
    {
        "bearing": np.deg2rad(0),
        "shape": "striped_rectangle",
        "colour": "blue",
        "width": 0.18,
        "height": 0.28,
    },
    {
        "bearing": np.deg2rad(70),
        "shape": "circle",
        "colour": "red",
        "width": 0.18,
        "height": 0.18,
    },
    {
        "bearing": np.deg2rad(190),
        "shape": "triangle",
        "colour": "green",
        "width": 0.22,
        "height": 0.20,
    },
    {
        "bearing": np.deg2rad(285),
        "shape": "checkerboard",
        "colour": "black",
        "width": 0.22,
        "height": 0.22,
    },
]


class FlyEnv:
    def __init__(self, landmarks=None):
        self.heading = 0.0
        self.landmarks = landmarks if landmarks is not None else LANDMARKS

    def reset(self):
        self.heading = 0.0
        return self.heading

    def step(self, action):
        if action not in ACTIONS:
            raise ValueError(f"Invalid action: {action}. Use 0, 1, or 2.")

        delta_theta = ACTIONS[action]
        self.heading = wrap_angle(self.heading + delta_theta)
        return self.heading