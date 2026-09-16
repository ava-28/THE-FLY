import numpy as np

from camera import FlyCamera
from utils import wrap_angle


# ============================================================
# ACTIONS
#
# 0 = left
# 1 = stay
# 2 = right
# ============================================================

TURN_ANGLE = np.deg2rad(5)

ACTIONS = {
    0: -TURN_ANGLE,
    1: 0.0,
    2: TURN_ANGLE,
}


# ============================================================
# LANDMARKS
# ============================================================

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


# ============================================================
# ENVIRONMENT
# ============================================================

class FlyEnv:
    def __init__(
        self,
        turn_noise_std=0.0,
        seed=None,
        image_width=128,
        image_height=64,
        fov_degrees=120
    ):
        self.heading = 0.0
        self.landmarks = LANDMARKS

        self.camera = FlyCamera(
            width=image_width,
            height=image_height,
            fov_degrees=fov_degrees
        )

        self.turn_noise_std = turn_noise_std
        self.rng = np.random.default_rng(seed)

    def reset(
        self,
        initial_heading=None,
        randomize_heading=False,
        darkness=False
    ):
        if initial_heading is not None:
            self.heading = wrap_angle(float(initial_heading))
        elif randomize_heading:
            self.heading = self.rng.uniform(-np.pi, np.pi)
        else:
            self.heading = 0.0

        image = self._get_image(darkness=darkness)

        return {
            "image": image,
            "turn_estimate": 0.0,
            "true_heading": self.heading,
            "true_turn": 0.0,
        }

    def step(
        self,
        action,
        darkness=False
    ):
        if action not in ACTIONS:
            raise ValueError(
                f"Invalid action {action}. Use 0=left, 1=stay, 2=right."
            )

        # True turn
        true_turn = ACTIONS[action]

        # Update true heading
        self.heading = wrap_angle(self.heading + true_turn)

        # Noisy turning estimate
        noise = self.rng.normal(
            loc=0.0,
            scale=self.turn_noise_std
        )
        turn_estimate = true_turn + noise

        # Image observation
        image = self._get_image(darkness=darkness)

        return {
            "image": image,
            "turn_estimate": turn_estimate,
            "true_heading": self.heading,
            "true_turn": true_turn,
        }

    def _get_image(self, darkness=False):
        if darkness:
            return np.zeros(
                (
                    self.camera.height,
                    self.camera.width,
                    3
                ),
                dtype=np.uint8
            )

        return self.camera.render(
            heading=self.heading,
            landmarks=self.landmarks
        )