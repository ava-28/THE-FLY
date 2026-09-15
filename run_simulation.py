import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

from env import FlyEnv
from render import FlyRenderer


# ----------------------------------
# Create environment
# ----------------------------------

env = FlyEnv()


# ----------------------------------
# Create visualization
# ----------------------------------

renderer = FlyRenderer(
    env.landmarks
)


# ----------------------------------
# Actions
#
# 0 = left
# 1 = stay
# 2 = right
#
# This sequence makes the fly:
#
# 0° → +60°
# +60° → -60°
# -60° → 0°
#
# Then repeats.
# ----------------------------------

actions = (

    [2] * 12

    + [1] * 4

    + [0] * 24

    + [1] * 4

    + [2] * 12
)


# ----------------------------------
# Initial frame
# ----------------------------------

def initialize():

    heading = env.reset()

    renderer.update_heading(
        heading
    )

    return (
        renderer.heading_line,
        renderer.heading_text
    )


# ----------------------------------
# Animation update
# ----------------------------------

def update(frame):

    action = actions[frame]

    heading = env.step(action)

    print(
        f"action={action}   "
        f"heading="
        f"{np.rad2deg(heading):.1f}°"
    )

    renderer.update_heading(
        heading
    )

    return (
        renderer.heading_line,
        renderer.heading_text
    )


# ----------------------------------
# Create animation
# ----------------------------------

animation = FuncAnimation(

    renderer.fig,

    update,

    frames=len(actions),

    init_func=initialize,

    interval=250,

    repeat=True,

    blit=False,

    cache_frame_data=False
)


# ----------------------------------
# Start GUI
# ----------------------------------

plt.show()