from pathlib import Path
import json
import numpy as np

from env import FlyEnv


ACTION_NAMES = {
    0: "left",
    1: "stay",
    2: "right",
}


def sample_action_sequence(
    length,
    rng,
    min_run=4,
    max_run=16,
    pause_probability=0.20,
    reverse_probability=0.35,
    pause_min=1,
    pause_max=4
):
    """
    Generate a sequence with:
    - several consecutive turns in one direction
    - occasional pauses
    - occasional reversals
    """

    actions = []

    # Start by choosing a direction: left or right
    current_direction = int(rng.choice([0, 2]))

    while len(actions) < length:
        # A run of repeated turning in the same direction
        run_length = int(rng.integers(min_run, max_run + 1))
        actions.extend([current_direction] * run_length)

        if len(actions) >= length:
            break

        # Optional pause
        if rng.random() < pause_probability:
            pause_length = int(rng.integers(pause_min, pause_max + 1))
            actions.extend([1] * pause_length)

            if len(actions) >= length:
                break

        # Optional reversal
        if rng.random() < reverse_probability:
            current_direction = 2 if current_direction == 0 else 0

    return np.array(actions[:length], dtype=np.int64)


def make_darkness_mask(
    length,
    darkness_start=None,
    darkness_length=0
):
    """
    Return a boolean mask of length `length`.
    True means the camera image is blank at that timestep.
    """
    mask = np.zeros(length, dtype=bool)

    if darkness_start is None or darkness_length <= 0:
        return mask

    start = max(0, int(darkness_start))
    end = min(length, start + int(darkness_length))

    mask[start:end] = True
    return mask


def generate_episode(
    env,
    length=200,
    darkness_start=80,
    darkness_length=40,
    min_run=4,
    max_run=16,
    pause_probability=0.20,
    reverse_probability=0.35
):
    """
    Generate one episode.

    IMPORTANT ALIGNMENT:
    actions[t], images[t], true_headings[t], turn_estimates[t]
    all refer to the state AFTER taking actions[t].
    """

    # Random initial heading
    reset_output = env.reset(
        randomize_heading=True,
        darkness=False
    )

    initial_heading = reset_output["true_heading"]

    # Generate action program
    actions = sample_action_sequence(
        length=length,
        rng=env.rng,
        min_run=min_run,
        max_run=max_run,
        pause_probability=pause_probability,
        reverse_probability=reverse_probability
    )

    # Darkness interval
    darkness_mask = make_darkness_mask(
        length=length,
        darkness_start=darkness_start,
        darkness_length=darkness_length
    )

    # Allocate storage
    images = np.zeros(
        (
            length,
            env.camera.height,
            env.camera.width,
            3
        ),
        dtype=np.uint8
    )

    turn_estimates = np.zeros(length, dtype=np.float32)
    true_headings = np.zeros(length, dtype=np.float32)
    true_turns = np.zeros(length, dtype=np.float32)

    timestamps = np.arange(length, dtype=np.int32)

    # Roll out episode
    for t in range(length):
        action = int(actions[t])

        output = env.step(
            action=action,
            darkness=bool(darkness_mask[t])
        )

        images[t] = output["image"]
        turn_estimates[t] = output["turn_estimate"]
        true_headings[t] = output["true_heading"]
        true_turns[t] = output["true_turn"]

    episode = {
        "timestamps": timestamps,
        "actions": actions,
        "images": images,
        "turn_estimates": turn_estimates,
        "true_headings": true_headings,
        "true_turns": true_turns,
        "darkness": darkness_mask,
        "initial_heading": np.float32(initial_heading),
        "episode_length": int(length),
        "darkness_start": None if darkness_start is None else int(darkness_start),
        "darkness_length": int(darkness_length),
        "turn_noise_std": np.float32(env.turn_noise_std),
    }

    return episode


def save_episode(
    episode,
    output_path
):
    """
    Save one episode to:
    - .npz for arrays
    - .json for metadata
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        timestamps=episode["timestamps"],
        actions=episode["actions"],
        images=episode["images"],
        turn_estimates=episode["turn_estimates"],
        true_headings=episode["true_headings"],
        true_turns=episode["true_turns"],
        darkness=episode["darkness"],
        initial_heading=episode["initial_heading"]
    )

    metadata = {
        "episode_length": episode["episode_length"],
        "image_height": int(episode["images"].shape[1]),
        "image_width": int(episode["images"].shape[2]),
        "channels": int(episode["images"].shape[3]),
        "turn_noise_std_radians": float(episode["turn_noise_std"]),
        "turn_noise_std_degrees": float(np.rad2deg(episode["turn_noise_std"])),
        "initial_heading_radians": float(episode["initial_heading"]),
        "initial_heading_degrees": float(np.rad2deg(episode["initial_heading"])),
        "darkness_start": episode["darkness_start"],
        "darkness_length": episode["darkness_length"],
        "action_mapping": ACTION_NAMES,
        "alignment_note": (
            "At index t, action[t], image[t], turn_estimate[t], "
            "true_heading[t], and true_turn[t] describe the state "
            "after taking action[t]."
        ),
    }

    metadata_path = output_path.with_suffix(".json")

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main():
    output_dir = Path("episodes")
    output_dir.mkdir(exist_ok=True)

    # Zero-noise debugging version first
    env = FlyEnv(
        turn_noise_std=0.0,
        seed=42
    )

    episode = generate_episode(
        env=env,
        length=200,
        darkness_start=80,
        darkness_length=40,
        min_run=4,
        max_run=16,
        pause_probability=0.20,
        reverse_probability=0.35
    )

    save_path = output_dir / "episode_000.npz"
    save_episode(episode, save_path)

    print("Saved episode to:", save_path)
    print("Initial heading:", round(np.rad2deg(episode["initial_heading"]), 2), "degrees")
    print("Images shape:", episode["images"].shape)
    print("Number of steps:", len(episode["timestamps"]))
    print("Darkness steps:", int(np.sum(episode["darkness"])))

    print("\nFirst 10 steps:")
    for t in range(10):
        action = int(episode["actions"][t])
        print(
            f"t={t:3d} | "
            f"action={ACTION_NAMES[action]:>5s} | "
            f"turn_est={np.rad2deg(episode['turn_estimates'][t]):6.2f}° | "
            f"heading={np.rad2deg(episode['true_headings'][t]):7.2f}° | "
            f"dark={bool(episode['darkness'][t])}"
        )


if __name__ == "__main__":
    main()