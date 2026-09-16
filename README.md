# THE FLY

A visual heading-estimation experiment, with a planned fly-inspired memory circuit.
Stage 1 supplies a rotating agent and camera. Stage 2 trains an image-only CNN.
The current model contains no fly connectome, internal compass, or RL policy yet.

## Stage 2: run it

From the repository folder, with your Python environment activated (Python 3.10+):

```bash
python -m pip install -r requirements-vision.txt
python prepare_vision_data.py
python train_vision.py
python evaluate_vision.py
```

The preparation step calls the existing `FlyEnv`, `generate_episode`, and
`save_episode`; it does not replace the simulator or change the episode format.
Defaults generate 60 training, 15 validation, and 15 test episodes of 200 steps.
Every episode has a distinct recorded seed. All frames are illuminated initially.
The loader supports existing darkness arrays and excludes dark frames and uniform
views without visible landmarks. Excluded frame counts are reported.

Outputs:

- `data/vision/manifest.json`: episode assignments, seeds, and content hashes.
- `runs/vision/best.pt`: checkpoint with the lowest validation angular error.
- `runs/vision/config.json` and `history.json`: settings and learning history.
- `runs/vision/test_metrics.json`: mean, median, 95th percentile angular error,
  percentage within 5 degrees, and per-episode mean errors.
- `runs/vision/test_predictions.csv`: episode/frame identifiers, labels, predictions.
- `runs/vision/test_results.png`: heading trace and test error distribution.

Generation and training refuse to overwrite nonempty output directories. For a
new experiment, choose a new data directory and run directory explicitly:

```bash
python prepare_vision_data.py --output data/experiment2 --seed 10000
python train_vision.py --data data/experiment2 --output runs/experiment2
python evaluate_vision.py --data data/experiment2 --checkpoint runs/experiment2/best.pt
```

A small integration check:

```bash
python prepare_vision_data.py --output data/check --train 12 --val 3 --test 3
python train_vision.py --data data/check --output runs/check --epochs 12 --device cpu
python evaluate_vision.py --data data/check --checkpoint runs/check/best.pt --device cpu
python -m unittest discover -s tests
```

## What the model learns

Input is RGB pixels only, converted from uint8 HWC to float CHW in [0, 1].
The label is `(cos(theta), sin(theta))`, trained with mean squared error.
Predicted heading is `atan2(output[1], output[0])`. Error uses the shortest
angular distance, so 179 and -179 degrees are two degrees apart.
Three convolutional layers feed a position-preserving flattened feature map and
small dense head. A global average over the image would discard useful position
information. The model uses the standard PyTorch Dataset/DataLoader and module
training pattern: https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

`turn_estimates` remain in the episodes but are not passed to this baseline.
`true_headings` are labels only; `true_turns` are never inputs. The original
after-action timestep alignment is preserved without shifting any arrays.

## Interpreting results

This evaluates new trajectories in the **same fixed arena**, not transfer to
new landmark layouts. Nearby headings can produce identical pixel images due
to rasterization, including across episodes; separate episodes prevent sequence
mixing but do not establish unseen-image or unseen-environment generalization.
Do not tune the model using test results: use validation error for development.
For research comparisons, repeat training across seeds and compare episode-level
results rather than treating correlated frames as independent experiments.

The vector magnitude is not calibrated confidence; nearly zero vectors have an
unreliable angle and are counted in evaluation. Empty views are excluded explicitly,
not claimed as successful navigation. Darkness, blur, memory and RL come later.

Automatic device selection prefers CUDA, then Apple MPS, then CPU. Use
`--device cpu` if GPU support is unavailable. Identical seeds do not guarantee
bitwise-identical results across different hardware/PyTorch versions.
Default images occupy roughly 450 MB of RAM for training and validation, plus
model/batch overhead. Larger datasets need a streaming loader.

## Initial integration result

The small integration run above (12 train / 3 validation / 3 test episodes,
12 epochs, CPU, seed 7) selected epoch 11 and achieved 0.822 degrees test mean
angular error, 0.695 degrees median, and 2.099 degrees 95th percentile across
600 test frames. All were within 5 degrees. This is a single-seed same-arena
check, not a robustness claim. Metrics, settings, history, and plot are in
`results/stage2-check/`. The two automated tests passed.
