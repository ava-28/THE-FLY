"""Evaluate the selected checkpoint once on held-out test episodes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from vision_dataset import VisionDataset
from vision_model import HeadingCNN, angular_error_degrees, get_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/vision'))
    parser.add_argument('--checkpoint', type=Path, default=Path('runs/vision/best.pt'))
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'mps', 'cuda'])
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    digest = hashlib.sha256((args.data / 'manifest.json').read_bytes()).hexdigest()
    if checkpoint['manifest_sha256'] != digest:
        raise ValueError('This checkpoint was trained on a different dataset manifest.')
    data = VisionDataset(args.data, 'test')
    model = HeadingCNN().to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    predictions, errors, norms = [], [], []
    with torch.no_grad():
        for images, targets in DataLoader(data, batch_size=128):
            output = model(images.to(device)).cpu()
            predictions.extend(torch.atan2(output[:, 1], output[:, 0]).numpy())
            errors.extend(angular_error_degrees(output, targets).numpy())
            norms.extend(output.norm(dim=1).numpy())
    predictions, errors, norms = map(np.asarray, (predictions, errors, norms))
    episode_maes = [float(errors[data.episode_ids == i].mean())
                    for i in np.unique(data.episode_ids)]
    metrics = {'frames': data.summary(), 'checkpoint_epoch': checkpoint['epoch'],
               'mean_error_deg': float(errors.mean()), 'median_error_deg': float(np.median(errors)),
               'p95_error_deg': float(np.percentile(errors, 95)),
               'within_5_deg_percent': float(100 * np.mean(errors <= 5)),
               'episode_mean_errors_deg': episode_maes,
               'near_zero_output_count': int((norms < 1e-6).sum()),
               'scope': 'Illuminated landmark frames; same arena, held-out episodes. No memory or RL.'}
    output_dir = args.checkpoint.parent
    (output_dir / 'test_metrics.json').write_text(json.dumps(metrics, indent=2))
    with (output_dir / 'test_predictions.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'step', 'true_deg', 'predicted_deg', 'error_deg', 'output_norm'])
        for i in range(len(data)):
            writer.writerow([data.files[data.episode_ids[i]], data.steps[i],
                             np.rad2deg(data.headings[i]), np.rad2deg(predictions[i]), errors[i], norms[i]])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    first = data.episode_ids == data.episode_ids[0]
    # Points avoid misleading connecting lines across the -180/180 boundary.
    axes[0].scatter(data.steps[first], np.rad2deg(data.headings[first]), s=12, label='True')
    axes[0].scatter(data.steps[first], np.rad2deg(predictions[first]), s=8, label='Predicted')
    axes[0].set(xlabel='Step', ylabel='Heading (degrees)', title='First retained test episode')
    axes[0].legend()
    axes[1].hist(errors, bins=40)
    axes[1].set(xlabel='Shortest angular error (degrees)', ylabel='Frames', title='Test error distribution')
    fig.savefig(output_dir / 'test_results.png', dpi=160)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))
    print('Saved test_metrics.json, test_predictions.csv, and test_results.png in', output_dir)


if __name__ == '__main__':
    main()
