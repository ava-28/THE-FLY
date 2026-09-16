"""Train on train episodes; select checkpoints using validation episodes only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from vision_dataset import VisionDataset, read_manifest
from vision_model import HeadingCNN, angular_error_degrees, get_device


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    loss_sum = error_sum = count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            predictions = model(images)
            loss = nn.functional.mse_loss(predictions, targets)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss.')
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            n = len(images)
            count += n
            loss_sum += loss.item() * n
            error_sum += angular_error_degrees(predictions.detach(), targets).sum().item()
    return {'mse': loss_sum / count, 'mae_deg': error_sum / count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/vision'))
    parser.add_argument('--output', type=Path, default=Path('runs/vision'))
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'mps', 'cuda'])
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1 or args.lr <= 0:
        parser.error('Epochs, batch size, threads, and learning rate must be positive.')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Run directory is not empty. Choose a new --output.')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device(args.device)
    manifest = read_manifest(args.data)
    train = VisionDataset(args.data, 'train', manifest)
    val = VisionDataset(args.data, 'val', manifest)
    generator = torch.Generator().manual_seed(args.seed)
    loaders = [DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=generator),
               DataLoader(val, batch_size=args.batch_size)]
    model = HeadingCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(device_used=str(device), torch_version=str(torch.__version__),
                  train=train.summary(), val=val.summary(),
                  manifest_sha256=hashlib.sha256((args.data / 'manifest.json').read_bytes()).hexdigest())
    (args.output / 'config.json').write_text(json.dumps(config, indent=2))
    print(f'Device: {device}; train: {train.summary()}; val: {val.summary()}', flush=True)
    best = float('inf')
    history = []
    for epoch in range(1, args.epochs + 1):
        training = run_epoch(model, loaders[0], device, optimizer)
        validation = run_epoch(model, loaders[1], device)
        history.append({'epoch': epoch, 'train': training, 'val': validation})
        if validation['mae_deg'] < best:
            best = validation['mae_deg']
            torch.save({'model_state': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        'epoch': epoch, 'val_mae_deg': best,
                        'manifest_sha256': config['manifest_sha256']}, args.output / 'best.pt')
        (args.output / 'history.json').write_text(json.dumps(history, indent=2))
        print(f"Epoch {epoch:02d}: train {training['mae_deg']:.2f}°, val {validation['mae_deg']:.2f}°", flush=True)
    print(f'Best validation MAE: {best:.2f}°. Saved {args.output / "best.pt"}')


if __name__ == '__main__':
    main()
