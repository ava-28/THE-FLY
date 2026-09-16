"""Generate stage-two episodes using the original environment and file format."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from env import FlyEnv
from generate_sequences import generate_episode, save_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/vision'))
    parser.add_argument('--train', type=int, default=60)
    parser.add_argument('--val', type=int, default=15)
    parser.add_argument('--test', type=int, default=15)
    parser.add_argument('--length', type=int, default=200)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    if min(args.train, args.val, args.test, args.length) < 1 or args.seed < 0:
        parser.error('Counts and length must be positive; seed must be nonnegative.')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output is not empty. Use a new directory to preserve existing data.')
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {'seed': args.seed, 'length': args.length, 'splits': {},
                'scope': 'New trajectories in the same fixed landmark arena.'}
    offset = 0
    for split, count in [('train', args.train), ('val', args.val), ('test', args.test)]:
        records = []
        for i in range(count):
            seed = args.seed + offset
            offset += 1
            env = FlyEnv(seed=seed, turn_noise_std=0.0)
            episode = generate_episode(env, length=args.length,
                                       darkness_start=None, darkness_length=0)
            path = args.output / split / f'episode_{i:04d}.npz'
            save_episode(episode, path)
            records.append({'file': str(path.relative_to(args.output)), 'seed': seed,
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        manifest['splits'][split] = records
        print(f'{split}: {count} episodes, {count * args.length} frames', flush=True)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Saved:', args.output)


if __name__ == '__main__':
    main()
