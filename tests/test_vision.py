import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from env import FlyEnv
from generate_sequences import generate_episode, save_episode
from vision_dataset import VisionDataset
from vision_model import HeadingCNN, angular_error_degrees
from utils import wrap_angle


class VisionTests(unittest.TestCase):
    def test_boundary_error(self):
        angles = torch.deg2rad(torch.tensor([179., -179.]))
        vectors = torch.stack((angles.cos(), angles.sin()), dim=1)
        self.assertAlmostEqual(angular_error_degrees(vectors[:1], vectors[1:]).item(), 2, places=4)

    def test_original_episode_alignment_and_filter(self):
        env = FlyEnv(seed=91)
        episode = generate_episode(env, length=20, darkness_start=5, darkness_length=3)
        expected = wrap_angle(episode['initial_heading'] + np.cumsum(episode['true_turns']))
        np.testing.assert_allclose(episode['true_headings'], expected, atol=1e-6)
        self.assertTrue((episode['images'][5:8] == 0).all())
        episode['images'][0] = 235  # Deliberately blank, illuminated view.
        with tempfile.TemporaryDirectory() as folder:
            save_episode(episode, Path(folder) / 'episode.npz')
            manifest = {'splits': {'train': [{'file': 'episode.npz'}]}}
            data = VisionDataset(folder, 'train', manifest)
            self.assertEqual(data.excluded_dark, 3)
            self.assertGreaterEqual(data.excluded_blank, 1)
            self.assertNotIn(0, data.steps)
            self.assertFalse(np.isin(data.steps, [5, 6, 7]).any())
            image, target = data[0]
            self.assertEqual(tuple(image.shape), (3, 64, 128))
            self.assertTrue(0 <= image.min() <= image.max() <= 1)
            self.assertAlmostEqual(target.norm().item(), 1, places=5)
            first_step = data.steps[0]
            np.testing.assert_allclose(target.numpy(),
                [np.cos(episode['true_headings'][first_step]), np.sin(episode['true_headings'][first_step])])
            model = HeadingCNN()
            output = model(image.unsqueeze(0))
            self.assertEqual(tuple(output.shape), (1, 2))
            (output - target).square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
