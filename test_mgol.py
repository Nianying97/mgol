"""Synthetic correctness tests. All numeric settings below are toy fixtures."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from config import validate_config
from data_process import (TabularTransform, assign_units,
                          inference_dataset, prepare_domain)
from mgol import MGOL, FisherRaoTemporalEnhancement, OverlappingLocalGeometry
from trainer import train
from utils import classification_metrics, seed_everything


def toy_config():
    """Intentionally small synthetic settings, unrelated to manuscript experiments."""
    return {'seed': 7,
            'split': {'test_fraction': 0.25, 'validation_fraction_of_train': 0.25},
            'model': {'latent_dim': 6, 'probability_dim': 3, 'num_units': 3,
                      'num_heads': 2, 'num_layers': 1, 'seq_len': 2, 'output_dim': 8,
                      'encoder_hidden_dims': [10, 6], 'classifier_hidden_dims': [9, 5],
                      'ffn_dim': 11, 'dropout': 0.0, 'affinity_weight': 0.37},
            'training': {'batch_size': 8, 'eval_batch_size': 17, 'num_workers': 0,
                         'pretrain_epochs': 1, 'finetune_epochs': 1,
                         'lr_pretrain': 0.002, 'lr_private_finetune': 0.0003,
                         'weight_decay': 0.0, 'grad_clip': 2.0, 'amp': False},
            'domains': {'toy': {'csv': 'unused.csv', 'label_column': 'Label',
                                'benign_label': 'Benign', 'feature_columns': ['x', 'v'],
                                'group_columns': ['run'], 'block_size': None,
                                'window_size': 3, 'stride': 1}}}


def toy_frame():
    rng = np.random.default_rng(11)
    n = 24 * 18
    return pd.DataFrame({'run': np.repeat(np.arange(24), 18),
                         'x': rng.normal(size=n), 'v': rng.uniform(size=n),
                         'Label': np.where(np.arange(n) % 4 < 2, 'Benign', 'Attack')})


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_fisher_rao_equations_and_gradient(self):
        module = FisherRaoTemporalEnhancement(6, 3)
        z = torch.randn(4, 2, 6, requires_grad=True)
        output, distance, attention = module(z)
        p = (module.projection(z) / (F.softplus(module.theta_p) + 1e-6)).softmax(-1)
        phi = F.normalize(p.clamp_min(1e-12).sqrt(), dim=-1)
        expected = 2 * torch.acos((phi[:, 1] * phi[:, 0]).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6))
        torch.testing.assert_close(distance[:, 0], torch.zeros(4))
        torch.testing.assert_close(distance[:, 1], expected)
        torch.testing.assert_close(attention.sum(1), torch.ones(4))
        torch.testing.assert_close(output, z * (1 + F.softplus(module.eta_g) * attention[..., None]))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        one, d, a = module(torch.randn(2, 1, 6))
        torch.testing.assert_close(d, torch.zeros_like(d))
        torch.testing.assert_close(a, torch.ones_like(a))

    def test_angular_range_and_double_routing(self):
        module = OverlappingLocalGeometry(2, 2, 0.37)
        with torch.no_grad():
            module.centers.copy_(torch.tensor([[1., 0.], [-1., 0.]]))
            module.semantic_gate.weight.zero_()
            module.semantic_gate.bias.zero_()
            module.local_weight[0].copy_(torch.tensor([[2., 0.], [0., 1.]]))
            module.scale.copy_(torch.tensor([[1., 2.], [3., 4.]]))
        z = torch.tensor([[[1., 0.5], [0., 0.]]], requires_grad=True)
        output, routing, affinity = module(z)
        self.assertTrue((affinity.abs() <= 1 + 1e-6).all())
        torch.testing.assert_close(routing.sum(-1), torch.ones(1, 2))
        torch.testing.assert_close(routing, (0.37 * affinity).softmax(-1))
        linear = torch.einsum('bti,uoi->btuo', z, module.local_weight)
        expected = (F.gelu(linear * routing[..., None]) * routing[..., None]).sum(-2)
        expected = module.output(module.norm(expected * (routing @ module.scale)))
        torch.testing.assert_close(output, expected)
        output.sum().backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        _, _, opposite = module(torch.tensor([[[1., 0.]]]))
        torch.testing.assert_close(opposite, torch.tensor([[[1., -1.]]]))

    def test_joint_matches_independent_inference_and_private_gradients(self):
        cfg = toy_config()['model']
        model = MGOL({'a': {'input_dim': 4, 'num_classes': 2},
                      'b': {'input_dim': 7, 'num_classes': 3}}, cfg).eval()
        batches = {'a': torch.randn(3, 2, 4), 'b': torch.randn(5, 2, 7)}
        joint = model.forward_joint(batches)
        for domain in batches:
            torch.testing.assert_close(joint[domain][0], model(batches[domain], domain)[0], atol=1e-6, rtol=1e-5)
        model.train()
        model(batches['a'], 'a')[0].square().sum().backward()
        self.assertTrue(all(p.grad is None for p in model.encoders['b'].parameters()))
        self.assertTrue(all(p.grad is None for p in model.heads['b'].parameters()))
        self.assertIsInstance(model.encoders['a'].network[-1], nn.Linear)
        self.assertIsInstance(model.heads['a'].network[1], nn.BatchNorm1d)


class PreprocessingTests(unittest.TestCase):
    def test_disjoint_units_and_contiguous_block_fallback(self):
        cfg = toy_config()
        for grouped in [True, False]:
            spec = copy.deepcopy(cfg['domains']['toy'])
            if not grouped:
                spec.update(group_columns=[], block_size=18)
            split = assign_units(toy_frame(), spec, cfg['split'], cfg['seed'])
            for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
                self.assertFalse(set(split[a]['__unit__']) & set(split[b]['__unit__']))
                self.assertFalse(set(split[a]['__row__']) & set(split[b]['__row__']))
            self.assertEqual(sum(map(len, split.values())), len(toy_frame()))

    def test_train_only_transform_unknowns_and_roundtrip(self):
        frame = pd.DataFrame({'n': [1., np.nan, 3., np.inf], 'c': ['a', 'b', 'a', None]})
        transform = TabularTransform().fit(frame, ['n', 'c'])
        original = copy.deepcopy(transform.state_dict())
        heldout = pd.DataFrame({'n': [1000., np.inf], 'c': ['unseen', None]})
        got = transform.transform(heldout)
        self.assertEqual(original, transform.state_dict())
        self.assertTrue(np.isfinite(got).all())
        self.assertEqual(transform._encode(heldout)[0, 1], -1)
        restored = TabularTransform.from_state_dict(original)
        np.testing.assert_allclose(got, restored.transform(heldout))

    def test_sequences_never_cross_units_and_final_label(self):
        cfg = toy_config()
        spec = cfg['domains']['toy']
        bundle = prepare_domain(toy_frame(), spec, cfg)
        for split, dataset in bundle.datasets.items():
            self.assertEqual(len(dataset), bundle.split_summary[split]['units'] * 15)
            offset = 0
            for x, y in dataset.segments:
                for i in range(len(x) - cfg['model']['seq_len'] + 1):
                    got_x, got_y = dataset[offset + i]
                    self.assertEqual(got_y, y[i + 1])
                    np.testing.assert_allclose(got_x.numpy(), x[i:i + 2])
                offset += len(x) - 1
        inference = inference_dataset(toy_frame().drop(columns='Label'), bundle.preprocessing, 2)
        self.assertEqual(len(inference), 24 * 15)

    def test_raw_window_flattening_without_reconstruction(self):
        cfg = toy_config()
        spec = cfg['domains']['toy']
        spec['feature_columns'] = None
        bundle = prepare_domain(toy_frame(), spec, cfg)
        self.assertEqual(bundle.datasets['train'].input_dim, 3 * 2)
        parts = assign_units(toy_frame(), spec, cfg['split'], cfg['seed'])
        transform = TabularTransform.from_state_dict(bundle.preprocessing['tabular'])
        first_unit = next(iter(parts['train'].groupby('__unit__', sort=False)))[1]
        expected = transform.transform(first_unit)[:3].reshape(-1)
        np.testing.assert_allclose(bundle.datasets['train'][0][0][0].numpy(), expected)
        self.assertEqual(transform.columns, ['x', 'v'])

    def test_impossible_split_and_bad_config_fail(self):
        cfg = toy_config()
        validate_config(cfg)
        bad = copy.deepcopy(cfg)
        bad['model']['latent_dim'] = 5
        with self.assertRaises(ValueError):
            validate_config(bad)
        frame = toy_frame()
        frame['run'] = 0
        with self.assertRaises(ValueError):
            assign_units(frame, cfg['domains']['toy'], cfg['split'], 7)


class TrainingTests(unittest.TestCase):
    def test_metrics_security_fpr(self):
        metrics = classification_metrics([0, 0, 1, 1], [1, 0, 0, 1], 2, 0)
        self.assertEqual(metrics['fpr'], 0.5)
        self.assertEqual(metrics['accuracy'], 0.5)
        self.assertEqual(classification_metrics([1], [1], 2, 0)['fpr'], None)

    def test_two_stage_checkpoint_and_inference(self):
        torch.set_num_threads(1)
        cfg = toy_config()
        seed_everything(cfg['seed'])
        spec = cfg['domains']['toy']
        a = prepare_domain(toy_frame(), spec, cfg)
        raw_spec = copy.deepcopy(spec)
        raw_spec['feature_columns'] = ['x']
        b = prepare_domain(toy_frame(), raw_spec, cfg)
        data = {'a': a, 'b': b}
        info = {d: {'input_dim': v.datasets['train'].input_dim, 'num_classes': len(v.classes)}
                for d, v in data.items()}
        model = MGOL(info, cfg['model'])
        with tempfile.TemporaryDirectory() as tmp:
            result = train(model, data, info, cfg, torch.device('cpu'), tmp)
            self.assertEqual(set(result), {'a', 'b'})
            saved = torch.load(Path(tmp) / 'finetune_best.pt', weights_only=True)
            loaded = MGOL(saved['domain_info'], saved['model_config']).eval()
            loaded.load_state_dict(saved['state_dict'])
            x = torch.stack([a.datasets['test'][0][0], a.datasets['test'][1][0]])
            model.eval()
            torch.testing.assert_close(model(x, 'a')[0], loaded(x, 'a')[0])


if __name__ == '__main__':
    unittest.main()
