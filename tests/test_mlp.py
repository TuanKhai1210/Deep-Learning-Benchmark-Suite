import unittest

import torch
from torch import nn

from dlbench.a1.metrics import count_parameters
from dlbench.a1.models.mlp import MLPClassifier


class TestMLPClassifier(unittest.TestCase):
	@staticmethod
	def _parameters(**overrides: object) -> dict[str, object]:
		parameters: dict[str, object] = {
			"input_dim": 784,
			"num_classes": 10,
			"hidden_dims": [16, 8],
			"hidden_activation": "relu",
			"dropout": 0.0,
		}
		parameters.update(overrides)
		return parameters

	def test_forward_accepts_image_batches_and_returns_logits(self) -> None:
		model = MLPClassifier(self._parameters()).eval()

		for batch_size in (1, 7):
			with self.subTest(batch_size=batch_size):
				images = torch.randn(batch_size, 1, 28, 28)
				logits = model(images)

				self.assertEqual(tuple(logits.shape), (batch_size, 10))
				self.assertTrue(torch.isfinite(logits).all().item())

	def test_flattening_preserves_results_for_equivalent_inputs(self) -> None:
		model = MLPClassifier(self._parameters()).eval()
		flat_images = torch.randn(3, 784)

		with torch.inference_mode():
			flat_logits = model(flat_images)
			image_logits = model(flat_images.reshape(3, 1, 28, 28))

		self.assertTrue(torch.equal(flat_logits, image_logits))

	def test_linear_layers_have_expected_dimensions(self) -> None:
		model = MLPClassifier(
			self._parameters(input_dim=12, hidden_dims=[7, 5], num_classes=3)
		)
		linear_layers = [layer for layer in model.network if isinstance(layer, nn.Linear)]

		self.assertEqual(
			[(layer.in_features, layer.out_features) for layer in linear_layers],
			[(12, 7), (7, 5), (5, 3)],
		)

	def test_parameter_count_matches_all_linear_weights_and_biases(self) -> None:
		model = MLPClassifier(
			self._parameters(input_dim=12, hidden_dims=[7, 5], num_classes=3)
		)

		# (12*7 + 7) + (7*5 + 5) + (5*3 + 3)
		self.assertEqual(count_parameters(model), {
			"total_parameters": 149,
			"trainable_parameters": 149,
		})

	def test_dropout_does_not_change_parameter_count(self) -> None:
		without_dropout = MLPClassifier(self._parameters(dropout=0.0))
		with_dropout = MLPClassifier(self._parameters(dropout=0.5))

		self.assertEqual(
			count_parameters(without_dropout), count_parameters(with_dropout)
		)

	def test_single_hidden_layer_builds_without_dropout_by_default(self) -> None:
		model = MLPClassifier(
			{"input_dim": 4, "num_classes": 2, "hidden_dims": [3], "hidden_activation": "gelu"}
		)

		self.assertEqual(sum(isinstance(layer, nn.Dropout) for layer in model.network), 0)
		self.assertEqual(tuple(model(torch.randn(5, 4)).shape), (5, 2))

	def test_network_has_expected_layers_and_dropout(self) -> None:
		model = MLPClassifier(
			self._parameters(hidden_dims=[16, 8], dropout=0.25)
		)

		self.assertIsInstance(model.network[0], nn.Flatten)
		self.assertIsInstance(model.network[1], nn.Linear)
		self.assertEqual(model.network[1].in_features, 784)
		self.assertEqual(model.network[1].out_features, 16)
		self.assertIsInstance(model.network[2], nn.ReLU)
		self.assertIsInstance(model.network[3], nn.Dropout)
		self.assertEqual(model.network[3].p, 0.25)
		self.assertEqual(model.network[4].out_features, 8)
		self.assertIsInstance(model.network[-1], nn.Linear)
		self.assertEqual(model.network[-1].in_features, 8)
		self.assertEqual(model.network[-1].out_features, 10)

	def test_positive_dropout_is_applied_in_train_mode_only(self) -> None:
		model = MLPClassifier(self._parameters(dropout=0.5))
		images = torch.ones(16, 1, 28, 28)

		model.eval()
		eval_first = model(images)
		eval_second = model(images)
		self.assertTrue(torch.equal(eval_first, eval_second))

		model.train()
		torch.manual_seed(0)
		train_first = model(images)
		train_second = model(images)
		self.assertFalse(torch.equal(train_first, train_second))

	def test_backward_updates_parameters(self) -> None:
		model = MLPClassifier(self._parameters()).train()
		optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
		images = torch.randn(7, 1, 28, 28)
		targets = torch.arange(7)
		before = [parameter.detach().clone() for parameter in model.parameters()]

		optimizer.zero_grad()
		loss = torch.nn.functional.cross_entropy(model(images), targets)
		self.assertTrue(torch.isfinite(loss).item())
		loss.backward()
		optimizer.step()

		self.assertTrue(
			any(
				not torch.equal(old, new.detach())
				for old, new in zip(before, model.parameters())
			)
		)

	def test_supported_hidden_activations_construct(self) -> None:
		for activation in ("tanh", "sigmoid", "relu", "leakyrelu", "gelu"):
			with self.subTest(activation=activation):
				model = MLPClassifier(self._parameters(hidden_activation=activation))
				self.assertEqual(tuple(model(torch.randn(2, 1, 28, 28)).shape), (2, 10))

	def test_missing_required_parameters_raise(self) -> None:
		required = self._parameters()
		for parameter in ("input_dim", "num_classes", "hidden_dims", "hidden_activation"):
			with self.subTest(parameter=parameter):
				parameters = required.copy()
				del parameters[parameter]
				with self.assertRaises(ValueError):
					MLPClassifier(parameters)

	def test_invalid_configuration_raises(self) -> None:
		with self.assertRaises(TypeError):
			MLPClassifier(self._parameters(hidden_dims=(16, 8)))

		with self.assertRaises(ValueError):
			MLPClassifier(self._parameters(hidden_activation="unknown"))


if __name__ == "__main__":
	unittest.main()
