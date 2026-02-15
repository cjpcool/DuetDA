"""Centralized default configuration for CGCNN training.

This module mirrors the argparse defaults defined in cgcnn/main.py so other
scripts can import a single source of truth for the canonical hyperparameters.
"""

from dataclasses import dataclass, asdict, replace
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class CGCNNConfig:
	workers: int = 0
	epochs: int = 50
	start_epoch: int = 0
	batch_size: int = 256
	lr: float = 0.01
	lr_milestones: Tuple[int, ...] = (100,)
	momentum: float = 0.9
	weight_decay: float = 0.0
	print_freq: int = 2
	resume: str = ''
	use_wandb: bool = False
	wandb_project: str = 'cgcnn'
	wandb_entity: Optional[str] = None
	wandb_mode: str = 'online'
	wandb_name: Optional[str] = None
	train_ratio: Optional[float] = None
	train_size: Optional[int] = None
	val_ratio: float = 0.1
	val_size: Optional[int] = None
	test_ratio: float = 0.1
	test_size: Optional[int] = None
	optim: str = 'SGD'
 
	orig_atom_fea_len: int = 92
	nbr_fea_len: int = 41
	atom_fea_len: int = 64
	h_fea_len: int = 128
	n_conv: int = 3
	n_h: int = 1
	seed: int = 42

	def to_dict(self) -> Dict[str, Any]:
		"""Return the configuration as a plain dictionary."""
		return asdict(self)

	def override(self, **overrides: Any) -> 'CGCNNConfig':
		"""Return a copy with specific fields replaced."""
		return replace(self, **overrides)


CGCNN_DEFAULTS = CGCNNConfig()


def build_cgcnn_config(overrides: Optional[Dict[str, Any]] = None) -> CGCNNConfig:
	"""Create a CGCNNConfig, optionally overriding selected fields."""
	if not overrides:
		return CGCNN_DEFAULTS
	return CGCNN_DEFAULTS.override(**overrides)
