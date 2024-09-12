"""
Main ProtoCBM model implementation
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import load, nn
from torchvision import models
from networks import PPNet, ProtoPool 

from utils.resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features
from utils.densenet_features import densenet121_features, densenet161_features, densenet169_features, densenet201_features
from utils.vgg_features import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, vgg16_bn_features,\
						 vgg19_features, vgg19_bn_features

from utils.proto_models import compute_proto_layer_rf_info_v2

base_architecture_to_features = {'resnet18': resnet18_features,
								 'resnet34': resnet34_features,
								 'resnet50': resnet50_features,
								 'resnet101': resnet101_features,
								 'resnet152': resnet152_features,
								 'densenet121': densenet121_features,
								 'densenet161': densenet161_features,
								 'densenet169': densenet169_features,
								 'densenet201': densenet201_features,
								 'vgg11': vgg11_features,
								 'vgg11_bn': vgg11_bn_features,
								 'vgg13': vgg13_features,
								 'vgg13_bn': vgg13_bn_features,
								 'vgg16': vgg16_features,
								 'vgg16_bn': vgg16_bn_features,
								 'vgg19': vgg19_features,
								 'vgg19_bn': vgg19_bn_features}

class SequentialModified(nn.Sequential):
	def forward(self, x):
		output = super().forward(x)
		return output, None, None

class ProtoCBM(nn.Module):
	"""
	Vanilla CBM + Prototype Networks implementation (ProtoPNet and ProtoPool)
	"""
	
	def __init__(self, config, device='cuda'):
		super(ProtoCBM, self).__init__()

		# Configuration arguments
		self.num_concepts = config['num_concepts']
		self.num_classes = config['num_classes']

		self.head_arch = config['head_arch'] # Architecture of concept predictor
		self.encoder_arch = config['encoder_arch']

		self.prototype_mode = config['prototype_mode']
		self.training_mode = config['training_mode']

		self.prototype_shape=config['prototype_shape']

		if self.prototype_mode == 'cbm':
			if self.encoder_arch == 'resnet18':
				encoder_res = models.resnet18(weights=None)
				encoder_res.load_state_dict(
					torch.load(os.path.join(config['model_dir'], 'resnet18-5c106cde.pth')))

				n_features = encoder_res.fc.in_features
				encoder_res.fc = nn.Identity()
				projector = nn.Sequential(nn.Linear(n_features, self.num_concepts, bias=True))
				self.encoder = SequentialModified(encoder_res, projector)	
			else:
				raise ValueError('ERROR: CBM encoder architecture not supported!')

		elif self.prototype_mode == 'ppnet':
			# Construct PPNet
			features = base_architecture_to_features[self.encoder_arch](pretrained=True)
			layer_filter_sizes, layer_strides, layer_paddings = features.conv_info()
			proto_layer_rf_info = compute_proto_layer_rf_info_v2(img_size=config['img_size'],
														layer_filter_sizes=layer_filter_sizes,
														layer_strides=layer_strides,
														layer_paddings=layer_paddings,
														prototype_kernel_size=self.prototype_shape[2])

			self.encoder = PPNet(features=features,
							img_size=config['img_size'],
							prototype_shape=self.prototype_shape,
							proto_layer_rf_info=proto_layer_rf_info,
							num_classes=self.num_concepts, # NOTE!!! what we called num_classes in CG-PPNet is num_concepts here
							init_weights=True,
							prototype_activation_function=config['prototype_activation_function'],
							add_on_layers_type=config['add_on_layers_type'],
							device=device)
						
			if config['prototype_pretrained']:
				checkpoint = torch.load(config['pretrained_dir'], map_location=device)
				self.encoder.load_state_dict(checkpoint['model_state_dict'])

		elif self.prototype_mode == 'ppool':
			# Construct ProtoPools
			self.encoder = ProtoPool(
					num_prototypes=config['num_prototypes'],
					num_descriptive=config['num_descriptive'],
					num_classes=self.num_concepts, # NOTE!!! what we called num_classes in ProtoPool is num_concepts here
					use_thresh=config['use_thresh'],
					arch=config['encoder_arch'],
					pretrained=config['pretrained'],
					add_on_layers_type=config['add_on_layers_type'],
					prototype_activation_function=config['prototype_activation_function'],
					proto_depth=config['proto_depth'],
					use_last_layer=config['last_layer'],
					inat=config['inat'],
					device=device
				)

			if config['prototype_pretrained']:
				# Load pretrained PPNet or ProtoPool model
				checkpoint = torch.load(config['pretrained_dir'], map_location=device)
				self.encoder.load_state_dict(checkpoint['model_state_dict'])
		
		else:
			raise ValueError('ERROR: MODE NOT SUPPORTED')

		# Assume binary concepts (sigmoid to get predictions)
		self.act_c = nn.Sigmoid()

		# Common target/label predictor for all training modes
		# Link function g(.)
		if self.num_classes == 2:
			pred_dim = 1
		elif self.num_classes > 2:
			pred_dim = self.num_classes

		if self.head_arch == 'linear':
			fc_y = nn.Linear(self.num_concepts, pred_dim)
			self.head = nn.Sequential(fc_y)
		elif self.head_arch == 'nonlinear':
			fc1_y = nn.Linear(self.num_concepts, 256)
			fc2_y = nn.Linear(256, pred_dim)
			self.head = nn.Sequential(fc1_y, nn.ReLU(), fc2_y)
		else:
			raise ValueError('ERROR: CBM head architecture not supported!')

	def forward(self, x, c_true=None, gumbel_scalar=None):
		"""
		Forward pass

		:param x: covariates
		:param c_true: true concepts
		:param gumbel_scalar: for ProtoPool assignment
		:return: predicted concepts, probabilities and logits for the target variable
		"""
		
		# Get intermediate representations
		if self.prototype_mode == 'ppool':
			c_pred_logits, min_distances, proto_presence = self.encoder(x, gumbel_scalar)
		else:
			c_pred_logits, min_distances, proto_presence = self.encoder(x)
		
		c_pred_probs = self.act_c(c_pred_logits)

		# Get predicted targets
		assert c_true.shape == c_pred_probs.shape

		if self.training:
			if self.training_mode == 'independent':
				y_pred_logits = self.head(c_true.float())
			elif self.training_mode in ('sequential', 'joint'):
				y_pred_logits = self.head(c_pred_probs)
			else:
				raise ValueError
		else:
			y_pred_logits = self.head(c_pred_probs)

		return c_pred_logits, min_distances, proto_presence, y_pred_logits