import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from torchmetrics import Metric
import wandb
import logging

from utils.metrics import calc_target_metrics, calc_concept_metrics
from networks import ProtoPool, PPNet

def train_one_epoch_cbm(train_loader, model, optimizer, mode, metrics, epoch, config, loss_fn, device):
	"""
	Train for one epoch for standard CBM
	@param train_loader: DataLoader object for training data
	@param model: Model
	@param optimizer: Optimizer
	@param mode: 'j' for joint, 'c' for concept, 't' for target
	@param metrics: Custom_Metrics object
	@param epoch: Current epoch
	@param config: Config dictionary
	@param loss_fn: Loss function
	@param device: Device
	"""
	
	model.train()
	metrics.reset()

	if config['training_mode'] == 'sequential':
		if mode == 'c':
			model.head.eval()
		elif mode == 't':
			model.encoder.eval()

	for k, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}', position=0, leave=True)):
		batch_features, target_true = batch['features'].to(device), batch['labels'].to(device) 
		concepts_true = batch['concepts'].to(device)
		
		# Forward pass
		concepts_pred_logits, _, _, target_pred_logits = model(batch_features, concepts_true)

		# Backward pass depends on the training mode of the model
		optimizer.zero_grad()
		# Compute the loss
		target_loss, concepts_loss, summed_concepts_loss, total_loss = loss_fn(
			concepts_pred_logits, concepts_true, target_pred_logits, target_true)

		if mode == 'j':
			total_loss.backward()
		elif mode == 'c':
			summed_concepts_loss.backward()
		elif mode == 't':
			target_loss.backward()
		else:
			raise ValueError
		optimizer.step()  # perform an update

		# Store predictions
		concepts_pred_probs = F.sigmoid(concepts_pred_logits)
		metrics.update(target_loss, concepts_loss, summed_concepts_loss, total_loss,
						0, 0, 0, 0, 0, 0, # Not used in CBM
						target_true, target_pred_logits, concepts_true, concepts_pred_probs)

	# Calculate and log metrics
	metrics_dict = metrics.compute()
	wandb.log({f'train/{k}': v for k, v in metrics_dict.items()})
	prints = f"Epoch {epoch + 1}, Train     : "
	for key, value in metrics_dict.items():
		if key == 'concepts_loss':
			continue
		prints += f"{key}: {value:.3f} "
	logging.info(prints)
	metrics.reset()
	return


def validate_one_epoch_cbm(loader, model, metrics, epoch, config, loss_fn, device, test=False):
	"""
	Validating one epoch for standard CBM
	@param loader: DataLoader object for validation data
	@param model: Model
	@param metrics: Custom_Metrics object
	@param epoch: Current epoch
	@param config: Config dictionary
	@param loss_fn: Loss function
	@param device: Device
	@param test: Boolean, whether the function is called for testing
	"""
	model.eval()

	with torch.no_grad():

		for k, batch in enumerate(loader):
			batch_features, target_true = batch['features'].to(device), batch['labels'].to(device) 
			concepts_true = batch['concepts'].to(device)

			concepts_pred_logits, _, _, target_pred_logits = model(batch_features, concepts_true)

			target_loss, concepts_loss, summed_concepts_loss, total_loss = loss_fn(
				concepts_pred_logits, concepts_true, target_pred_logits, target_true)
			
			# Store predictions
			concepts_pred_probs = F.sigmoid(concepts_pred_logits)
			metrics.update(target_loss, concepts_loss, summed_concepts_loss,
						0, 0, 0, 0, 0, 0, # Not used in CBM
						total_loss, target_true, target_pred_logits, concepts_true, concepts_pred_probs)

	# Calculate and log metrics
	metrics_dict = metrics.compute(validation=True, config=config)
	if not test:
		wandb.log({f'validation/{k}': v for k, v in metrics_dict.items()})
		prints = f"Epoch {epoch}, Validation: "
	else:
		wandb.log({f'test/{k}': v for k, v in metrics_dict.items()})
		prints = f"Test: "
	for key, value in metrics_dict.items():
		if key == 'concepts_loss':
			continue
		prints += f"{key}: {value:.3f} "
	logging.info(prints)
	metrics.reset()
	return

def train_cbm(config, model, epochs, mode, train_loader, val_loader, metrics, loss_fn, device):
	"""
	Wrapper function for training and testing a standard CBM model
	@param config: Config dictionary
	@param model: Model
	@param epochs: Number of epochs
	@param mode: 'j' for joint, 'c' for concept, 't' for target
	@param train_loader: DataLoader object for training data
	@param val_loader: DataLoader object for validation data
	@param metrics: Custom_Metrics object
	@param loss_fn: Loss function
	@param device: Device
	"""
	cbm_optimizer = create_optimizer(config, model)
	lr_scheduler = optim.lr_scheduler.StepLR(cbm_optimizer, step_size=config['decrease_every'],gamma=1/config['lr_divisor'])

	for epoch in range(epochs):
		# Validate the model periodically
		if epoch % config['validate_per_epoch'] == 0:
			logging.info("EVALUATION ON THE VALIDATION SET:")
			validate_one_epoch_cbm(val_loader, model, metrics, epoch, config, loss_fn, device)
		train_one_epoch_cbm(train_loader, model, cbm_optimizer, mode, metrics, epoch, config, loss_fn, device)
		lr_scheduler.step()

def train_or_test_(model, dataloader, device, loss_fn, mode, metrics, epoch, config, optimizer=None, gumbel_scalar=None, test=False):
	'''
	Wrapper function for training and testing a prototype model
	@param model: Model
	@param dataloader: DataLoader object
	@param device: Device
	@param loss_fn: Loss function
	@param mode: 'j' for joint, 'c' for concept, 't' for target
	@param metrics: Custom_Metrics object
	@param epoch: Current epoch
	@param config: Config dictionary
	@param optimizer: Optimizer
	@param gumbel_scalar: Gumbel scalar for PPools
	@param test: Boolean, whether the function is called for testing
    '''

	is_train = optimizer is not None
	
	for k, batch in enumerate(tqdm(dataloader, leave=True)):
		batch_features = batch['features'].to(device)
		concepts_true = batch['concepts'].to(torch.float).to(device)
		target_true = batch['labels'].to(device)
		
		grad_req = torch.enable_grad() if is_train else torch.no_grad()
		# compute the loss
		with grad_req:
			if isinstance(model.module.encoder, PPNet):
				concepts_pred_logits, min_distances, proto_presence, target_pred_logits = model(batch_features, c_true=concepts_true)
			else:
				concepts_pred_logits, min_distances, proto_presence, target_pred_logits = model(batch_features, c_true=concepts_true, gumbel_scalar=gumbel_scalar)

			target_loss, concepts_loss, summed_concepts_loss, total_loss, cross_entropy, cluster_cost, separation_cost, l1, orthogonal_loss_p, orthogonal_loss_c = loss_fn(
				concepts_pred_logits, concepts_true, min_distances, proto_presence, target_pred_logits, target_true, model)

        # compute gradient and do SGD step
		if is_train:
			optimizer.zero_grad()
			if mode == 'j':
				total_loss.backward()
			elif mode == 'c':
				summed_concepts_loss.backward()
			elif mode == 't':
				target_loss.backward()
			else:
				raise ValueError
			optimizer.step()

		# get predicted probabilities from logits
		concepts_pred_probs = F.sigmoid(concepts_pred_logits)

		metrics.update(target_loss, concepts_loss, summed_concepts_loss, total_loss, 
						cross_entropy, cluster_cost, separation_cost, l1, orthogonal_loss_p, orthogonal_loss_c,
						target_true, target_pred_logits, concepts_true, concepts_pred_probs)

	# Calculate and log metrics
	if is_train:
		metrics_dict = metrics.compute()
		wandb.log({f'train/{k}': v for k, v in metrics_dict.items()})
		prints = f"Epoch {epoch}, Train     : "
	else:
		metrics_dict = metrics.compute(validation=True, config=config)
		if test:
			wandb.log({f'test/{k}': v for k, v in metrics_dict.items()})
			prints = f"Test: "
		else:
			prints = f"Epoch {epoch}, Validation: "
			wandb.log({f'validation/{k}': v for k, v in metrics_dict.items()})

	for key, value in metrics_dict.items():
		if key == 'concepts_loss':
			continue
		prints += f"{key}: {value:.3f} "

	logging.info(prints)
	metrics.reset()

def train_proto(model, dataloader, device, loss_fn, mode, metrics, epoch, config, optimizer, gumbel_scalar=None): 
    assert(optimizer is not None)
    model.train()
    train_or_test_(
		model=model, dataloader=dataloader, device=device, loss_fn=loss_fn, mode=mode, metrics=metrics, epoch=epoch, config=config, 
		optimizer=optimizer, gumbel_scalar=gumbel_scalar)

def test_proto(model, dataloader, device, loss_fn, mode, metrics, epoch, config, gumbel_scalar=None, test=False): 
    model.eval()
    train_or_test_(
		model=model, dataloader=dataloader, device=device, loss_fn=loss_fn, mode=mode, metrics=metrics, epoch=epoch, 
		config=config, optimizer=None, gumbel_scalar=gumbel_scalar, test=test)

def last_only(model): 
	# Freeze all layers except the last layer
	logging.info('last layer prototype training')
	for p in model.module.encoder.features.parameters():
		p.requires_grad = False
	for p in model.module.encoder.add_on_layers.parameters():
		p.requires_grad = False
	model.module.encoder.prototype_vectors.requires_grad = False
	for p in model.module.encoder.last_layer.parameters():
		p.requires_grad = True
	if isinstance(model.module.encoder, ProtoPool):
		model.module.encoder.proto_presence.requires_grad = False

def warm_only(model): 
	# Freeze all layers except the last layer and the add-on layers
	logging.info('warm prototype training')
	for p in model.module.encoder.features.parameters():
		p.requires_grad = False
	for p in model.module.encoder.add_on_layers.parameters():
		p.requires_grad = True
	model.module.encoder.prototype_vectors.requires_grad = True
	for p in model.module.encoder.last_layer.parameters():
		p.requires_grad = True 
	if isinstance(model.module.encoder, ProtoPool):
		model.module.encoder.proto_presence.requires_grad = True

def joint(model): 
	# Unfreeze all layers
	logging.info('joint prototype training')
	for p in model.module.encoder.features.parameters():
		p.requires_grad = True
	for p in model.module.encoder.add_on_layers.parameters():
		p.requires_grad = True
	model.module.encoder.prototype_vectors.requires_grad = True
	for p in model.module.encoder.last_layer.parameters():
		p.requires_grad = True
	if isinstance(model.module.encoder, ProtoPool):
		model.module.encoder.proto_presence.requires_grad = True

def create_optimizer(config, model, joint_with_target=False):
	"""
	Parse the configuration file and return a relevant optimizer object
	@param config: Config dictionary
	@param model: Model
	@param joint_with_target: Boolean, whether to jointly optimize with target
	@return: Optimizer object
	"""
	assert config['optimizer'] in ['sgd', 'adam'], 'Only SGD and Adam optimizers are available!'

	# For joint optimization in prototype models
	if joint_with_target:
		target_optimizer_specs = [{'params': model.head.parameters(), 'lr': config['learning_rate'] / 10,
			'weight_decay': config['weight_decay']}]
	else:
		target_optimizer_specs = []

	if config['prototype_mode'] == 'cbm':
		optim_params = [
			{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': config['learning_rate'],
			'weight_decay': config['weight_decay']}
		]

		if config['optimizer'] == 'sgd':
			return torch.optim.SGD(optim_params)
		elif config['optimizer'] == 'adam':
			return torch.optim.Adam(optim_params)

	elif config['prototype_mode'] == 'ppnet':
		joint_optimizer_specs = \
			[{'params': model.encoder.features.parameters(), 'lr': config['joint_optimizer_lrs']['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
			{'params': model.encoder.add_on_layers.parameters(), 'lr': config['joint_optimizer_lrs']['add_on_layers'], 'weight_decay': 1e-3},
			{'params': model.encoder.prototype_vectors, 'lr': config['joint_optimizer_lrs']['prototype_vectors']},
			] + target_optimizer_specs
		joint_optimizer = torch.optim.Adam(joint_optimizer_specs)

		warm_optimizer_specs = \
			[{'params': model.encoder.add_on_layers.parameters(), 'lr': config['warm_optimizer_lrs']['add_on_layers'], 'weight_decay': 1e-3},
			{'params': model.encoder.prototype_vectors, 'lr': config['warm_optimizer_lrs']['prototype_vectors']},
			] + target_optimizer_specs

		warm_optimizer = torch.optim.Adam(warm_optimizer_specs)
		
		last_layer_optimizer_specs = [{'params': model.encoder.last_layer.parameters(), 'lr': float(config['last_layer_optimizer_lr'])}] + target_optimizer_specs
		last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

		return {'joint': joint_optimizer, 'warm': warm_optimizer, 'last_layer': last_layer_optimizer}

	elif config['prototype_mode'] == 'ppool':
		warm_optimizer = torch.optim.Adam(
        	[{'params': model.encoder.add_on_layers.parameters(), 'lr': config['warm_optimizer_lrs']['add_on_layers'], 'weight_decay': 1e-3},
         	{'params': model.encoder.proto_presence, 'lr': config['warm_optimizer_lrs']['proto_presence']},
         	{'params': model.encoder.prototype_vectors, 'lr': config['warm_optimizer_lrs']['prototype_vectors']}] + target_optimizer_specs
		)

		joint_optimizer = torch.optim.Adam(
			[{'params': model.encoder.features.parameters(), 'lr': config['joint_optimizer_lrs']['features'], 'weight_decay': 1e-3},
			{'params': model.encoder.add_on_layers.parameters(), 'lr': config['joint_optimizer_lrs']['add_on_layers'], 'weight_decay': 1e-3},
			{'params': model.encoder.proto_presence, 'lr': config['joint_optimizer_lrs']['proto_presence']},
			{'params': model.encoder.prototype_vectors, 'lr': config['joint_optimizer_lrs']['prototype_vectors']}] + target_optimizer_specs
		)

		last_layer_optimizer = torch.optim.Adam(
			[{'params': model.encoder.last_layer.parameters(), 'lr': float(config['last_layer_optimizer_lr']), 'weight_decay': 1e-3}] + target_optimizer_specs
		)

		return {'joint': joint_optimizer, 'warm': warm_optimizer, 'last_layer': last_layer_optimizer}

def create_proto_target_optimizer(config, model):
	"""
	Parse the configuration file and return a relevant optimizer object for the target
	@param config: Config dictionary
	@param model: Model
	@return: Optimizer object
	"""
	optim_params = [
			{'params': model.module.head.parameters(), 'lr': config['learning_rate'],
			'weight_decay': config['weight_decay']}
		]

	return torch.optim.Adam(optim_params)

class Custom_Metrics(Metric):
	"""
	Custom metric class for storing and computing metrics
	"""
	def __init__(self, config, device):
		super().__init__()
		self.n_concepts = config['num_concepts']
		self.prototype_mode = config['prototype_mode']
		self.add_state("target_loss", default=torch.tensor(0., device=device))
		self.add_state("concepts_loss", default=torch.tensor([0.] * self.n_concepts, device=device))
		self.add_state("summed_concepts_loss", default=torch.tensor(0., device=device))
		self.add_state("total_loss", default=torch.tensor(0., device=device))
		self.add_state("y_true", default=[])
		self.add_state("y_pred_logits", default=[])
		self.add_state("c_true", default=[])
		self.add_state("c_pred_probs", default=[])
		self.add_state("n_samples", default=torch.tensor(0, dtype=torch.int, device=device))

		self.add_state("n_tp_batch", default=torch.tensor([0.] * self.n_concepts, device=device))
		self.add_state("n_tp_fp_batch", default=torch.tensor([0.] * self.n_concepts, device=device))
		self.add_state("n_tp_fn_batch", default=torch.tensor([0.] * self.n_concepts, device=device))
		self.add_state("n_instance_batch", default=torch.tensor(0., device=device))
		self.add_state("n_tp_total", default=torch.tensor(0., device=device))
		self.add_state("n_tp_fn_total", default=torch.tensor(0., device=device))
		self.add_state("n_tp_fp_total", default=torch.tensor(0., device=device))
		self.add_state("total_cross_entropy", default=torch.tensor(0., device=device))
		self.add_state("total_cluster_cost", default=torch.tensor(0., device=device))
		self.add_state("total_separation_cost", default=torch.tensor(0., device=device))
		self.add_state("l1", default=torch.tensor(0., device=device))
		self.add_state("orthogonal_loss_p", default=torch.tensor(0., device=device))
		self.add_state("orthogonal_loss_c", default=torch.tensor(0., device=device))

	def update(self, target_loss: torch.Tensor, concepts_loss: torch.Tensor, summed_concepts_loss: torch.Tensor, total_loss: torch.Tensor, 
				cross_entropy: torch.Tensor, cluster_cost: torch.Tensor, separation_cost: torch.Tensor, l1: torch.Tensor, 
				orthogonal_loss_p: torch.Tensor, orthogonal_loss_c: torch.Tensor, 
				y_true: torch.Tensor, y_pred_logits: torch.Tensor, c_true: torch.Tensor, c_pred_probs: torch.Tensor):
		assert c_true.shape == c_pred_probs.shape

		n_samples = y_true.size(0)
		self.n_samples += n_samples
		self.target_loss += target_loss * n_samples
		for concept_idx in range(self.n_concepts):
			self.concepts_loss[concept_idx] += concepts_loss[concept_idx] * n_samples
		self.summed_concepts_loss += summed_concepts_loss * n_samples
		self.total_loss += total_loss * n_samples
		self.y_true.append(y_true)
		self.y_pred_logits.append(y_pred_logits.detach())
		self.c_true.append(c_true)
		self.c_pred_probs.append(c_pred_probs.detach())

		c_predicted = (c_pred_probs > 0.5).float()

		# macro-averaged F1
		self.n_tp_batch += torch.sum(c_predicted*c_true, axis = 0)            
		self.n_tp_fn_batch += torch.sum(c_true, axis = 0)
		self.n_tp_fp_batch += torch.sum(c_predicted, axis = 0)
		
		# instance-averaged F1
		self.n_instance_batch += torch.sum(2*torch.sum(c_predicted*c_true, axis = 1)/(torch.sum(c_true, axis = 1) + torch.sum(c_predicted, axis = 1))).item()

		# micro-averaged F1
		self.n_tp_total += torch.sum(c_predicted*c_true).item()
		self.n_tp_fn_total += torch.sum(c_true).item()
		self.n_tp_fp_total += torch.sum(c_predicted).item()

		if self.prototype_mode == 'ppnet':    	
			self.total_cross_entropy += cross_entropy.item() * n_samples
			self.total_cluster_cost += cluster_cost.item() * n_samples
			self.total_separation_cost += separation_cost.item() * n_samples
			self.l1 += l1 * n_samples
			if self.prototype_mode == 'ppool':
				self.orthogonal_loss_p += orthogonal_loss_p * n_samples
				self.orthogonal_loss_c += orthogonal_loss_c * n_samples

	def compute(self, validation=False, config=None):
		self.y_true = torch.cat(self.y_true, dim=0).cpu()
		self.y_pred_logits = torch.cat(self.y_pred_logits, dim=0).cpu()
		self.c_true = torch.cat(self.c_true, dim=0).cpu().numpy()
		self.c_pred_probs = torch.cat(self.c_pred_probs, dim=0).cpu().numpy()
		c_pred = self.c_pred_probs>0.5
		if self.y_pred_logits.ndim==1:
			y_pred_probs = nn.Sigmoid()(self.y_pred_logits) 
			y_pred = self.y_pred_logits>0
		else:
			y_pred_probs = nn.Softmax(dim=1)(self.y_pred_logits)
			y_pred = self.y_pred_logits.argmax(dim=-1)

		target_acc = (self.y_true==y_pred).sum()/self.n_samples
		concept_acc = (self.c_true==c_pred).sum()/(self.n_samples*self.n_concepts)
		concept_hamming = (self.c_true!=c_pred).sum()/(self.n_samples*self.n_concepts)

		concept_macro_f1 = (1/self.n_concepts)*torch.sum((2*self.n_tp_batch/(self.n_tp_fn_batch + self.n_tp_fp_batch))).item()
		concept_instance_f1 = self.n_instance_batch / self.n_samples
		concept_micro_f1 = 2*self.n_tp_total / (self.n_tp_fn_total + self.n_tp_fp_total)


		metrics = dict({'target_loss': self.target_loss / self.n_samples, 'concepts_loss': self.concepts_loss / self.n_samples,
						'summed_concepts_loss': self.summed_concepts_loss / self.n_samples, 'total_loss': self.total_loss / self.n_samples, 
						'concept f1 macro': concept_macro_f1, 'concept f1 micro': concept_micro_f1, 'concept f1 instance': concept_instance_f1,
						'label accuracy': target_acc, 'concept accuracy': concept_acc, 'concept hamming loss': concept_hamming})

		if self.prototype_mode in ('ppnet', 'ppool'):
			metrics = metrics | {'cross entropy': self.total_cross_entropy / self.n_samples, 'cluster cost': self.total_cluster_cost / self.n_samples, 
								'separation cost': self.total_separation_cost / self.n_samples, 'l1 loss': self.l1 / self.n_samples, 
								'orthogonal loss within prototypes': self.orthogonal_loss_p / self.n_samples, 
								'orthogonal loss positive vs negative': self.orthogonal_loss_c / self.n_samples}
		
		if validation is True:
			c_pred_probs = []
			# TODO: Check if necessary, i.e. diff structure
			for j in range(self.n_concepts):
				c_pred_probs.append(
					np.hstack((np.expand_dims(1 - self.c_pred_probs[:, j], 1), np.expand_dims(self.c_pred_probs[:, j], 1))))

			y_metrics = calc_target_metrics(self.y_true.numpy(), y_pred_probs.numpy(), config)
			c_metrics, c_metrics_per_concept = calc_concept_metrics(self.c_true, c_pred_probs, config)	
			metrics = metrics | {f'y_{k}': v for k, v in y_metrics.items()} | {f'c_{k}': v for k, v in c_metrics.items()} # | c_metrics_per_concept # Update dict

		return metrics

def freeze_module(m):
	m.eval()
	for param in m.parameters():
		param.requires_grad = False


def unfreeze_module(m):
	m.train()
	for param in m.parameters():
		param.requires_grad = True