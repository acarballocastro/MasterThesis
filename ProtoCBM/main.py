"""
Run this file, giving a configuration file as input, to train models, e.g.:
	python main.py --config configfile.yaml
"""

import argparse
import os
import sys
from collections import Counter
from os.path import join
from pathlib import Path
import wandb
import numpy as np
import torch
import torch.optim as optim
import time
import uuid
import distutils

from tqdm import tqdm
import logging

from ProtoCBLoss import create_loss
from ProtoCBM import ProtoCBM

from utils.training import (
	freeze_module, 
	unfreeze_module, 
	create_optimizer,
	create_proto_target_optimizer, 
	train_one_epoch_cbm, 
	validate_one_epoch_cbm,
	train_cbm, 
	train_proto,
	test_proto,
	warm_only,
	joint,
	last_only,
	Custom_Metrics
)
import utils.push as push
from utils.utils import prepare_config, reset_random_seeds, str_to_tuple
from utils.proto_models import preprocess_input_function

from datasets.CUB_dataset import get_CUB_dataloaders
from datasets.AwA2_dataset import get_AwA_dataloaders

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

def train(config):
	"""
	Train and test routine
	"""
	# ---------------------------------
	#       Setup
	# ---------------------------------
	# WandB init
	wandb.init() # replace by log in to wandb
	logging.info(f"Running stored under name: {config['run_name']}")
	
	# Setting device on GPU if available, else CPU
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Additional info when using cuda
	if device.type == 'cuda':
		logging.info(f"Using {torch.cuda.get_device_name(0)}")
	else:
		logging.info("No GPU available")

	# Set paths and directories 
	save_path = Path(os.path.join(config['save_dir'], config['run_name']))
	if config['save_dir']:
		save_path.mkdir(exist_ok=True) #TODO: change back to False before start running experiments
		logging.info(f"Save directory: {save_path}")
	prototype_img_filename_prefix = 'prototype-img'
	prototype_self_act_filename_prefix = 'prototype-self-act'
	proto_bound_boxes_filename_prefix = 'bb'

	# --------------------------------------------
	#       Prepare data and initialization
	# --------------------------------------------
	if config['dataset'] == 'CUB':
		train_loader, val_loader, test_loader, train_push_loader = get_CUB_dataloaders(config)
	elif config['dataset'] == 'awa':
		train_loader, val_loader, test_loader, train_push_loader = get_AwA_dataloaders(
			classes_file=os.path.join(config['data_path'], 'Animals_with_Attributes2/classes.txt'),
			data_path=config['data_path'],
			batch_size=config['train_batch_size'],
			train_ratio=0.6,
			val_ratio=0.2,
			seed=config['seed'],
			img_aug=config['img_aug'], 
			img_size=config['img_size'],
			partial_predicates=False,
			num_predicates=config['num_concepts'],
			preload=True,
		)

	if config['prototype_pretrained']:
			assert config['training_mode'] in ('sequential', 'independent'), "Cannot train joint CBM from a pretrained prototype model"
			assert config['prototype_mode'] in ('ppnet', 'ppool'), "Only PPNet and PPool support pretrained models"

	# Initialize model and training objects
	model = ProtoCBM(config=config, device=device)
	model.to(device)
	if config['prototype_mode'] in ('ppnet', 'ppool'):
		model = torch.nn.DataParallel(model)

	loss_fn = create_loss(config=config)

	# Initialize Metrics
	metrics = Custom_Metrics(config, device).to(device)

	logging.info('TRAINING: ' + str(config['prototype_mode']) + ' in ' + str(config['training_mode']) + ' mode')

 	# -------------------------------------------------------------------------
	#       Concept learning (only sequential or independent training mode)
	# -------------------------------------------------------------------------
	if config['training_mode'] in ('sequential', 'independent'):
		mode = 'c'

		if config['prototype_pretrained']:
			logging.info(f"Concept predictor already trained, loading model from {config['pretrained_dir']}")
			test_proto(model=model, dataloader=val_loader, device=device, loss_fn=loss_fn, mode=mode, 
						metrics=metrics, epoch=0, config=config, gumbel_scalar=10e-3)

		else:
			logging.info('Starting concepts training!')

			# Freeze the target prediction part
			if config['prototype_mode'] in ('ppnet', 'ppool'): 
				model.module.head.apply(freeze_module)
			else:
				model.head.apply(freeze_module)

			if config['prototype_mode'] == 'cbm':
				train_cbm(config=config, model=model, epochs=config['c_epochs'], mode=mode, train_loader=train_loader, val_loader=val_loader, metrics=metrics, loss_fn=loss_fn, device=device)

			elif config['prototype_mode'] == 'ppnet':
				ppnet_optimizers = create_optimizer(config, model.module)
				joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(ppnet_optimizers['joint'], step_size=config['joint_lr_step_size'], gamma=0.1)

				for epoch in range(config['c_epochs']):
					if epoch < config['num_warm_epochs']:
						# Warm-up period
						warm_only(model=model)
						train_proto(model=model, dataloader=train_loader, device=device, loss_fn=loss_fn, mode=mode, 
									metrics=metrics, epoch=epoch, config=config, optimizer=ppnet_optimizers['warm'])

					else:
						# Joint prototype training
						joint(model=model)
						train_proto(model=model, dataloader=train_loader, device=device, loss_fn=loss_fn, mode=mode, 
									metrics=metrics, epoch=epoch, config=config, optimizer=ppnet_optimizers['joint'])
						joint_lr_scheduler.step()

					test_proto(model=model, dataloader=val_loader, device=device, loss_fn=loss_fn, mode=mode, 
							metrics=metrics, epoch=epoch, config=config)

					if epoch >= config['push_start'] and epoch in config['push_epochs']:
						# Last-layer optimization and prototype pushing
						push.push_prototypes_ppnet(
							train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
							prototype_network_parallel=model, # pytorch network with prototype_vectors
							preprocess_input_function=preprocess_input_function, # normalize if needed
							prototype_layer_stride=1,
							root_dir_for_saving_prototypes=save_path, # if not None, prototypes will be saved here
							epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
							prototype_img_filename_prefix=prototype_img_filename_prefix,
							prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
							proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
							save_prototype_class_identity=True)
						test_proto(model=model, dataloader=val_loader, device=device, loss_fn=loss_fn, mode=mode, 
								metrics=metrics, epoch=epoch, config=config)

						if config['prototype_activation_function'] != 'linear':
							# Last layer only
							last_only(model=model)
							# Training of the last layer
							for i in tqdm(range(20)):
								train_proto(model=model, dataloader=train_loader, device=device, loss_fn=loss_fn, mode=mode, 
											metrics=metrics, epoch=epoch, config=config, optimizer=ppnet_optimizers['last_layer'])
								test_proto(model=model, dataloader=val_loader, device=device, loss_fn=loss_fn, mode=mode, 
										metrics=metrics, epoch=epoch, config=config)

			elif config['prototype_mode'] == 'ppool':
				ppool_optimizers = create_optimizer(config, model.module)
				joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(ppool_optimizers['joint'], step_size=config['joint_lr_step_size'], gamma=0.1)

				start_val, end_val = 1.3, 10 ** 3
				gumbel_alpha = (end_val / start_val) ** 2 / config['gumbel_time']	

				for epoch in range(config['c_epochs']):				
					gumbel_scalar = start_val * np.sqrt(gumbel_alpha * (epoch)) if epoch < config['gumbel_time'] else end_val
					if epoch < config['num_warm_epochs']:
						# Warm-up period
						warm_only(model=model)
						train_proto(model=model, dataloader=train_loader, device=device, loss_fn=loss_fn, mode=mode, 
									metrics=metrics, epoch=epoch, config=config, 
									optimizer=ppool_optimizers['warm'], gumbel_scalar=gumbel_scalar)
					else:
						# Joint training
						joint(model=model)
						train_proto(model=model, dataloader=train_loader, device=device, loss_fn=loss_fn, mode=mode, 
									metrics=metrics, epoch=epoch, config=config, 
									optimizer=ppool_optimizers['joint'], gumbel_scalar=gumbel_scalar)
						joint_lr_scheduler.step()
					
					test_proto(model=model, dataloader=val_loader, device=device, loss_fn=loss_fn, mode=mode, 
							metrics=metrics, epoch=epoch, config=config, gumbel_scalar=gumbel_scalar)

				# Last-layer optimization and prototype pushing
				# Prototype pushing
				push.push_prototypes_ppool(
					train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
                    model, # pytorch network with prototype_vectors
                    preprocess_input_function=None, # normalize if needed
                    prototype_layer_stride=1,
                    root_dir_for_saving_prototypes=save_path, # if not None, prototypes will be saved here
                    prototype_img_filename_prefix='prototype-img',
                    prototype_self_act_filename_prefix='prototype-self-act',
                    proto_bound_boxes_filename_prefix=None,
                    prototype_activation_function_in_numpy=None)
				test_proto(model=model, dataloader=val_loader, device=device, loss_fn=loss_fn, mode=mode, 
						metrics=metrics, epoch=epoch, config=config, gumbel_scalar=gumbel_scalar)
				# Last-layer optimization
				last_only(model=model)
				for i in tqdm(range(25)):
					train_proto(model=model, dataloader=train_loader, device=device, loss_fn=loss_fn, mode=mode, 
								metrics=metrics, epoch=epoch, config=config, 
								optimizer=ppool_optimizers['last_layer'], gumbel_scalar=10e-3)
					test_proto(model=model, dataloader=val_loader, device=device, loss_fn=loss_fn, mode=mode, 
							metrics=metrics, epoch=epoch, config=config, gumbel_scalar=10e-3)
					
			else:
				raise NotImplementedError		

			logging.info('Storing encoder model...')
			if config['prototype_mode'] == 'cbm':
				torch.save(model.encoder.state_dict(), join(save_path, 'model.pth'))
			else:
				torch.save(model.module.encoder.state_dict(), join(save_path, 'model.pth'))

	# ----------------------------------
	#       Target / Joint learning 
	# ----------------------------------
	if config['training_mode'] in ('sequential', 'independent'):
		logging.info('Starting target training!')
		mode = 't'
		# Preparing paramters for training
		if config['prototype_mode'] == 'cbm':
			model.head.apply(unfreeze_module)
			model.encoder.apply(freeze_module)
			train_cbm(config=config, model=model, epochs=config['t_epochs'], mode=mode, train_loader=train_loader, val_loader=val_loader, metrics=metrics, loss_fn=loss_fn, device=device)
		else:
			model.module.head.apply(unfreeze_module)
			model.module.encoder.apply(freeze_module)

			target_optimizer = create_proto_target_optimizer(config, model)
			target_lr_scheduler = optim.lr_scheduler.StepLR(target_optimizer, step_size=config['decrease_every'],gamma=1/config['lr_divisor'])

			for epoch in range(config['t_epochs']):
				if epoch % config['validate_per_epoch'] == 0:
					logging.info("EVALUATION ON THE VALIDATION SET:")
					test_proto(model=model, dataloader=val_loader, device=device, loss_fn=loss_fn, mode=mode, 
							metrics=metrics, epoch=epoch, config=config, gumbel_scalar=10e-3)
				train_proto(model=model, dataloader=train_loader, device=device, loss_fn=loss_fn, mode=mode, 
									metrics=metrics, epoch=epoch, config=config, optimizer=target_optimizer, gumbel_scalar=10e-3)
				# Gumbel scalar fixed since it does not matter for target training
				target_lr_scheduler.step()
		
	elif config['training_mode'] == 'joint':
		logging.info('Starting joint training!')
		mode = 'j'
		# Preparing paramters for training
		if config['prototype_mode'] == 'cbm':
			model.head.apply(unfreeze_module)
			model.encoder.apply(unfreeze_module)
			train_cbm(config=config, model=model, epochs=config['j_epochs'], mode=mode, train_loader=train_loader, val_loader=val_loader, metrics=metrics, loss_fn=loss_fn, device=device)
		elif config['prototype_mode'] == 'ppnet':
			model.module.head.apply(unfreeze_module)
			model.module.encoder.apply(unfreeze_module)

			ppnet_optimizers = create_optimizer(config, model.module, joint_with_target=True)
			joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(ppnet_optimizers['joint'], step_size=config['joint_lr_step_size'], gamma=0.1)

			for epoch in range(config['j_epochs']):
				if epoch < config['num_warm_epochs']:
					# Warm-up period
					warm_only(model=model)
					train_proto(model=model, dataloader=train_loader, device=device, loss_fn=loss_fn, mode=mode, 
								metrics=metrics, epoch=epoch, config=config, optimizer=ppnet_optimizers['warm'])

				else:
					# Joint prototype training
					joint(model=model)
					train_proto(model=model, dataloader=train_loader, device=device, loss_fn=loss_fn, mode=mode, 
								metrics=metrics, epoch=epoch, config=config, optimizer=ppnet_optimizers['joint'])
					joint_lr_scheduler.step()

				test_proto(model=model, dataloader=val_loader, device=device, loss_fn=loss_fn, mode=mode, 
						metrics=metrics, epoch=epoch, config=config)

				if epoch >= config['push_start'] and epoch in config['push_epochs']:
					# Last-layer optimization and prototype pushing
					push.push_prototypes_ppnet(
						train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
						prototype_network_parallel=model, # pytorch network with prototype_vectors
						preprocess_input_function=preprocess_input_function, # normalize if needed
						prototype_layer_stride=1,
						root_dir_for_saving_prototypes=save_path, # if not None, prototypes will be saved here
						epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
						prototype_img_filename_prefix=prototype_img_filename_prefix,
						prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
						proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
						save_prototype_class_identity=True)
					test_proto(model=model, dataloader=val_loader, device=device, loss_fn=loss_fn, mode=mode, 
							metrics=metrics, epoch=epoch, config=config)

					if config['prototype_activation_function'] != 'linear':
						# Last layer only
						last_only(model=model)
						# Training of the last layer
						for i in tqdm(range(20)):
							train_proto(model=model, dataloader=train_loader, device=device, loss_fn=loss_fn, mode=mode, 
										metrics=metrics, epoch=epoch, config=config, optimizer=ppnet_optimizers['last_layer'])
							test_proto(model=model, dataloader=val_loader, device=device, loss_fn=loss_fn, mode=mode, 
									metrics=metrics, epoch=epoch, config=config)

		elif config['prototype_mode'] == 'ppool':
			model.module.head.apply(unfreeze_module)
			model.module.encoder.apply(unfreeze_module)

			ppool_optimizers = create_optimizer(config, model.module, joint_with_target=True)
			joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(ppool_optimizers['joint'], step_size=config['joint_lr_step_size'], gamma=0.1)

			start_val, end_val = 1.3, 10 ** 3
			gumbel_alpha = (end_val / start_val) ** 2 / config['gumbel_time']	

			for epoch in range(config['c_epochs']):				
				gumbel_scalar = start_val * np.sqrt(gumbel_alpha * (epoch)) if epoch < config['gumbel_time'] else end_val
				if epoch < config['num_warm_epochs']:
					# Warm-up period
					warm_only(model=model)
					train_proto(model=model, dataloader=train_loader, device=device, loss_fn=loss_fn, mode=mode, 
								metrics=metrics, epoch=epoch, config=config, 
								optimizer=ppool_optimizers['warm'], gumbel_scalar=gumbel_scalar)
				else:
					# Joint training
					joint(model=model)
					train_proto(model=model, dataloader=train_loader, device=device, loss_fn=loss_fn, mode=mode, 
								metrics=metrics, epoch=epoch, config=config, 
								optimizer=ppool_optimizers['warm'], gumbel_scalar=gumbel_scalar)
					joint_lr_scheduler.step()
				
				test_proto(model=model, dataloader=val_loader, device=device, loss_fn=loss_fn, mode=mode, 
						metrics=metrics, epoch=epoch, config=config, gumbel_scalar=gumbel_scalar)

			# Last-layer optimization and prototype pushing
			# Prototype pushing
			push.push_prototypes_ppool(
				train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
				model, # pytorch network with prototype_vectors
				preprocess_input_function=None, # normalize if needed
				prototype_layer_stride=1,
				root_dir_for_saving_prototypes=save_path, # if not None, prototypes will be saved here
				prototype_img_filename_prefix='prototype-img',
				prototype_self_act_filename_prefix='prototype-self-act',
				proto_bound_boxes_filename_prefix=None,
				prototype_activation_function_in_numpy=None)
			test_proto(model=model, dataloader=val_loader, device=device, loss_fn=loss_fn, mode=mode, 
					metrics=metrics, epoch=epoch, config=config, gumbel_scalar=gumbel_scalar)
			# # Last-layer optimization
			# last_only(model=model)
			for i in tqdm(range(25)):
				train_proto(model=model, dataloader=train_loader, device=device, loss_fn=loss_fn, mode=mode, 
							metrics=metrics, epoch=epoch, config=config, 
							optimizer=ppool_optimizers['last_layer'], gumbel_scalar=10e-3)
				test_proto(model=model, dataloader=val_loader, device=device, loss_fn=loss_fn, mode=mode, 
						metrics=metrics, epoch=epoch, config=config, gumbel_scalar=10e-3)

	else:
		raise ValueError

	if config['prototype_mode'] == 'cbm':
		logging.info("EVALUATION ON THE TEST SET:")
		validate_one_epoch_cbm(test_loader, model, metrics, config['t_epochs'], config, loss_fn, device, test=True)
	else:
		test_proto(model=model, dataloader=test_loader, device=device, loss_fn=loss_fn, mode=mode, 
						metrics=metrics, epoch=epoch, config=config, gumbel_scalar=10e-3, test=True) 

	model.apply(freeze_module)
	logging.info('I got to the end :)')
	torch.save(model.state_dict(), join(save_path, 'model.pth'))
	logging.info('TRAINING FINISHED, MODEL SAVED!')

	return None

def main():
	project_dir = Path(__file__).absolute().parent
	logging.info(f"Project directory: {project_dir}")

	parser = argparse.ArgumentParser()

	# Model parameters
	parser.add_argument('--run_name', type=str, help='wandb run name')
	parser.add_argument('--model_dir', type=str, help='pre-trained models directory')
	parser.add_argument('--data_path', type=str, help='data directory')
	parser.add_argument('--train_path', type=str, help='train data directory')
	parser.add_argument('--test_path', type=str, help='test data directory')
	parser.add_argument('--val_path', type=str, help='validation data directory')
	parser.add_argument('--pretrained_dir', type=str, help='pre-trained prototype models directory')

	parser.add_argument('--dataset', type=str, help='the dataset')
	parser.add_argument('--workers', type=int, help='number of workers in dataloader')
	parser.add_argument('--seed', type=int, help='random number generator seed')
	parser.add_argument('--train_batch_size', type=int, help='train batch size')
	parser.add_argument('--val_batch_size', type=int, help='validation batch size')

	parser.add_argument('--num_concepts', type=int, help='number of concepts')
	parser.add_argument('--num_classes', type=int, help='number of classes')
	parser.add_argument('--img_aug', type=bool, help='augment datasets') 
	parser.add_argument('--img_size', type=int, help='image size') 

	parser.add_argument('--alpha', type=int, help='weight in joint')
	parser.add_argument('--encoder_arch', type=str, help='encoder architecture')
	parser.add_argument('--head_arch', type=str, help='linear or non-linear classifier')
	parser.add_argument('--training_mode', type=str, help='sequential or joint or independent') 
	parser.add_argument('--prototype_mode', type=str, help='sequential or joint or independent')
	parser.add_argument('--prototype_pretrained', type=bool, help='load pretrained prototype model or not') 

	parser.add_argument('--prototype_shape', type=str, help='protoype shape')
	parser.add_argument('--num_prototypes', type=int, help='number of prototypes')
	parser.add_argument('--prototype_activation_function', type=str, help='activation function')
	parser.add_argument('--add_on_layers_type', type=str, help='add on layers type')
	parser.add_argument('--num_warm_epochs', type=int, help='number of warm epochs')
	parser.add_argument('--joint_optimizer_lrs_features', type=float, help='joint stage: learning rate pretrained layers')
	parser.add_argument('--joint_optimizer_lrs_add_on_layers', type=float, help='joint stage: learning rate add-on convolutional layers')
	parser.add_argument('--joint_optimizer_prototype_vectors', type=float, help='joint stage: learning rate prototype layer')
	parser.add_argument('--joint_lr_step_size', type=int, help='joint optimizer step size')
	parser.add_argument('--warm_optimizer_lrs_add_on_layers', type=float, help='warm-up stage: learning rate add-on convolutional layers')
	parser.add_argument('--warm_optimizer_prototype_vectors', type=float, help='warm-up stage: learning rate prototype layer')
	parser.add_argument('--last_layer_optimizer_lr', type=float, help='last layer learning rate')
	parser.add_argument('--coefs_crs_ent', type=float, help='coefficients: cross entropy')
	parser.add_argument('--coefs_clst', type=float, help='coefficients: cluster cost')
	parser.add_argument('--coefs_sep', type=float, help='coefficients: separation cost')
	parser.add_argument('--coefs_l1', type=float, help='coefficients: l1 regularization term')
	parser.add_argument('--push_start', type=int, help='push start')

	parser.add_argument('--optimizer', type=str, help='optimizer: sgd or adam')
	parser.add_argument('--learning_rate', type=str, help='learning rate in the joint optimization')  
	parser.add_argument('--weight_decay', type=str, help='weight decay')  
	parser.add_argument('--decrease_every', type=str, help='frequency of the learning rate decrease')  
	parser.add_argument('--lr_divisor', type=str, help='rate of the learning rate decrease')
	parser.add_argument('--validate_per_epoch', type=str, help='periodicity to evaluate the model')

	parser.add_argument('--c_epochs', type=int, help='epochs of concept training') 
	parser.add_argument('--t_epochs', type=int, help='epochs of target training')
	parser.add_argument('--j_epochs', type=int, help='epochs of joint training')  

	# Specify config name
	parser.add_argument('--config_name', default='CUB_ProtoCBM', type=str, choices=['CUB_ProtoCBM', 'AwA_ProtoCBM'], help='the override file name for config.yml')

	args = parser.parse_args()
	config = prepare_config(args, project_dir)
	config['prototype_shape'] = str_to_tuple(config['prototype_shape'])
	config['push_epochs'] = [i for i in range(config['j_epochs']) if i % 10 == 0] # stores every 10 epochs (starting from epoch push_start)
	config['joint_optimizer_lrs'] = {'features': float(config['joint_optimizer_lrs_features']),
                                'add_on_layers': float(config['joint_optimizer_lrs_add_on_layers']),
                                'prototype_vectors': float(config['joint_optimizer_prototype_vectors']),
								'proto_presence': float(config['joint_optimizer_proto_presence'])}
	config['warm_optimizer_lrs'] = {'add_on_layers': float(config['warm_optimizer_lrs_add_on_layers']),
                                'prototype_vectors': float(config['warm_optimizer_prototype_vectors']),
								'proto_presence': float(config['warm_optimizer_proto_presence'])}
	config['coefs'] = {
        'crs_ent': float(config['coefs_crs_ent']),
        'clst': float(config['coefs_clst']),
        'sep': float(config['coefs_sep']),
        'l1': float(config['coefs_l1']),
		'ortho_p': float(config['coefs_ortho_p']),
		'ortho_c': float(config['coefs_ortho_c'])
    	}
	train(config)


if __name__ == "__main__":

    # os.environ["WANDB_MODE"] = "offline"
    wandb.login() # login to wandb
    main()

    wandb.finish(quiet=True)