"""
Utility methods for constructing loss functions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Optional
import itertools

def create_loss(config):
	"""
	Parse configuration file and return a relevant loss function
	"""
	if config['prototype_mode'] == 'cbm':
		return CBMLoss(
			num_classes=config['num_classes'], reduction='mean', 
			alpha=config['alpha'], training_mode=config['training_mode']
			)
	elif config['prototype_mode'] == 'ppnet':
		return PPNetLoss(
			num_classes=config['num_classes'], reduction='mean', 
			alpha=config['alpha'], training_mode=config['training_mode'], 
			use_l1_mask=True, coefs=config['coefs']
		)
	elif config['prototype_mode'] == 'ppool':
		return PPoolLoss(
			num_classes=config['num_classes'], reduction='mean', 
			alpha=config['alpha'], training_mode=config['training_mode'], 
			use_l1_mask=True, coefs=config['coefs']
		)
	elif config['prototype_mode'] == 'ppnetdata':
		return
	else:
		raise NotImplementedError

class CBMLoss(nn.Module):
	"""
	Loss function for the concept bottleneck model
	"""

	def __init__(
			self,
			num_classes: int,
			training_mode: str,
			reduction: str = 'mean',
			alpha: float = 1) -> None:
		"""
		Initializes the loss object

		@param num_classes: the number of the classes of the target variable
		@param training_mode: the mode of training, either 'joint' or 'sequential'
		@param reduction: reduction to apply to the output of the CE loss
		@param alpha: parameter controlling the trade-off between the target and concept prediction during the joint
						optimization. The higher the @alpha, the higher the weight of the concept prediction loss
		"""
		super().__init__()
		self.num_classes = num_classes
		self.alpha = alpha if training_mode == 'joint' else 1.
		self.reduction = reduction

	def forward(self, concepts_pred_logits: Tensor, concepts_true: Tensor, target_pred_logits: Tensor, target_true: Tensor,) -> Tensor:
		"""
		Computes the loss for the given predictions

		@param concepts_pred_logits: predicted logits for the concept values
		@param concepts_true: ground-truth concept values
		@param target_pred_logits: predicted logits for the target variable
		@param target_true: ground-truth target variable values
		@return: target prediction loss, a tensor of prediction losses for each of the concepts, summed concept
					prediction loss and the total loss
		"""

		summed_concepts_loss = 0
		concepts_loss = []

		# NOTE: all concepts are assumed to be binary
		for concept_idx in range(concepts_true.shape[1]):
			c_loss = F.binary_cross_entropy_with_logits(
				concepts_pred_logits[:, concept_idx], concepts_true[:, concept_idx].float(), reduction=self.reduction)
			concepts_loss.append(c_loss)
			summed_concepts_loss += c_loss

		if self.num_classes == 2:
			target_loss = F.binary_cross_entropy_with_logits(
				target_pred_logits.squeeze(1), target_true, reduction=self.reduction)
		else:
			target_loss = F.cross_entropy(
				target_pred_logits, target_true.long(), reduction=self.reduction)

		total_loss = target_loss + self.alpha * summed_concepts_loss 

		return target_loss, concepts_loss, summed_concepts_loss, total_loss

class PPNetLoss(nn.Module):
	"""
	Loss function for the Prototype Concept Bottleneck Model
	"""

	def __init__(
			self,
			num_classes: int,
			training_mode: str,
			use_l1_mask: bool, 
			coefs: dict,
			reduction: str = 'mean',
			alpha: float = 1) -> None:
		"""
		Initializes the loss object

		@param num_classes: the number of the classes of the target variable
		@param training_mode: the mode of training, either 'joint' or 'sequential'
		@param use_l1_mask: whether to use the L1 mask for the L1 loss
		@param coefs: dictionary with coefficients to calculate ProtoPNet loss
		@param reduction: reduction to apply to the output of the CE loss
		@param alpha: parameter controlling the trade-off between the target and concept prediction during the joint
						optimization. The higher the @alpha, the higher the weight of the concept prediction loss
		"""
		super().__init__()
		self.num_classes = num_classes
		self.alpha = alpha if training_mode == 'joint' else 1.0
		self.reduction = reduction
		self.coefs = coefs
		self.use_l1_mask = use_l1_mask

	def forward(self, concepts_pred_logits: Tensor, concepts_true: Tensor, min_distances: Tensor, proto_presence: Tensor, target_pred_logits: Tensor, target_true: Tensor, model) -> Tensor:
		"""
		Computes the loss for the given predictions

		@param concepts_pred_logits: predicted concept values (logits)
		@param concepts_true: ground-truth concept values
		@param target_pred_logits: predicted logits for the target variable
		@param target_true: ground-truth target variable values
		@param model: ProtoCBM model
		@param is_train: train or test mode
		@param use_l1_mask: use mask for l1 loss
		@param coefs: dictionary with coefficients to calculate ProtoPNet loss
		@return: target prediction loss, a tensor of prediction losses for each of the concepts, summed concept
					prediction loss and the total loss
		"""

		concepts_loss = []

		# NOTE: all concepts are assumed to be binary
		for concept_idx in range(concepts_true.shape[1]):
			c_loss = F.binary_cross_entropy_with_logits(
				concepts_pred_logits[:, concept_idx], concepts_true[:, concept_idx].float(), reduction=self.reduction)
			concepts_loss.append(c_loss)

		# compute cross_entropy loss
		cross_entropy = torch.nn.functional.binary_cross_entropy_with_logits(concepts_pred_logits, concepts_true)

		max_dist = (model.module.prototype_shape[1]
					* model.module.prototype_shape[2]
					* model.module.prototype_shape[3])

		# indices: according to whether concept is positive or negative
		indices = concepts_true.cpu() * torch.arange(model.module.num_concepts) + (1 - concepts_true.cpu()) * (torch.arange(model.module.num_concepts) + model.module.num_concepts)
		indices = indices.to(int).numpy()
		# indices_flipped: opposite of indices for the other prototypes
		indices_flipped = (1 - concepts_true.cpu()) * torch.arange(model.module.num_concepts) + concepts_true.cpu() * (torch.arange(model.module.num_concepts) + model.module.num_concepts)
		indices_flipped = indices_flipped.to(int).numpy()

		batch_size = concepts_pred_logits.shape[0]
		min_distances_clust = torch.zeros((batch_size*model.module.num_concepts, model.module.encoder.num_prototypes_per_class))
		min_distances_sep = torch.zeros((batch_size*model.module.num_concepts, model.module.encoder.num_prototypes_per_class))

		for n, k in itertools.product(range(batch_size), range(model.module.num_concepts)):
			start_index_clust = indices[n, k]*model.module.encoder.num_prototypes_per_class
			start_index_sep = indices_flipped[n, k]*model.module.encoder.num_prototypes_per_class    

			min_distances_clust[n*model.module.num_concepts+k] = min_distances[n, start_index_clust:start_index_clust+model.module.encoder.num_prototypes_per_class]
			min_distances_sep[n*model.module.num_concepts+k] = min_distances[n, start_index_sep:start_index_sep+model.module.encoder.num_prototypes_per_class]

		cluster_cost = torch.mean(torch.min(min_distances_clust, axis = 1)[0])
		separation_cost = torch.mean(torch.min(min_distances_sep, axis = 1)[0])
			
		if self.use_l1_mask:
			l1 = (model.module.encoder.last_layer.weight * model.module.encoder.l1_mask).norm(p=1)
		else:
			l1 = model.module.encoder.last_layer.weight.norm(p=1)

		if self.coefs is not None:
			summed_concepts_loss = (self.coefs['crs_ent'] * cross_entropy
					+ self.coefs['clst'] * cluster_cost
					+ self.coefs['sep'] * separation_cost
					+ self.coefs['l1'] * l1)
		else:
			summed_concepts_loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1

		if self.num_classes == 2:
			# Logits to probs
			target_pred_probs = nn.Sigmoid(target_pred_logits.squeeze(1))
			target_loss = F.binary_cross_entropy(
				target_pred_probs, target_true, reduction=self.reduction)
		else:
			target_loss = F.cross_entropy(
				target_pred_logits, target_true.long(), reduction=self.reduction)

			total_loss = target_loss + self.alpha * summed_concepts_loss 
		return target_loss, concepts_loss, summed_concepts_loss, total_loss, cross_entropy, cluster_cost, separation_cost, l1, None, None

def ppool_dist_loss(model, min_distances, proto_presence, top_k):
    #               model, [b, p],        [b, c, p, n],      [scalar]
    max_dist = (model.module.prototype_shape[1]
                * model.module.prototype_shape[2]
                * model.module.prototype_shape[3])

    # In this implementation, we add all probabilities for a different prototype across slots
    # and then pick the top k prototypes (all will be different)
    basic_proto = proto_presence.sum(dim=-1).detach()  # [b, c, p]
    _, idx = torch.topk(basic_proto, top_k, dim=2)  # [b, c, n]

    # In this implementation, for each distribution we pick the prototype with the highest
    # probability and that way we get our top k prototypes (some might be the same)
    # idx = torch.argmax(proto_presence, dim=2)

	# In this implementation, we use the method get_map_class_to_prototypes
    # which is already built in the model
    # idx = torch.from_numpy(model.get_map_class_to_prototypes())[indices.long()].cuda()

    binarized_top_k = torch.zeros_like(basic_proto)
    binarized_top_k.scatter_(dim=2, src=torch.ones_like(basic_proto), index=idx)  # [b, c, p]

    inverted_distances, _ = torch.max((max_dist - min_distances).unsqueeze(1) * binarized_top_k, dim=2)  # [b, c]
    inverted_distances = inverted_distances.mean(dim = 1) # [b], mean of all concept
    cost = torch.mean(max_dist - inverted_distances)
    return cost

class PPoolLoss(nn.Module):
	"""
	Loss function for the Prototype Concept Bottleneck Model
	"""

	def __init__(
			self,
			num_classes: int,
			training_mode: str,
			use_l1_mask: bool, 
			coefs: dict,
			reduction: str = 'mean',
			alpha: float = 1) -> None:
		"""
		Initializes the loss object

		@param num_classes: the number of the classes of the target variable
		@param training_mode: the mode of training, either 'joint' or 'sequential'
		@param use_l1_mask: whether to use the L1 mask for the L1 loss
		@param coefs: dictionary with coefficients to calculate ProtoPNet loss
		@param reduction: reduction to apply to the output of the CE loss
		@param alpha: parameter controlling the trade-off between the target and concept prediction during the joint
						optimization. The higher the @alpha, the higher the weight of the concept prediction loss
		"""
		super().__init__()
		self.num_classes = num_classes
		self.alpha = alpha if training_mode == 'joint' else 1.0
		self.reduction = reduction
		self.pp_ortho = True
		self.coefs = coefs
		self.use_l1_mask = use_l1_mask

	def forward(self, concepts_pred_logits: Tensor, concepts_true: Tensor, min_distances: Tensor, proto_presence: Tensor, target_pred_logits: Tensor, target_true: Tensor, model) -> Tensor:
		"""
		Computes the loss for the given predictions

		@param concepts_pred_logits: predicted concept values (logits)
		@param concepts_true: ground-truth concept values
		@param target_pred_logits: predicted logits for the target variable
		@param target_true: ground-truth target variable values
		@return: target prediction loss, a tensor of prediction losses for each of the concepts, summed concept
					prediction loss and the total loss
		"""

		concepts_loss = []

		# NOTE: all concepts are assumed to be binary
		for concept_idx in range(concepts_true.shape[1]):
			c_loss = F.binary_cross_entropy_with_logits(
				concepts_pred_logits[:, concept_idx], concepts_true[:, concept_idx].float(), reduction=self.reduction)
			concepts_loss.append(c_loss)

		# criterion = torch.nn.BCEWithLogitsLoss()
		# entropy_loss = criterion(concepts_pred_logits, concepts_true)
		entropy_loss = torch.nn.functional.binary_cross_entropy_with_logits(concepts_pred_logits, concepts_true)

		orthogonal_loss_p = torch.Tensor([0]).cuda()
		orthogonal_loss_c = torch.Tensor([0]).cuda()
		if self.pp_ortho:
			# Orthogonal loss per class
			orthogonal_loss_p += \
				torch.nn.functional.cosine_similarity(model.module.encoder.proto_presence.unsqueeze(2),
														model.module.encoder.proto_presence.unsqueeze(-1), dim=1).sum()				
			# Orthogonal loss positive-negative concepts
			positive_concepts = model.module.encoder.proto_presence[:model.module.num_concepts].unsqueeze(2)
			negative_concepts = model.module.encoder.proto_presence[model.module.num_concepts:].unsqueeze(-1)
			orthogonal_loss_c += torch.nn.functional.cosine_similarity(positive_concepts, negative_concepts, dim=1).sum()

			orthogonal_loss_p = (orthogonal_loss_p / (model.module.encoder.num_descriptive * model.module.num_concepts * 2) - 1)[0] 
			orthogonal_loss_c = (orthogonal_loss_c / (model.module.encoder.num_descriptive * model.module.num_concepts))[0]

		indices = concepts_true.cpu() * torch.arange(model.module.num_concepts) + (1 - concepts_true.cpu()) * (torch.arange(model.module.num_concepts) + model.module.num_concepts)
		inverted_indices = (1 - concepts_true.cpu()) * torch.arange(model.module.num_concepts) + concepts_true.cpu() * (torch.arange(model.module.num_concepts) + model.module.num_concepts)
		
		inverted_proto_presence = 1 - proto_presence

		proto_presence = proto_presence[indices.long()] #label_p
		inverted_proto_presence = inverted_proto_presence[inverted_indices.long()] # label_p

		clst_loss_val = ppool_dist_loss(model, min_distances, proto_presence, model.module.encoder.num_descriptive)  
		sep_loss_val = ppool_dist_loss(model, min_distances, inverted_proto_presence, model.module.encoder.num_prototypes - model.module.encoder.num_descriptive)  
		prototypes_of_correct_class = proto_presence.sum(dim=-1).detach()
		prototypes_of_wrong_class = 1 - prototypes_of_correct_class
		avg_separation_cost = torch.sum(min_distances.unsqueeze(1).repeat(1,model.module.num_concepts,1) * prototypes_of_wrong_class, dim=-1) / torch.sum(prototypes_of_wrong_class, dim=-1)
		avg_separation_cost = torch.mean(avg_separation_cost)

		l1 = (model.module.encoder.last_layer.weight * model.module.encoder.l1_mask).norm(p=1)

		if self.coefs is not None:
			summed_concepts_loss = self.coefs['crs_ent'] * entropy_loss + clst_loss_val * self.coefs['clst'] + \
                        sep_loss_val * self.coefs['sep'] + self.coefs['l1'] * l1 + \
							self.coefs['ortho_p'] * orthogonal_loss_p + self.coefs['ortho_c'] * orthogonal_loss_c
		else:
			summed_concepts_loss = entropy_loss + 0.8 * clst_loss_val - 0.08 * sep_loss_val + 1e-4 * l1 + orthogonal_loss_p + orthogonal_loss_c
			

		if self.num_classes == 2:
			# Logits to probs
			target_pred_probs = nn.Sigmoid(target_pred_logits.squeeze(1))
			target_loss = F.binary_cross_entropy(
				target_pred_probs, target_true, reduction=self.reduction)
		else:
			target_loss = F.cross_entropy(
				target_pred_logits, target_true.long(), reduction=self.reduction)

		total_loss = target_loss + self.alpha * summed_concepts_loss 

		return target_loss, concepts_loss, summed_concepts_loss, total_loss, entropy_loss, clst_loss_val, sep_loss_val, l1, orthogonal_loss_p, orthogonal_loss_c