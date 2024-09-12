"""
Neural network architectures
"""

import os
from typing import Optional, Tuple
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch import load, nn
from torch.nn import utils
from torch.nn.functional import gumbel_softmax
from torchvision import models

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

class PPNet(nn.Module):

    def __init__(self, features, img_size, prototype_shape,
                 proto_layer_rf_info, num_classes, init_weights=True,
                 prototype_activation_function='log',
                 add_on_layers_type='bottleneck', device='cuda'):

        super(PPNet, self).__init__()
        self.img_size = img_size
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.num_classes = num_classes
        self.epsilon = 1e-4
        self.device = device
        
        # prototype_activation_function could be 'log', 'linear',
        # or a generic function that converts distance to similarity score
        self.prototype_activation_function = prototype_activation_function

        '''
        Here we are initializing the class identities of the prototypes
        Without domain specific knowledge we allocate the same number of
        prototypes for each class
        '''
        assert(self.num_prototypes % self.num_classes == 0)
        self.prototype_class_identity = torch.zeros(self.num_prototypes, self.num_classes*2)

        self.num_prototypes_per_class = self.num_prototypes // (self.num_classes*2)
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // self.num_prototypes_per_class] = 1

        self.proto_layer_rf_info = proto_layer_rf_info

        # this has to be named features to allow the precise loading
        self.features = features

        features_name = str(self.features).upper()
        if features_name.startswith('VGG') or features_name.startswith('RES'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        elif features_name.startswith('DENSE'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
        else:
            raise Exception('other base base_architecture NOT implemented')

        if add_on_layers_type == 'bottleneck':
            add_on_layers = []
            current_in_channels = first_add_on_layer_in_channels
            while (current_in_channels > self.prototype_shape[1]) or (len(add_on_layers) == 0):
                current_out_channels = max(self.prototype_shape[1], (current_in_channels // 2))
                add_on_layers.append(nn.Conv2d(in_channels=current_in_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                add_on_layers.append(nn.ReLU())
                add_on_layers.append(nn.Conv2d(in_channels=current_out_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                if current_out_channels > self.prototype_shape[1]:
                    add_on_layers.append(nn.ReLU())
                else:
                    assert(current_out_channels == self.prototype_shape[1])
                    add_on_layers.append(nn.Sigmoid())
                current_in_channels = current_in_channels // 2
            self.add_on_layers = nn.Sequential(*add_on_layers)
        else:
            self.add_on_layers = nn.Sequential(
                nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_shape[1], kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1),
                nn.Sigmoid()
                )
        
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape),
                                              requires_grad=True)

        # do not make this just a tensor,
        # since it will not be moved automatically to gpu
        self.ones = nn.Parameter(torch.ones(self.prototype_shape),
                                 requires_grad=False)

        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes,
                                    bias=False) # do not use bias

        if init_weights:
            self._initialize_weights()

    def conv_features(self, x):
        '''
        the feature input to prototype layer
        '''
        x = self.features(x)
        x = self.add_on_layers(x)
        return x

    @staticmethod
    def _weighted_l2_convolution(input, filter, weights):
        '''
        input of shape N * c * h * w
        filter of shape P * c * h1 * w1
        weight of shape P * c * h1 * w1
        '''
        input2 = input ** 2
        input_patch_weighted_norm2 = F.conv2d(input=input2, weight=weights)

        filter2 = filter ** 2
        weighted_filter2 = filter2 * weights
        filter_weighted_norm2 = torch.sum(weighted_filter2, dim=(1, 2, 3))
        filter_weighted_norm2_reshape = filter_weighted_norm2.view(-1, 1, 1)

        weighted_filter = filter * weights
        weighted_inner_product = F.conv2d(input=input, weight=weighted_filter)

        # use broadcast
        intermediate_result = \
            - 2 * weighted_inner_product + filter_weighted_norm2_reshape
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(input_patch_weighted_norm2 + intermediate_result)

        return distances

    def _l2_convolution(self, x):
        '''
        apply self.prototype_vectors as l2-convolution filters on input x
        '''
        x2 = x ** 2
        x2_patch_sum = F.conv2d(input=x2, weight=self.ones)

        p2 = self.prototype_vectors ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1)
        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(input=x, weight=self.prototype_vectors)
        intermediate_result = - 2 * xp + p2_reshape  # use broadcast
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)

        return distances

    def prototype_distances(self, x):
        '''
        x is the raw input
        '''
        conv_features = self.conv_features(x)
        distances = self._l2_convolution(conv_features)
        return distances

    def distance_2_similarity(self, distances):
        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        else:
            return self.prototype_activation_function(distances)

    def forward(self, x):
        distances = self.prototype_distances(x)
        '''
        we cannot refactor the lines below for similarity scores
        because we need to return min_distances
        '''
        # global min pooling
        min_distances = -F.max_pool2d(-distances,
                                      kernel_size=(distances.size()[2],
                                                   distances.size()[3]))
        min_distances = min_distances.view(-1, self.num_prototypes)
        prototype_activations = self.distance_2_similarity(min_distances)
        logits = self.last_layer(prototype_activations)

        proto_presence = None
        return logits, min_distances, proto_presence

    def push_forward(self, x):
        '''this method is needed for the pushing operation'''
        conv_output = self.conv_features(x)
        distances = self._l2_convolution(conv_output)
        return conv_output, distances

    def __repr__(self):
        # PPNet(self, features, img_size, prototype_shape,
        # proto_layer_rf_info, num_classes, init_weights=True):
        rep = (
            'PPNet(\n'
            '\tfeatures: {},\n'
            '\timg_size: {},\n'
            '\tprototype_shape: {},\n'
            '\tproto_layer_rf_info: {},\n'
            '\tnum_classes: {},\n'
            '\tepsilon: {}\n'
            ')'
        )

        return rep.format(self.features,
                          self.img_size,
                          self.prototype_shape,
                          self.proto_layer_rf_info,
                          self.num_classes,
                          self.epsilon)

    def set_last_layer_incorrect_connection(self):
        '''
        1 if positive concept, -1 if negative concept
        '''
        weights = torch.zeros((self.num_classes, self.num_prototypes))
        weights[:, :self.num_prototypes//2] = torch.t(self.prototype_class_identity)[:self.num_classes, :self.num_prototypes//2]
        weights[:, self.num_prototypes//2:] = -1*torch.t(self.prototype_class_identity)[self.num_classes:, self.num_prototypes//2:]
        self.last_layer.weight.data.copy_(weights)
        self.l1_mask = 1 - torch.abs(weights).to(self.device)

    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.set_last_layer_incorrect_connection()

class ProtoPool(nn.Module):

    def __init__(self, num_prototypes: int, num_descriptive: int, num_classes: int,
                 use_thresh: bool = False, arch: str = 'resnet34', pretrained: bool = True,
                 add_on_layers_type: str = 'linear', prototype_activation_function: str = 'log',
                 proto_depth: int = 128, use_last_layer: bool = False, inat: bool = False, device: str = 'cuda') -> None:
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = 1e-4
        self.num_descriptive = num_descriptive
        self.num_prototypes = num_prototypes
        self.proto_depth = proto_depth
        self.prototype_shape = (self.num_prototypes, self.proto_depth, 1, 1)
        self.use_thresh = use_thresh
        self.arch = arch
        self.pretrained = pretrained
        self.prototype_activation_function = prototype_activation_function
        self.inat = inat
        if self.use_thresh:
            self.alfa = Parameter(torch.Tensor(1, num_classes, num_descriptive))
            nn.init.xavier_normal_(self.alfa, gain=1.0)
        else:
            self.alfa = 1
            self.beta = 0

        self.proto_presence = torch.zeros(num_classes*2, num_prototypes, num_descriptive)  # [c, p, K]

        self.proto_presence = Parameter(self.proto_presence, requires_grad=True)
        nn.init.xavier_normal_(self.proto_presence, gain=1.0)

        if self.inat:
            self.features = base_architecture_to_features['resnet50'](pretrained=pretrained, inat=True)
        else:
            self.features = base_architecture_to_features[self.arch](pretrained=pretrained)

        features_name = str(self.features).upper()
        if features_name.startswith('VGG') or features_name.startswith('RES'):
            first_add_on_layer_in_channels = \
                [i for i in self.features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        elif features_name.startswith('DENSE'):
            first_add_on_layer_in_channels = \
                [i for i in self.features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
        else:
            raise Exception('other base base_architecture NOT implemented')

        if add_on_layers_type == 'bottleneck':
            raise NotImplementedError
        else:
            add_on_layers = [
                nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_shape[1],
                          kernel_size=1),
                # nn.ReLU(),
                # nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1),
                nn.Sigmoid(),
            ]

            self.add_on_layers = nn.Sequential(*add_on_layers)

        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)

        self.ones = nn.Parameter(torch.ones(self.prototype_shape), requires_grad=False)

        # initial weights
        for m in self.add_on_layers.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.use_last_layer = use_last_layer
        if self.use_last_layer:
            prototype_class_identity = torch.zeros(self.num_descriptive * self.num_classes * 2, self.num_classes * 2)

            for j in range(self.num_descriptive * self.num_classes * 2):
                prototype_class_identity[j, j // self.num_descriptive] = 1

            self.last_layer = nn.Linear(self.num_descriptive * self.num_classes * 2, self.num_classes, bias=False)

            weights = torch.zeros((self.num_classes, self.num_descriptive * self.num_classes * 2))
            weights[:, :(self.num_descriptive * self.num_classes * 2)//2] = torch.t(prototype_class_identity)[:self.num_classes, :(self.num_descriptive * self.num_classes * 2)//2]
            weights[:, (self.num_descriptive * self.num_classes * 2)//2:] = -1*torch.t(prototype_class_identity)[self.num_classes:, (self.num_descriptive * self.num_classes * 2)//2:]
            self.last_layer.weight.data.copy_(weights)
            self.l1_mask = 1 - torch.abs(weights).to(device)

        else:
            self.last_layer = nn.Identity()

    def conv_features(self, x):
        '''
        the feature input to prototype layer
        '''
        x = self.features(x)
        x = self.add_on_layers(x)
        return x

    def forward(self, x: torch.Tensor, gumbel_scale: int = 0) -> \
            Tuple[torch.Tensor, torch.LongTensor]:
        if gumbel_scale == 0:
            proto_presence = torch.softmax(self.proto_presence, dim=1)
        else:
            proto_presence = gumbel_softmax(self.proto_presence * gumbel_scale, dim=1, tau=0.5)

        distances = self.prototype_distances(x)  # [b, C, H, W] -> [b, p, h, w]
        '''
        we cannot refactor the lines below for similarity scores because we need to return min_distances
        '''
        # global min pooling
        min_distances = -F.max_pool2d(-distances,
                                      kernel_size=(distances.size()[2],
                                                   distances.size()[3])).squeeze()  # [b, p]
        avg_dist = F.avg_pool2d(distances, kernel_size=(distances.size()[2],
                                                        distances.size()[3])).squeeze()  # [b, p]
        min_mixed_distances = self._mix_l2_convolution(min_distances, proto_presence)  # [b, c, n]
        avg_mixed_distances = self._mix_l2_convolution(avg_dist, proto_presence)  # [b, c, n]
        x = self.distance_2_similarity(min_mixed_distances)  # [b, c, n]
        x_avg = self.distance_2_similarity(avg_mixed_distances)  # [b, c, n]
        x = x - x_avg # Focal similarity (max - avg)
        if self.use_last_layer:
            logits = self.last_layer(x.flatten(start_dim=1))
        else:
            logits = x.sum(dim=-1)

        return logits, min_distances, proto_presence  # [b,c,n] [b, p] [c, p, n]

    def _l2_convolution(self, x):
        '''
        apply self.prototype_vectors as l2-convolution filters on input x
        '''
        x2 = x ** 2
        x2_patch_sum = F.conv2d(input=x2, weight=self.ones)

        p2 = self.prototype_vectors ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1)
        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(input=x, weight=self.prototype_vectors)
        intermediate_result = - 2 * xp + p2_reshape  # use broadcast
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)

        return distances

    def _mix_l2_convolution(self, distances, proto_presence):
        '''
        apply self.prototype_vectors as l2-convolution filters on input x
        '''
        # distances [b, p]
        # proto_presence [c, p, n]
        mixed_distances = torch.einsum('bp,cpn->bcn', distances, proto_presence)

        return mixed_distances  # [b, c, n]

    def prototype_distances(self, x):
        '''
        x is the raw input
        '''
        conv_features = self.conv_features(x)
        distances = self._l2_convolution(conv_features)  # [b, p, h, w]
        return distances  # [b, n, h, w], [b, p, h, w]

    def distance_2_similarity(self, distances):  # [b,c,n]
        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            if self.use_thresh:
                distances = distances  # * torch.exp(self.alfa)  # [b, c, n]
            return 1 / (distances + 1)
        else:
            raise NotImplementedError

    def get_map_class_to_prototypes(self):
        pp = gumbel_softmax(self.proto_presence * 10e3, dim=1, tau=0.5).detach()
        return np.argmax(pp.cpu().numpy(), axis=1)

    def __repr__(self):
        res = super(ProtoPool, self).__repr__()
        return res