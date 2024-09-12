import os 
import numpy as np
import random
import yaml
import torch
from pathlib import Path

def str_to_tuple(s):
    return tuple(map(int, s.strip('()').split(',')))

def reset_random_seeds(seed):
	os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.use_deterministic_algorithms(True)
	torch.backends.cudnn.benchmark = False
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	gen = torch.manual_seed(seed)
	return gen
	

def merge_yaml_args(configs, args):
	arg_dict = args.__dict__
	configs['parser'] = dict()
	for key, value in arg_dict.items():
		flag = True
		# Replace/Create values in config if they are defined by arg in parser.
		if arg_dict[key] is not None:
			for key_config in configs.keys():
				# If value of config is dict itself, then search key-value pairs inside this dict for matching the arg
				if type(configs[key_config]) is dict:
					for key2, value2 in configs[key_config].items():
						if key == key2:
							configs[key_config][key2] = value
							flag = False
				# If value of config is not a dict, check whether key matches to the arg
				else:
					if key == key_config:
						configs[key_config] = value
						flag = False
				# Break out of loop if key got replaced
				if flag == False:
					break
			# If arg does not match any keys of config, define a new key
			else:
				print("Could not find this key in config, therefore adding it:", key)
				configs['parser'][key] = arg_dict[key]
	return configs

def prepare_config(args, project_dir):
	# Load config
	config_name = args.config_name +'.yaml'
	config_path = project_dir / 'configs' / config_name

	with config_path.open(mode='r') as yamlfile:
		configs = yaml.safe_load(yamlfile)

	# Override config if args in parser
	configs = merge_yaml_args(configs, args)
	# configs['experiment_dir'] = Path(project_dir, 'experiments').absolute()
	return configs

def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.
    index = torch.randperm(x.shape[0], dtype=x.dtype, device=x.device).to(torch.long)
    mixed_x = lam * x + (1 - lam) * x[index, ...]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam