import os
from collections import OrderedDict
from datetime import datetime
import json


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def parse(opt_path):
    # remove comments starting with '//'
    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)

    opt['timestamp'] = get_timestamp()
    scale = opt['scale']
    rgb_range = opt['rgb_range']

    # datasets
    for phase, dataset in opt['datasets'].items():
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        dataset['scale'] = scale
        dataset['rgb_range'] = rgb_range
        
    # for network initialize
    if opt['mode'] == 'sr' or opt['mode'] == 'sr_curriculum':
        opt['networks']['G']['scale'] = opt['scale']
    elif opt['mode'] == 'srgan':
        opt['networks']['G']['scale'] = opt['scale']
        opt['networks']['D']['scale'] = opt['scale']

    network_opt = opt['networks']
    path_opt = OrderedDict()

    if opt['mode'] == 'sr' or opt['mode'] == 'sr_curriculum':
        config_str = '%s_in%df%d_x%d'%(network_opt['G']['which_model'].upper(), network_opt['G']['in_channels'],
                                                        network_opt['G']['num_features'], opt['scale'])
        exp_path = os.path.join(os.getcwd(), 'experiments', config_str)

    elif opt['mode'] == 'srgan':
        # TODO: the pretrain_model and config_str need to be indentical
        # TODO: the .json can combine with the pretrain_model
        config_str = '%s_%s_p%df%d_x%d'%(network_opt['G']['which_model'].upper(), network_opt['D']['which_model'].upper(), opt['train']['pixel_weight'],
                                                        opt['train']['feature_weight'], opt['scale'])
        exp_path = os.path.join(os.getcwd(), 'experiments', config_str)

    else:
        raise NotImplementedError("The mode [%s] of networks is not recognized." % opt['mode'])

    if opt['train']['resume']:
        exp_path = opt['train']['resume_path']

    path_opt['exp_root'] = exp_path
    path_opt['epoch'] = os.path.join(exp_path, 'epoch')
    path_opt['log'] = os.path.join(exp_path, 'log')
    path_opt['vis'] = os.path.join(exp_path, 'vis')
    path_opt['results'] = os.path.join(exp_path, 'results')
    opt['path'] = path_opt

    return opt

def save(opt):
    #dump_dir = opt['path']['experiments_root'] if opt['is_train'] else opt['path']['results_root']
    dump_dir = opt['path']['exp_root']
    dump_path = os.path.join(dump_dir, 'options.json')
    with open(dump_path, 'w') as dump_file:
        json.dump(opt, dump_file, indent=2)


class NoneDict(dict):
    def __missing__(self, key):
        return None

# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt
