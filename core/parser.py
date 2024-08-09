import os
from collections import OrderedDict
import json
from pathlib import Path
from datetime import datetime
from functools import partial
import importlib
from types  import FunctionType
from sklearn.model_selection import train_test_split
import shutil

    
def init_obj(opt, logger, *args, default_file_name='default file', given_module=None, init_type='Network', **modify_kwargs):
    """
    Finds a function handle with the name given as 'name' in config,
    and returns the instance initialized with corresponding args.
    """ 
    
    if opt is None or len(opt)<1:
        logger.info('Option is None when initialize {}'.format(init_type))
        return None
    
    ''' default format is dict with name key '''
    if isinstance(opt, str):
        opt = {'name': opt}
        logger.warning('Config is a str, converts to a dict {}'.format(opt))

    name = opt['name']
    ''' name can be list, indicates the file and class name of function '''
    if isinstance(name, list):
        file_name, class_name = name[0], name[1]
    else:
        file_name, class_name = default_file_name, name
    try:
        if given_module is not None:
            module = given_module
            
        else:
            module = importlib.import_module(file_name)
        print(f"[INFO] Module: {module} class_name: {class_name}")
        attr = getattr(module, class_name)  # Get the class object out of classname in module!
        print(f"[INFO] Object of class {class_name}: {attr}")
        kwargs = opt.get('args', {})
        kwargs.update(modify_kwargs)
        
        ''' import class or function with args '''
        print(f"[INFO] isinstance of {class_name}: {type(attr)}") # A user-defined class (or the class "object") is an instance of the class "type".
        if isinstance(attr, type): 
            ret = attr(*args, **kwargs)  # Initialize an instance of the class if we detect that attr is an user-defined class
            # print(f"[INFO] ret.__name__")
            ret.__name__  = ret.__class__.__name__
            # print(f"[INFO] ret.__name__")
        elif isinstance(attr, FunctionType): # For dynamically creating a function....wat. Any user-defined function is of type FunctionType.
            ret = partial(attr, *args, **kwargs)
            ret.__name__  = attr.__name__
            # ret = attr
        logger.info('{} [{:s}() form {:s}] is created.'.format(init_type, class_name, file_name))
    except:
        raise NotImplementedError('{} [{:s}() form {:s}] not recognized.'.format(init_type, class_name, file_name))
    return ret


def mkdirs(paths):
    if isinstance(paths, str):
        os.makedirs(paths, exist_ok=True)
    else:
        for path in paths:
            os.makedirs(path, exist_ok=True)

def get_timestamp():
    return datetime.now().strftime('%y%m%d_%H%M%S')


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

class NoneDict(dict):
    def __missing__(self, key):
        return None

def dict_to_nonedict(opt):
    """ 
    Convert an existing dictionary or list opt into a NoneDict, 
    which returns None for missing key(s). 
    """
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt

def dict2str(opt, indent_l=1):
    """ dict to string for logger """
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg

def parse(args):
    """Parse the JSON file in one of the arguments used for the CLI command
    Args:
        args (argparse.Namespace): object containing the arguments used in the CLI command

    Returns:
        NoneDict: A NoneDict version of the parsed JSON file
    """
    json_str = ''
    with open(args.config, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)
    ''' replace the config context using args '''
    opt['phase'] = args.phase
    if args.gpu_ids is not None:
        opt['gpu_ids'] = [int(id) for id in args.gpu_ids.split(',')]
    if args.batch is not None:
        opt['datasets'][opt['phase']]['dataloader']['args']['batch_size'] = args.batch
 
    ''' set cuda environment '''
    if len(opt['gpu_ids']) > 1:
        opt['distributed'] = True
    else:
        opt['distributed'] = False

    ''' update name '''
    if args.debug:
        opt['name'] = 'debug_{}'.format(opt['name'])
    elif opt['finetune_norm']:
        opt['name'] = 'finetune_{}'.format(opt['name'])
    else:
        opt['name'] = '{}_{}'.format(opt['phase'], opt['name'])
    
    # Check if the flist of the appropriate phase exists. If not, then create a new one.
    if "flist_make" in opt:
        if opt["flist_make"]:
            flist_exist = []
            flist_paths = []
            for phase in opt["datasets"]:  # Only works with train and test
                flist_path = Path(opt["datasets"][phase]["which_dataset"]["args"]["data_root"])
                if not flist_path.exists():
                    flist_exist.append(False)
                else:
                    flist_exist.append(True)
                flist_paths.append(flist_path)
            if False in flist_exist:  # Assume that both of them does not have flist
                make_flist(opt=opt, flist_paths=flist_paths)
    


    ''' set log directory '''
    experiments_root = os.path.join(opt['path']['base_dir'], f"{opt['name']}_{get_timestamp()}")
    mkdirs(experiments_root)

    ''' save json config for this experiment in particular '''
    write_json(opt, '{}/config.json'.format(experiments_root))
    ''' change folder relative hierarchy '''
    opt['path']['experiments_root'] = experiments_root
    for key, path in opt['path'].items():
        if 'resume' not in key and 'base' not in key and 'root' not in key:
            opt['path'][key] = os.path.join(experiments_root, path)
            mkdirs(opt['path'][key])

    ''' debug mode '''
    if 'debug' in opt['name']:
        opt['train'].update(opt['debug'])

    ''' code backup in case stuff goes down ''' 
    for name in os.listdir('.'):
        if name in ['config', 'models', 'core', 'slurm', 'data']:
            shutil.copytree(name, os.path.join(opt['path']['code'], name), ignore=shutil.ignore_patterns("*.pyc", "__pycache__"))
        if '.py' in name or '.sh' in name:
            shutil.copy(name, opt['path']['code'])
    return dict_to_nonedict(opt)


def make_flist(opt: dict, flist_paths: tuple[str, str]):
    """
    Create separate flists for training and testing purposes. Only for datasets containing 
    all of their data into a single folder.
    
    Args:
        opt (dict): the dictionary of parameters for the entire model.
        flist_paths (str): tuple of all flist paths in the config file. 
    Returns:
        None
    """
    all_files = []
    if (test_size := opt.get("train_test_split", 0)) == 0:
        test_size = 0.3  # Set default values if the config file doesn't have the field.
    raw_data_path = opt["path"]["root"]
    for _, _, filenames in os.walk(raw_data_path):
        all_files = filenames
    all_files = [os.path.join(raw_data_path, filename) for filename in all_files]
    all_subsets = train_test_split(all_files, test_size=test_size)
    
    # Write train to train flist, test to test flist    
    for i in range(len(flist_paths)):
        f = open(flist_paths[i], "w")
        f.write("\n".join(all_subsets[i]))
        f.close()
    return
         
            

