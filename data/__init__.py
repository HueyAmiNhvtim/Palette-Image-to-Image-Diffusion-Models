from functools import partial
import numpy as np

from torch.utils.data.distributed import DistributedSampler
from torch import Generator, randperm
from torch.utils.data import DataLoader, Subset

import core.util as Util
from core.parser import init_obj
# [TODO] Understand what validation_split is. 
# [TODO] Automatically generate a flist file if not existed and not a directory. You may have to create a train-test-split thing.
 

def define_dataloader(logger, opt):
    """ Create train/test dataloader and validation dataloader,  
        Validation dataloader is None when phase is test or not GPU 0 """
    '''create dataset and set random seed'''
    # Grab the appropriate parameter values for the dataloaders

    dataloader_args = opt['datasets'][opt['phase']]['dataloader']['args']
    worker_init_fn = partial(Util.set_seed, gl_seed=opt['seed'])

    phase_dataset, val_dataset = define_dataset(logger, opt)

    '''create datasampler'''
    data_sampler = None
    if opt['distributed']:
        data_sampler = DistributedSampler(phase_dataset, shuffle=dataloader_args.get('shuffle', False), num_replicas=opt['world_size'], rank=opt['global_rank'])
        dataloader_args.update({'shuffle':False}) # sampler option is mutually exclusive with shuffle 
    
    ''' create dataloader and validation dataloader '''
    dataloader = DataLoader(phase_dataset, sampler=data_sampler, worker_init_fn=worker_init_fn, **dataloader_args)
    ''' val_dataloader don't use DistributedSampler to run only GPU 0! '''
    if opt['global_rank']==0 and val_dataset is not None:
        dataloader_args.update(opt['datasets'][opt['phase']]['dataloader'].get('val_args',{}))
        val_dataloader = DataLoader(val_dataset, worker_init_fn=worker_init_fn, **dataloader_args) 
    else:
        val_dataloader = None
    return dataloader, val_dataloader


def define_dataset(logger, opt):
    ''' loading Dataset() class from given file's name '''
    dataset_opt = opt['datasets'][opt['phase']]['which_dataset']  # Get the approriate dataset parameters given the phase (train or test)
    phase_dataset = init_obj(dataset_opt, logger, default_file_name='data.dataset', init_type='Dataset')
    val_dataset = None

    valid_len = 0
    data_len = len(phase_dataset)
    if 'debug' in opt['name']:  
        debug_split = opt['debug'].get('debug_split', 1.0)
        if isinstance(debug_split, int):
            data_len = debug_split
        else:
            data_len *= debug_split

    dataloader_opt = opt['datasets'][opt['phase']]['dataloader']
    valid_split = dataloader_opt.get('validation_split', 0)  # [TODO] Understand why validation_split is here....  
    
    ''' divide validation dataset, valid_split==0 when phase is test or validation_split is itself 0 when extracted out of dict '''
    if valid_split > 0.0 or 'debug' in opt['name']: 
        if isinstance(valid_split, int):
            assert valid_split < data_len, "Validation set size is configured to be larger than entire dataset."
            valid_len = valid_split
        else:   
            valid_len = int(data_len * valid_split)
        data_len -= valid_len
        phase_dataset, val_dataset = subset_split(dataset=phase_dataset, lengths=[data_len, valid_len], generator=Generator().manual_seed(opt['seed']))
    
    logger.info(f"Dataset for {opt['phase']} have {data_len} samples.")
    if opt['phase'] == 'train':
        logger.info(f"Dataset for {'val'} have {valid_len} samples.")   
    return phase_dataset, val_dataset

def subset_split(dataset, lengths, generator):
    """
    Split a dataset into non-overlapping new datasets of given lengths. 
    Main code is from random_split function in pytorch.
    """
    indices = randperm(sum(lengths), generator=generator).tolist()
    Subsets = []
    for offset, length in zip(np.add.accumulate(lengths), lengths):
        if length == 0:
            Subsets.append(None)
        else:
            Subsets.append(Subset(dataset, indices[offset - length : offset]))
    return Subsets
