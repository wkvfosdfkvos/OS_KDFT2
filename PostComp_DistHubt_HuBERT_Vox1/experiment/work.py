from tqdm import tqdm
from itertools import cycle

import torch
import torch.distributed as dist

from exp_lib import test

def train_kd(current_epoch, model, data_loader, logger):
    model.train()
    
    count = 0
    loss_sum_ssl = 0
    loss_sum_clf = 0

    with tqdm(total=len(data_loader), ncols=90) as pbar:
        for x_kd in data_loader:
            # to GPU
            x_kd = x_kd.to(dtype=torch.float32, device=model.device)

            # train
            loss_ssl, loss_clf = model.tune_kd(x_kd)
            
            # logging
            if logger is not None:
                count += 1
                loss_sum_ssl += loss_ssl
                loss_sum_clf += loss_clf
                if len(data_loader) * 0.02 <= count:
                    logger.log_metric('Loss/KD/SSL', loss_sum_ssl / count)
                    logger.log_metric('Loss/KD/CLF', loss_sum_clf / count)
                    loss_sum_ssl = 0
                    loss_sum_clf = 0
                    count = 0

                desc = f'train-[{current_epoch}|(loss): {loss_ssl + loss_clf:.4f}'
                pbar.set_description(desc)
                pbar.update(1)

    _synchronize()

def train_ft(current_epoch, model, data_loader, logger):
    model.train()
    
    count = 0
    loss_sum = 0

    with tqdm(total=len(data_loader), ncols=90) as pbar:
        for x_ft, ft_label in data_loader:
            # to GPU
            x_ft = x_ft.to(dtype=torch.float32, device=model.device)
            ft_label = ft_label.to(dtype=torch.int64, device=model.device)

            # train
            loss = model.tune_ft(x_ft, ft_label)
            
            # logging
            if logger is not None:
                count += 1
                loss_sum += loss
                if len(data_loader) * 0.02 <= count:
                    logger.log_metric('Loss/FT', loss_sum / count)
                    loss_sum = 0
                    count = 0

                desc = f'train-[{current_epoch}|(loss): {loss:.4f}'
                pbar.set_description(desc)
                pbar.update(1)

    _synchronize()

def val(model, data_loader_TTA, trials):
    model.eval()
    
    # enrollment
    embeddings_TTA = test.SV_enrollment(model, data_loader_TTA, use_TTA=True, run_on_ddp=True)

    # EER
    eer = test.test_SV_EER(trials, multi_embedding=embeddings_TTA)
    _synchronize()

    return eer

def eval(model, data_loader, data_loader_TTA, trials):
    model.eval()
    
    # enrollment
    embeddings_full = test.SV_enrollment(model, data_loader, use_TTA=False, run_on_ddp=True)
    embeddings_TTA = test.SV_enrollment(model, data_loader_TTA, use_TTA=True, run_on_ddp=True)

    # EER
    eer = test.test_SV_EER(trials, mono_embedding=embeddings_full, multi_embedding=embeddings_TTA)
    _synchronize()

    return eer

def _synchronize():
    torch.cuda.empty_cache()
    dist.barrier()