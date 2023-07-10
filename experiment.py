import argparse
import os
from functools import partial

import ray
import torch

import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from ray import tune

from lib.cfg import update_cfg
from lib.models.hmr import HMR
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger




def single_experiment(config, base_config, wd, name):
    os.chdir(wd)
    sweep_config_list = []
    for k, v in config.items():
        sweep_config_list.append(k)
        sweep_config_list.append(v)
    exp_config = base_config.clone()
    exp_config.merge_from_list(sweep_config_list)

    wandb.init(project='debug', name=name)
    wandblogger = WandbLogger(name=name, project='DDA', log_model=True)
    wandblogger.log_hyperparams(exp_config)

    if exp_config.SEED_VALUE >= 0:
        pl.seed_everything(exp_config.SEED_VALUE, workers=True)

    checkpoint_callback = ModelCheckpoint(
        monitor=f'val_{exp_config.DATASET.VAL.SET[0]}/val_loss',
        mode='min', save_top_k=1, verbose=True)

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        logger=wandblogger,
        log_every_n_steps=1,
        enable_progress_bar=False,
        max_epochs=1,
        val_check_interval=400,
        num_sanity_val_steps=0,
        callbacks=checkpoint_callback
    )

    model = HMR(hparams=exp_config)
    trainer.fit(model)
    trainer.test(ckpt_path='best')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file path', required=True)
    parser.add_argument('--exp_name', type=str, help='cfg file path', required=True)
    args = parser.parse_args()
    base_config = update_cfg(args.cfg)
    if torch.cuda.device_count() == 1:
        single_experiment({}, base_config, wd=os.getcwd(), name=args.exp_name)
    else:
        exp_config = {
            "SEED_VALUE": tune.grid_search([
                56249,
                # 921758,
                # 486239,
                # 358053,
                # 173094,
                # 17393,
                # 4,
                # 5
            ]),
        }
        ray.init(num_gpus=torch.cuda.device_count())
        analysis = tune.run(
            partial(single_experiment,base_config=base_config, wd=os.getcwd(), name=args.exp_name),
            config=exp_config,
            metric=None,
            mode='min',
            num_samples=1,
            name=args.exp_name,
            resources_per_trial={'cpu': 16, 'gpu': 1}
        )
