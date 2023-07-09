import argparse

import wandb
from pytorch_lightning.callbacks import ModelCheckpoint

from lib.cfg import update_cfg
from lib.models.hmr import HMR
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, help='cfg file path', required=True)
parser.add_argument('--exp_name', type=str, help='cfg file path', required=True)
args = parser.parse_args()
base_config = update_cfg(args.cfg)


def single_experiment():
    wandb.init(project='DDA')
    sweep_config_list = []
    for k, v in wandb.config.items():
        sweep_config_list.append(k)
        sweep_config_list.append(v)
    exp_config = base_config.clone()
    exp_config.merge_from_list(sweep_config_list)

    if exp_config.SEED_VALUE >= 0:
        pl.seed_everything(exp_config.SEED_VALUE, workers=True)

    wandblogger = WandbLogger(name=args.exp_name, project='DDA', log_model=True)
    wandblogger.log_hyperparams(exp_config)

    checkpoint_callback = ModelCheckpoint(
        monitor=f'val_{exp_config.DATASET.VAL.SET[0]}/val_loss',
        mode='min', save_top_k=1, verbose=True)

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        logger=wandblogger,
        log_every_n_steps=1,
        enable_progress_bar=True,
        max_epochs=1,
        val_check_interval=400,
        num_sanity_val_steps=0,
        callbacks=checkpoint_callback
    )

    model = HMR(hparams=exp_config)
    trainer.fit(model)
    trainer.test(ckpt_path='best')


if __name__ == '__main__':
    sweep_config = {
        'method': 'grid',
        'name': args.exp_name,
        'metric': {
            'goal': 'minimize',
            'name': f'val_{base_config.DATASET.VAL.SET[0]}/val_loss'
        },
        'parameters': {
            "SEED_VALUE": {'values': [1, 2, 3]}
        }
    }
    single_experiment()
    # sweep_id = wandb.sweep(sweep_config, project="DDA")
    # wandb.agent(sweep_id=sweep_id, function=single_experiment, count=3)
