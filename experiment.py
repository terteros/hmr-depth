import argparse

import wandb
from pytorch_lightning.callbacks import ModelCheckpoint

from lib.cfg import update_cfg
from lib.models.hmr import HMR
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, help='cfg file path')
parser.add_argument('--exp_name', type=str, help='cfg file path')
args = parser.parse_args()
cfg = update_cfg(args.cfg)
if cfg.SEED_VALUE >= 0:
    pl.seed_everything(cfg.SEED_VALUE, workers=True)

wandblogger = WandbLogger(name=args.exp_name, project='DDA', log_model=True)
wandblogger.log_hyperparams(cfg)

checkpoint_callback = ModelCheckpoint(
    monitor=f'val_{cfg.DATASET.VAL.SET[0]}/val_loss',
    mode='min', save_top_k=1, verbose=True)

trainer = pl.Trainer(
    gpus=1,
    logger=wandblogger,
    log_every_n_steps=1,
    enable_progress_bar=True,
    progress_bar_refresh_rate=1,
    max_epochs=1,
    val_check_interval=400,
    callbacks=checkpoint_callback
)

model = HMR(hparams=cfg)
trainer.fit(model)
print(trainer.test(ckpt_path='best'))
