import argparse
from typing import Any

from pytorch_lightning.utilities import rank_zero_only

from lib.cfg import update_cfg
from lib.models.hmr import HMR
from lib.dataset.dataset import DatasetDepth
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import LightningLoggerBase

ORIGINAL_CHKPT = '/home/batuhan/ssl-part-render/results/cviu/3dpwn_l3d_drl_cp/DEPTH_LOSS_9.0_SEED_VALUE_4_' \
                 '/3dpwn_l3d_drl_cp/a0f9ee9a629e4aba95ac83adef5b9b12/checkpoints/epoch=4-val_loss=198.6940.ckpt'
def update_hmr_checkpoint(in_file, out_file):
    chkpt = torch.load(in_file)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for name, param in chkpt['state_dict'].items():
        if 'model' in name:
            name = name.replace('model.', '')
        new_state_dict[name] = param
    chkpt['state_dict'] = new_state_dict
    torch.save(chkpt, out_file)


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, help='cfg file path')
args = parser.parse_args()
cfg = update_cfg(args.cfg)
if cfg.SEED_VALUE >= 0:
    pl.seed_everything(cfg.SEED_VALUE, workers=True)

# model = HMR.load_from_checkpoint('epoch=4-val_loss=198.6940.ckpt', strict=False, hparams=cfg)
# model.eval()

class MyLogger(LightningLoggerBase):
    @property
    def experiment(self) -> Any:
        pass

    def log_text(self, *args, **kwargs) -> None:
        pass

    def log_image(self, *args, **kwargs) -> None:
        pass

    @property
    def name(self):
        return "MyLogger"

    @property
    def version(self):
        # Return the experiment version, int or str.
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        print(params)

    @rank_zero_only
    def log_metrics(self, metrics, step):
        print(metrics)


trainer = pl.Trainer(
    gpus=1,
    flush_logs_every_n_steps=1,
    enable_progress_bar=True,
    progress_bar_refresh_rate=1,
    max_epochs=1,
    logger=MyLogger()
)

model = HMR(hparams=cfg)
trainer.fit(model)
print(trainer.test())
