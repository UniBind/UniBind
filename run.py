#!/usr/bin/env python3
import json
import shutil

import fire
from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping, ModelCheckpoint

import pandas as pd
import numpy as np
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer

from unibind.data.dataset import UnibindDataset
from unibind.model.models.unibind import UniBind
from unibind.model.mono.monotonic import MonoRegularLayer
import diskcache


class ProteinDataModule(pl.LightningDataModule):
    def __init__(self, df_path, data_root, max_length=256, n_neighbors=32, cols_label=None, col_group='dataset',
                 batch_size=32, num_workers=0, pin_memory=True, shuffle=True, cache_dir=None,
                 ):
        super().__init__()
        self.df_path = df_path
        self.data_root = data_root

        self.max_length = max_length
        self.n_neighbors = n_neighbors
        self.cols_label = cols_label
        self.col_group = col_group
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.cache_dir = cache_dir

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if self.cache_dir is None:
            cache = None
        else:
            cache = diskcache.Cache(directory=self.cache_dir, eviction_policy='none')

        # Assign train/val datasets for use in dataloaders
        if isinstance(self.df_path, pd.DataFrame):
            df = self.df_path
        else:
            df = pd.read_csv(self.df_path)
        df_train = df[df[self.col_group].isin(['train'])]
        df_valid = df[df[self.col_group].isin(['valid'])]
        df_test = df[df[self.col_group].isin(['test'])]
        self.ds_train = UnibindDataset(df_train, self.data_root, cols_label=self.cols_label, max_length=self.max_length,
                                       n_neighbors=self.n_neighbors, train=True, diskcache=cache)
        self.ds_valid = UnibindDataset(df_valid, self.data_root, cols_label=self.cols_label, max_length=self.max_length,
                                       n_neighbors=self.n_neighbors, train=False, diskcache=cache)
        self.ds_test = UnibindDataset(df_test, self.data_root, cols_label=self.cols_label, max_length=self.max_length,
                                      n_neighbors=self.n_neighbors, train=False, diskcache=cache)

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.ds_valid, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, shuffle=False)

    def teardown(self, stage=None):
        pass


def get_model(num_classes=1, model_path=None, model_args={}):
    if model_path is None:
        model_args_ori = dict(output_dim=num_classes, max_relpos=32, pair_feat_dim=64, node_feat_dim=14 * 16,
                              geo_attn_num_layers=3, twotrack=True)
        model_args_ori.update(model_args)
        model = UniBind(**model_args_ori)
    else:
        model = torch.load(model_path)
    return model


class ProteinModelModule(pl.LightningModule):
    def __init__(self, model, cols_label=None, num_classes=None, output_dir=None,
                 lr=0.001, model_args=None, data_args=None,
                 ):
        super().__init__()
        if model_args is None:
            model_args = {}
        if data_args is None:
            data_args = {}
        self.lr = lr
        self.cols_label = cols_label
        self.num_classes = num_classes if num_classes is not None else model.num_classes
        self.output_dir = output_dir
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir) / 'pred'
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.reg_weight = model_args.get('reg_weight', np.ones(self.num_classes) * 0.05)
        self.class_dir = model_args.get('class_dir', np.ones(self.num_classes))
        self.model_args = model_args
        self.model = model
        self.data_args = data_args

        if isinstance(self.reg_weight, int) or isinstance(self.reg_weight, float):
            self.reg_weight = np.ones(self.num_classes) * self.reg_weight
        weights = torch.tensor(self.reg_weight, dtype=torch.float32)
        self.reg_weights = nn.Parameter(weights, requires_grad=False)
        self.regular_layer = MonoRegularLayer(output_dim=self.num_classes)

        weights = torch.tensor(self.class_dir, dtype=torch.float32)
        self.class_dir = nn.Parameter(weights, requires_grad=False)
        self.train_loss = None

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop('v_num', None)
        return tqdm_dict

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, x):
        out = self.model(*x)
        return out

    def on_train_start(self):
        log_hyperparams = {
            "num_classes": self.num_classes,
            "cols_label": self.cols_label,
            "data_args": self.data_args,
            "lr": self.lr,
        }
        log_hyperparams.update(self.model_args)
        self.logger.log_hyperparams(log_hyperparams)

    def _get_reg_loss(self, preds, y):
        preds = self.regular_layer(preds * self.class_dir)
        losses = F.mse_loss(preds, (y * self.class_dir), reduction='none')
        return losses

    def training_step(self, batch, batch_idx):
        x, (y, mask) = batch
        preds = self.model(*x)

        if len(preds.shape) - len(y.shape) == 1:
            y = y[..., None]
            mask = mask[..., None]
        losses = F.mse_loss(preds, y, reduction='none')
        loss = (losses * mask).sum() / (mask.sum().clip(1))

        if self.num_classes > 1:
            losses_reg = self._get_reg_loss(preds, y)
            losses_reg = losses_reg[..., 1:] * self.reg_weights[1:]
            loss_reg = (losses_reg * mask[..., 1:]).sum() / (mask[..., 1:].sum().clip(1))
            loss = (loss + loss_reg)

        self.train_loss = loss.detach()
        return loss

    def validation_step(self, batch, batch_idx):
        x, (y, mask) = batch
        preds = self.model(*x)
        if len(preds.shape) - len(y.shape) == 1:
            y = y[..., None]
            mask = mask[..., None]
        losses = F.mse_loss(preds, y, reduction='none')
        loss = (losses * mask).sum() / (mask.sum().clip(1))
        if self.num_classes > 1:
            losses_reg = self._get_reg_loss(preds, y)
            losses_reg = losses_reg[..., 1:] * self.reg_weights[1:]
            loss_reg = (losses_reg * mask[..., 1:]).sum() / (mask[..., 1:].sum().clip(1))
            loss = (loss + loss_reg)

        if self.train_loss is None:
            self.train_loss = 0
        self.log("train_loss", self.train_loss, prog_bar=False, sync_dist=True)
        self.log("val_loss", loss, prog_bar=False, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, (y, mask) = batch
        preds = self.model(*x)
        return {'pred': preds}

    def test_epoch_end(self, outputs):
        preds = torch.cat([x['pred'] for x in outputs], dim=0)
        preds = preds.to('cpu')
        global_rank = self.global_rank
        indices = np.arange(len(preds)).astype('int')
        output_path = self.output_dir / f'test_pred.{global_rank}.csv'
        result = {'sample_id': indices}
        print(self.num_classes)
        for i in range(self.num_classes):
            result[f'pred_{i}'] = preds[..., i]
        df_pred = pd.DataFrame(result)
        df_pred.to_csv(output_path, index=False)


class ProteinLightRunner(object):
    def __init__(self, gpus=0,batch_size=8, num_workers=0,pin_memory=True,):
        super(ProteinLightRunner, self).__init__()
        self._gpus = gpus
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._pin_memory = pin_memory

    def train(self, df_path, data_root, output_dir, col_group='dataset',
              cols_label=None,
              max_length=128, n_neighbors=None,
              shuffle=True, cache_dir=None,
              lr=0.001, n_epoch=10, patience=10,
              model_args={},
              ):
        """
        Training
        """
        if isinstance(cols_label, str):
            cols_label = cols_label.split(',')
        cols_label = list(cols_label)
        output_dir = Path(output_dir)

        # data module
        data_module = ProteinDataModule(df_path, data_root, col_group=col_group, cols_label=cols_label,
                                        max_length=max_length, n_neighbors=n_neighbors,
                                        batch_size=self._batch_size, num_workers=self._num_workers,
                                        pin_memory=self._pin_memory, shuffle=shuffle,
                                        cache_dir=cache_dir, )
        data_module.setup()
        num_classes = data_module.ds_train.num_classes

        # model module
        model = get_model(num_classes=num_classes, model_args=model_args.get('model', {}))
        data_args_log = {
            'max_length': max_length,
            'cols_label': cols_label,
            'n_neighbors': n_neighbors,
        }
        model_kargs = dict(model=model, lr=lr, cols_label=cols_label,
                           output_dir=output_dir, model_args=model_args, data_args=data_args_log)
        model_module = ProteinModelModule(**model_kargs)

        # trainer
        log_dir = output_dir / 'log'
        logger_csv = CSVLogger(str(log_dir))
        version_dir = Path(logger_csv.log_dir)
        trainer = Trainer(
            gpus=self._gpus,
            max_epochs=n_epoch,
            logger=[
                logger_csv,
            ],
            callbacks=[
                EarlyStopping(monitor="val_loss", mode="min", patience=patience),
                ModelCheckpoint(dirpath=(version_dir / 'checkpoint'), filename='{epoch}-{val_loss:.3f}',
                                monitor="val_loss", mode="min", save_last=True),
                TQDMProgressBar(refresh_rate=1),
            ],
            strategy='ddp',
        )
        trainer.fit(model_module, datamodule=data_module)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        model_module = model_module.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, **model_kargs)

        if trainer.global_rank == 0:
            model = model_module.model
            (output_dir / 'model_data.json').write_text(json.dumps(data_args_log, indent=2))
            torch.save(model, str(output_dir / 'model.pt'))

    def make_input(self, path_pdb, path_mutation, output_dir, path_bin_evoef2='EvoEF2', return_json=False):
        from tools import ProteinTools
        tools = ProteinTools()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True,exist_ok=True)
        path_wt = output_dir/'wt.pdb'
        path_mut = output_dir/'mut.pdb'
        path_mut_list = output_dir/'mutation.txt'

        shutil.copy(path_pdb, path_wt)
        shutil.copy(path_mutation, path_mut_list)
        tools.evoef2(path_wt, path_mut_list, path_mut, path_bin=path_bin_evoef2)

        record = {
            'path_wt': str(path_wt.name),
            'path_mut': str(path_mut.name),
            'mutation': path_mut_list.read_text().strip(';'),
            'dataset': 'test',
        }
        if return_json:
            return record
        else:
            return None

    def inference(self, path_pdb, path_mutation, model_path, output_dir, chainsid=None, path_bin_evoef2='EvoEF2', col_group='dataset'):
        output_dir = Path(output_dir)

        path_csv = output_dir/'input.csv'
        case = self.make_input(path_pdb, path_mutation, output_dir, path_bin_evoef2, return_json=True)
        case['chainids'] = chainsid
        df = pd.DataFrame([case])
        df.to_csv(path_csv)

        model_data_path = Path(model_path).parent / 'model_data.json'

        model = get_model(model_path=model_path)
        model_module = ProteinModelModule(model=model, output_dir=output_dir)

        if model_data_path.exists():
            data_args = json.loads(model_data_path.read_text())
        else:
            data_args = {}
        print(data_args)
        max_length = data_args.get('max_length', 256)
        n_neighbors = data_args.get('n_neighbors', 32)
        cols_label = data_args.get('cols_label', ['ddg'])

        # data module
        data_module = ProteinDataModule(path_csv, output_dir, col_group=col_group, n_neighbors=n_neighbors,
                                        max_length=max_length, cols_label=cols_label,
                                        batch_size=self._batch_size, num_workers=self._num_workers,
                                        pin_memory=self._pin_memory, shuffle=False,
                                        cache_dir=None,
                                        )

        # trainer
        log_dir = output_dir / 'log'
        logger_csv = CSVLogger(str(log_dir))
        trainer = Trainer(
            gpus=self._gpus,
            max_epochs=0,
            logger=[
                logger_csv,
            ],
            callbacks=[
                TQDMProgressBar(refresh_rate=1),
            ],
            strategy='ddp',
        )

        data_module.setup()
        dl_test = data_module.test_dataloader()
        dl_test.dataset.df.to_csv(output_dir / 'test_data.csv', index=False)
        trainer.test(model_module, dataloaders=dl_test)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if trainer.global_rank == 0:
            paths = sorted((output_dir / 'pred').glob('*.csv'))
            dfs_pred = []
            for path in paths:
                global_rank = int(path.name.split('.')[1])
                df_pred = pd.read_csv(path)
                df_pred['sample_id'] = df_pred['sample_id'] * len(paths) + global_rank
                dfs_pred += [df_pred]
            df_pred = pd.concat(dfs_pred).sort_values('sample_id')
            df_pred.to_csv(output_dir / 'test_pred.csv', index=False)

if __name__ == '__main__':
    fire.Fire(ProteinLightRunner)
