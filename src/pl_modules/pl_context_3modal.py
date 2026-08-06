import torch
import wandb
import copy
from torchmetrics import MeanMetric

from . import ContextMixerModule


class Pl3Modal(ContextMixerModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        metrics: dict,
        criterion: torch.nn.Module,
        **kwargs,
    ):
        super().__init__(model, optimizer, scheduler, metrics, criterion, kwargs=kwargs)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.evaluation_mode = kwargs.get("evaluation_mode", "full_modalities")
        for key in self.hparams.metrics.val:
            setattr(self, f"test_{key}", copy.deepcopy(self.hparams.metrics.val[key]))
        self.out_dict = {
            'inputs': [], 'outputs': [], 'targets': [], 'site_ids': [],
            'input_start_timestamps': [], 'input_end_timestamps': [],
            'forecast_start_timestamps': [], 'forecast_end_timestamps': [],
            'forecast_timestamps': [], 'forecast_horizons': [],
        }

    def forward(self, batch, mask=True):
        (
            x_ts,
            x_ctx,
            y_ts,
            y_prev_ts,
            ctx_coords,
            ts_coords,
            time_coords,
            x_ec
        ) = self.prepare_batch(batch)

        out = self.model(
            x_ctx, ctx_coords, x_ts, ts_coords, time_coords, x_ec, mask,
            modality_availability=batch.get('modality_availability'),
            evaluation_mode=self.evaluation_mode,
        )
        return out, y_ts, y_prev_ts, x_ts

    def prepare_batch(self, batch, use_target=True):
        # batch return_tensors = {
        #     'ts_input': ts_input, [T, C2]
        #     'ts_target': ts_target,  [T, C2]
        #     'ts_time': ts_time,  [T, C3, H, W]
        #     'ts_coords': ts_coords,  [2, 1, 1]
        #     'stl_input': stl_input,  [T, C1, H, W]
        #     'stl_coords': stl_coords,  [2, H, W]
        #     'ec_input': ec_input,  [T, C4]
        # }
        x_ts = batch['ts_input'].float()
        x_ctx = batch['stl_input'].float()
        y_ts = batch['ts_target'].unsqueeze(-1).float()
        y_previous_ts = torch.randn_like(x_ts).float()
        ctx_coords = batch['stl_coords'].float()
        ts_coords = batch['ts_coords'].float()
        time_coords = batch['ts_time'].float()
        x_ec = batch['ec_input'].float()

        if use_target:
            return x_ts, x_ctx, y_ts, y_previous_ts, ctx_coords, ts_coords, time_coords, x_ec
        return x_ts, x_ctx, ctx_coords, ts_coords, time_coords, x_ec

    def training_step(self, train_batch, batch_idx):
        y_hat, y_ts, y_prev_ts, x_ts = self(train_batch, mask=True)
        y_hat, vq_loss = y_hat[0], y_hat[1]
        y_hat = y_hat.mean(dim=2)
        # print('y_hat', y_hat.shape, y_ts.shape)
        loss = self.criterion(y_hat, y_ts)
        loss = loss + vq_loss

        # print('train loss', loss)
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, prog_bar=True)

        for key in self.hparams.metrics.train:
            metric = getattr(self, f"train_{key}")
            if hasattr(metric, "needs_previous") and metric.needs_previous:
                metric(y_hat, y_ts, y_prev_ts)
            else:
                metric(y_hat, y_ts)
            self.log(
                f"train/{key}",
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        y_hat, y_ts, y_prev_ts, x_ts = self(val_batch, mask=False)
        y_hat = y_hat.mean(dim=2)

        # print('y_hat valid', y_hat.shape, y_ts.shape)
        loss = self.criterion(y_hat, y_ts)
        # print('mzq, loss got')

        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=True, prog_bar=True)

        for key in self.hparams.metrics.val:
            metric = getattr(self, f"val_{key}")
            if hasattr(metric, "needs_previous") and metric.needs_previous:
                metric(y_hat.clone(), y_ts.clone(), y_prev_ts.clone())
            else:
                metric(y_hat.clone(), y_ts.clone())
            self.log(
                f"val/{key}",
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        return {"predictions": y_hat.clone().requires_grad_(), "ground_truth": y_ts.clone().requires_grad_()}
    
    def test_step(self, val_batch, batch_idx):
        y_hat, y_ts, y_prev_ts, x_ts = self(val_batch, mask=False)
        y_hat = y_hat.mean(dim=2)

        loss = self.criterion(y_hat, y_ts)

        self.out_dict['inputs'] += [x_ts.cpu().detach().numpy()]
        self.out_dict['outputs'] += [y_hat.cpu().detach().numpy()]
        self.out_dict['targets'] += [y_ts.cpu().detach().numpy()]
        metadata_mapping = {
            'site_ids': 'site_id',
            'input_start_timestamps': 'input_start_timestamp',
            'input_end_timestamps': 'input_end_timestamp',
            'forecast_start_timestamps': 'forecast_start_timestamp',
            'forecast_end_timestamps': 'forecast_end_timestamp',
            'forecast_timestamps': 'forecast_timestamps',
        }
        for output_key, batch_key in metadata_mapping.items():
            self.out_dict[output_key].append(val_batch[batch_key].cpu().detach().numpy())
        self.out_dict['forecast_horizons'].append(
            torch.arange(1, y_ts.shape[1] + 1).repeat(y_ts.shape[0], 1).numpy()
        )

        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

        for key in self.hparams.metrics.val:
            metric = getattr(self, f"test_{key}")
            if hasattr(metric, "needs_previous") and metric.needs_previous:
                metric(y_hat, y_ts, y_prev_ts)
            else:
                metric(y_hat, y_ts)
            self.log(
                f"test/{key}",
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        return {"predictions": y_hat, "ground_truth": y_ts}
