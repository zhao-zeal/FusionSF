import copy
import json
from typing import Dict, List, Tuple, Union

import os
# import deeplake
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from einops import rearrange, repeat

from src.datasets.split_utils import build_target_time_windows, chronological_boundaries


def get_data_spower(data_dir, solar_power_file, num_sites=10, num_ignored_sites=0, strict=False):
    """
    Parameters
    -----------
    data_dir : str
            Absolute directory path where dataset is located.
    solar_power_file : str
            Relative directory path where solar power data is located in data_dir.
    num_sites : int
            number of power sites to be employed in the dataset, max=88
    num_ignored_sites : int
            number of power sites ignored, so the sites employed will be [num_ignored_sites: num_sites]

    Returns
    -----------
    data_sp : tensor
            solar power data tensor with shape [N, d]
    time : tensor
            timestamp tensor (month, day, hour) with shape [N, 3]
    time_dt : pd.Series
            timestamp series
    length : int
            dataset length
    """
    data_df = pd.read_csv(os.path.join(data_dir, solar_power_file),
                          parse_dates=['datetime'])
    if strict and data_df[['datetime', 'power', 'site']].isna().any().any():
        raise ValueError("fixed_v1 does not silently replace missing power/timestamp/site values")
    if not strict:
        data_df = data_df.fillna(0)
    data_sp = []
    reference_times = None
    for i, (k, v) in enumerate(data_df.groupby('site')):
        if i >= num_sites:
            break
        if i < num_ignored_sites:
            continue
        print('dataset k,v len', k, len(v))
        site_times = pd.DatetimeIndex(v['datetime'])
        if reference_times is None:
            reference_times = site_times
        elif len(site_times) != len(reference_times) or not site_times.equals(reference_times):
            raise ValueError(f"site {k} timestamps are not aligned with the first selected site")
        lats_lons = torch.tensor([v['lat'].values[0], v['lon'].values[0]])
        # v[v['power'] < 0]['power'] = 0
        values = torch.from_numpy(v['power'].values)
        if values.isnan().any():
            print('dataset contains nan len', k, len(v))
        data_sp += [{'site': k, 'values': values, 'lats_lons': lats_lons}]
    length = len(v)
    month = v['datetime'].apply(lambda x: x.month)
    day = v['datetime'].apply(lambda x: x.day)
    hour = v['datetime'].apply(lambda x: x.hour)
    time = torch.from_numpy(np.stack([month, day, hour], axis=0))
    time_dt = v['datetime']
    return data_sp, time, time_dt, length


def get_data_satellite(
    data_dir, satellite_dir, norm_stl=True, num_feature=1, fit_end_time=None,
    return_scaler=False, external_scaler_state=None,
):
    """
    Parameters
    -----------
    data_dir : str
            Absolute directory path where dataset is located.
    satellite_dir : str
            Relative directory path where satellite images is located in data_dir.
    """
    print('load satellite from: [{}]'.format(os.path.join(data_dir, satellite_dir)))
    array_satellite = np.load(os.path.join(data_dir, satellite_dir, 'satellite.npy'))
    T, H, W, C = array_satellite.shape
    scaler_state = None
    if norm_stl:
        flattened = array_satellite.reshape(T, -1)
        if external_scaler_state is not None:
            mean = np.asarray(external_scaler_state["mean"])
            scale = np.asarray(external_scaler_state["scale"])
            if mean.shape != (flattened.shape[1],) or scale.shape != mean.shape:
                raise ValueError("external satellite scaler shape does not match satellite features")
            if np.any(scale == 0):
                raise ValueError("external satellite scaler contains a zero scale")
            array_satellite = ((flattened - mean) / scale).reshape(T, H, W, C)
            scaler_state = copy.deepcopy(external_scaler_state)
            scaler_state["source"] = "external_training_dataset"
        elif fit_end_time is None:
            scaler = StandardScaler()
            fit_values = flattened
            fit_range = "full_dataset_legacy"
            scaler.fit(fit_values)
            array_satellite = scaler.transform(flattened).reshape(T, H, W, C)
            scaler_state = {"mean": scaler.mean_, "scale": scaler.scale_, "fit_range": fit_range}
        else:
            scaler = StandardScaler()
            satellite_times = pd.to_datetime(np.load(os.path.join(data_dir, satellite_dir, 'satellite_times.npy')))
            fit_mask = satellite_times < pd.Timestamp(fit_end_time)
            if not fit_mask.any():
                raise ValueError("no satellite samples fall inside the training scaler range")
            fit_values = flattened[fit_mask]
            fit_range = f"{satellite_times[fit_mask].min()}..{satellite_times[fit_mask].max()}"
            scaler.fit(fit_values)
            array_satellite = scaler.transform(flattened).reshape(T, H, W, C)
            scaler_state = {"mean": scaler.mean_, "scale": scaler.scale_, "fit_range": fit_range}
    data_satellite = torch.from_numpy(array_satellite)
    data_satellite_coords = torch.from_numpy(
        np.load(os.path.join(data_dir, satellite_dir, 'satellite_coords.npy')))
    data_satellite_times = np.load(os.path.join(data_dir, satellite_dir, 'satellite_times.npy'))
    result = (data_satellite[..., :num_feature], data_satellite_times, data_satellite_coords)
    return result + (scaler_state,) if return_scaler else result


def get_data_nwp(
    data_dir, nwp_file, norm_nwp=True, fit_end_time=None, return_scaler=False,
    fit_coordinates=None, external_scaler_state=None,
):
    """
    Parameters
    -----------
    data_dir : str
            Absolute directory path where dataset is located.
    nwp_file : str
            Relative directory path where the numerical weather prediction (nwp) file is located in data_dir.
    """
    df = pd.read_csv(os.path.join(data_dir, nwp_file), parse_dates=['fcst_date'])
    df['lat'] = np.round(df['lat'], 1)
    df['lon'] = np.round(df['lon'], 1)
    value_columns = df.columns.drop(['fcst_date', 'lat', 'lon'])
    # fixed_v1 interpolates independently within each grid point. The legacy call keeps
    # the original whole-frame interpolation behavior for exact reproducibility.
    fixed_processing = fit_end_time is not None or external_scaler_state is not None
    if not fixed_processing:
        df = df.interpolate()
    else:
        df = df.sort_values(['lat', 'lon', 'fcst_date'])
        df[value_columns] = df.groupby(['lat', 'lon'], sort=False)[value_columns].transform(
            lambda group: group.interpolate(limit_direction='both')
        )
    # get nwp start time
    nwp_start_time = df['fcst_date'].iloc[0]

    # process nwp dataframe
    times = df['fcst_date'].copy()
    df = df.drop(columns=['fcst_date'])
    # normalize nwp dataframe
    columns = df.columns.drop(['lat', 'lon'])
    scaler_state = None
    if norm_nwp:
        if external_scaler_state is not None:
            expected_features = list(external_scaler_state.get("feature_names", columns.tolist()))
            if expected_features != columns.tolist():
                raise ValueError("external NWP scaler feature order does not match test NWP data")
            mean = np.asarray(external_scaler_state["mean"])
            scale = np.asarray(external_scaler_state["scale"])
            if mean.shape != (len(columns),) or scale.shape != mean.shape:
                raise ValueError("external NWP scaler shape does not match NWP features")
            if np.any(scale == 0):
                raise ValueError("external NWP scaler contains a zero scale")
            df.loc[:, columns] = (df[columns].to_numpy() - mean) / scale
            scaler_state = copy.deepcopy(external_scaler_state)
            scaler_state["source"] = "external_training_dataset"
        else:
            scaler = StandardScaler()
            fit_mask = np.ones(len(df), dtype=bool) if fit_end_time is None else times < pd.Timestamp(fit_end_time)
            if fit_coordinates is not None:
                allowed = {(round(float(lat), 1), round(float(lon), 1)) for lat, lon in fit_coordinates}
                coordinate_mask = np.array([
                    (lat, lon) in allowed for lat, lon in zip(df["lat"], df["lon"])
                ])
                fit_mask &= coordinate_mask
            if not fit_mask.any():
                raise ValueError("no NWP samples fall inside the training-site scaler range")
            scaler.fit(df.loc[fit_mask, columns])
            df.loc[:, columns] = scaler.transform(df[columns])
            scaler_state = {
                "mean": scaler.mean_, "scale": scaler.scale_,
                "feature_names": columns.tolist(),
                "fit_range": "full_dataset_legacy" if fit_end_time is None else f"{times[fit_mask].min()}..{times[fit_mask].max()}",
                "fit_coordinates": sorted(allowed) if fit_coordinates is not None else "all_coordinates",
            }
    if not fixed_processing:
        # Exact legacy representation includes the two grouping coordinates as guide channels.
        data_nwp_grouped = df.groupby(['lat', 'lon'])
    else:
        # fixed_v1 keeps coordinates out of weather features; station coordinates are supplied
        # separately to the model.
        data_nwp_grouped = df.set_index(['lat', 'lon']).groupby(level=['lat', 'lon'])
    result = (data_nwp_grouped, nwp_start_time)
    return result + (scaler_state,) if return_scaler else result


class Ts3MDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        satellite_dir: str = 'satellite',
        seq_len: int = 24 * 2,
        label_len: int = 12,
        pred_len: int = 24 * 2,
        num_sites: int = 10,
        num_ignored_sites: int = 0,
        norm_nwp: bool = True,
        norm_stl: bool = True,
        modality_mode: str = "all",
        data_pipeline: dict = None,
        train_ratio: float = 0.6,
        valid_ratio: float = 0.2,
        test_ratio: float = 0.2,
        stride: int = 1,
        precomputed_scaler_state: dict = None,
        **kwargs

    ) -> None:
        """
        Parameters
        ----------
        data_dir : str
            Absolute directory path where the deeplake dataset is located.
        seq_len: int
            Number of frames in the input sequence.
        label_len: int
            Number of frames in the label sequence.
        pred_len: int
            Number of frames in the prediction sequence.
        """

        self.data_dir = data_dir
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

        valid_modality_modes = {"power", "power_nwp", "all"}
        if modality_mode not in valid_modality_modes:
            raise ValueError(
                f"modality_mode must be one of {sorted(valid_modality_modes)}, "
                f"but got {modality_mode!r}"
            )
        self.modality_mode = modality_mode
        self.use_nwp = modality_mode in {"power_nwp", "all"}
        self.use_satellite = modality_mode == "all"
        self.data_pipeline = dict(data_pipeline or {"version": "legacy_v0"})
        self.pipeline_version = self.data_pipeline.get("version", "legacy_v0")
        if self.pipeline_version not in {"legacy_v0", "fixed_v1"}:
            raise ValueError("data_pipeline.version must be legacy_v0 or fixed_v1")
        if precomputed_scaler_state is not None and self.pipeline_version != "fixed_v1":
            raise ValueError("precomputed scalers are supported only by fixed_v1")

        self.n_samples = []
        self.year_mapping = {}

        # mzq
        self.num_sites = num_sites
        self.num_ignored_sites = num_ignored_sites
        self.data_dir = './data'

        self.data_sp, self.data_sp_time, self.data_sp_time_dt, self.data_sp_length = (
            get_data_spower(data_dir=data_dir,
                            solar_power_file='solar_power/solar_power.csv',
                            num_sites=num_sites,
                            num_ignored_sites=num_ignored_sites,
                            strict=self.pipeline_version == "fixed_v1"))

        self.window_records = None
        self.scaler_state = copy.deepcopy(precomputed_scaler_state or {})
        fit_end_time = None
        if self.pipeline_version == "fixed_v1":
            self.window_records = build_target_time_windows(
                self.data_sp_time_dt, self.seq_len, self.pred_len,
                train_ratio, valid_ratio, test_ratio, stride,
            )
            fit_end_time = chronological_boundaries(
                self.data_sp_time_dt, train_ratio, valid_ratio, test_ratio
            )["train"][1]
            self.scaler_state["fit_end_exclusive"] = str(fit_end_time)

        if precomputed_scaler_state is not None:
            required_scalers = []
            if self.use_satellite:
                required_scalers.append("satellite")
            if self.use_nwp:
                required_scalers.append("nwp")
            missing = [name for name in required_scalers if name not in precomputed_scaler_state]
            if missing:
                raise ValueError(f"precomputed scaler state is missing: {missing}")
            self.scaler_state["source"] = "external_training_dataset"

        training_coordinates = [
            tuple(float(value) for value in site["lats_lons"].tolist()) for site in self.data_sp
        ]

        if self.use_satellite:
            satellite_result = get_data_satellite(
                data_dir=data_dir, satellite_dir=satellite_dir, norm_stl=norm_stl,
                fit_end_time=fit_end_time, return_scaler=self.pipeline_version == "fixed_v1",
                external_scaler_state=(precomputed_scaler_state or {}).get("satellite"),
            )
            self.data_stl, self.data_stl_times, self.data_stl_coords = satellite_result[:3]
            if len(satellite_result) == 4:
                self.scaler_state["satellite"] = satellite_result[3]
        else:
            self.data_stl = self.data_stl_times = self.data_stl_coords = None

        if self.use_nwp:
            nwp_result = get_data_nwp(
                data_dir=data_dir, nwp_file='nwp/nwp.csv', norm_nwp=norm_nwp,
                fit_end_time=fit_end_time, return_scaler=self.pipeline_version == "fixed_v1",
                fit_coordinates=training_coordinates,
                external_scaler_state=(precomputed_scaler_state or {}).get("nwp"),
            )
            self.data_ec_grouped, self.ec_start_time = nwp_result[:2]
            if len(nwp_result) == 3:
                self.scaler_state["nwp"] = nwp_result[2]
            first_group_key = next(iter(self.data_ec_grouped.groups))
            self.nwp_channels = self.data_ec_grouped.get_group(first_group_key).shape[1]
        else:
            self.data_ec_grouped = self.ec_start_time = None
            self.nwp_channels = 17

        print('ts context nwp 3 modal dataset prepared, num_sites={}, sites_ignored={}'.format(num_sites,
                                                                                               num_ignored_sites))

    def __len__(self) -> int:
        if self.window_records is not None:
            return len(self.window_records) * len(self.data_sp)
        return (self.data_sp_length - self.seq_len - self.pred_len + 1) * (self.num_sites - self.num_ignored_sites)

    def __getitem__(self, idx: int):
        if self.window_records is not None:
            windows_per_site = len(self.window_records)
            site_id = idx // windows_per_site
            record = self.window_records[idx % windows_per_site]
            x_begin_index = record.start_index
        else:
            site_id = idx // (self.data_sp_length-self.seq_len-self.pred_len+1)

            # get index
            x_begin_index = idx % (self.data_sp_length-self.seq_len-self.pred_len+1)
        x_end_index = x_begin_index + self.seq_len
        y_begin_index = x_end_index
        y_end_index = y_begin_index + self.pred_len

        if self.use_satellite:
            stl_begin_index = (
                self.data_sp_time_dt.iloc[x_begin_index]
                - pd.to_datetime(self.data_stl_times[0])
            )
            stl_begin_index = int(stl_begin_index.total_seconds() // 3600)
            stl_end_index = stl_begin_index + self.seq_len
            stl_input = rearrange(
                self.data_stl[stl_begin_index: stl_end_index], 't h w c -> t c h w'
            )
            stl_coords = rearrange(self.data_stl_coords, 'h w c -> c h w')
            _, _, H, W = stl_input.shape
        else:
            # Preserve the batch interface without materializing unused 64x64 data.
            H = W = 1
            stl_input = torch.zeros(self.seq_len, 1, H, W)
            stl_coords = torch.zeros(2, H, W)
        
        # print('data_sp_len', self.data_sp_length)
        ts_input = self.data_sp[site_id]['values'][x_begin_index: x_end_index].unsqueeze(-1)
        ts_target = self.data_sp[site_id]['values'][y_begin_index: y_end_index]  #.unsqueeze(-1)
        ts_time = repeat(self.data_sp_time[:, x_begin_index: x_end_index], 'c t -> t c h w', h=H, w=W)
        ts_time_dt = self.data_sp_time_dt.iloc[x_begin_index: x_end_index]
        ts_coords = self.data_sp[site_id]['lats_lons'].unsqueeze(-1).unsqueeze(-1)

        lat = np.round(float(ts_coords[0, 0, 0]), 1)
        lon = np.round(float(ts_coords[1, 0, 0]), 1)
        if self.use_nwp:
            ec_begin_index = (
                self.data_sp_time_dt.iloc[x_begin_index] - self.ec_start_time
            )
            ec_begin_index = int(ec_begin_index.total_seconds() // 3600) + self.seq_len
            ec_end_index = ec_begin_index + self.pred_len
            ec_input = self.data_ec_grouped.get_group((lat, lon)).values
            ec_input = torch.from_numpy(ec_input[ec_begin_index: ec_end_index])
        else:
            ec_input = torch.zeros(self.pred_len, self.nwp_channels)

        return_tensors = {
            'ts_input': ts_input,  # (torch.Tensor): Station timeseries of shape [T, C2]
            'ts_target': ts_target,  # (torch.Tensor): Target station timeseries of shape [T, C2]
            'ts_time': ts_time,  # (torch.Tensor): Time coordinates of shape [T, C3, H, W]
            'ts_coords': ts_coords,  # (torch.Tensor): Station coordinates of shape [2, 1, 1]
            'stl_input': stl_input,  # (torch.Tensor): Satellite image (stl) Context frames of shape [T, C1, H, W]
            'stl_coords': stl_coords,  # (torch.Tensor): Coordinates of context frames of shape [2, H, W]
            'ec_input': ec_input,  # (torch.Tensor): NWP guide [pred_len, channels]
            # [satellite_available, nwp_available]; values are distinct from data zeros.
            'modality_availability': torch.tensor(
                [float(self.use_satellite), float(self.use_nwp)], dtype=torch.float32
            ),
            'site_id': torch.tensor(int(self.data_sp[site_id]['site']), dtype=torch.long),
            'input_start_timestamp': torch.tensor(pd.Timestamp(self.data_sp_time_dt.iloc[x_begin_index]).value),
            'input_end_timestamp': torch.tensor(pd.Timestamp(self.data_sp_time_dt.iloc[x_end_index - 1]).value),
            'forecast_start_timestamp': torch.tensor(pd.Timestamp(self.data_sp_time_dt.iloc[y_begin_index]).value),
            'forecast_end_timestamp': torch.tensor(pd.Timestamp(self.data_sp_time_dt.iloc[y_end_index - 1]).value),
            'forecast_timestamps': torch.tensor(
                pd.DatetimeIndex(self.data_sp_time_dt.iloc[y_begin_index:y_end_index]).asi8.copy(),
                dtype=torch.long,
            ),
        }
        if self.window_records is not None:
            return_tensors['split_id'] = torch.tensor(
                {'train': 0, 'validation': 1, 'test': 2}[record.split], dtype=torch.int8
            )
        return return_tensors
