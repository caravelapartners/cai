import gcsfs
import jax
import numpy as np
import pickle

import pandas as pd
import xarray
import time
import os, argparse


from dinosaur import horizontal_interpolation
from dinosaur import spherical_harmonic
from dinosaur import xarray_utils
import neuralgcm



gcs = gcsfs.GCSFileSystem() # use default credentials

def load_checkpoint(model_path):
    with gcs.open(model_path, 'rb') as f:
        ckpt = pickle.load(f)

    model = neuralgcm.PressureLevelModel.from_checkpoint(ckpt)
    return model

def load_initial_conditions(model, modeldate, base_path='gs://weather_website/zarr/'):
    init_path = f"{base_path}{modeldate.strftime('%Y%m%d%H')}/"
    init_ds = xarray.open_zarr(gcs.get_mapper(init_path), chunks=None)
    init_ds = init_ds.sortby('level')

    ic_grid = spherical_harmonic.Grid(
        latitude_nodes=init_ds.sizes['latitude'],
        longitude_nodes=init_ds.sizes['longitude'],
        latitude_spacing='equiangular_with_poles',
        longitude_offset=0,
    )
    regridder = horizontal_interpolation.ConservativeRegridder(
        ic_grid, model.data_coords.horizontal, skipna=True
    )
    out_ds = xarray_utils.regrid(init_ds, regridder)
    out_ds = xarray_utils.fill_nan_with_nearest(out_ds)

    return out_ds

def run_model(modeldate, model, base_path, output_path):
    start_time = time.time()  # Record the start time
    inner_steps = 24  # save model outputs once every 24 hours
    outer_steps = 15 * 24 // inner_steps
    timedelta = np.timedelta64(1, 'h') * inner_steps
    times = (np.arange(outer_steps) * inner_steps)  # time axis in hours

    ic_ds = load_initial_conditions(model, modeldate, base_path+'/analysis/zarr/')

    # initialize model state
    inputs = model.inputs_from_xarray(ic_ds.isel(time=0))
    input_forcings = model.forcings_from_xarray(ic_ds.isel(time=0))
    rng_key = jax.random.key(42)  # optional for deterministic models
    print(rng_key)
    initial_state = model.encode(inputs, input_forcings, rng_key)

    # use persistence for forcing variables (SST and sea ice cover)
    all_forcings = model.forcings_from_xarray(ic_ds.head(time=1))

    # make forecast
    final_state, predictions = model.unroll(initial_state, all_forcings,
                                            steps=outer_steps, timedelta=timedelta, start_with_input=True)
    predictions_ds = model.data_to_xarray(predictions, times=times)

    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time  # Calculate the execution time
    print(f"Ran {modeldate.strftime('%Y-%m-%d %HZ')} \t Execution time: {execution_time:.3f} seconds")

    with gcs.open(f"{output_path}{modeldate.strftime('%Y%m%d%H')}.nc", 'wb') as f:
        predictions_ds.to_netcdf(f)


def run(modeldates, base_path= '/mnt/disks/era5land/', output_path = 'gs://compostus-stor/weather/cai/fcast/'):
    model = load_checkpoint(base_path + '/model/ens1pt4.pkl')
    for modeldate in modeldates:
        print(f'Running model for {modeldate}')
        run_model(modeldate, model, base_path, output_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelhour", help="modelhour", action="store", default='0')
    parser.add_argument("--startdate", help="startdate", action="store", default='2024-01-01')
    parser.add_argument("--enddate", help="enddate", action="store", default='2024-08-30')
    args = parser.parse_args()

    model_dts = pd.date_range(args.startdate, args.enddate, freq='12H')
    run(model_dts)

if __name__ == "__main__":
    main()
