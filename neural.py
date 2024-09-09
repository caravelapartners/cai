import gcsfs
import jax
import numpy as np
import pickle
import xarray
import time

start_time = time.time()  # Record the start time                                                                                                                                                                                


from dinosaur import horizontal_interpolation
from dinosaur import spherical_harmonic
from dinosaur import xarray_utils
import neuralgcm

gcs = gcsfs.GCSFileSystem(token='anon')

model_name = ''

with gcs.open(f'gs://weather_website/zarr/ens1pt4.pkl', 'rb') as f:
  ckpt = pickle.load(f)

model = neuralgcm.PressureLevelModel.from_checkpoint(ckpt)

init_path = 'gs://weather_website/zarr/2024010100/'
init_ds = xarray.open_zarr(gcs.get_mapper(init_path), chunks=None)
init_ds = init_ds.sortby('level')

#init_ds = init_ds.expand_dims({'time': [np.datetime64('2024-08-29')]})                                                                                                                                                          

#era5_path = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'                                                                                                                                              
#full_era5 = xarray.open_zarr(gcs.get_mapper(era5_path), chunks=None)                                                                                                                                                            

# sliced_era5 = (                                                                                                                                                                                                                
#     full_era5                                                                                                                                                                                                                  
#     [model.input_variables + model.forcing_variables]                                                                                                                                                                          
#     .pipe(                                                                                                                                                                                                                     
#         xarray_utils.selective_temporal_shift,                                                                                                                                                                                 
#         variables=model.forcing_variables,                                                                                                                                                                                     
#         time_shift='24 hours',                                                                                                                                                                                                 
#     )                                                                                                                                                                                                                          
#     .sel(time=slice(demo_start_time, demo_end_time, data_inner_steps))                                                                                                                                                         
#     .compute()                                                                                                                                                                                                                 
# )                                                                                                                                                                                                                              

#lat_spacing = xarray_utils.infer_latitude_spacing(init_ds.latitude)                                                                                                                                                             
#lon_offset = xarray_utils.infer_longitude_offset(init_ds.longitude)            

era5_grid = spherical_harmonic.Grid(
    latitude_nodes=init_ds.sizes['latitude'],
    longitude_nodes=init_ds.sizes['longitude'],
    latitude_spacing='equiangular_with_poles',
    longitude_offset=0,
)
regridder = horizontal_interpolation.ConservativeRegridder(
    era5_grid, model.data_coords.horizontal, skipna=True
)
eval_era5 = xarray_utils.regrid(init_ds, regridder)
eval_era5 = xarray_utils.fill_nan_with_nearest(eval_era5)

inner_steps = 24  # save model outputs once every 24 hours                                                                                                                                                                       
outer_steps = 15 * 24 // inner_steps
timedelta = np.timedelta64(1, 'h') * inner_steps
times = (np.arange(outer_steps) * inner_steps)  # time axis in hours                                                                                                                                                             

# use persistence for forcing variables (SST and sea ice cover)                                                                                                                                                                  
all_forcings = model.forcings_from_xarray(eval_era5.head(time=1))

# initialize model state                                                                                                                                                                                                         
inputs = model.inputs_from_xarray(eval_era5.isel(time=0))
input_forcings = model.forcings_from_xarray(eval_era5.isel(time=0))
rng_key = jax.random.key(42)  # optional for deterministic models                                                                                                                                                                
initial_state = model.encode(inputs, input_forcings, rng_key)

# use persistence for forcing variables (SST and sea ice cover)                                                                                                                                                                  
all_forcings = model.forcings_from_xarray(eval_era5.head(time=1))

# make forecast                                                                                                                                                                                                                  
final_state, predictions = model.unroll(
    initial_state,
    all_forcings,
    steps=outer_steps,
    timedelta=timedelta,
    start_with_input=True,
)
predictions_ds = model.data_to_xarray(predictions, times=times)

end_time = time.time()  # Record the end time                                                                                                                                                                                    
execution_time = end_time - start_time  # Calculate the execution time                                                                                                                                                           
print(f"Execution time: {execution_time:.3f} seconds")

predictions_ds.to_netcdf('/home/miguelf/forecast.nc')



                     
