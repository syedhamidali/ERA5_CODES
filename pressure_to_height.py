"""
Transform ERA5 Pressure Levels to Geometric Height using Geopotential (z)
Author: [Hamid Ali Syed](https://github.com/syedhamidali) (@syedhamidali)
References: https://confluence.ecmwf.int/pages/viewpage.action?pageId=151531383
"""

print(__doc__)

__all__ = [
    "pressure_levels_to_geometric_height",
    "pressure_to_height",
]

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d

# Constants
Re = 6371000  # Earth's radius in meters
g = 9.80665  # Gravity constant in m/s^2


def pressure_levels_to_geometric_height(ds, alt_max=15000, alt_res=250):
    """
    Transform ERA5 pressure level data to geometric height.

    Parameters
    ----------
    ds : xarray.Dataset
        Pressure Level Dataset containing variables like 'z' (geopotential in m²/s²).
    alt_max : int, optional
        Maximum altitude in meters for interpolation. Default is 15000 m.
    alt_res : int, optional
        Resolution between altitude levels in meters. Default is 250 m.

    Returns
    -------
    xarray.Dataset
        Interpolated dataset on the new (alt, lat, lon) grid, with attributes preserved.

    Example
    -------
    >>> ds = xr.open_dataset("path_to_era5_data.nc")
    >>> ds_interpolated = pressure_levels_to_geometric_height(ds, alt_max=30000, alt_res=250)
    >>> print(ds_interpolated)

    Author
    ------
    Hamid Ali Syed (https://syedha.com, @syedhamidali)

    References
    ----------
    .. [1] https://confluence.ecmwf.int/pages/viewpage.action?pageId=151531383

    See Also
    --------
    pressure_to_height : Transform ERA5 Reanalysis dataset with multiple time steps.
    """
    # Ensure 'z' exists in the dataset to calculate geopotential height
    if "z" not in ds.data_vars:
        raise ValueError(
            "Dataset must contain 'z' (geopotential in m²/s²) to calculate altitude."
        )

    # Calculate geopotential height (h) in geopotential meters
    Re = 6371000  # Earth's radius in meters
    g = 9.80665  # Gravity constant in m/s^2
    ds["geopotential_height"] = ds["z"] / g
    geometric_height = Re * ds["geopotential_height"] / (Re - ds["geopotential_height"])
    altitude_levels = np.arange(0, alt_max + alt_res, alt_res)

    # Interpolate each variable along the altitude dimension
    interpolated_vars = {}
    for var in ds.data_vars:
        if var in ["z", "geopotential_height"]:
            continue  # Skip processed variables

        # Interpolate along altitude with apply_ufunc
        interp_data = xr.apply_ufunc(
            lambda x, y: interp1d(y, x, bounds_error=False, fill_value="extrapolate")(
                altitude_levels
            ),
            ds[var],
            geometric_height,
            input_core_dims=[["level"], ["level"]],
            output_core_dims=[["alt"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[ds[var].dtype],
        )

        # Store with attributes and add to dictionary
        interp_data.attrs = ds[var].attrs
        interpolated_vars[var] = interp_data

    # Create dataset with interpolated variables and coordinates
    coords = {
        "time": ds.time if "time" in ds.coords else None,
        "alt": altitude_levels,
        "lat": ds.lat,
        "lon": ds.lon,
    }

    # Initialize interpolated dataset and set correct order of coordinates and data_vars
    ds_interpolated = xr.Dataset(interpolated_vars, coords=coords).transpose(
        "time", "alt", "lat", "lon"
    )

    # Add attributes to alt coordinate
    ds_interpolated["alt"].attrs = {
        "standard_name": "altitude",
        "units": "m",
        "long_name": "Geometric Height",
        "positive": "up",
    }

    ds_interpolated["lon"].attrs = ds["lon"].attrs
    ds_interpolated["lat"].attrs = ds["lat"].attrs
    ds_interpolated["time"].attrs = ds["time"].attrs

    return ds_interpolated


def pressure_to_height(ds, alt_max=15000, alt_res=250):
    """
    Transform ERA5 pressure level data to geometric height for a dataset with multiple time steps.

    Parameters
    ----------
    ds : xarray.Dataset
        Pressure Level Dataset containing variables like 'z' (geopotential in m²/s²).
    alt_max : int, optional
        Maximum altitude in meters for interpolation. Default is 15000 m.
    alt_res : int, optional
        Resolution between altitude levels in meters. Default is 250 m.

    Returns
    -------
    xarray.Dataset
        Interpolated dataset on the new (alt, lat, lon) grid, with attributes preserved.

    Author
    ------
    Hamid Ali Syed (https://syedha.com, @syedhamidali)

    References
    ----------
    .. [1] https://confluence.ecmwf.int/pages/viewpage.action?pageId=151531383

    See Also
    --------
    pressure_levels_to_geometric_height : Transform ERA5 Reanalysis dataset with single time step.
    """
    # Rename dimensions and drop unused variables
    rename_dict = {
        "valid_time": "time",
        "pressure_level": "level",
        "latitude": "lat",
        "longitude": "lon",
    }
    ds = ds.rename({k: v for k, v in rename_dict.items() if k in ds})

    # Drop unnecessary variables if present
    ds = ds.drop_vars(["number", "expver"], errors="ignore")

    # Apply transformation for the entire dataset
    ds_interpolated = pressure_levels_to_geometric_height(ds, alt_max, alt_res)

    # Set the correct order for coordinates in `ds_interpolated` by reassigning them in the desired order
    ds_interpolated = ds_interpolated.assign_coords(
        time=ds_interpolated.time,
        alt=ds_interpolated.alt,
        lat=ds_interpolated.lat,
        lon=ds_interpolated.lon,
    )

    return ds_interpolated
