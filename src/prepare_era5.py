import argparse
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

import numcodecs
import numpy as np
import scipy.sparse as sp
import xarray as xr
import zarr
from tqdm.auto import tqdm

from .dataset import SST_INDEX, STATIC_NAMES
from .model import CHANNEL_NAMES

SOURCE = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
)
TARGET_LATITUDE = np.arange(-89.75, 90.0, 0.5, dtype=np.float32)
TARGET_LONGITUDE = np.arange(0.0, 360.0, 0.5, dtype=np.float32)


SURFACE_VARIABLES = [
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation_6hr",
    "sea_surface_temperature",
    "total_column_water_vapour",
    "total_cloud_cover",
]


PRESSURE_VARIABLES = [
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "geopotential",
    "specific_humidity",
]


PRESSURE_LEVELS_HPA = [1000, 925, 850, 700]


def nonperiodic_bounds(
    centers: np.ndarray,
    lower_bound: float,
    upper_bound: float,
) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float64)
    interior = 0.5 * (centers[:-1] + centers[1:])
    return np.concatenate([[lower_bound], interior, [upper_bound]])


def build_latitude_conservative_weights(
    source_latitude: np.ndarray,
    target_latitude: np.ndarray,
) -> sp.csr_matrix:
    source_latitude = np.asarray(source_latitude, dtype=np.float64)
    target_latitude = np.asarray(target_latitude, dtype=np.float64)
    source_bounds = nonperiodic_bounds(source_latitude, -90.0, 90.0)
    target_bounds = nonperiodic_bounds(target_latitude, -90.0, 90.0)

    source_measure_bounds = np.sin(np.deg2rad(source_bounds))
    target_measure_bounds = np.sin(np.deg2rad(target_bounds))

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    iterator = tqdm(
        range(len(target_latitude)),
        desc="Latitude conservative weights",
    )
    for target_index in iterator:
        target_low = target_measure_bounds[target_index]
        target_high = target_measure_bounds[target_index + 1]
        overlap = np.maximum(
            0.0,
            np.minimum(source_measure_bounds[1:], target_high)
            - np.maximum(source_measure_bounds[:-1], target_low),
        )
        source_indices = np.flatnonzero(overlap > 0)
        denominator = target_high - target_low
        for source_index in source_indices:
            rows.append(target_index)
            cols.append(int(source_index))
            data.append(float(overlap[source_index] / denominator))

    return sp.csr_matrix(
        (data, (rows, cols)),
        shape=(len(target_latitude), len(source_latitude)),
    )


def build_longitude_conservative_weights(
    source_longitude: np.ndarray,
    target_longitude: np.ndarray,
    period: float = 360.0,
) -> sp.csr_matrix:
    source_longitude = np.asarray(source_longitude, dtype=np.float64)
    target_longitude = np.asarray(target_longitude, dtype=np.float64)
    source_step = float(np.median(np.diff(source_longitude)))
    target_step = float(np.median(np.diff(target_longitude)))

    source_low = source_longitude - source_step / 2
    source_high = source_longitude + source_step / 2

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for target_index, center in enumerate(
        tqdm(target_longitude, desc="Longitude conservative weights")
    ):
        target_low = center - target_step / 2
        target_high = center + target_step / 2
        accumulated: dict[int, float] = {}
        for shift in (-period, 0.0, period):
            overlap = np.maximum(
                0.0,
                np.minimum(source_high + shift, target_high)
                - np.maximum(source_low + shift, target_low),
            )
            for source_index in np.flatnonzero(overlap > 0):
                accumulated[int(source_index)] = (
                    accumulated.get(int(source_index), 0.0)
                    + float(overlap[source_index])
                )
        for source_index, overlap_value in accumulated.items():
            rows.append(target_index)
            cols.append(source_index)
            data.append(overlap_value / target_step)

    return sp.csr_matrix(
        (data, (rows, cols)),
        shape=(len(target_longitude), len(source_longitude)),
    )


def conservative_regrid_2d(
    field: np.ndarray,
    latitude_weights: sp.csr_matrix,
    longitude_weights: sp.csr_matrix,
    valid_weight: np.ndarray | None = None,
) -> np.ndarray:
    field = np.asarray(field, dtype=np.float64)
    if valid_weight is None:
        valid_weight = np.isfinite(field).astype(np.float64)
    else:
        valid_weight = np.asarray(valid_weight, dtype=np.float64)
        valid_weight = valid_weight * np.isfinite(field)

    safe_field = np.nan_to_num(field, nan=0.0)
    numerator_lat = latitude_weights.dot(safe_field * valid_weight)
    numerator = longitude_weights.dot(numerator_lat.T).T
    denominator_lat = latitude_weights.dot(valid_weight)
    denominator = longitude_weights.dot(denominator_lat.T).T
    result = numerator / np.maximum(denominator, 1e-12)
    result[denominator <= 1e-12] = np.nan
    return result.astype(np.float32)


def read_weather_frame_0p25(
    dataset: xr.Dataset,
    timestamp: np.datetime64,
) -> np.ndarray:
    surface = (
        dataset[SURFACE_VARIABLES]
        .sel(time=timestamp)
        .sortby("latitude")
        .compute()
    )
    pressure = (
        dataset[PRESSURE_VARIABLES]
        .sel(time=timestamp, level=PRESSURE_LEVELS_HPA)
        .sortby("latitude")
        .compute()
    )
    selected_arrays = [
        np.asarray(
            surface[name].transpose("latitude", "longitude").values,
            dtype=np.float32,
        )
        for name in SURFACE_VARIABLES
    ]
    for name in PRESSURE_VARIABLES:
        values = pressure[name].transpose(
            "level", "latitude", "longitude"
        ).values
        selected_arrays.extend(
            np.asarray(values[index], dtype=np.float32)
            for index in range(len(PRESSURE_LEVELS_HPA))
        )
    return np.stack(selected_arrays, axis=0)


def read_static_fields_0p25(
    dataset: xr.Dataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    land_sea_mask = (
        dataset["land_sea_mask"]
        .sortby("latitude")
        .transpose("latitude", "longitude")
        .compute()
        .values
    )
    surface_geopotential = (
        dataset["geopotential_at_surface"]
        .sortby("latitude")
        .transpose("latitude", "longitude")
        .compute()
        .values
    )
    latitude = np.sort(np.asarray(dataset.latitude.values, dtype=np.float32))
    longitude = np.asarray(dataset.longitude.values, dtype=np.float32)
    orography_m = np.asarray(surface_geopotential, dtype=np.float32) / 9.80665
    return (
        np.clip(np.asarray(land_sea_mask, dtype=np.float32), 0.0, 1.0),
        orography_m,
        latitude,
        longitude,
    )


def normalize_orography(orography_m: np.ndarray) -> np.ndarray:
    finite = np.isfinite(orography_m)
    mean = float(orography_m[finite].mean())
    std = float(orography_m[finite].std())
    return ((orography_m - mean) / max(std, 1e-6)).astype(np.float32)


def build_static_tensor(
    land_sea_mask: np.ndarray,
    orography_m: np.ndarray,
    latitude: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    latitude_2d = np.broadcast_to(
        latitude[:, None], land_sea_mask.shape
    ).astype(np.float32)
    static = np.stack(
        [
            land_sea_mask,
            normalize_orography(orography_m),
            np.sin(np.deg2rad(latitude_2d)),
            np.cos(np.deg2rad(latitude_2d)),
        ],
        axis=0,
    ).astype(np.float32)
    ocean_weight = np.clip(1.0 - land_sea_mask, 0.0, 1.0).astype(np.float32)
    return static, ocean_weight


def season_index(day):
    return (date.fromisoformat(str(day)).month % 12) // 3


def select_timestamps(available, start, stop, count, seed):
    times = np.sort(available.astype("datetime64[ns]"))
    times = times[(times >= np.datetime64(start, "ns")) & (times < np.datetime64(stop, "ns"))]
    if count is None:
        return times
    days, starts, sizes = np.unique(times.astype("datetime64[D]"), return_index=True, return_counts=True)

    rng = np.random.default_rng(seed)
    by_season = [[], [], [], []]
    for day, first, size in zip(days, starts, sizes):
        by_season[season_index(day)].append(times[first + rng.integers(size)])
    for group in by_season:
        rng.shuffle(group)
    ordered = []
    for position in range(max(map(len, by_season))):
        for group in by_season:
            if position < len(group):
                ordered.append(group[position])
    return np.sort(np.asarray(ordered[:count], dtype=times.dtype))


def build_cache(source, output, start, stop, count=None, seed=42):
    source = str(source)
    output = Path(output)
    sampling = {"source": source, "start": start, "stop": stop, "frames": count, "seed": seed}
    if output.exists():
        return output
    partial = output.with_name(output.name + ".partial")
    root = None
    completed = 0
    if partial.exists():
        root = zarr.open_group(str(partial), mode="a")
        source = root.attrs["source_uri"]
        times = root["time_ns"][:].astype("datetime64[ns]")
        completed = root.attrs["completed_frames"]
        if completed == len(times):
            partial.rename(output)
            return output

    store_options = {"storage_options": {"token": "anon"}} if source.startswith("gs://") else {}
    with xr.open_zarr(source, consolidated=True, chunks={}, **store_options) as dataset:
        if root is None:
            times = select_timestamps(dataset.time.values, start, stop, count, seed)
        source_lsm, source_orography, source_lat, source_lon = read_static_fields_0p25(dataset)
        target_lat, target_lon = TARGET_LATITUDE, TARGET_LONGITUDE
        lat_weights = build_latitude_conservative_weights(source_lat, target_lat)
        lon_weights = build_longitude_conservative_weights(source_lon, target_lon)
        ocean_source = 1.0 - source_lsm
        if root is None:
            lsm = conservative_regrid_2d(source_lsm, lat_weights, lon_weights)
            orography = conservative_regrid_2d(source_orography, lat_weights, lon_weights)
            static, ocean = build_static_tensor(np.clip(lsm, 0.0, 1.0), orography, target_lat)
            compressor = numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
            output.parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(dir=output.parent, prefix=f".{output.name}.init-") as directory:
                staging = Path(directory) / "store"
                root = zarr.open_group(str(staging), mode="w-")
                root.create_dataset("weather", shape=(len(times), 28, len(target_lat), len(target_lon)),
                                    chunks=(1, 28, 128, 128), dtype="f4", compressor=compressor)
                root.create_dataset("static", data=static, chunks=(4, 128, 128), compressor=compressor)
                root.create_dataset("latitude", data=target_lat)
                root.create_dataset("longitude", data=target_lon)
                root.create_dataset("ocean_weight", data=ocean)
                root.create_dataset("time_ns", data=times.astype(np.int64))
                root.attrs.update(
                    channel_names=CHANNEL_NAMES, static_names=STATIC_NAMES, source_uri=source,
                    target_resolution="0p5", pressure_levels_hpa=PRESSURE_LEVELS_HPA, completed_frames=0,
                    sampling=sampling,
                )
                staging.rename(partial)
            root = zarr.open_group(str(partial), mode="a")

        ocean = root["ocean_weight"][:] > 0.5
        weather = root["weather"]
        for index in tqdm(range(completed, len(times)), total=len(times), initial=completed,
                          desc="ERA5 → Zarr 0.5°", unit="кадр"):
            frame_chunks = (
                partial / "weather" / f"{index}.0.{y}.{x}"
                for y in range((weather.shape[-2] + 127) // 128)
                for x in range((weather.shape[-1] + 127) // 128)
            )
            if all(chunk.is_file() for chunk in frame_chunks):
                root.attrs["completed_frames"] = index + 1
                continue
            frame = read_weather_frame_0p25(dataset, times[index])
            remapped = np.stack([
                conservative_regrid_2d(field, lat_weights, lon_weights,
                                       ocean_source if channel == SST_INDEX else None)
                for channel, field in enumerate(frame)
            ])
            remapped[SST_INDEX] = np.where(ocean, np.nan_to_num(remapped[SST_INDEX], nan=0.0), 0.0)
            weather[index] = remapped
            root.attrs["completed_frames"] = index + 1
    partial.rename(output)
    return output


def main():
    parser = argparse.ArgumentParser(description="Подготовка локального Zarr из WeatherBench 2 ERA5")
    parser.add_argument("--source", default=SOURCE,
                        help="Сырой WeatherBench 2 Zarr, локальный путь или gs://")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", required=True, help="Начало интервала UTC")
    parser.add_argument("--stop", required=True, help="Конец интервала UTC, исключительно")
    parser.add_argument("--frames", type=int, help="Число уникальных дат, равномерно по сезонам")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(build_cache(args.source, args.output, args.start, args.stop, args.frames, args.seed))


if __name__ == "__main__":
    main()
