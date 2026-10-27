# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import importlib.util
import os

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

import anemoi.transform.commands as commands  # noqa: E402


def _load_module():
    # the module file name has hyphens, so it cannot be imported with a plain import statement
    path = os.path.join(os.path.dirname(commands.__file__), "make-regrid-file.py")
    spec = importlib.util.spec_from_file_location("make_regrid_file", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_array(group, name, data):
    if hasattr(group, "create_array"):  # zarr 3
        group.create_array(name, data=data)
    else:  # zarr 2
        group.create_dataset(name, data=data)


def _write_store(path, **arrays):
    group = zarr.open_group(str(path), mode="w")
    for name, data in arrays.items():
        _write_array(group, name, data)


def test_path_to_lat_lon_anemoi_zarr(tmp_path):
    """The latitudes and longitudes of an anemoi dataset are read with zarr, so the zarr format does not matter."""
    latitudes = np.linspace(-80.0, 80.0, 12)
    longitudes = np.linspace(0.0, 330.0, 12)
    _write_store(tmp_path / "anemoi.zarr", latitudes=latitudes, longitudes=longitudes)

    lat, lon = _load_module()._path_to_lat_lon(str(tmp_path / "anemoi.zarr"))

    np.testing.assert_array_equal(lat, latitudes)
    np.testing.assert_array_equal(lon, longitudes)


def test_path_to_lat_lon_zarr_without_anemoi_arrays_uses_fallback(tmp_path, monkeypatch):
    """A zarr store without ``latitudes``/``longitudes`` arrays goes to the earthkit-data fallback, as before."""
    _write_store(tmp_path / "other.zarr", x=np.zeros(3))
    module = _load_module()
    expected = (np.zeros(1), np.ones(1))
    monkeypatch.setattr(module, "_ds_to_lat_lon", lambda path: expected)

    assert module._path_to_lat_lon(str(tmp_path / "other.zarr")) is expected
