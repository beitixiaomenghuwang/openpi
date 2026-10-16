import dataclasses
import os
import pathlib
import pickle
import sys
import threading
import types
import typing

import datasets as hf_datasets
import datasets.data_files as hf_data_files
from datasets.fingerprint import Hasher
import jax
from lerobot.common.datasets import lerobot_dataset
import numpy as np
import pytest
import torch.utils.data

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


class _FakeLeRobotDatasetMeta:
    def __init__(self, repo_id: str):
        self.repo_id = repo_id
        self.fps = 10
        self.tasks = {0: "do the thing"}


class _FakeLeRobotDataset:
    """Stand-in for lerobot_dataset.LeRobotDataset, keyed by repo_id, that avoids touching real data."""

    LENGTHS: typing.ClassVar = {"repo_a": 5, "repo_b": 7}

    def __init__(self, repo_id: str, delta_timestamps=None):
        self._repo_id = repo_id

    def __len__(self) -> int:
        return self.LENGTHS[self._repo_id]

    def __getitem__(self, index):
        return {"task_index": 0, "value": index}


def test_create_torch_dataset_concatenates_multiple_repo_ids(monkeypatch):
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata", _FakeLeRobotDatasetMeta)
    monkeypatch.setattr(_data_loader, "FileListLeRobotDataset", _FakeLeRobotDataset)

    data_config = _config.DataConfig(repo_id=["repo_a", "repo_b"], action_sequence_keys=("actions",))
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon=1, model_config=None)

    assert isinstance(dataset, torch.utils.data.ConcatDataset)
    assert len(dataset) == sum(_FakeLeRobotDataset.LENGTHS.values())
    assert dataset[0] == {"task_index": 0, "value": 0}
    # Index 5 falls in the second (repo_b) dataset, at local index 0.
    assert dataset[5] == {"task_index": 0, "value": 0}


def _write_lerobot_dataset(tmp_path, monkeypatch) -> pathlib.Path:
    """Writes a 3-episode, 12-frame LeRobot dataset without videos and returns its root."""
    monkeypatch.setattr(hf_datasets.config, "HF_DATASETS_CACHE", str(tmp_path / "hf_cache"))
    root = tmp_path / "dataset"
    features = {
        "observation.state": {"dtype": "float32", "shape": (2,), "names": None},
        "action": {"dtype": "float32", "shape": (2,), "names": None},
    }
    writer = lerobot_dataset.LeRobotDataset.create("local/dataset", fps=10, root=root, features=features)
    for episode_length in (3, 5, 4):
        for _ in range(episode_length):
            value = np.full(2, writer.meta.total_frames + len(writer.episode_buffer["timestamp"]), np.float32)
            writer.add_frame({"observation.state": value, "action": -value, "task": "do the thing"})
        writer.save_episode()
    return root


def test_resolve_data_files_preserves_order_and_cache_metadata(tmp_path):
    paths = [tmp_path / f"episode_{i}.parquet" for i in range(70)]
    for path in paths:
        path.touch()
    files = [str(path) for path in reversed(paths)] + [str(paths[0])]
    expected = hf_data_files.DataFilesList.from_patterns(files)
    actual = _data_loader._resolve_data_files(files)  # noqa: SLF001

    assert actual == expected == files
    assert actual.origin_metadata == expected.origin_metadata
    assert Hasher.hash(actual) == Hasher.hash(expected)

    stat = paths[0].stat()
    os.utime(paths[0], ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))
    updated = _data_loader._resolve_data_files(files)  # noqa: SLF001
    assert Hasher.hash(updated) != Hasher.hash(actual)
    assert updated.origin_metadata == hf_data_files.DataFilesList.from_patterns(files).origin_metadata


def test_resolve_data_files_runs_in_parallel(tmp_path, monkeypatch):
    files = [str(tmp_path / f"episode_{i}.parquet") for i in range(2)]
    for path in files:
        pathlib.Path(path).touch()
    barrier = threading.Barrier(2, timeout=10)
    original = hf_data_files.resolve_pattern

    def resolve(*args, **kwargs):
        barrier.wait()
        return original(*args, **kwargs)

    monkeypatch.setattr(hf_data_files, "resolve_pattern", resolve)
    assert _data_loader._resolve_data_files(files) == files  # noqa: SLF001


def test_resolve_data_files_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        _data_loader._resolve_data_files([str(tmp_path / "missing.parquet")])  # noqa: SLF001


def test_file_list_lerobot_dataset_matches_lerobot(tmp_path, monkeypatch):
    root = _write_lerobot_dataset(tmp_path, monkeypatch)
    delta_timestamps = {"action": [0.0, 0.1, 0.2]}
    expected = lerobot_dataset.LeRobotDataset("local/dataset", root=root, delta_timestamps=delta_timestamps)
    dataset = _data_loader.FileListLeRobotDataset(str(root), delta_timestamps=delta_timestamps)
    files = [str(root / dataset.meta.get_data_file_path(i)) for i in range(dataset.meta.total_episodes)]
    serial = hf_datasets.load_dataset("parquet", data_files=files, split="train")
    assert dataset.hf_dataset.cache_files == serial.cache_files

    # DataLoader workers get a pickled copy, which leaves out the per-episode metadata.
    worker_copy = pickle.loads(pickle.dumps(dataset))
    assert worker_copy.meta.episodes_stats is None
    assert dataset.meta.episodes_stats is not None
    assert "hf_dataset" not in vars(worker_copy)  # Reopened on first use.

    assert len(dataset) == len(worker_copy) == len(expected) == 12
    for index in range(len(expected)):
        expected_item = expected[index]
        for item in (dataset[index], worker_copy[index]):
            assert item.keys() == expected_item.keys()
            for key, value in expected_item.items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(item[key], value)
                else:
                    assert item[key] == value


def test_file_list_lerobot_dataset_downloads_missing_files(tmp_path, monkeypatch):
    root = _write_lerobot_dataset(tmp_path, monkeypatch)
    missing = root / "data/chunk-000/episode_000001.parquet"
    contents = missing.read_bytes()
    missing.unlink()
    downloads = []

    def download_episodes(self, *, download_videos=True):
        downloads.append(download_videos)
        missing.write_bytes(contents)

    monkeypatch.setattr(_data_loader.FileListLeRobotDataset, "download_episodes", download_episodes)
    monkeypatch.setattr(lerobot_dataset, "get_safe_version", lambda repo_id, version: version)
    dataset = _data_loader.FileListLeRobotDataset(str(root))

    assert downloads == [True]
    assert len(dataset) == 12


def test_main_module_imports(monkeypatch):
    main_module = types.ModuleType("__main__")
    main_module.np = np
    main_module.lerobot_dataset = lerobot_dataset
    main_module.value = 1
    monkeypatch.setitem(sys.modules, "__main__", main_module)

    assert _data_loader._main_module_imports() == ["lerobot.common.datasets.lerobot_dataset", "numpy"]  # noqa: SLF001


def test_collate_fn_returns_tensors_for_numeric_data():
    items = [
        {"image": np.full((2, 2), i, np.uint8), "mask": np.ones((), bool), "name": np.array("x")} for i in range(3)
    ]
    batch = _data_loader._collate_fn(items)  # noqa: SLF001

    assert isinstance(batch["image"], torch.Tensor)
    assert batch["image"].dtype == torch.uint8
    np.testing.assert_array_equal(batch["image"].numpy()[:, 0, 0], [0, 1, 2])
    assert isinstance(batch["mask"], torch.Tensor)
    assert isinstance(batch["name"], np.ndarray)


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
