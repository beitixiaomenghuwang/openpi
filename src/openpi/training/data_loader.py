from collections.abc import Iterator, Sequence
import concurrent.futures
import copy
import logging
import multiprocessing
import multiprocessing.forkserver
import os
import pickle
import sys
import time
import types
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import datasets as hf_datasets
import datasets.data_files as hf_data_files
import jax
import jax.numpy as jnp
from lerobot.common.constants import HF_LEROBOT_HOME
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import lerobot.common.datasets.utils as lerobot_utils
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def _resolve_data_files(files: list[str]) -> hf_data_files.DataFilesList:
    # HF resolves patterns serially, even for explicit paths: ~23 min for 320k files on OSS.
    # Contiguous chunks put workers in different directories, avoiding same-directory FUSE locks.
    # HF still supplies the file order and origin metadata used for cache invalidation.
    chunk_size = max(1, (len(files) + 63) // 64)
    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as pool:
        chunks = list(
            pool.map(
                hf_data_files.DataFilesList.from_patterns,
                (files[i : i + chunk_size] for i in range(0, len(files), chunk_size)),
            )
        )
    return hf_data_files.DataFilesList(
        [path for chunk in chunks for path in chunk],
        [metadata for chunk in chunks for metadata in chunk.origin_metadata],
    )


class FileListLeRobotDataset(lerobot_dataset.LeRobotDataset):
    """LeRobotDataset for a complete local dataset with ~300k episodes on a network file system.

    LeRobotDataset has five steps that scale badly there (timed on CPFS, 320k episodes, 59M frames):
    - load_dataset("parquet", data_dir=...) makes HF datasets infer the splits by globbing the whole data/
      tree once per candidate split pattern (> 1 h). load_hf_dataset lists the files instead, which is
      LeRobot's own path for a subset of episodes. Path resolution runs in parallel for OSS mounts.
    - The timestamp check builds one torch tensor per frame (~15 min, ~70 GB of RAM). Here it reads the
      Arrow columns directly.
    - The file existence check stats ~1M files one after another (~6 min). Here it runs in threads.
    - Each DataLoader worker gets a pickled copy of the dataset: 370 MB that take 30 s, almost all of it
      per-episode metadata that only __init__ reads. __getstate__ leaves that out, and the worker reopens
      the Arrow tables on first use instead of while the DataLoader waits for it to start.
    - Reading an action chunk that runs past the end of its episode (see _query_hf_dataset).
    Otherwise this matches LeRobotDataset.__init__ without `root` and `episodes`.
    """

    def __init__(self, repo_id: str, *, delta_timestamps: dict[str, list[float]] | None = None, tolerance_s=1e-4):
        self.repo_id = repo_id
        self.root = HF_LEROBOT_HOME / repo_id
        self.image_transforms = None
        self.delta_timestamps = delta_timestamps
        self.episodes = None
        self.tolerance_s = tolerance_s
        self.revision = lerobot_dataset.CODEBASE_VERSION
        self.video_backend = lerobot_dataset.get_safe_default_codec()
        self.delta_indices = None
        self.image_writer = None
        self.episode_buffer = None
        self.meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, self.root, self.revision)

        paths = [self.root / path for path in self.get_episodes_file_paths()]

        def find_missing(batch):
            return [path for path in batch if not os.path.isfile(path)]

        # Spread workers across directories; adjacent OSS lookups serialize on FUSE inode locks.
        chunk_size = max(1, (len(paths) + 63) // 64)
        logging.info("Checking %d dataset files with 64 threads", len(paths))
        start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=64) as pool:
            missing = [
                path
                for batch in pool.map(
                    find_missing, (paths[i : i + chunk_size] for i in range(0, len(paths), chunk_size))
                )
                for path in batch
            ]
        logging.info("Checked %d dataset files in %.1f s", len(paths), time.monotonic() - start)
        if missing:
            # Not (completely) downloaded yet: fetch it from the Hub, as LeRobotDataset does.
            logging.info(
                f"{len(missing)} of {len(paths)} files of {repo_id} are missing, e.g. {missing[0]}; downloading"
            )
            self.revision = lerobot_dataset.get_safe_version(repo_id, self.revision)
            self.download_episodes()
        self.hf_dataset = self.load_hf_dataset()

        self.episode_data_index = lerobot_utils.get_episode_data_index(self.meta.episodes, self.episodes)
        table = self.hf_dataset.data
        lerobot_utils.check_timestamps_sync(
            table.column("timestamp").to_numpy(),
            table.column("episode_index").to_numpy(),
            {key: index.numpy() for key, index in self.episode_data_index.items()},
            self.fps,
            self.tolerance_s,
        )
        if delta_timestamps is not None:
            lerobot_utils.check_delta_timestamps(delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = lerobot_utils.get_delta_indices(delta_timestamps, self.fps)

    def __getstate__(self) -> dict:
        # __getitem__ only reads `info` and `tasks` from the metadata.
        meta = copy.copy(self.meta)
        meta.episodes = meta.episodes_stats = meta.stats = None
        # The DataLoader starts one worker at a time and waits until it has unpickled the dataset. Reopening
        # the Arrow tables takes ~6 s of CPU at 59M frames, so a worker does that on first use (__getattr__).
        state = {**self.__dict__, "meta": meta, "_pickled_hf_dataset": pickle.dumps(self.hf_dataset)}
        del state["hf_dataset"]
        return state

    def __getattr__(self, name: str):
        # Only reached for attributes that are not set, so only for hf_dataset of an unpickled copy.
        if name == "hf_dataset" and "_pickled_hf_dataset" in self.__dict__:
            self.hf_dataset = pickle.loads(self.__dict__.pop("_pickled_hf_dataset"))
            return self.hf_dataset
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def load_hf_dataset(self) -> hf_datasets.Dataset:
        episodes = self.episodes if self.episodes is not None else range(self.meta.total_episodes)
        files = [str(self.root / self.meta.get_data_file_path(ep_idx)) for ep_idx in episodes]
        logging.info("Resolving %d parquet paths with 64 threads", len(files))
        start = time.monotonic()
        data_files = _resolve_data_files(files)
        logging.info("Resolved %d parquet paths in %.1f s", len(data_files), time.monotonic() - start)
        # One process per ~10k files, at most 32; only used when the cache has to be built.
        num_proc = min(32, len(files) // 10_000) or None
        hf_dataset = hf_datasets.load_dataset("parquet", data_files=data_files, split="train", num_proc=num_proc)
        hf_dataset.set_transform(lerobot_utils.hf_transform_to_torch)
        return hf_dataset

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        # LeRobot reads an action chunk with hf_dataset.select, which slices for consecutive indices but
        # otherwise rebuilds the index of the whole table: ~6 s of CPU at 59M frames, hit by every chunk that
        # LeRobot clamps at the end of its episode. The indices lie in one episode, so read their span instead.
        result = {}
        for key, indices in query_indices.items():
            if key in self.meta.video_keys:
                continue
            start = min(indices)
            rows = self.hf_dataset[start : max(indices) + 1][key]
            result[key] = torch.stack([rows[index - start] for index in indices])
        return result


def _create_single_torch_dataset(repo_id: str, data_config: _config.DataConfig, action_horizon: int) -> Dataset:
    """Create a torch dataset for a single LeRobot repo id."""
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = FileListLeRobotDataset(
        repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon + data_config.action_sequence_extra_steps)]
            for key in data_config.action_sequence_keys
        },
    )

    if data_config.prompt_from_task:
        # task_index is local to each dataset, so apply the mapping before concatenating.
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def _load_lerobot_episodes(repo_id: str | Sequence[str] | None) -> dict[int, dict]:
    """Reads meta/episodes.jsonl of an already downloaded LeRobot dataset."""
    if not isinstance(repo_id, str) or repo_id == "fake":
        raise ValueError(f"data_config.sampler needs a single LeRobot repo id, got {repo_id!r}.")
    return lerobot_utils.load_episodes(HF_LEROBOT_HOME / repo_id)


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    repo_ids = [repo_id] if isinstance(repo_id, str) else list(repo_id)
    datasets = [_create_single_torch_dataset(rid, data_config, action_horizon) for rid in repo_ids]
    if len(datasets) == 1:
        return datasets[0]
    return torch.utils.data.ConcatDataset(datasets)


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        filter_dict_path=data_config.filter_dict_path,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if data_config.sampler is not None:
        if framework == "pytorch" and torch.distributed.is_initialized():
            raise NotImplementedError("data_config.sampler is not supported with PyTorch DDP.")
        episodes = _load_lerobot_episodes(data_config.repo_id)
        if sum(ep["length"] for ep in episodes.values()) != len(dataset):
            raise ValueError("Episode lengths in meta/episodes.jsonl do not add up to the dataset length.")
        sampler = data_config.sampler.create(episodes, seed=seed)
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            # The DataLoader starts its workers one after another, and each worker first runs the main
            # module, whose imports take ~30 s for scripts/train.py on a network file system: an hour for
            # 128 spawned workers. A fork server imports them once and forks the workers from there, so a
            # worker's imports are cache hits. It is a fresh process that never initializes JAX, so the
            # workers are as fork-safe as spawned ones (none of the preloaded modules starts a thread).
            # Python 3.11 accepts "__main__" as a preload but then skips it, so name the modules instead.
            mp_context = multiprocessing.get_context("forkserver")
            mp_context.set_forkserver_preload([__name__, *_main_module_imports()])

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(
                        lambda x: jax.make_array_from_process_local_data(self._sharding, np.asarray(x)), batch
                    )
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _main_module_imports() -> list[str]:
    """Names of the modules that the main module binds at its top level, e.g. through `import wandb`."""
    main_module = sys.modules.get("__main__")
    if main_module is None:
        return []
    return sorted({value.__name__ for value in vars(main_module).values() if isinstance(value, types.ModuleType)})


def _collate_fn(items):
    """Collate the batch elements into batched torch tensors (numpy arrays for non-numeric data).

    A worker sends a tensor to the main process through shared memory, but pickles a numpy array
    through a pipe. The main process reads that pipe at ~500 MB/s, and a batch of 1024 frames with
    three 224x224 images is ~460 MB. It also only reads when the next batch in order is missing, and
    then reads all the batches queued before it: on 8x B300 that idled the GPUs for 1-2 min every
    128 steps (with 128 workers), slowing training by ~25%.
    """

    def stack(*xs):
        # Make sure to convert to numpy arrays before stacking since some of the incoming elements
        # may be JAX arrays.
        array = np.stack([np.asarray(x) for x in xs], axis=0)
        return torch.from_numpy(array) if array.dtype.kind in "biuf" else array

    return jax.tree.map(stack, *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    # Workers come from a fork server (see TorchDataLoader) and inherit the descriptor that keeps it running.
    # Close it, so that the fork server exits with the main process and the workers notice their parent is gone;
    # otherwise killing the main process leaves the fork server and the workers behind.
    forkserver = multiprocessing.forkserver._forkserver  # noqa: SLF001
    if forkserver._forkserver_alive_fd is not None:  # noqa: SLF001
        os.close(forkserver._forkserver_alive_fd)  # noqa: SLF001
        forkserver._forkserver_alive_fd = None  # noqa: SLF001


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
