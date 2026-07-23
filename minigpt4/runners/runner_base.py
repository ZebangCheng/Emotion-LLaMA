"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import datetime
import json
import logging
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import webdataset as wds
from minigpt4.common.dist_utils import (
    download_cached_file,
    get_rank,
    get_world_size,
    is_main_process,
    main_process,
)
from minigpt4.common.registry import registry
from minigpt4.common.utils import is_url
from minigpt4.datasets.data_utils import concat_datasets, reorg_datasets_by_split, ChainDataset
from minigpt4.datasets.datasets.dataloader_utils import (
    IterLoader,
    MultiIterLoader,
    PrefetchLoader,
)
from minigpt4.evaluation.tracker import (
    ValidationTracker,
    collapse_single_eval_datasets,
    resolve_best_checkpoint_path,
    validate_split_configuration,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler


@registry.register_runner("runner_base")
class RunnerBase:
    """
    A runner class to train and evaluate a model given a task and datasets.

    The runner uses pytorch distributed data parallel by default. Future release
    will support other distributed frameworks.
    """

    def __init__(self, cfg, task, model, datasets, job_id):
        self.config = cfg
        self.job_id = job_id

        self.task = task
        self.datasets = datasets

        self._model = model

        self._wrapped_model = None
        self._device = None
        self._optimizer = None
        self._scaler = None
        self._dataloaders = None
        self._lr_sched = None

        self.start_epoch = 0
        self._splits_validated = False
        self._best_model_split = None
        self._best_checkpoint_path = None
        self.validation_tracker = ValidationTracker(
            metric_name=self.config.run_cfg.get(
                "metric_for_best_model", "agg_metrics"
            ),
            greater_is_better=self.config.run_cfg.get(
                "greater_is_better", True
            ),
            patience=self.config.run_cfg.get("early_stopping_patience", None),
            min_delta=self.config.run_cfg.get("early_stopping_min_delta", 0.0),
        )

        # self.setup_seeds()
        self.setup_output_dir()

    @property
    def device(self):
        if self._device is None:
            self._device = torch.device(self.config.run_cfg.device)

        return self._device

    @property
    def use_distributed(self):
        return self.config.run_cfg.distributed

    @property
    def model(self):
        """
        A property to get the DDP-wrapped model on the device.
        """
        # Moving and wrapping are separate: a model may already be on the
        # requested device but still need its initial wrapper assignment.
        if self._model.device != self.device:
            self._model = self._model.to(self.device)

        if self._wrapped_model is None:
            if self.use_distributed:
                device_ids = (
                    [self.config.run_cfg.gpu]
                    if self.device.type == "cuda"
                    else None
                )
                self._wrapped_model = DDP(
                    self._model,
                    device_ids=device_ids,
                    find_unused_parameters=True,
                )
            else:
                self._wrapped_model = self._model

        return self._wrapped_model

    @property
    def optimizer(self):
        # TODO make optimizer class and configurations
        if self._optimizer is None:
            num_parameters = 0
            p_wd, p_non_wd = [], []
            attention = []
            for n, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue  # frozen weights
                print(n)
                if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                    p_non_wd.append(p)
                else:
                    p_wd.append(p)
                num_parameters += p.data.nelement()

            logging.info("number of trainable parameters: %d" % num_parameters)
            optim_params = [
                {
                    "params": p_wd,
                    "weight_decay": float(self.config.run_cfg.weight_decay),
                    "lr": float(self.config.run_cfg.init_lr)
                },
                {"params": p_non_wd, "weight_decay": 0, "lr": float(self.config.run_cfg.init_lr)},
            ]

            beta2 = self.config.run_cfg.get("beta2", 0.999)
            self._optimizer = torch.optim.AdamW(
                optim_params,
                lr=float(self.config.run_cfg.init_lr),
                weight_decay=float(self.config.run_cfg.weight_decay),
                betas=(0.9, beta2),
            )

        return self._optimizer

    @property
    def scaler(self):
        amp = self.config.run_cfg.get("amp", False)

        if amp:
            if self._scaler is None:
                self._scaler = torch.cuda.amp.GradScaler()

        return self._scaler

    @property
    def lr_scheduler(self):
        """
        A property to get and create learning rate scheduler by split just in need.
        """
        if self._lr_sched is None:
            lr_sched_cls = registry.get_lr_scheduler_class(self.config.run_cfg.lr_sched)

            # max_epoch = self.config.run_cfg.max_epoch
            max_epoch = self.max_epoch
            # min_lr = self.config.run_cfg.min_lr
            min_lr = self.min_lr
            # init_lr = self.config.run_cfg.init_lr
            init_lr = self.init_lr

            # optional parameters
            decay_rate = self.config.run_cfg.get("lr_decay_rate", None)
            warmup_start_lr = self.config.run_cfg.get("warmup_lr", -1)
            warmup_steps = self.config.run_cfg.get("warmup_steps", 0)
            iters_per_epoch = self.config.run_cfg.get("iters_per_epoch", None)

            if iters_per_epoch is None:
                try:
                    iters_per_epoch = len(
                        self.dataloaders[self.train_splits[0]]
                    )
                except (AttributeError, TypeError):
                    iters_per_epoch = 10000

            self._lr_sched = lr_sched_cls(
                optimizer=self.optimizer,
                max_epoch=max_epoch,
                iters_per_epoch=iters_per_epoch,
                min_lr=min_lr,
                init_lr=init_lr,
                decay_rate=decay_rate,
                warmup_start_lr=warmup_start_lr,
                warmup_steps=warmup_steps,
            )

        return self._lr_sched

    @property
    def dataloaders(self) -> dict:
        """
        A property to get and create dataloaders by split just in need.

        If no train_dataset_ratio is provided, concatenate map-style datasets and
        chain wds.DataPipe datasets separately. Training set becomes a tuple
        (ConcatDataset, ChainDataset), both are optional but at least one of them is
        required. The resultant ConcatDataset and ChainDataset will be sampled evenly.

        If train_dataset_ratio is provided, create a MultiIterLoader to sample
        each dataset by ratios during training.

        Currently do not support multiple datasets for validation and test.

        Returns:
            dict: {split_name: (tuples of) dataloader}
        """
        if self._dataloaders is None:
            self._validate_splits()

            # concatenate map-style datasets and chain wds.DataPipe datasets separately
            # training set becomes a tuple (ConcatDataset, ChainDataset), both are
            # optional but at least one of them is required. The resultant ConcatDataset
            # and ChainDataset will be sampled evenly.
            logging.info(
                "dataset_ratios not specified, datasets will be concatenated (map-style datasets) or chained (webdataset.DataPipeline)."
            )

            batch_sizes = {dataset_name: getattr(self.config.datasets_cfg, dataset_name).batch_size
                           for dataset_name in self.datasets.keys()}
            datasets, batch_sizes = reorg_datasets_by_split(self.datasets, batch_sizes)
            datasets, batch_sizes = collapse_single_eval_datasets(
                datasets, batch_sizes, self.train_splits
            )
            self.datasets = datasets
            # self.datasets = concat_datasets(datasets)

            # print dataset statistics after concatenation/chaining
            for split_name in self.datasets:
                if isinstance(self.datasets[split_name], tuple) or isinstance(
                    self.datasets[split_name], list
                ):
                    # mixed wds.DataPipeline and torch.utils.data.Dataset
                    num_records = sum(
                        [
                            len(d)
                            if not type(d) in [wds.DataPipeline, ChainDataset]
                            else 0
                            for d in self.datasets[split_name]
                        ]
                    )

                else:
                    if hasattr(self.datasets[split_name], "__len__"):
                        # a single map-style dataset
                        num_records = len(self.datasets[split_name])
                    else:
                        # a single wds.DataPipeline
                        num_records = -1
                        logging.info(
                            "Only a single wds.DataPipeline dataset, no __len__ attribute."
                        )

                if num_records >= 0:
                    logging.info(
                        "Loaded {} records for {} split from the dataset.".format(
                            num_records, split_name
                        )
                    )

            # create dataloaders
            split_names = sorted(self.datasets.keys())

            datasets = [self.datasets[split] for split in split_names]
            batch_sizes = [batch_sizes[split] for split in split_names]
            is_trains = [split in self.train_splits for split in split_names]

            print("batch sizes", batch_sizes)

            collate_fns = []
            for dataset in datasets:
                if isinstance(dataset, tuple) or isinstance(dataset, list):
                    collate_fns.append([getattr(d, "collater", None) for d in dataset])
                else:
                    collate_fns.append(getattr(dataset, "collater", None))

            dataloaders = self.create_loaders(
                datasets=datasets,
                num_workers=self.config.run_cfg.num_workers,
                batch_sizes=batch_sizes,
                is_trains=is_trains,
                collate_fns=collate_fns,
            )

            self._dataloaders = {k: v for k, v in zip(split_names, dataloaders)}

        return self._dataloaders

    @property
    def cuda_enabled(self):
        return self.device.type == "cuda"

    @property
    def max_epoch(self):
        return int(self.config.run_cfg.max_epoch)

    @property
    def log_freq(self):
        log_freq = self.config.run_cfg.get("log_freq", 50)
        return int(log_freq)

    @property
    def init_lr(self):
        return float(self.config.run_cfg.init_lr)

    @property
    def min_lr(self):
        return float(self.config.run_cfg.min_lr)

    @property
    def accum_grad_iters(self):
        return int(self.config.run_cfg.get("accum_grad_iters", 1))

    @property
    def valid_splits(self):
        valid_splits = list(self.config.run_cfg.get("valid_splits", []))

        if len(valid_splits) == 0:
            logging.info("No validation splits found.")

        return valid_splits

    @property
    def test_splits(self):
        test_splits = list(self.config.run_cfg.get("test_splits", []))

        return test_splits

    @property
    def train_splits(self):
        train_splits = list(self.config.run_cfg.get("train_splits", []))

        if len(train_splits) == 0:
            logging.info("Empty train splits.")

        return train_splits

    @property
    def evaluate_only(self):
        """
        Set to True to skip training.
        """
        return self.config.run_cfg.get("evaluate", False)

    @property
    def best_model_split(self):
        if self._best_model_split is not None:
            return self._best_model_split
        return self.config.run_cfg.get("best_model_split", None)

    @property
    def use_dist_eval_sampler(self):
        return self.config.run_cfg.get("use_dist_eval_sampler", True)

    @property
    def resume_ckpt_path(self):
        return self.config.run_cfg.get("resume_ckpt_path", None)

    @property
    def train_loader(self):
        train_dataloader = self.dataloaders[self.train_splits[0]]

        return train_dataloader

    def _available_dataset_splits(self):
        available = set()
        for dataset in self.datasets.values():
            if isinstance(dataset, dict):
                available.update(dataset.keys())
            else:
                available.update(self.datasets.keys())
                break
        return available

    def _validate_splits(self):
        if self._splits_validated:
            return
        self._best_model_split = validate_split_configuration(
            available_splits=self._available_dataset_splits(),
            train_splits=self.train_splits,
            valid_splits=self.valid_splits,
            test_splits=self.test_splits,
            evaluate_only=self.evaluate_only,
            best_model_split=self.config.run_cfg.get("best_model_split", None),
        )
        self._splits_validated = True

    def setup_output_dir(self):
        lib_root = Path(registry.get_path("library_root"))

        output_dir = lib_root / self.config.run_cfg.output_dir / self.job_id
        # output_dir = lib_root / self.config.run_cfg.output_dir
        result_dir = output_dir / "result"

        output_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)

        registry.register_path("result_dir", str(result_dir))
        registry.register_path("output_dir", str(output_dir))

        self.result_dir = result_dir
        self.output_dir = output_dir

    def train(self):
        start_time = time.time()

        self.log_config()
        self._validate_splits()

        if self.evaluate_only:
            self.evaluate(cur_epoch="provided", skip_reload=True)
            total_time = time.time() - start_time
            logging.info(
                "Evaluation time {}".format(
                    str(datetime.timedelta(seconds=int(total_time)))
                )
            )
            return

        # resume from checkpoint if specified
        if self.resume_ckpt_path is not None:
            self._load_checkpoint(self.resume_ckpt_path)

        last_epoch = self.start_epoch - 1
        for cur_epoch in range(self.start_epoch, self.max_epoch):
            last_epoch = cur_epoch
            # training phase
            logging.info("Start training")
            train_stats = self.train_epoch(cur_epoch)
            self.log_stats(split_name="train", stats=train_stats)

            # evaluation phase
            should_stop = False
            if len(self.valid_splits) > 0:
                validation_logs = {}
                for split_name in self.valid_splits:
                    logging.info("Evaluating on {}.".format(split_name))

                    val_log = self.eval_epoch(
                        split_name=split_name, cur_epoch=cur_epoch
                    )
                    if val_log is None:
                        raise RuntimeError(
                            "evaluation returned no metrics for split {!r}".format(
                                split_name
                            )
                        )
                    validation_logs[split_name] = val_log

                primary_log = validation_logs[self.best_model_split]
                metric_name = self.validation_tracker.metric_name
                if metric_name not in primary_log:
                    raise KeyError(
                        "metric_for_best_model {!r} is missing from {} metrics: {}".format(
                            metric_name,
                            self.best_model_split,
                            ", ".join(sorted(primary_log.keys())),
                        )
                    )
                tracker_update = self.validation_tracker.update(
                    primary_log[metric_name], cur_epoch
                )
                if tracker_update["improved"]:
                    self._mark_best_checkpoint_saved()
                    self._save_checkpoint(cur_epoch, is_best=True)
                self._save_checkpoint(cur_epoch, checkpoint_name="last")
                should_stop = tracker_update["should_stop"]

                for split_name, val_log in validation_logs.items():
                    if split_name == self.best_model_split:
                        val_log.update(
                            {
                                "best_epoch": self.validation_tracker.best_epoch,
                                "best_metric": self.validation_tracker.best_metric,
                                "bad_epochs": self.validation_tracker.bad_epochs,
                            }
                        )
                    self.log_stats(val_log, split_name)

            else:
                # if no validation split is provided, we just save the checkpoint at the end of each epoch.
                self._save_checkpoint(cur_epoch, is_best=False)

            should_stop = self._broadcast_early_stop(should_stop)

            if self.config.run_cfg.distributed:
                dist.barrier()

            if should_stop:
                logging.info(
                    "Early stopping at epoch %d after %d non-improving validation epoch(s).",
                    cur_epoch,
                    self.validation_tracker.bad_epochs,
                )
                break

        # testing phase
        if len(self.test_splits) > 0:
            test_epoch = (
                "best"
                if self.validation_tracker.has_best
                and self._best_checkpoint_path is not None
                else last_epoch
            )
            self.evaluate(cur_epoch=test_epoch, skip_reload=False)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Training time {}".format(total_time_str))

    def _broadcast_early_stop(self, should_stop):
        if not dist.is_available() or not dist.is_initialized():
            return bool(should_stop)
        flag = torch.tensor(
            [int(bool(should_stop)) if is_main_process() else 0],
            dtype=torch.int32,
            device=self.device,
        )
        dist.broadcast(flag, src=0)
        return bool(flag.item())

    def _mark_best_checkpoint_saved(self):
        self._best_checkpoint_path = os.path.abspath(
            os.path.join(self.output_dir, "checkpoint_best.pth")
        )

    def evaluate(self, cur_epoch="best", skip_reload=False):
        test_logs = dict()

        if len(self.test_splits) > 0:
            for split_name in self.test_splits:
                test_logs[split_name] = self.eval_epoch(
                    split_name=split_name, cur_epoch=cur_epoch, skip_reload=skip_reload
                )

            return test_logs

    def train_epoch(self, epoch):
        # train
        self.model.train()

        return self.task.train_epoch(
            epoch=epoch,
            model=self.model,
            data_loader=self.train_loader,
            optimizer=self.optimizer,
            scaler=self.scaler,
            lr_scheduler=self.lr_scheduler,
            cuda_enabled=self.cuda_enabled,
            log_freq=self.log_freq,
            accum_grad_iters=self.accum_grad_iters,
        )

    @torch.no_grad()
    def eval_epoch(self, split_name, cur_epoch, skip_reload=False):
        """
        Evaluate the model on a given split.

        Args:
            split_name (str): name of the split to evaluate on.
            cur_epoch (int): current epoch.
            skip_reload_best (bool): whether to skip reloading the best checkpoint.
                During training, we will reload the best checkpoint for validation.
                During testing, we will use provided weights and skip reloading the best checkpoint .
        """
        data_loader = self.dataloaders.get(split_name, None)
        assert data_loader, "data_loader for split {} is None.".format(split_name)

        # TODO In validation, you need to compute loss as well as metrics
        # TODO consider moving to model.before_evaluation()
        model = self.unwrap_dist_model(self.model)
        if not skip_reload and cur_epoch == "best":
            model = self._reload_best_model(model)
        model.eval()

        self.task.before_evaluation(
            model=model,
            dataset=self.datasets[split_name],
        )
        results = self.task.evaluation(
            model, data_loader, cuda_enabled=self.cuda_enabled
        )

        if results is not None:
            return self.task.after_evaluation(
                val_result=results,
                split_name=split_name,
                epoch=cur_epoch,
                dataset=self.datasets[split_name],
            )

    def unwrap_dist_model(self, model):
        if self.use_distributed:
            return model.module
        else:
            return model

    def create_loaders(
        self,
        datasets,
        num_workers,
        batch_sizes,
        is_trains,
        collate_fns,
        dataset_ratios=None,
    ):
        """
        Create dataloaders for training and validation.
        """

        def _create_loader(dataset, num_workers, bsz, is_train, collate_fn):
            # create a single dataloader for each split
            if isinstance(dataset, ChainDataset) or isinstance(
                dataset, wds.DataPipeline
            ):
                # wds.WebdDataset instance are chained together
                # webdataset.DataPipeline has its own sampler and collate_fn
                loader = iter(
                    DataLoader(
                        dataset,
                        batch_size=bsz,
                        num_workers=num_workers,
                        pin_memory=True,
                    )
                )
            else:
                # map-style dataset are concatenated together
                # setup distributed sampler

                if self.use_distributed:
                    sampler = DistributedSampler(
                        dataset,
                        shuffle=is_train,
                        num_replicas=get_world_size(),
                        rank=get_rank(),
                    )
                    if not self.use_dist_eval_sampler:
                        # e.g. retrieval evaluation
                        sampler = sampler if is_train else None
                else:
                    sampler = None

                loader = DataLoader(
                    dataset,
                    batch_size=bsz,
                    num_workers=num_workers,
                    pin_memory=True,
                    sampler=sampler,
                    shuffle=sampler is None and is_train,
                    collate_fn=collate_fn,
                    drop_last=True if is_train else False,
                )
                if self.cuda_enabled:
                    loader = PrefetchLoader(loader)

                if is_train:
                    loader = IterLoader(loader, use_distributed=self.use_distributed)

            return loader

        loaders = []

        for dataset, bsz, is_train, collate_fn in zip(
            datasets, batch_sizes, is_trains, collate_fns
        ):
            if isinstance(dataset, list) or isinstance(dataset, tuple):
                if hasattr(dataset[0], 'sample_ratio') and dataset_ratios is None:
                    dataset_ratios = [d.sample_ratio for d in dataset]
                loader = MultiIterLoader(
                    loaders=[
                        _create_loader(d, num_workers, bsz[i], is_train, collate_fn[i])
                        for i, d in enumerate(dataset)
                    ],
                    ratios=dataset_ratios,
                )
            else:
                loader = _create_loader(dataset, num_workers, bsz, is_train, collate_fn)

            loaders.append(loader)

        return loaders

    @main_process
    def _save_checkpoint(self, cur_epoch, is_best=False, checkpoint_name=None):
        """
        Save the checkpoint at the current epoch.
        """

        model_no_ddp = self.unwrap_dist_model(self.model)
        param_grad_dic = {
            k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()
        }
        state_dict = model_no_ddp.state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                # delete parameters that do not require gradient
                del state_dict[k]

        save_obj = {
            "model": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "epoch": cur_epoch,
            "runner_state": self.validation_tracker.state_dict(),
            "best_checkpoint_path": self._best_checkpoint_path,
        }
        if checkpoint_name is not None and is_best:
            raise ValueError("checkpoint_name and is_best cannot be combined")
        checkpoint_suffix = (
            checkpoint_name
            if checkpoint_name is not None
            else ("best" if is_best else cur_epoch)
        )
        save_to = os.path.join(
            self.output_dir,
            "checkpoint_{}.pth".format(checkpoint_suffix),
        )
        logging.info("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
        torch.save(save_obj, save_to)

    def _reload_best_model(self, model):
        """
        Load the best checkpoint for evaluation.
        """
        checkpoint_path = self._best_checkpoint_path
        if checkpoint_path is None or not os.path.isfile(checkpoint_path):
            raise RuntimeError(
                "best checkpoint is unavailable; resume with checkpoint_best.pth "
                "beside the last checkpoint or run another improving validation epoch"
            )

        logging.info("Loading checkpoint from {}.".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        try:
            model.load_state_dict(checkpoint["model"])
        except RuntimeError as e:
            logging.warning(
                """
                Key mismatch when loading checkpoint. This is expected if only part of the model is saved.
                Trying to load the model with strict=False.
                """
            )
            model.load_state_dict(checkpoint["model"], strict=False)
        return model

    def _load_checkpoint(self, url_or_filename):
        """
        Resume from a checkpoint.
        """
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location=self.device)
            checkpoint_source = cached_file
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location=self.device)
            checkpoint_source = url_or_filename
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]
        message = self.unwrap_dist_model(self.model).load_state_dict(state_dict,strict=False)

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scaler and checkpoint.get("scaler") is not None:
            self.scaler.load_state_dict(checkpoint["scaler"])

        self.validation_tracker.load_state_dict(checkpoint.get("runner_state"))
        if self.validation_tracker.has_best:
            self._best_checkpoint_path = resolve_best_checkpoint_path(
                resume_checkpoint_path=checkpoint_source,
                stored_best_checkpoint_path=checkpoint.get("best_checkpoint_path"),
                output_dir=self.output_dir,
            )
            if self._best_checkpoint_path is None:
                logging.warning(
                    "The resumed tracker has a historical best metric, but no "
                    "checkpoint_best.pth is available. Final testing will use "
                    "the latest in-memory model unless validation improves."
                )

        self.start_epoch = checkpoint["epoch"] + 1
        print("resume the checkpoint")
        logging.info("Resume checkpoint from {}".format(url_or_filename))

    @main_process
    def log_stats(self, stats, split_name):
        if isinstance(stats, dict):
            log_stats = {**{f"{split_name}_{k}": v for k, v in stats.items()}}
            with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
        elif isinstance(stats, list):
            pass

    @main_process
    def log_config(self):
        with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(self.config.to_dict(), indent=4) + "\n")
