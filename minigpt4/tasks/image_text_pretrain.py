"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from pathlib import Path

import torch

from minigpt4.common.dist_utils import is_main_process
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import CONV_VISION_minigptv2
from minigpt4.evaluation.distributed import gather_evaluation_records
from minigpt4.evaluation.evaluator import evaluate_records, write_evaluation_report
from minigpt4.evaluation.prompting import prepare_conversation_texts
from minigpt4.tasks.base_task import BaseTask


@registry.register_task("image_text_pretrain")
class ImageTextPretrainTask(BaseTask):
    def __init__(self):
        super().__init__()

    def _evaluation_config(self):
        if not self.cfg:
            return {}
        return self.cfg.run_cfg.get("evaluation", {})

    @staticmethod
    def _as_list(value, size, default=None):
        if value is None:
            return [default] * size
        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value] * size

    def valid_step(self, model, samples):
        evaluation_cfg = self._evaluation_config()
        generation_cfg = evaluation_cfg.get("generation", {})
        generation_options = {"max_new_tokens": 20, "do_sample": False}
        for key in (
            "max_new_tokens",
            "min_length",
            "num_beams",
            "top_p",
            "repetition_penalty",
            "length_penalty",
            "temperature",
            "do_sample",
        ):
            value = generation_cfg.get(key, None)
            if value is not None:
                generation_options[key] = value

        conversation = CONV_VISION_minigptv2.copy()
        conversation.system = ""
        texts = prepare_conversation_texts(
            samples["instruction_input"], conversation
        )
        predictions = model.generate(
            samples["image"],
            samples["video_features"],
            texts,
            **generation_options,
        )
        batch_size = len(predictions)
        instance_ids = self._as_list(samples.get("instance_id"), batch_size)
        if any(instance_id is None for instance_id in instance_ids):
            raise ValueError(
                "validation datasets must provide stable instance_id values"
            )
        sample_ids = self._as_list(
            samples.get("sample_id", samples.get("image_id")), batch_size
        )
        sample_indexes = self._as_list(samples.get("sample_index"), batch_size)
        datasets = self._as_list(samples.get("dataset"), batch_size, default="")
        splits = self._as_list(samples.get("split"), batch_size, default="")
        tasks = self._as_list(samples.get("task"), batch_size, default="emotion")
        targets = self._as_list(
            samples.get("target_raw", samples.get("answer")), batch_size
        )

        records = []
        for index, prediction in enumerate(predictions):
            records.append(
                {
                    "instance_id": str(instance_ids[index]),
                    "sample_id": str(sample_ids[index]),
                    "dataset": str(datasets[index]),
                    "split": str(splits[index]),
                    "sample_index": int(sample_indexes[index]),
                    "task": str(tasks[index]),
                    "target": str(targets[index]),
                    "prediction": str(prediction),
                }
            )
        return records

    def after_evaluation(
        self,
        val_result,
        split_name,
        epoch,
        dataset=None,
        **kwargs
    ):
        records = gather_evaluation_records(val_result)
        evaluation_cfg = self._evaluation_config()
        task = evaluation_cfg.get("task", "auto")
        labels = evaluation_cfg.get("labels", None)
        if labels is None and dataset is not None:
            labels = getattr(dataset, "labels", getattr(dataset, "emos", None))
        aliases = evaluation_cfg.get("label_aliases", None)
        report = evaluate_records(records, task=task, labels=labels, aliases=aliases)
        report["metrics"].update({"split": split_name, "epoch": epoch})

        if is_main_process():
            result_root = Path(registry.get_path("result_dir"))
            output_dir = result_root / str(split_name) / "epoch_{}".format(epoch)
            write_evaluation_report(report, output_dir)
        return report["metrics"]
