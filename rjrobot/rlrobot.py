# -*- coding: utf-8 -*-

# @Description:
# @Author: CaptainHu
# @Date: 2025-04-14 12:38:42
# @LastEditors: CaptainHu
from typing import Dict, Any
from copy import deepcopy
from collections import defaultdict

import torch

from rjrobot.utils import build_obj_from_dict
import rjrobot.models.expert_tools_models as rmetm
import rjrobot.models.encoders as rme
import rjrobot.models.vlm as rmv
import rjrobot.models.act_experts as rmae
import rjrobot.models.act_safe_guard as rmasg
from rjrobot.constants import SUBTASK_SYS_PROMPT, ACTIONPLAN_SYS_PROMPT
from rjrobot.saver import Stack
from rjrobot.common import ActionFlag, ErrorCode


class RJRobotPolicy(object):
    def __init__(self, cfg: dict):
        # init each module model
        self.expert_tools = defaultdict(list)
        for etm_args in cfg["expert_tools_models"]:
            m = build_obj_from_dict(etm_args, parent=rmetm)
            self.expert_tools[m.inputdata_modality].append(m)

        self.encoders = {
            x["inputdata_modality"]: build_obj_from_dict(x, parent=rme)
            for x in cfg["encoders"]
        }
        self.act_planer: rmv.OnlineVLM = build_obj_from_dict(cfg["vlm"], parent=rmv)
        self.action_expert = build_obj_from_dict(cfg["action_expert"], parent=rmae)
        self.act_safe_guard = build_obj_from_dict(cfg["act_safe_guard"], parent=rmasg)

        self.sub_tasks = Stack()
        self.current_sub_task_atom_actions = Stack()

    def __call__(self, inputs: Dict[str, Dict[str, Any]], **kwds):
        tmp_inputs: dict = deepcopy(inputs)
        # expert_tools deal
        for modality, datas in inputs.items():
            if modality in self.expert_tools.keys():
                for expert_tool in self.expert_tools[modality]:
                    tmp_inputs.setdefault(expert_tool.outdata_modality, {}).update(
                        {
                            data_name
                            + "_"
                            + expert_tool.__class__.__name__: expert_tool.predict(data)
                            for data_name, data in datas.items()
                        }
                    )
        # encoder deal
        encoder_outputs = defaultdict(dict)
        for modality, datas in tmp_inputs.items():
            if modality in self.encoders.keys():
                encoder = self.encoders[modality]
                encoder_outputs[modality].update(
                    {
                        data_name + "_" + encoder.__class__.__name__:  # ruff: noqa
                        encoder.predict(data)  # ruff: noqa
                        for data_name, data in datas.items()
                    }
                )
        prompt_info = tuple(encoder_outputs.pop("text").values())[0][0]
        observations = encoder_outputs
        if self.sub_tasks.is_empty():
            # plan sub-task
            self.act_planer.clear_history()
            self.act_planer.set_system_prompt(SUBTASK_SYS_PROMPT)
            subs_task = self.act_planer.predict(observations, prompt_info)
            if subs_task:
                self.sub_tasks.batch_push(subs_task)
            else:
                return None, ErrorCode.PLAN_SUB_TASK_FAIL

        print("current sub-task: ", self.sub_tasks.peek())

        if self.current_sub_task_atom_actions.is_empty():
            # plan atomic action
            self.act_planer.clear_history()
            self.act_planer.set_system_prompt(ACTIONPLAN_SYS_PROMPT)

            atom_actions = self.act_planer.predict(observations, self.sub_tasks.peek())
            if atom_actions:
                self.current_sub_task_atom_actions.batch_push(atom_actions)
            else:
                return torch.Tensor([0]*7), ErrorCode.PLAN_SUB_ATOM_ACTION_FAIL 
        print("current atomic action: ", self.current_sub_task_atom_actions.peek())

        current_atomic_action = self.current_sub_task_atom_actions.peek()
        action = self.action_expert.predict(observations, current_atomic_action)
        safe_flag = self.act_safe_guard.predict(observations,prompt_info, action)
        if safe_flag ==ActionFlag.SAFE:
            self.current_sub_task_atom_actions.pop()
            if self.current_sub_task_atom_actions.is_empty():
                self.sub_tasks.pop()
                if self.sub_tasks.is_empty():
                    # all sub-tasks are done
                    return action, ErrorCode.TASK_SUCCESS 
        elif safe_flag == ActionFlag.STOP:
            # replan sub-task
            self.current_sub_task_atom_actions.clear()
        return action, safe_flag
