from enum import Enum


class ActionFlag(Enum):
    SAFE = 1
    WARNING = 2
    STOP = 3


class ErrorCode(Enum):
    PLAN_SUB_TASK_FAIL = 1
    PLAN_SUB_ATOM_ACTION_FAIL = 2
    TASK_SUCCESS = 3
