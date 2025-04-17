# -*- coding: utf-8 -*-

# @Description: 存档
# @Author: CaptainHu
# @Date: 2025-04-15 15:04:09
# @LastEditors: CaptainHu
from collections import deque

class Stack:
    def __init__(self,initial_items=None):
        self.items = deque()
        if initial_items is not None:
            self.items.extend(initial_items[::-1])

    def __repr__(self):
        return f"Stack({list(self.items)})"

    def batch_push(self, items):
        self.items.extend(items[::-1])
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        raise IndexError("Pop from empty stack")
    
    def peek(self):
        if not self.is_empty():
            return self.items[-1]
        raise IndexError("Peek from empty stack")
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)

    def clear(self):
        self.items.clear()

class TaskSaver:
    pass

class AtomicActionSaver:
    pass