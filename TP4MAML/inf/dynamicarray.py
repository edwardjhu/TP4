import torch
from torch import nn, optim
from copy import deepcopy
import numpy as np

class DynArr():
  def __init__(self, d=None, resizemult=2, initsize=0, initbuffersize=10, device='cpu', **kw):
    self.d = d
    self.device = device
    self.resizemult = resizemult
    if d is not None:
      self.arr = torch.zeros([initbuffersize, d], device=device)
    else:
      self.arr = torch.zeros([initbuffersize], device=device)
    self.size = initsize
  def isempty(self):
    return self.size == 0
  def _cat(self, arr):
    size = arr.shape[0]
    if self.size + size > len(self.arr):
      if self.d is not None:
        assert arr.shape[1] == self.d
        new_arr = torch.zeros([int((self.resizemult-1) * (self.size + size)), self.d], device=self.arr.device)
      else:
        assert len(arr.shape) == 1
        new_arr = torch.zeros([int((self.resizemult-1) * (self.size + size))], device=self.arr.device)
      self.arr = torch.cat([self.arr, new_arr])
    self.arr[self.size:self.size+size] = arr
    self.size += size
  def cat(self, *arrs):
    for arr in arrs:
      self._cat(arr)
  def checkpoint(self):
    self._checkpoint = self.size
  def restore(self):
    self.size = self._checkpoint
  @property
  def a(self):
    return self.arr[:self.size]
  def cuda(self):
    self.arr = self.arr.cuda()
    return self
  def cpu(self):
    self.arr = self.arr.cpu()
    return self
  def half(self):
    self.arr = self.arr.half()
    return self
  def float(self):
    self.arr = self.arr.float()
    return self

class CycArr():
  def __init__(self, d=None, maxsize=10000, initsize=0, device='cpu', **kw):
    assert initsize <= maxsize
    self.size = initsize
    if initsize == maxsize:
      self.end = 0
    else:
      self.end = None
    self.d = d
    self.maxsize = maxsize
    self.device = device
    if d is not None:
      self.arr = torch.zeros([maxsize, d], device=device)
    else:
      self.arr = torch.zeros([maxsize], device=device)

  @property
  def a(self):
    if self.size == self.maxsize:
      return self.arr
    return self.arr[:self.size]
  
  def cuda(self):
    self.arr = self.arr.cuda()
    return self
    
  def half(self):
    self.arr = self.arr.half()
    return self

  def isempty(self):
    return self.size == 0

  def _cat(self, arr):
    size = arr.shape[0]
    if self.size == self.maxsize:
      # cyclic writing
      if self.end + size < self.maxsize:
        self.arr[self.end:self.end+size] = arr
        self.end += size
      else:
        p1size = self.maxsize - self.end
        p2size = size - p1size
        self.arr[self.end:] = arr[:p1size]
        self.arr[:p2size] = arr[p1size:]
        self.end = p2size
    elif self.size + size >= self.maxsize:
      assert size < self.maxsize
      # writing at the end and spill over to beginning
      p1size = self.maxsize - self.size
      p2size = size - p1size
      self.arr[self.size:] = arr[:p1size]
      self.arr[:p2size] = arr[p1size:]
      self.end = p2size
      self.size = self.maxsize
    else:
      # noncyclic writing
      self.arr[self.size:self.size+size] = arr
      self.size += size

  def cat(self, *arrs):
    for arr in arrs:
      self._cat(arr)

