import torch
from torch import nn, optim
import torch.nn.functional as F
from copy import deepcopy
import numpy as np

from inf.dynamicarray import DynArr, CycArr
from inf.utils import safe_sqrt, safe_acos, F00ReLUsqrt, F11ReLUsqrt, F02ReLUsqrt, VReLUmatrix, ABnorm, F11norm, VStepmatrix, J0

# NOTE: Here we only implement the GP and NTK limits.
# The linear 1-hidden-layer infinite-width feature learning network `MetaInfLin1LP` is implemented in `meta.maml.model` directly for the metalearning setting

class InfGP1LP():
  def __init__(self, d, dout, varw=1, varb=0, initbuffersize=None, device='cpu', arrbackend=DynArr):
    self.d = d
    self.dout = dout
    self.device = device
    self.varw = varw
    self.varb = varb
    def arr(dim):
      return arrbackend(dim, initsize=1, initbuffersize=initbuffersize, device=device)
    self.A = arr(d)
    self.B = arr(1)
    self.C = arr(dout)
    self.zero_grad()
    self.initialize()

  def initialize(self):
    for arr in [self.A, self.B, self.C]:
      arr.a[:] = 0

  def zero_grad(self):
    self.dA = []
    self.dB = []
    self.dC = []

  def checkpoint(self):
    for arr in [self.A, self.B, self.C]:
      arr.checkpoint()

  def restore(self):
    for arr in [self.A, self.B, self.C]:
      arr.restore()

  def __call__(self, X, doreshape=True):
    return self.forward(X, doreshape=doreshape)

  def forward(self, X, doreshape=True):
    if doreshape:
      self.X = X = X.reshape(X.shape[0], -1)
    else:
      self.X = X
    # B x 1
    s = (self.varw * X.norm(dim=1, keepdim=True)**2 + self.varb)**0.5
    # B x d
    At = X / s
    # B x 1
    Bt = 1 / s
    q = self.varw * At @ self.A.a.T + self.varb * Bt @ self.B.a.T
    self.At = At
    self.Bt = Bt
    self.s = s
    out = F00ReLUsqrt(q, 1, s) @ self.C.a
    out.requires_grad_()
    out.retain_grad()
    return out

  def backward(self, delta, buffer=None):
    '''
    Input:
      delta: shape B x dout
    '''
    dA = self.dA
    dB = self.dB
    dC = self.dC
    if buffer is not None:
      dA, dB, dC = buffer
    dA.append(self.At)
    dB.append(self.Bt)
    dC.append(delta * self.s)

  def newgradbuffer(self):
    return [], [], []

  def resetbuffer(self, buffer):
    for d in buffer:
      d[:] = []
    return buffer

  def step(self, lr, wd=0, momentum=None, buffer=None, **kw):
    # TODO: momentum not implemented
    dA, dB, dC = self.dA, self.dB, self.dC
    if buffer is not None:
      dA, dB, dC = buffer
    if wd > 0:
      factor = 1 - lr * wd
      self.C.a[:] *= factor
    self.A.cat(*dA)
    self.B.cat(*dB)
    self.C.cat(*[-lr * d for d in dC])

  def wnorms(self):
    W = torch.cat([self.varw**0.5 * self.A.a, self.varb**0.5 * self.B.a], dim=1)
    # doing [1:] to avoid the 0 entries at init
    return [ABnorm(self.C.a[1:], W[1:])]

  def gnorms(self, buffer=None):
    dA, dB, dC = self.dA, self.dB, self.dC
    if buffer is not None:
      dA, dB, dC = buffer
    C = torch.cat(dC)
    A = torch.cat(dA)
    B = torch.cat(dB)
    W = torch.cat([self.varw**0.5 * A, self.varb**0.5 * B], dim=1)
    return [ABnorm(C, W)]

  def gnorm(self, buffer=None):
    return self.gnorms(buffer=buffer)[0]

  def gclip(self, norm, buffer=None, per_param=False):
    gnorm = self.gnorm(buffer)
    ratio = 1
    if gnorm > norm:
      ratio = norm / gnorm
    dA, dB, dC = self.dA, self.dB, self.dC
    if buffer is not None:
      dA, dB, dC = buffer
    for d in dC:
      d *= ratio

  def state_dict(self):
    d = {'A': self.A, 'B': self.B, 'C': self.C,
        'varw': self.varw, 'varb': self.varb}
    return d

  def load_state_dict(self, d):
    self.A = d['A']
    self.B = d['B']
    self.C = d['C']
    self.varw = d['varw']
    self.varb = d['varb']



  def train(self):
    pass

  def eval(self):
    pass


  ### format conversion

  def cuda(self):
    self.A = self.A.cuda()
    self.B = self.B.cuda()
    self.C = self.C.cuda()
    self.device = 'cuda'
    return self

  def cpu(self):
    self.A = self.A.cpu()
    self.B = self.B.cpu()
    self.C = self.C.cpu()
    self.device = 'cpu'
    return self

  def half(self):
    self.A = self.A.half()
    self.B = self.B.half()
    self.C = self.C.half()
    return self

  def float(self):
    self.A = self.A.float()
    self.B = self.B.float()
    self.C = self.C.float()
    return self
    
  def to(self, device):
    if 'cpu' in device.type:
      self.cpu()
    else:
      self.cuda()
    return self

  def sample(self, width, fincls=None):
    if fincls is None:
      fincls = FinGP1LP
    finnet = fincls(self.d, width, self.dout)
    finnet.initialize(self)
    return finnet



class FinGP1LP(nn.Module):
  def __init__(self, datadim, width, ncls, bias=True, nonlin=nn.ReLU, varw=1, varb=1, lincls=nn.Linear):
    super().__init__()
    self.lin1 = lincls(datadim, width, bias=bias)
    self.readout = lincls(width, ncls, bias=False)
    self.nonlin = nonlin()
    self.width = width
    self.datadim = datadim
    self._initialize(varw, varb)

  def _initialize(self, varw, varb):
    with torch.no_grad():
      weight, bias = self.lin1.weight, self.lin1.bias
      weight.normal_()
      weight *= (varw/self.width)**0.5
      weight.requires_grad = False
      if bias is not None:
        bias.normal_()
        bias *= varb**0.5
        bias.requires_grad = False
      self.readout.weight *= 0

  def initialize(self, infnet, keepomegas=False):
    n = self.width
    d = self.datadim
    sigw = infnet.varw**0.5
    sigb = infnet.varb**0.5
    dev = infnet.device
    if not keepomegas:
      self.omegaw = omegaw = torch.randn(n, d, device=dev)
      self.omegab = omegab = torch.randn(n, 1, device=dev)
    else:
      omegaw = self.omegaw
      omegab = self.omegab
    with torch.no_grad():
      self.lin1.weight[:] = sigw * omegaw
      if self.lin1.bias is not None:
        self.lin1.bias[:] = sigb * omegab.reshape(-1)
      self.readout.weight[:] = (self.nonlin(
        sigw * omegaw @ infnet.A.a.T
        + sigb * omegab @ infnet.B.a.T
        ) @ infnet.C.a).T / n**0.5

  def forward(self, X, doreshape=True):
    if doreshape:
      X = X.reshape(X.shape[0], -1)
    return self.readout(self.nonlin(self.lin1(X))) / self.width**0.5


class InfNTK1LP(InfGP1LP):
  def __init__(self, d, dout, varw=1, varb=0, varw2=1, initbuffersize=None, device='cpu', arrbackend=DynArr):
    super().__init__(d=d, dout=dout, varw=varw, varb=varb,  
                    initbuffersize=initbuffersize,
                    device=device, arrbackend=arrbackend)
    self.varw2 = varw2

  def forward(self, X, doreshape=True):
    if doreshape:
      self.X = X = X.reshape(X.shape[0], -1)
    else:
      self.X = X
    # B x 1
    s = (self.varw * X.norm(dim=1, keepdim=True)**2 + self.varb)**0.5
    # B x d
    At = X / s
    # B x 1
    Bt = 1 / s
    c = At @ self.A.a.T
    q = self.varw * c + self.varb * Bt @ self.B.a.T
    self.At = At
    self.Bt = Bt
    self.s = s
    g2 = F00ReLUsqrt(q, 1, s)
    g1 = self.varw2 * F11ReLUsqrt(q) * c * s
    out = (g1 + g2) @ self.C.a
    out.requires_grad_()
    out.retain_grad()
    return out

  def wnorms(self):
    raise NotImplementedError

  def gnorms(self, buffer=None):
    dA, dB, dC = self.dA, self.dB, self.dC
    if buffer is not None:
      dA, dB, dC = buffer
    C = torch.cat(dC)
    A = torch.cat(dA)
    B = torch.cat(dB)
    W = torch.cat([self.varw**0.5 * A, self.varb**0.5 * B], dim=1)
    return [F11norm(C, W, A), ABnorm(C, W)]

  def gnorm(self, buffer=None):
    return np.linalg.norm(self.gnorms(buffer=buffer))

  def sample(self, *args, **kw):
    raise NotImplementedError


  def state_dict(self):
    d = {'A': self.A, 'B': self.B, 'C': self.C,
        'varw': self.varw, 'varb': self.varb,
        'varw2': self.varw2}
    return d

  def load_state_dict(self, d):
    self.A = d['A']
    self.B = d['B']
    self.C = d['C']
    self.varw = d['varw']
    self.varb = d['varb']
    self.varw2 = d['varw2']

