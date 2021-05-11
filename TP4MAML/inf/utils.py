import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def safe_sqrt(arr, eps=1e-20):
  return torch.sqrt(torch.clamp(arr, min=eps))

def safe_acos(arr, eps=1e-20):
  # print(torch.min(arr), torch.max(arr))
  return torch.acos(torch.clamp(arr, min=-1+eps, max=1-eps))

def F00ReLU(c, v, v2, eps=1e-5):
  '''
  Computes E[ReLU(z1) ReLU(z2)]
  where (z1, z2) is sampled from a 2-dimensional Gaussian
  with mean 0 and covariance
  |v    cov|
  |cov  v  |
  or
  |v    cov|
  |cov  v2 |
  
  Inputs:
      `c`: correlation of input matrix, c=cov/sqrt(v v2)
      `v`: the first diagonal variance
      `v2`: the second diagonal variance
  The inputs can be tensors, in which case they need to have the same shape
  '''
  return (0.5 / np.pi) * (safe_sqrt(1 - c**2) + safe_acos(-c) * c) * torch.sqrt(v * v2)


def F11ReLU(c, v, v2, eps=1e-5):
  '''
  Computes E[ReLU'(z1) ReLU'(z2)]
  where (z1, z2) is sampled from a 2-dimensional Gaussian
  with mean 0 and covariance
  |v    cov|
  |cov  v  |
  or
  |v    cov|
  |cov  v2 |
  
  Inputs:
      `c`: correlation of input matrix, c=cov/sqrt(v v2)
      `v`: the first diagonal variance
      `v2`: the second diagonal variance
  The inputs can be tensors, in which case they need to have the same shape
  '''
  return (0.5 / np.pi) * safe_acos(-c)

def F02ReLU(c, v, v2, eps=1e-5):
  '''
  Computes E[ReLU(z1) ReLU''(z2)]
  where (z1, z2) is sampled from a 2-dimensional Gaussian
  with mean 0 and covariance
  |v    cov|
  |cov  v  |
  or
  |v    cov|
  |cov  v2 |
  
  Inputs:
      `c`: correlation of input matrix, c=cov/sqrt(v v2)
      `v`: the first diagonal variance
      `v2`: the second diagonal variance
  The inputs can be tensors, in which case they need to have the same shape
  '''
  return (0.5 / np.pi) * safe_sqrt(v/v2 * (1 - c**2))

def F00ExpSqrt(c, vsqrt=1, v2sqrt=1, sigma=1):
  return torch.exp(
    (vsqrt**2 + v2sqrt**2 + 2 * c * vsqrt * v2sqrt)
      / 2 / sigma**2)


def safe_randn(*args, **kw):
  return torch.randn(*args, **kw, dtype=float).type(torch.get_default_dtype())


def J1(c, eps=1e-20):
  # c[c > 1-eps] = 1-eps
  # c[c < -1+eps] = -1+eps 
  # c = torch.tanh(c / 10.) * 10. # soft clip
  return (safe_sqrt(1-c**2, eps=eps) + (np.pi - safe_acos(c, eps=eps)) * c) / np.pi


def VReLUmatrix(cov, eps=1e-20):
  '''
  Given a covariance matrix, computes
  E[relu(x)relu(x)^T]
  where x is drawn from N(0, cov)
  '''
  d = torch.sqrt(torch.diag(cov))
  c = d[:, None]**(-1) * cov * d**(-1)
  return 0.5 * d[:, None] * J1(c, eps=eps) * d

def F00ReLUsqrt(c, vsqrt, v2sqrt, eps=1e-5):
  '''
  Computes E[ReLU(z1) ReLU(z2)]
  where (z1, z2) is sampled from a 2-dimensional Gaussian
  with mean 0 and covariance
  |v    cov|
  |cov  v  |
  or
  |v    cov|
  |cov  v2 |
  
  Inputs:
      `c`: correlation of input matrix, c=cov/sqrt(v v2)
      `v`: the first diagonal variance
      `v2`: the second diagonal variance
  The inputs can be tensors, in which case they need to have the same shape
  '''
  return (0.5 / np.pi) * (safe_sqrt(1 - c**2) + safe_acos(-c) * c) * (vsqrt * v2sqrt)


def F11ReLUsqrt(c, vsqrt=None, v2sqrt=None, eps=1e-5):
  '''
  Computes E[ReLU'(z1) ReLU'(z2)]
  where (z1, z2) is sampled from a 2-dimensional Gaussian
  with mean 0 and covariance
  |v    cov|
  |cov  v  |
  or
  |v    cov|
  |cov  v2 |
  
  Inputs:
      `c`: correlation of input matrix, c=cov/sqrt(v v2)
      `v`: the first diagonal variance
      `v2`: the second diagonal variance
  The inputs can be tensors, in which case they need to have the same shape
  '''
  return (0.5 / np.pi) * safe_acos(-c)

def F02ReLUsqrt(c, vsqrt, v2sqrt, eps=1e-5):
  '''
  Computes E[ReLU(z1) ReLU''(z2)]
  where (z1, z2) is sampled from a 2-dimensional Gaussian
  with mean 0 and covariance
  |v    cov|
  |cov  v  |
  or
  |v    cov|
  |cov  v2 |
  
  Inputs:
      `c`: correlation of input matrix, c=cov/sqrt(v v2)
      `v`: the first diagonal variance
      `v2`: the second diagonal variance
  The inputs can be tensors, in which case they need to have the same shape
  '''
  return (0.5 / np.pi) * vsqrt/v2sqrt * safe_sqrt(1 - c**2)
    
def J0(c, eps=1e-20):
  # c[c > 1-eps] = 1-eps
  # c[c < -1+eps] = -1+eps 
  # c = torch.tanh(c / 10.) * 10. # soft clip
  return safe_acos(-c, eps=eps) / np.pi

def VStepmatrix(cov, eps=1e-20):
  '''
  Given a covariance matrix, computes
  E[step(x)step(x)^T]
  where x is drawn from N(0, cov)
  '''
  d = torch.sqrt(torch.diag(cov))
  c = d[:, None]**(-1) * cov * d**(-1)
  return 0.5 * J0(c, eps=eps)

def F11norm(A, B, C):
  return torch.einsum('ij,jk,ki->', A.T, VStepmatrix(B @ B.T) * (C @ C.T), A).item()**0.5

def ABnorm(A, B):
  return torch.einsum('ij,jk,ki->', A.T, VReLUmatrix(B @ B.T), A).item()**0.5


class MyLinear(nn.Linear):

  def __init__(self, *args, **kw):
    self.device = kw.pop('device', 'cpu')
    self.bias_alpha = kw.pop('bias_alpha', 1)
    super().__init__(*args, **kw)

  def reset_parameters(self) -> None:
    self.to(self.device)
    super().reset_parameters()

  def forward(self, input):
    return F.linear(input, self.weight,
      self.bias * self.bias_alpha if self.bias is not None else self.bias)