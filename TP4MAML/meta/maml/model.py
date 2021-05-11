import torch.nn as nn
import torch.nn.functional as F
import torch

from collections import OrderedDict
from torchmeta.modules import (MetaModule, MetaConv2d, MetaBatchNorm2d,
                 MetaSequential, MetaLinear)
# from inf.pimlp import FinPiMLP, InfPiMLP
from inf.utils import MyLinear
from inf.inf1lp import FinGP1LP


def conv_block(in_channels, out_channels, **kwargs):
  return MetaSequential(OrderedDict([
    ('conv', MetaConv2d(in_channels, out_channels, **kwargs)),
    ('norm', nn.BatchNorm2d(out_channels, momentum=1.,
      track_running_stats=False)),
    ('relu', nn.ReLU()),
    ('pool', nn.MaxPool2d(2))
  ]))

class SafeMetaLinear(MyLinear, MetaModule):
  def forward(self, input, params=None):
    if params is None:
      params = OrderedDict(self.named_parameters())
    bias = params.get('bias', None)
    return F.linear(input, params['weight'],
      bias * self.bias_alpha if bias is not None else bias)

class MetaConvModel(MetaModule):
  """4-layer Convolutional Neural Network architecture from [1].

  Parameters
  ----------
  in_channels : int
    Number of channels for the input images.

  out_features : int
    Number of classes (output of the model).

  hidden_size : int (default: 64)
    Number of channels in the intermediate representations.

  feature_size : int (default: 64)
    Number of features returned by the convolutional head.

  References
  ----------
  .. [1] Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic Meta-Learning
       for Fast Adaptation of Deep Networks. International Conference on
       Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)
  """
  def __init__(self, in_channels, out_features, hidden_size=64, feature_size=64):
    super(MetaConvModel, self).__init__()
    self.in_channels = in_channels
    self.out_features = out_features
    self.hidden_size = hidden_size
    self.feature_size = feature_size

    self.features = MetaSequential(OrderedDict([
      ('layer1', conv_block(in_channels, hidden_size, kernel_size=3,
                  stride=1, padding=1, bias=True)),
      ('layer2', conv_block(hidden_size, hidden_size, kernel_size=3,
                  stride=1, padding=1, bias=True)),
      ('layer3', conv_block(hidden_size, hidden_size, kernel_size=3,
                  stride=1, padding=1, bias=True)),
      ('layer4', conv_block(hidden_size, hidden_size, kernel_size=3,
                  stride=1, padding=1, bias=True))
    ]))
    self.classifier = MetaLinear(feature_size, out_features, bias=True)

  def forward(self, inputs, params=None):
    features = self.features(inputs, params=self.get_subdict(params, 'features'))
    features = features.view((features.size(0), -1))
    logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
    return logits

class MetaMLPModel(MetaModule):
  """Multi-layer Perceptron architecture from [1].

  Parameters
  ----------
  in_features : int
    Number of input features.

  out_features : int
    Number of classes (output of the model).

  hidden_sizes : list of int
    Size of the intermediate representations. The length of this list
    corresponds to the number of hidden layers.

  References
  ----------
  .. [1] Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic Meta-Learning
       for Fast Adaptation of Deep Networks. International Conference on
       Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)
  """
  def __init__(self, in_features, out_features, hidden_sizes, normalize=None, bias=True, nonlin=nn.ReLU, train_last_layer_only=False):
    super(MetaMLPModel, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.hidden_sizes = hidden_sizes
    self.train_last_layer_only = train_last_layer_only

    def makeblock(din, dout, bias=True):
      block = [
        ('linear', MetaLinear(din, dout, bias=bias)),
        ('relu', nonlin())
      ]
      if normalize == 'BN':
        block.insert(1, 
          ('norm', torch.nn.BatchNorm1d(
          dout,
          affine=True,
          momentum=0.999,
          eps=1e-3,
          track_running_stats=False,
        )))
      elif normalize == 'LN':
        block.insert(1, ('norm', torch.nn.LayerNorm(
          dout,
          elementwise_affine=True
        )))
      elif not normalize in [False, None]:
        raise ValueError(f'invalid `normalize` value {normalize}')
      return OrderedDict(block)

    layer_sizes = [in_features] + hidden_sizes
    self.features = MetaSequential(OrderedDict([('layer{0}'.format(i + 1),
      MetaSequential(
        makeblock(hidden_size, layer_sizes[i+1], bias=bias)
      )) for (i, hidden_size) in enumerate(layer_sizes[:-1])]))
    if not hidden_sizes:
      width = in_features
    else:
      width = hidden_sizes[-1]
    # TODO: classifier bias = bias?
    self.classifier = MetaLinear(width, out_features, bias=bias)
    
  def forward(self, inputs, params=None):
    inputs = inputs.reshape(inputs.shape[0], -1)
    if not self.train_last_layer_only:
      features = self.features(inputs, params=self.get_subdict(params, 'features'))
    else:
      features = self.features(inputs)
    logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
    return logits

class MetaInfLin1LP(MetaModule):

  def __init__(self, in_features, out_features, sigma1=1, sigma2=1, alpha=1, bias_alpha1=0, bias_alpha2=0):
    # no biases
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    h = in_features + out_features
    self.alpha = alpha
    havebias1 = bias_alpha1!=0
    havebias2 = bias_alpha2!=0
    self.features = SafeMetaLinear(in_features, h, bias=havebias1, bias_alpha=bias_alpha1)
    self.classifier = SafeMetaLinear(h, out_features, bias=havebias2, bias_alpha=bias_alpha2)
    with torch.no_grad():
      self.features.weight[:in_features] = torch.eye(in_features) * sigma1
      self.features.weight[in_features:] = 0
      self.classifier.weight[:, :in_features] = 0
      self.classifier.weight[:, in_features:] = torch.eye(out_features) * sigma2
      if havebias1:
        self.features.bias[:] = 0
      if havebias2:
        self.classifier.bias[:] = 0

  def forward(self, inputs, params=None):
    inputs = inputs.reshape(inputs.shape[0], -1)
    features = self.features(inputs, params=self.get_subdict(params, 'features'))
    logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
    return logits * self.alpha

  def sample(self, width):
    net = MetaFinLin1LP(self.in_features, self.out_features, width,
                      alpha=self.alpha, bias_alpha1=self.features.bias_alpha,
                      bias_alpha2=self.classifier.bias_alpha)
    net.initialize(self)
    return net

class MetaFinLin1LP(MetaModule):

  def __init__(self, in_features, out_features, width, alpha=1, bias_alpha1=0, bias_alpha2=0):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.width = width
    self.alpha = alpha
    
    havebias1 = bias_alpha1!=0
    havebias2 = bias_alpha2!=0
    self.features = SafeMetaLinear(in_features, width, bias=havebias1, bias_alpha=bias_alpha1)
    self.classifier = SafeMetaLinear(width, out_features, bias=havebias2, bias_alpha=bias_alpha2)

  def initialize(self, infnet):
    self.alpha = infnet.alpha
    self.features.bias_alpha = infnet.features.bias_alpha
    self.classifier.bias_alpha = infnet.classifier.bias_alpha
    n = self.width
    d = self.in_features
    do = self.out_features
    self.omegas = torch.randn(n, d + do, device=infnet.features.weight.device)
    with torch.no_grad():
      self.features.weight[:] = n**-0.5 * self.omegas @ infnet.features.weight
      self.classifier.weight[:] = n**-0.5 * infnet.classifier.weight @ self.omegas.T
      if infnet.features.bias is not None:
        self.features.bias[:] = n**-0.5 * self.omegas @ infnet.features.bias
      if infnet.classifier.bias is not None:
        self.classifier.bias[:] = infnet.classifier.bias

  def forward(self, inputs, params=None):
    inputs = inputs.reshape(inputs.shape[0], -1)
    features = self.features(inputs, params=self.get_subdict(params, 'features'))
    logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
    return logits * self.alpha

class MetaFinGP1LP(FinGP1LP, MetaModule):

  def __init__(self, *args, **kw):
    kw['lincls'] = SafeMetaLinear
    FinGP1LP.__init__(self, *args, **kw)
    self._children_modules_parameters_cache = dict()
    
  def forward(self, X, params=None, doreshape=True):
    if doreshape:
      X = X.reshape(X.shape[0], -1)
    return self.readout(self.nonlin(self.lin1(X)),
          params=self.get_subdict(params, f'readout')) / self.width**0.5


# class MetaFinPiMLP(FinPiMLP, MetaModule):
#   def __init__(self, *args, **kw):
#     kw['lincls'] = SafeMetaLinear
#     no_adapt_readout = kw.pop('no_adapt_readout', False)
#     FinPiMLP.__init__(self, *args, **kw)
#     self._children_modules_parameters_cache = dict()
#     self.no_adapt_readout = no_adapt_readout

#   def Gproj(self, grads=None):
#     if self.r >= self.width:
#       return
#     if grads is None:
#       wtgrads = {l: self.linears[l].weight.grad for l in range(1, self.L+1)}
#     else:
#       wtgrads = {l: grads[f'_linears.{l-1}.weight'] for l in range(1, self.L+1)}
#     biasgrads = None
#     if self.linears[1].bias is not None:
#       if grads is None:
#         biasgrads = {l: self.linears[l].bias.grad for l in range(1, self.L+1)}
#       else:
#         biasgrads = {l: grads[f'_linears.{l-1}.bias'] for l in range(1, self.L+1)}
#     with torch.no_grad():
#       for l in range(1, self.L+1):
#         grad = wtgrads[l]
#         om = self.omegas[l]
#         grad[:] = om @ (self.Gcovinvs[l] @ (om.T @ grad))
#         if biasgrads:
#           biasgrads[l][:] = om @ (self.Gcovinvs[l] @ (om.T @ biasgrads[l]))

#   def cuda(self):
#     if hasattr(self, 'omegas'):
#       for l in range(1, self.L+1):
#         self.omegas[l] = self.omegas[l].cuda()
#         self.Gcovinvs[l] = self.Gcovinvs[l].cuda()
#     return super().cuda()

#   def forward(self, x, params=None):
#     x = x.reshape(x.shape[0], -1)
#     L = self.L
#     for l in range(1, L+1):
#       if l == 1:
#         x = self.nonlin(self.first_layer_alpha * self.linears[l](x, params=self.get_subdict(params, f'_linears.{l-1}')))
#       else:
#         x = self.nonlin(self.linears[l](x, params=self.get_subdict(params, f'_linears.{l-1}')))
#     if self.no_adapt_readout:
#       return self.linears[L+1](x)
#     return self.linears[L+1](x, params=self.get_subdict(params, f'_linears.{L}'))

#   def backward(self, *args, **kw):
#     super().backward(*args, **kw)
#     self.Gproj()

#   def state_dict(self):
#     d = super().state_dict()
#     d['no_adapt_readout'] = self.no_adapt_readout
#     d['omegas'] = self.omegas
#     d['Gcovinvs'] = self.Gcovinvs
#     return d

#   def load_state_dict(self, d):
#     self.no_adapt_readout = d.pop('no_adapt_readout', False)
#     self.omegas = d.pop('omegas', {})
#     self.Gcovinvs = d.pop('Gcovinvs', {})
#     super().load_state_dict(d)

def ModelConvOmniglot(out_features, hidden_size=64):
  return MetaConvModel(1, out_features, hidden_size=hidden_size,
             feature_size=hidden_size)

def ModelConvMiniImagenet(out_features, hidden_size=64):
  return MetaConvModel(3, out_features, hidden_size=hidden_size,
             feature_size=5 * 5 * hidden_size)

def ModelMLPSinusoid(hidden_sizes=[40, 40], **kw):
  return MetaMLPModel(1, 1, hidden_sizes, **kw)

def ModelMLPOmniglot(out_features, hidden_sizes=[256, 128, 64, 64], normalize=None, nonlin=nn.ReLU, bias=True, **kw):
  return MetaMLPModel(28**2, out_features, hidden_sizes, normalize=normalize,
                      nonlin=nonlin, bias=bias, **kw)

class Id(nn.Module):
  def forward(self, x):
    return x

def ModelLinMLPOmniglot(out_features, hidden_sizes=[256, 128, 64, 64],
                        normalize=None, bias=True, **kw):
  return MetaMLPModel(28**2, out_features, hidden_sizes, normalize=normalize,
                      nonlin=Id, bias=bias, **kw)

def ModelInfLin1LPOmniglot(out_features, alpha=1, sigma1=1, sigma2=1, **kw):
  return MetaInfLin1LP(28**2, out_features, alpha=alpha, sigma1=sigma1, sigma2=sigma2, **kw)

if __name__ == '__main__':
  model = ModelMLPSinusoid()
