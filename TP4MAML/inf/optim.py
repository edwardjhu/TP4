class InfSGD():
  def __init__(self, model, lr, wd=0, momentum=0, bias_lr_mult=1):
    self.lr = lr
    self.bias_lr_mult = bias_lr_mult
    self.wd = wd
    self.momentum = momentum
    self.model = model
  def step(self, buffer=None):
    self.model.step(self.lr, wd=self.wd, buffer=buffer, momentum=self.momentum, bias_lr_mult=self.bias_lr_mult)
  def zero_grad(self):
    self.model.zero_grad()

class InfMultiStepLR():
  def __init__(self, optimizer, milestones, gamma):
    self.optimizer = optimizer
    self.milestones = milestones
    self.gamma = gamma
    self.epoch = 0

  def step(self):
    if self.epoch in self.milestones:
      self.optimizer.lr *= self.gamma
    self.epoch += 1
    