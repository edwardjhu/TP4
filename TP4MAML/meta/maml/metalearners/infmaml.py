import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from collections import OrderedDict
from torchmeta.utils import gradient_update_parameters
from meta.maml.utils import tensors_to_device, compute_accuracy
from meta.maml.metalearners.maml import MAML

__all__ = ['InfMAML']

class InfMAML(MAML):
  def __init__(self, *args, **kw):
    no_adapt_readout = kw.pop('no_adapt_readout', False)
    super().__init__(*args, **kw)
    self.no_adapt_readout = no_adapt_readout
    if self.first_order is False:
      raise NotImplementedError('second order MAML is not implemented')
    self.metagrads = self.model.newgradbuffer()

  def get_outer_loss(self, batch, dobackward=True):
    if 'test' not in batch:
      raise RuntimeError('The batch does not contain any test dataset.')
    _, test_targets = batch['test']
    num_tasks = test_targets.size(0)
    is_classification_task = (not test_targets.dtype.is_floating_point)
    results = {
      'num_tasks': num_tasks,
      'inner_losses': np.zeros((self.num_adaptation_steps,
        num_tasks), dtype=np.float32),
      'outer_losses': np.zeros((num_tasks,), dtype=np.float32),
      'mean_outer_loss': 0.
    }
    if is_classification_task:
      results.update({
        'accuracies_before': np.zeros((num_tasks,), dtype=np.float32),
        'accuracies_after': np.zeros((num_tasks,), dtype=np.float32)
      })

    mean_outer_loss = torch.tensor(0., device=self.device)
    for task_id, (train_inputs, train_targets, test_inputs, test_targets) \
        in enumerate(zip(*batch['train'], *batch['test'])):
      train_inputs = train_inputs.type(torch.get_default_dtype())
      test_inputs = test_inputs.type(torch.get_default_dtype())
      adaptation_results = self.adapt(train_inputs, train_targets,
        is_classification_task=is_classification_task,
        num_adaptation_steps=self.num_adaptation_steps,
        step_size=self.step_size, first_order=self.first_order)

      results['inner_losses'][:, task_id] = adaptation_results['inner_losses']
      if is_classification_task:
        results['accuracies_before'][task_id] = adaptation_results['accuracy_before']
      else:
        test_targets = test_targets.type(torch.get_default_dtype())
      # CHANGE
      with torch.no_grad():
        test_logits = self.model(test_inputs)
      # END CHANGE
      outer_loss = self.loss_function(test_logits, test_targets)
      results['outer_losses'][task_id] = outer_loss.item()
      mean_outer_loss += outer_loss
      # CHANGE
      if dobackward:
        outer_loss.backward()
        self.model.backward(test_logits.grad, buffer=self.metagrads)
      self.model.restore()
      # END CHANGE

      if is_classification_task:
        results['accuracies_after'][task_id] = compute_accuracy(
          test_logits, test_targets)

    mean_outer_loss.div_(num_tasks)
    results['mean_outer_loss'] = mean_outer_loss.item()
    results['median_outer_loss'] = np.median(results['outer_losses'])

    return mean_outer_loss, results
    

  def adapt(self, inputs, targets, is_classification_task=None,
        num_adaptation_steps=1, step_size=0.1, first_order=False):
    if is_classification_task is None:
      is_classification_task = (not targets.dtype.is_floating_point)
    if not is_classification_task:
      targets = targets.type(torch.get_default_dtype())

    results = {'inner_losses': np.zeros(
      (num_adaptation_steps,), dtype=np.float32)}

    for step in range(num_adaptation_steps):
      logits = self.model(inputs)
      inner_loss = self.loss_function(logits, targets)
      results['inner_losses'][step] = inner_loss.item()
      ###
      inner_loss.backward()
      self.model.zero_grad()
      self.model.backward(logits.grad)
      if self.no_adapt_readout:
        # print('no adapt')
        self.model.zero_readout_grad()
        # ###
        # self.model.zero_embed_grad()
        # ###
      self.model.step(step_size)
      ###

      if (step == 0) and is_classification_task:
        results['accuracy_before'] = compute_accuracy(logits, targets)

    return results

  def train_iter(self, dataloader, max_batches=500):
    if self.optimizer is None:
      raise RuntimeError('Trying to call `train_iter`, while the '
        'optimizer is `None`. In order to train `{0}`, you must '
        'specify a Pytorch optimizer as the argument of `{0}` '
        '(eg. `{0}(model, optimizer=torch.optim.SGD(model.'
        'parameters(), lr=0.01), ...).'.format(__class__.__name__))
    num_batches = 0
    self.model.train()
    while num_batches < max_batches:
      for batch in dataloader:
        if num_batches >= max_batches:
          break

        ### CHANGE
        # self.optimizer.zero_grad()
        self.model.resetbuffer(self.metagrads)
        self.model.checkpoint()
        ### END CHANGE


        batch = tensors_to_device(batch, device=self.device)
        outer_loss, results = self.get_outer_loss(batch)
        # infnet is restored to checkpoint at this point
        yield results

        ### CHANGE
        if self.grad_clip is not None and self.grad_clip > 0:
          self.model.gclip(self.grad_clip, buffer=self.metagrads)
        
        # TODO: InfSGD needs to take a grad buffer
        self.optimizer.step(self.metagrads)
        ### END CHANGE

        if self.scheduler is not None:
          self.scheduler.step()

        num_batches += 1

  def evaluate_iter(self, dataloader, max_batches=500):
    num_batches = 0
    self.model.eval()
    while num_batches < max_batches:
      for batch in dataloader:
        if num_batches >= max_batches:
          break

        batch = tensors_to_device(batch, device=self.device)
        ### CHANGE
        self.model.checkpoint()
        _, results = self.get_outer_loss(batch, dobackward=False)
        ### END CHANGE
        yield results

        num_batches += 1