import torch.nn.functional as F

from collections import namedtuple
from torchmeta.datasets import MiniImagenet
from torchmeta.toy import Sinusoid
from torchmeta.transforms import ClassSplitter, Categorical, Rotation
from torchvision.transforms import ToTensor, Resize, Compose

from meta.cached_omniglot import Omniglot as Omniglot
from meta.maml.model import ModelConvOmniglot, ModelConvMiniImagenet, ModelMLPSinusoid, ModelMLPOmniglot, ModelLinMLPOmniglot, ModelInfLin1LPOmniglot, MetaMLPModel, MetaFinGP1LP
from meta.maml.utils import ToTensor1D
# from inf.pimlp import InfPiMLP
from inf.inf1lp import InfGP1LP, InfNTK1LP

Benchmark = namedtuple('Benchmark', 'meta_train_dataset meta_val_dataset '
                                    'meta_test_dataset model loss_function')

def get_benchmark_by_name(name,
                          folder,
                          num_ways,
                          num_shots,
                          num_shots_test,
                          hidden_size=None,
                          normalize=None,
                          seed=None,
                          depth=2,
                        #   infnet_r=100,
                          bias_alpha=16,
                          last_bias_alpha=0,
                          first_layer_alpha=1,
                          infnet_device='cuda',
                          orig_model=False,
                          lin_model=False,
                          sigma1=1,
                          sigma2=1,
                          sigmab=1,
                          train_last_layer_only=False,
                          gp1lp=False,
                          ntk1lp=False):
    dataset_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=num_shots,
                                      num_test_per_class=num_shots_test)
    assert name == 'omniglot'
    class_augmentations = [Rotation([90, 180, 270])]
    transform = Compose([Resize(28), ToTensor()])

    meta_train_dataset = Omniglot(folder,
                                    transform=transform,
                                    target_transform=Categorical(num_ways),
                                    num_classes_per_task=num_ways,
                                    meta_train=True,
                                    class_augmentations=class_augmentations,
                                    dataset_transform=dataset_transform,
                                    download=True)
    meta_val_dataset = Omniglot(folder,
                                transform=transform,
                                target_transform=Categorical(num_ways),
                                num_classes_per_task=num_ways,
                                meta_val=True,
                                class_augmentations=class_augmentations,
                                dataset_transform=dataset_transform)
    meta_test_dataset = Omniglot(folder,
                                    transform=transform,
                                    target_transform=Categorical(num_ways),
                                    num_classes_per_task=num_ways,
                                    meta_test=True,
                                    dataset_transform=dataset_transform)
    if seed is not None:
        import random; random.seed(seed)
        import torch; torch.manual_seed(seed)
        import numpy as np; np.random.seed(seed)
    if gp1lp:
        model = InfGP1LP(28**2, num_ways, varw=sigma1**2, varb=sigmab**2, initbuffersize=10000)
        if hidden_size >= 0:
            model = model.sample(hidden_size, fincls=MetaFinGP1LP)
        model = model.cuda()
    elif ntk1lp:
        model = InfNTK1LP(28**2, num_ways, varw=sigma1**2, varb=sigmab**2, varw2=sigma2**2, initbuffersize=10000)
        if hidden_size >= 0:
            raise NotImplementedError
        model = model.cuda()
    elif orig_model:
        if lin_model:                
            # print(f'linear model: width {hidden_size}, depth {depth}')
            model = ModelLinMLPOmniglot(num_ways, [hidden_size]*depth, normalize=normalize, bias=bias_alpha!=0, train_last_layer_only=train_last_layer_only)
        else:
            model = MetaMLPModel(28**2, num_ways, [hidden_size]*depth, normalize=normalize,
                    bias=bias_alpha!=0, train_last_layer_only=train_last_layer_only)
    elif lin_model:
        # print(f'linear model: width {hidden_size}, depth {depth}')
        print(f'inflin: alpha={first_layer_alpha}, sigma1={sigma1}, sigma2={sigma2}')
        model = ModelInfLin1LPOmniglot(num_ways, alpha=first_layer_alpha, sigma1=sigma1, sigma2=sigma2, bias_alpha1=bias_alpha, bias_alpha2=last_bias_alpha)
        if hidden_size >= 0:
            print(f'finlin, width {hidden_size}')
            model = model.sample(hidden_size)
        model = model.cuda()
    else:
        raise ValueError()
    loss_function = F.cross_entropy

    return Benchmark(meta_train_dataset=meta_train_dataset,
                     meta_val_dataset=meta_val_dataset,
                     meta_test_dataset=meta_test_dataset,
                     model=model,
                     loss_function=loss_function)
