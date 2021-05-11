import torch
import math
import os
import time
import json
import logging

import cox.store
from cox import utils
from hashlib import sha1

from torchmeta.utils.data import BatchMetaDataLoader

import pandas as pd
from meta.maml.datasets import get_benchmark_by_name
from meta.maml.metalearners import ModelAgnosticMetaLearning
from meta.maml.metalearners import InfMAML
from inf.optim import InfSGD, InfMultiStepLR

def main(args, store, exp_id):
    torch.set_default_dtype(getattr(torch, args.dtype))
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    device = torch.device('cuda' if args.use_cuda
                          and torch.cuda.is_available() else 'cpu')
    
    # random.seed is for dataset shuffling
    import random; random.seed(args.seed)
    torch.manual_seed(args.seed)
    import numpy as np; np.random.seed(args.seed)
    
    if (args.output_folder is not None):
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder, exist_ok=True)
            logging.debug('Creating folder `{0}`'.format(args.output_folder))

        folder = os.path.join(args.output_folder, exp_id)
        os.makedirs(folder, exist_ok=True)
        logging.debug('Creating folder `{0}`'.format(folder))

        args.folder = os.path.abspath(args.folder)
        args.model_path = os.path.abspath(os.path.join(folder, 'model.th'))
        # Save the configuration in a config.json file
        with open(os.path.join(folder, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)
        if args.verbose:
            logging.info('Saving configuration file in `{0}`'.format(
                     os.path.abspath(os.path.join(folder, 'config.json'))))
    
    if args.train_loss_function == 'None':
        args.train_loss_function = None
    else:
        args.train_loss_function = getattr(torch.nn, args.train_loss_function)()
    
    benchmark = get_benchmark_by_name(args.dataset,
                                      args.folder,
                                      args.num_ways,
                                      args.num_shots,
                                      args.num_shots_test,
                                      hidden_size=args.hidden_size,
                                      normalize=args.normalize,
                                      bias_alpha=args.bias_alpha,
                                      last_bias_alpha=args.last_bias_alpha,
                                      first_layer_alpha=args.first_layer_alpha,
                                      depth=args.depth,
                                    #   infnet_r=args.infnet_r,
                                      seed=args.seed,
                                      orig_model=args.orig_model,
                                      lin_model=args.lin_model,
                                      sigma1=args.sigma1,
                                      sigma2=args.sigma2,
                                      sigmab=args.sigmab,
                                      train_last_layer_only=args.train_last_layer_only,
                                      gp1lp=args.gp1lp,
                                      ntk1lp=args.ntk1lp)
                                      
    # random.seed is for dataset shuffling
    import random; random.seed(args.seed)
    torch.manual_seed(args.seed)
    import numpy as np; np.random.seed(args.seed)
    
    meta_train_dataloader = BatchMetaDataLoader(benchmark.meta_train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=not args.no_train_shuffle,
                                                num_workers=args.num_workers,
                                                pin_memory=True)
    meta_val_dataloader = BatchMetaDataLoader(benchmark.meta_val_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers,
                                              pin_memory=True)

    milestones = []
    if args.lr_drop_milestones:
        milestones = [int(float(e) * args.num_batches) for e in args.lr_drop_milestones.split(',')]
    
    if args.hidden_size < 0 and not (args.lin_model and not args.orig_model):
        if args.optimizer == 'sgd':
            meta_optimizer = InfSGD(benchmark.model, lr=args.meta_lr, momentum=args.meta_momentum, bias_lr_mult=1)
        elif args.optimizer == 'adam':
            raise NotImplementedError('no infAdam yet')
        sch = None
        if args.scheduler == 'cosine':
            raise NotImplementedError()
        elif args.scheduler == 'multistep':
            if args.verbose:
                print('multistep scheduler')
                print('milestones', milestones)
            # milestones = list(np.array([1, 2, 10, 100]) * args.num_batches / 2)
            # milestones = list(np.arange(11, 100, 10) * args.num_batches)
            # milestones = np.array([1,5, 11, 21, 31, 41, 51, 61, 71, 81, 91]) * args.num_batches
            sch = InfMultiStepLR(meta_optimizer, milestones=milestones, gamma=args.lr_drop_ratio)

        metalearner = InfMAML(benchmark.model,
                            meta_optimizer,
                            first_order=args.first_order,
                            num_adaptation_steps=args.num_steps,
                            step_size=args.step_size,
                            loss_function=benchmark.loss_function,
                            train_loss_function=args.train_loss_function,
                            grad_clip=args.grad_clip,
                            device=device,
                            no_adapt_readout=args.no_adapt_readout,
                            scheduler=sch)
    else:
        if args.verbose:
            print(benchmark.model)
        parameters = benchmark.model.parameters()
        if args.train_last_layer_only:
            if args.verbose:
                print('training last layer only')
            parameters = benchmark.model.classifier.parameters()
        if args.optimizer == 'sgd':
            meta_optimizer = torch.optim.SGD(parameters, lr=args.meta_lr, momentum=args.meta_momentum)
        elif args.optimizer == 'adam':
            meta_optimizer = torch.optim.Adam(parameters, lr=args.meta_lr)
        sch = None
        if args.scheduler == 'cosine':
            if args.verbose:
                print('cosine scheduler')
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(meta_optimizer, T_max=args.num_batches * args.num_epochs)
        elif args.scheduler == 'multistep':
            if args.verbose:
                print('multistep scheduler')
                print('milestones', milestones)
            # # milestones = np.arange(1, 100, 8) * args.num_batches
            # milestones = list(np.array([1, 5]) * args.num_batches)
            # # milestones += list(np.arange(11, 100, 8) * args.num_batches)
            # milestones += list(np.arange(11, 51, 20) * args.num_batches)
            # milestones += list(np.arange(51, 100, 5) * args.num_batches)
            # # milestones = np.array([1,5, 11, 21, 31, 41, 51, 61, 71, 81, 91]) * args.num_batches
            sch = torch.optim.lr_scheduler.MultiStepLR(meta_optimizer, milestones=milestones, gamma=args.lr_drop_ratio)
        benchmark.model.no_adapt_readout = args.no_adapt_readout
        metalearner = ModelAgnosticMetaLearning(
            benchmark.model,
            meta_optimizer,
            first_order=args.first_order,
            num_adaptation_steps=args.num_steps,
            step_size=args.step_size,
            loss_function=benchmark.loss_function,
            train_loss_function=args.train_loss_function,
            grad_clip=args.grad_clip,
            device=device,
            scheduler=sch)
    if args.load_model_path:
        print(f'loading model {args.load_model_path}')
        with open(args.load_model_path, 'rb') as f:
            benchmark.model.load_state_dict(torch.load(f, map_location=device))

    best_value = None

    # Training loop
    epoch_desc = 'Epoch {{0: <{0}d}}'.format(1 + int(math.log10(args.num_epochs)))
    # allresults = []
    for epoch in range(args.num_epochs):
        train_results = metalearner.train(meta_train_dataloader,
                          max_batches=args.num_batches,
                          verbose=args.verbose,
                          desc='Training',
                          leave=False)
        results = metalearner.evaluate(meta_val_dataloader,
                                       max_batches=args.num_batches,
                                       verbose=args.verbose,
                                       desc=epoch_desc.format(epoch + 1))
        _results = dict(
            [('epoch', epoch+1)]
            + [('train_' + k, v) for k,v in train_results.items()]
            + [('val_' + k, v) for k,v in results.items()]
        )
        store['result'].append_row(_results)
        #allresults.append(_results)
        if args.verbose:
            print(*[f'{k}: {v:.4f}' for k, v in _results.items() if k != 'epoch'])
        
        # Save best model
        # print('best value', best_value)
        save_model = False
        if 'accuracies_after' in results:
            # print('acc after')
            if (best_value is None) or (best_value < results['accuracies_after']):
                # print(best_value, results['accuracies_after'])
                best_value = results['accuracies_after']
                save_model = True
        elif (best_value is None) or (best_value > results['mean_outer_loss']):
            # print('outer loss')
            best_value = results['mean_outer_loss']
            save_model = True

        if save_model and (args.output_folder is not None):
            # print('save model')
            with open(args.model_path, 'wb') as f:
                torch.save(benchmark.model.state_dict(), f)
    if hasattr(benchmark.meta_train_dataset, 'close'):
        benchmark.meta_train_dataset.close()
        benchmark.meta_val_dataset.close()

    return best_value

def parse_main(arglst=None, unittest=False, check_existing=True):
    import argparse

    parser = argparse.ArgumentParser('MAML')

    # General
    parser.add_argument('folder', type=str,
        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--dataset', type=str,
        choices=['sinusoid', 'omniglot', 'miniimagenet'], default='omniglot',
        help='Name of the dataset (default: omniglot).')
    parser.add_argument('--output-folder', type=str, default=None,
        help='Path to the output folder to save the model.')
    parser.add_argument('--load-model-path', type=str, default=None,
        help='Path to the model state dict.')
    parser.add_argument('--num-ways', type=int, default=5,
        help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--num-shots', type=int, default=5,
        help='Number of training example per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-shots-test', type=int, default=15,
        help='Number of test example per class. If negative, same as the number '
        'of training examples `--num-shots` (default: 15).')

    # Model
    parser.add_argument('--hidden-size', type=int, default=64,
        help='width. If negative, then treated as infinity.'
        '(default: 64).')
    parser.add_argument('--depth', type=int, default=2,
        help='number of hidden layers. (default: 2)')
    # parser.add_argument('--infnet_r', type=int, default=100,
    #     help='rank of probability space for infnet. (default: 100)')
    parser.add_argument('--no-adapt-readout', action='store_true',
        help='readout layer does not get gradients during adaptation.')
    parser.add_argument('--orig-model', action='store_true',
        help='use original model instead of infnet/finnet.')
    parser.add_argument('--lin-model', action='store_true',
        help='use linear model instead of infnet/finnet.')
    parser.add_argument('--normalize', type=str, default='None',
        help='normalization. BN | LN | None. (default: None)')
    parser.add_argument('--bias-alpha', type=float, default=1,
        help='bias is multiplied by this number during forward pass. (default: 1)')
    parser.add_argument('--last-bias-alpha', type=float, default=0,
        help='bias alpha for the last layer (logits), overriding --bias-alpha. (default: 0)')
    parser.add_argument('--first-layer-alpha', type=float, default=1,
        help='First layer preactivation is multiplied by this. (default: 1)')
    parser.add_argument('--sigma1', type=float, default=1,
        help='For Lin1LP, 1st layer weights are initialized as N(0, sigma1**2/width). (default: 1)')
    parser.add_argument('--sigma2', type=float, default=1,
        help='For Lin1LP, 2nd layer weights are initialized as N(0, sigma2**2/width) (default: 1)')
    parser.add_argument('--train-last-layer-only', action='store_true')
    parser.add_argument('--gp1lp', action='store_true',
        help='train the last layer of a relu 1LP')
    parser.add_argument('--ntk1lp', action='store_true',
        help='train via the NTK of a relu 1LP')
    parser.add_argument('--sigmab', type=float, default=1,
        help='For GP1LP/NTK1LP, 1st layer biases are initialized as N(0, sigmab**2). (default: 1)')
    
    # Optimization
    parser.add_argument('--optimizer', type=str, default='sgd',
        help='adam | sgd. (default: sgd)')
    parser.add_argument('--scheduler', type=str, default='None', choices=('None', 'cosine', 'multistep'),
        help='None | cosine. (default: None)')
    parser.add_argument('--lr-drop-ratio', type=float, default=0.5,
        help='if using multistep scheduler, lr is multiplied by this number at milestones')
    parser.add_argument('--lr-drop-milestones', type=str, default='',
        help='comma-separated list of epoch numbers. If using multistep scheduler, lr is dropped at after these epochs*num-batches steps.')
    parser.add_argument('--train-loss-function', type=str, default='None', choices=('None', 'SmoothL1Loss'),
        help='None | SmoothL1Loss. (default: None)')
    parser.add_argument('--batch-size', type=int, default=25,
        help='Number of tasks in a batch of tasks (default: 25).')
    parser.add_argument('--num-steps', type=int, default=1,
        help='Number of fast adaptation steps, ie. gradient descent '
        'updates (default: 1).')
    parser.add_argument('--num-epochs', type=int, default=50,
        help='Number of epochs of meta-training (default: 50).')
    parser.add_argument('--num-batches', type=int, default=100,
        help='Number of batch of tasks per epoch (default: 100).')
    parser.add_argument('--step-size', type=float, default=0.1,
        help='Size of the fast adaptation step, ie. learning rate in the '
        'gradient descent update (default: 0.1).')
    parser.add_argument('--first-order', action='store_true',
        help='Use the first order approximation, do not use higher-order '
        'derivatives during meta-optimization.')
    parser.add_argument('--meta-lr', type=float, default=0.001,
        help='Learning rate for the meta-optimizer (optimization of the outer '
        'loss). The default optimizer is SGD (default: 1e-3).')
    parser.add_argument('--meta-momentum', type=float, default=0,
        help='Momentum for the meta-optimizer (optimization of the outer '
        'loss). The default optimizer is SGD (default: 0).')
    parser.add_argument('--grad-clip', type=float, default=-1,
        help='Gradient clipping')

    # Misc
    parser.add_argument('--num-workers', type=int, default=1,
        help='Number of workers to use for data-loading (default: 1).')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--use-cuda', action='store_true')
    parser.add_argument('--no-train-shuffle', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dtype', type=str, default='float64', choices=('float16', 'float32', 'float64'))

    if arglst is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(arglst)

    if args.num_shots_test <= 0:
        args.num_shots_test = args.num_shots

    if args.normalize == 'None':
        args.normalize = None

    if args.scheduler == 'None':
        args.scheduler = None

    if args.verbose:
        print(args)

    # initializing cox
    args_dict = args.__dict__
    exp_id = sha1(repr(sorted(frozenset(args_dict.items()))).encode('ASCII')).hexdigest()
    store = cox.store.Store(args.output_folder, exp_id)
    if check_existing and 'finished' in store.keys:
        if args.verbose:
            print('result already exists; skipping...')
        exit(0)
    if 'metadata' not in store.keys:
        schema = cox.store.schema_from_dict(args_dict)
        store.add_table('metadata', schema)
        store['metadata'].append_row(args_dict)
    else:
        if args.verbose:
            print('[Found existing metadata in store. Skipping this part.]')
    if 'result' not in store.keys:
        store.add_table('result', {
            'epoch': int,
            'train_mean_outer_loss': float,
            'train_median_outer_loss': float,
            'train_accuracies_after': float,
            'val_mean_outer_loss': float,
            'val_median_outer_loss': float,
            'val_accuracies_after': float,
        })
    else:
        if args.verbose:
            print('[Found existing result in store. Skipping this part.]')


    best_value = None
    try:
        best_value = main(args, store, exp_id)
    except BrokenPipeError:
        pass
    
    # mark as done
    store.add_table('finished', {'foo': int})
    store['finished'].append_row({'foo': 0})

    if unittest:
        return best_value

if __name__ == '__main__':
    parse_main()