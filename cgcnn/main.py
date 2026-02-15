import argparse
import os
import shutil
import sys
import time
import warnings
from random import sample
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Subset, DataLoader

# Optional Weights & Biases integration
try:
    import wandb  # type: ignore
except Exception:
    wandb = None

from cgcnn.cgcnn.data import CIFData, collate_pool, collate_pool_val_with_meta, get_train_val_test_loader, collate_pool_val
from cgcnn.cgcnn.model import CrystalGraphConvNet
import warnings

from modules.data_select import DataSelectorCGCNN, DuetDALoader
from modules.meta_train_cgcnn import prepare_batches_list
from modules.data_val import DualDataValuator


from modules.utils import EMAGeneralizationStatus, ema_beta_schedule



warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser(description='Crystal Graph Convolutional Neural Networks')
# parser.add_argument('data_options', default='../cgcnn_data/mp_gap_toy', metavar='OPTIONS', nargs='+',
#                     help='dataset options, started with the path to root dir, '
                        #  'then other options')
parser.add_argument('--data_root', default='cgcnn_data/matbench_log_kvrh_difficultyOOD1', metavar='OPTIONS', nargs='+',
                    help='data root dir')
parser.add_argument('--fold', default=1, type=int, metavar='N',
                    help='fold number in matbench task')

parser.add_argument('--task', choices=['regression', 'classification'],
                    default='regression', help='complete a regression or '
                                                   'classification task (default: regression)')

parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run (default: 50)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate (default: '
                                       '0.01)')
parser.add_argument('--lr-milestones', default=[100], nargs='+', type=int,
                    metavar='N', help='milestones for scheduler (default: '
                                      '[100])')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=2, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--use-wandb', action='store_true',
                    help='Enable Weights & Biases logging (default: False)')
parser.add_argument('--wandb-project', default='cgcnn', type=str,
                    help='W&B project name (default: cgcnn)')
parser.add_argument('--wandb-entity', default=None, type=str,
                    help='W&B entity/team (optional)')
parser.add_argument('--wandb-mode', default='online', choices=['online', 'offline', 'disabled'],
                    help='W&B mode: online, offline, or disabled (default: online)')
parser.add_argument('--wandb-name', default=None, type=str,
                    help='W&B run name (optional)')
train_group = parser.add_mutually_exclusive_group()
train_group.add_argument('--train-ratio', default=None, type=float, metavar='N',
                    help='number of training data to be loaded (default none)')
train_group.add_argument('--train-size', default=None, type=int, metavar='N',
                         help='number of training data to be loaded (default none)')
valid_group = parser.add_mutually_exclusive_group()
valid_group.add_argument('--val-ratio', default=0.1, type=float, metavar='N',
                    help='percentage of validation data to be loaded (default '
                         '0.1)')
valid_group.add_argument('--val-size', default=None, type=int, metavar='N',
                         help='number of validation data to be loaded (default '
                              '1000)')
test_group = parser.add_mutually_exclusive_group()
test_group.add_argument('--test-ratio', default=0.1, type=float, metavar='N',
                    help='percentage of test data to be loaded (default 0.1)')
test_group.add_argument('--test-size', default=None, type=int, metavar='N',
                        help='number of test data to be loaded (default 1000)')

parser.add_argument('--optim', default='SGD', type=str, metavar='SGD',
                    help='choose an optimizer, SGD or Adam, (default: SGD)')
parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N',
                    help='number of hidden atom features in conv layers')
parser.add_argument('--h-fea-len', default=128, type=int, metavar='N',
                    help='number of hidden features after pooling')
parser.add_argument('--n-conv', default=3, type=int, metavar='N',
                    help='number of conv layers')
parser.add_argument('--n-h', default=1, type=int, metavar='N',
                    help='number of hidden layers after pooling')
parser.add_argument('--seed', default=42, type=int, metavar='N',
                    help='random seed (default: 0)')

parser.add_argument('--da-input-dim', default=5970, type=int, metavar='N',
                    help='input dimension for data valuator (default: 5970)')
parser.add_argument('--selection-ratio', default=0.7, type=float, metavar='N',
                    help='selection ratio for data selector (default: 0.7)')


args = parser.parse_args(sys.argv[1:])
# args = parser.parse_args()

args.cuda = not args.disable_cuda and torch.cuda.is_available()

# Normalize data root to an absolute string (nargs='+' returns a list)
if isinstance(args.data_root, list):
    args.data_root = args.data_root[0]
args.data_root = os.path.abspath(args.data_root)

if args.task == 'regression':
    best_mae_error = 1e10
else:
    best_mae_error = 0.

# track current epoch globally for logging
CURRENT_EPOCH = 0

# import math
# args.batch_size = math.ceil(args.batch_size / args.selection_ratio)


def set_seed(seed):
    """设置全局随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 确保 CUDA 操作的确定性行为
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_ids(p):
    return [x.strip() for x in open(p).read().splitlines() if x.strip()]


def main():
    global args, best_mae_error
    global CURRENT_EPOCH
    
    # Set random seeds for reproducibility
    set_seed(args.seed)
    print(f'Random seed set to: {args.seed}')

    # dataset = CIFData(args.data_root, csv_file='id_prop_all.csv')
    # #load data
    # dataset = CIFData(args.data_root, csv_file='id_prop_all.csv')    
    
    # collate_fn = collate_pool
    # train_loader, val_loader, test_loader = get_train_val_test_loader(
    #     dataset=dataset,
    #     collate_fn=collate_fn,
    #     batch_size=args.batch_size,
    #     train_ratio=args.train_ratio,
    #     num_workers=args.workers,
    #     val_ratio=args.val_ratio,
    #     test_ratio=args.test_ratio,
    #     pin_memory=args.cuda,
    #     train_size=args.train_size,
    #     val_size=args.val_size,
    #     test_size=args.test_size,
    #       return_test=True)
    
    ## Load ood data
    # CIF files are in root/cifs, split indices are in root/splits/fold[i]
    cif_dir = os.path.join(args.data_root, 'cifs')
    split_dir = os.path.join(args.data_root, 'splits', "fold"+str(args.fold))
    
    print('CIF data dir:', cif_dir)
    print('Split dir:', split_dir)
    
    # CIFData expects id_prop.csv, atom_init.json, and .cif files in same directory
    # Solution: Create directory symlink and modify id_prop.csv to use "cifs/" prefix
    # Copy atom_init.json if not in split_dir
    atom_init_path = os.path.join(split_dir, 'atom_init.json')
    if not os.path.exists(atom_init_path):
        root_atom_init = os.path.join(args.data_root, 'atom_init.json')
        if os.path.exists(root_atom_init):
            shutil.copy2(root_atom_init, atom_init_path)
            print(f'Copied atom_init.json from root to split_dir')
        else:
            print(f'Warning: atom_init.json not found in root directory')
    
    # Create symlink to cifs directory (recreate if broken or wrong target)
    cifs_symlink = os.path.join(split_dir, 'cifs')
    desired_target = os.path.abspath(cif_dir)
    if os.path.islink(cifs_symlink):
        current_target = os.path.realpath(cifs_symlink)
        if current_target != desired_target:
            os.unlink(cifs_symlink)
            os.symlink(desired_target, cifs_symlink, target_is_directory=True)
            print(f'Recreated directory symlink: cifs -> {desired_target}')
        else:
            print('cifs link already exists; skipping')
    elif not os.path.exists(cifs_symlink):
        os.symlink(desired_target, cifs_symlink, target_is_directory=True)
        print(f'Created directory symlink: cifs -> {desired_target}')
    else:
        print('cifs path exists and is not a symlink; leaving as is')
    
    # Modify id_prop.csv to prefix IDs with "cifs/"
    id_prop_path = os.path.join(split_dir, 'id_prop.csv')
    id_prop_original = id_prop_path + '.original'
    
    if not os.path.exists(id_prop_original):
        # Backup original and create modified version
        shutil.copy2(id_prop_path, id_prop_original)
        
        # Read, modify, and write back
        with open(id_prop_original, 'r') as f_in, open(id_prop_path, 'w') as f_out:
            for line in f_in:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    parts[0] = 'cifs/' + parts[0]
                    f_out.write(','.join(parts) + '\n')
        print(f'Modified id_prop.csv with cifs/ prefix')
    
    
    # 1) Load dataset from split directory (accesses .cif files via cifs/ prefix in id_prop.csv)
    dataset = CIFData(split_dir)
    print(f'Loaded dataset with {len(dataset)} samples')

    # 2) 建 id->idx 映射（用 id_prop.csv 的顺序）
    id_prop = [line.strip().split(",") for line in open(os.path.join(split_dir, "id_prop.csv"))]
    all_ids = [x[0] for x in id_prop]
    id_to_idx = {sid: i for i, sid in enumerate(all_ids)}
    
    def make_subset(ids):
        idx = [id_to_idx['cifs/' + sid] for sid in ids if 'cifs/' + sid in id_to_idx]
        return Subset(dataset, idx)
    
    # 3) Load split indices from split directory
    train_ids = read_ids(split_dir+"/train_candidates.txt")
    iid_ids   = read_ids(split_dir+"/iid_val.txt")
    ood_ids   = read_ids(split_dir+"/ood_val.txt")
    test_ids  = read_ids(split_dir+"/test.txt")
    
    #TODO: Temperary, need to delete later
    # train_ids = list(set(train_ids) | set(iid_ids) | set(ood_ids))

    print(train_ids[:5])
    print(list(id_to_idx.items())[:5])
    train_set = make_subset(train_ids)
    iid_val_set = make_subset(iid_ids)
    ood_val_set = make_subset(ood_ids)
    test_set = make_subset(test_ids)

    
    
    # build data selector
    DataValuator = DualDataValuator(input_dim=args.da_input_dim)
    DataValuator.load_state_dict(torch.load('/home/grads/jianpengc/projects/3d_molecule/data_value/checkpoints/data_attributor_meta_step_transformer.pt'))
    if args.selection_ratio <= 1.0:
        Dataselector = DataSelectorCGCNN(valuator=DataValuator, selection_ratio=args.selection_ratio)
    else:
        Dataselector = None
    
    
    # --- after you build train_set / val/test sets ---
    train_duet = DuetDALoader(
        dataset=train_set,
        valuator=DataValuator,     # your DualDataValuator instance (or callable)
        ratio=args.selection_ratio,    # e.g. 0.5
        num_epoch=args.epochs, # optional: stop pruning after N epochs
        method='duet'  # 'duet' or 'random'
    )

    train_sampler = train_duet.pruning_sampler()

    

    train_loader = DataLoader(
        train_duet,
        batch_size=args.batch_size,
        sampler=train_sampler,      # <-- key
        shuffle=False,              # <-- must be False when sampler is set
        num_workers=args.workers,
        collate_fn=collate_pool_val_with_meta,  # see below
    )
    # train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate_pool_val)
    iid_loader   = DataLoader(iid_val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_pool)
    ood_loader   = DataLoader(ood_val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_pool)
    test_loader  = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_pool)
    

    # obtain target value normalizer
    if args.task == 'classification':
        normalizer = Normalizer(torch.zeros(2))
        normalizer.load_state_dict({'mean': 0., 'std': 1.})
    else:
        if len(dataset) < 500:
            warnings.warn('Dataset has less than 500 data points. '
                          'Lower accuracy is expected. ')
            sample_data_list = [dataset[i] for i in range(len(dataset))]
        else:
            sample_data_list = [dataset[i] for i in
                                sample(range(len(dataset)), 500)]
        _, sample_target, _ = collate_pool(sample_data_list)
        normalizer = Normalizer(sample_target)

    # build model
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    print(orig_atom_fea_len, nbr_fea_len)
    model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                atom_fea_len=args.atom_fea_len,
                                n_conv=args.n_conv,
                                h_fea_len=args.h_fea_len,
                                n_h=args.n_h,
                                classification=True if args.task ==
                                                       'classification' else False)


    # ---- init EMA generalization status tracker ----
    status_expected_dim = DataValuator.gate.status_dim  # or gate.status_dim
    status_tracker = EMAGeneralizationStatus(
        beta_schedule=ema_beta_schedule(steps=args.epochs, max_beta=0.9),
        expected_dim=status_expected_dim,
        use_deltas=True,  # set False if you only want level terms
        device=next(DataValuator.parameters()).device,
        normalize=True,
    )
    
    
    
    # Dataselector = None
    
    if args.cuda:
        model.cuda()

    # initialize W&B if requested
    if args.use_wandb and args.wandb_mode != 'disabled':
        if wandb is None:
            print('W&B not installed. Proceeding without logging.')
            args.use_wandb = False
        else:
            print('Initializing Weights & Biases logging...')
            wandb.init(project=args.wandb_project,
                       entity=args.wandb_entity if args.wandb_entity else None,
                       name=f"{time.strftime('%Y%m%d_%H%M%S')}-{args.wandb_name}" if args.wandb_name else None,
                       config=vars(args),
                       mode=args.wandb_mode)
            try:
                wandb.watch(model, log='all')
            except Exception:
                pass

    # define loss func and optimizer
    if args.task == 'classification':
        criterion = nn.NLLLoss()
    else:
        criterion = nn.MSELoss()
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), args.lr,
                               weight_decay=args.weight_decay)
    else:
        raise NameError('Only SGD or Adam is allowed as --optim')

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_mae_error = checkpoint['best_mae_error']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            normalizer.load_state_dict(checkpoint['normalizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones,
                            gamma=0.1)
    print('---------Start Training---------------')
    print('Training set size: ', len(train_set))
    print('IID Validation set size: ', len(iid_val_set))
    print('OOD Validation set size: ', len(ood_val_set))
    print('Test set size: ', len(test_set))
    print(f'start epoch:{args.start_epoch}, total epochs:{args.epochs}, batch size:{args.batch_size}')
    # Prepare directory for per-epoch checkpoints
    checkpoints_dir = os.path.join(os.getcwd(), 'checkpoints', 'cgcnn_epochs')
    try:
        os.makedirs(checkpoints_dir, exist_ok=True)
    except Exception:
        pass
    # Track all epoch-wise metrics and checkpoints
    epoch_records = []  # list of dicts: { 'epoch': int, 'iid': metrics, 'ood': metrics, 'ckpt': path }
    for epoch in range(args.start_epoch, args.epochs):
        CURRENT_EPOCH = epoch
        # train for one epoch
        train_stats = train(train_loader, model, criterion, optimizer, epoch, normalizer, train_duet=train_duet, status_tracker=status_tracker)
        avg_train_loss = train_stats['avg_loss']
        avg_grad_norm = train_stats['avg_grad_norm']

        # evaluate on IID validation set
        print('\n--- Evaluating on IID Validation Set ---')
        iid_metrics, iid_stats = validate(iid_loader, model, criterion, normalizer, val_type='iid')
        avg_iid_val = iid_stats['avg_loss']
        
        # evaluate on OOD validation set
        print('\n--- Evaluating on OOD Validation Set ---')
        ood_metrics, ood_stats = validate(ood_loader, model, criterion, normalizer, val_type='ood')
        avg_ood_val = ood_stats['avg_loss']
        
        if status_tracker is not None:
            status_tracker.update(
                step=epoch,
                train_loss=avg_train_loss,
                iid_val_loss=avg_iid_val,
                ood_val_loss=avg_ood_val,
                meta_grad_norm=avg_grad_norm,
            )
            
        

        # compute generalization gaps (OOD - IID) for normalized errors
        if args.task == 'regression':
            nrmse_gap = ood_metrics['nrmse'] - iid_metrics['nrmse']
            nmae_gap = ood_metrics['nmae'] - iid_metrics['nmae']
        else:
            nrmse_gap = None
            nmae_gap = None

        # log validation metrics to W&B
        if args.use_wandb and wandb is not None:
            if args.task == 'regression':
                wandb.log({
                    'epoch': epoch, 
                    'val_iid/mae': iid_metrics['mae'],
                    'val_iid/rmse': iid_metrics['rmse'],
                    'val_iid/nrmse': iid_metrics['nrmse'],
                    'val_iid/nmae': iid_metrics['nmae'],
                    'val_iid/r2': iid_metrics['r2'],
                    'val_ood/mae': ood_metrics['mae'],
                    'val_ood/rmse': ood_metrics['rmse'],
                    'val_ood/nrmse': ood_metrics['nrmse'],
                    'val_ood/nmae': ood_metrics['nmae'],
                    'val_ood/r2': ood_metrics['r2'],
                    'val/gap_nrmse_ood_minus_iid': nrmse_gap,
                    'val/gap_nmae_ood_minus_iid': nmae_gap,
                })
            else:
                wandb.log({
                    'epoch': epoch, 
                    'val_iid/auc': iid_metrics,
                    'val_ood/auc': ood_metrics,
                })

        # Use IID validation error for model selection
        mae_error = iid_metrics['mae'] if args.task == 'regression' else iid_metrics
        if mae_error != mae_error:
            print('Exit due to NaN')
            sys.exit(1)

        scheduler.step()

        # remember the best mae_eror and save checkpoint (based on IID validation)
        if args.task == 'regression':
            is_best = mae_error < best_mae_error
            best_mae_error = min(mae_error, best_mae_error)
        else:
            is_best = mae_error > best_mae_error
            best_mae_error = max(mae_error, best_mae_error)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mae_error': best_mae_error,
            'optimizer': optimizer.state_dict(),
            'normalizer': normalizer.state_dict(),
            'args': vars(args)
        }, is_best)

        # Also save a per-epoch checkpoint
        epoch_ckpt_path = os.path.join(checkpoints_dir, f'epoch_{epoch + 1}.pth.tar')
        try:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'normalizer': normalizer.state_dict(),
                'args': vars(args)
            }, epoch_ckpt_path)
        except Exception:
            epoch_ckpt_path = None
        # Record this epoch's metrics
        epoch_records.append({
            'epoch': epoch + 1,
            'iid': iid_metrics,
            'ood': ood_metrics,
            'ckpt': epoch_ckpt_path
        })

    # test best model
    print('---------Evaluate Model on Test Set---------------')
    best_checkpoint = torch.load('model_best.pth.tar')
    model.load_state_dict(best_checkpoint['state_dict'])
    test_metrics, _ = validate(test_loader, model, criterion, normalizer, test=True)
    if args.use_wandb and wandb is not None:
        if args.task == 'regression':
            wandb.log({
                'test/mae': test_metrics['mae'],
                'test/rmse': test_metrics['rmse'],
                'test/nrmse': test_metrics['nrmse'],
                'test/nmae': test_metrics['nmae'],
                'test/r2': test_metrics['r2']
            })
        else:
            wandb.log({'test/auc': test_metrics})

    # Build ranked lists: IID, OOD, and average of IID+OOD
    if args.task == 'regression':
        top_iid = sorted(epoch_records, key=lambda x: x['iid']['mae'])[:10]
        top_ood = sorted(epoch_records, key=lambda x: x['ood']['mae'])[:10]
        top_avg = sorted(epoch_records, key=lambda x: (x['iid']['mae'] + x['ood']['mae']) / 2.0)[:10]
    else:
        top_iid = sorted(epoch_records, key=lambda x: x['iid'], reverse=True)[:10]
        top_ood = sorted(epoch_records, key=lambda x: x['ood'], reverse=True)[:10]
        top_avg = sorted(epoch_records, key=lambda x: (x['iid'] + x['ood']) / 2.0, reverse=True)[:10]

    # Evaluate test metrics for union of top epochs across all rankings
    epochs_to_eval = {e['epoch'] for e in (top_iid + top_ood + top_avg)}
    epoch_test_map = {}
    for e in epochs_to_eval:
        rec = next((r for r in epoch_records if r['epoch'] == e), None)
        if rec and rec['ckpt'] and os.path.isfile(rec['ckpt']):
            try:
                ckpt = torch.load(rec['ckpt'])
                model.load_state_dict(ckpt['state_dict'])
                if 'normalizer' in ckpt:
                    normalizer.load_state_dict(ckpt['normalizer'])
                epoch_test_map[e], _ = validate(test_loader, model, criterion, normalizer, test=True)
            except Exception:
                epoch_test_map[e] = None
        else:
            epoch_test_map[e] = None

    # Print concise summaries
    def print_ranked(title, ranked_list):
        print(f'---------Top-10 Epochs ({title})---------------')
        for idx, entry in enumerate(ranked_list, start=1):
            e = entry['epoch']; tst = epoch_test_map.get(e)
            if args.task == 'regression':
                iid = entry['iid']; ood = entry['ood']
                print(f"#{idx} Epoch {e} | IID MAE {iid['mae']:.4f}, RMSE {iid['rmse']:.4f}, R2 {iid['r2']:.4f} | "
                      f"OOD MAE {ood['mae']:.4f}, RMSE {ood['rmse']:.4f}, R2 {ood['r2']:.4f} | "
                      f"Test: " + (f"MAE {tst['mae']:.4f}, RMSE {tst['rmse']:.4f}, R2 {tst['r2']:.4f}" if tst else 'N/A'))
            else:
                iid_auc = entry['iid']; ood_auc = entry['ood']
                print(f"#{idx} Epoch {e} | IID AUC {iid_auc:.4f} | OOD AUC {ood_auc:.4f} | Test AUC " + (f"{tst:.4f}" if tst is not None else 'N/A'))

    print_ranked('by IID validation', top_iid)
    print_ranked('by OOD validation', top_ood)
    print_ranked('by IID+OOD average', top_avg)

    # Log W&B top-10 rankings as plain scalars (no Table)
    if args.use_wandb and wandb is not None:
        def _py(x):
            """Convert torch/numpy scalars to python scalars for wandb."""
            if x is None:
                return None
            try:
                import numpy as np
                if isinstance(x, np.generic):
                    return x.item()
            except Exception:
                pass
            if torch.is_tensor(x):
                return x.detach().cpu().item()
            if isinstance(x, (int, float)):
                return x
            try:
                return float(x)
            except Exception:
                return None

        # Log W&B top-10 rankings as plain scalars (no Table) + put into Summary
        if args.use_wandb and wandb is not None and wandb.run is not None:
            def _py(x):
                if x is None:
                    return None
                try:
                    import numpy as np
                    if isinstance(x, np.generic):
                        return x.item()
                except Exception:
                    pass
                if torch.is_tensor(x):
                    return x.detach().cpu().item()
                if isinstance(x, (int, float)):
                    return x
                try:
                    return float(x)
                except Exception:
                    return None

            def log_ranking(prefix, ranked_list):
                log_dict = {}
                for rank, it in enumerate(ranked_list, start=1):
                    e = int(it["epoch"])
                    tst = epoch_test_map.get(e, None)

                    if args.task == "regression":
                        iid = it["iid"]; ood = it["ood"]
                        # log keys
                        log_dict[f"{prefix}/rank_{rank}/epoch"] = e
                        log_dict[f"{prefix}/rank_{rank}/iid_mae"] = _py(iid.get("mae"))
                        log_dict[f"{prefix}/rank_{rank}/iid_rmse"] = _py(iid.get("rmse"))
                        log_dict[f"{prefix}/rank_{rank}/iid_r2"] = _py(iid.get("r2"))
                        log_dict[f"{prefix}/rank_{rank}/ood_mae"] = _py(ood.get("mae"))
                        log_dict[f"{prefix}/rank_{rank}/ood_rmse"] = _py(ood.get("rmse"))
                        log_dict[f"{prefix}/rank_{rank}/ood_r2"] = _py(ood.get("r2"))
                        log_dict[f"{prefix}/rank_{rank}/test_mae"] = _py(tst.get("mae")) if tst else None
                        log_dict[f"{prefix}/rank_{rank}/test_rmse"] = _py(tst.get("rmse")) if tst else None
                        log_dict[f"{prefix}/rank_{rank}/test_r2"] = _py(tst.get("r2")) if tst else None
                    else:
                        log_dict[f"{prefix}/rank_{rank}/epoch"] = e
                        log_dict[f"{prefix}/rank_{rank}/iid_auc"] = _py(it.get("iid"))
                        log_dict[f"{prefix}/rank_{rank}/ood_auc"] = _py(it.get("ood"))
                        log_dict[f"{prefix}/rank_{rank}/test_auc"] = _py(tst) if tst is not None else None

                # 1) normal log: do NOT set step (avoid step rollback being dropped)
                wandb.log(log_dict, commit=True)

                # 2) also write into Summary so it shows like test metrics in Overview
                for k, v in log_dict.items():
                    if isinstance(v, (int, float)) and v == v:  # numeric and not NaN
                        wandb.run.summary[k] = v

            log_ranking("top10/by_iid", top_iid)
            log_ranking("top10/by_ood", top_ood)
            log_ranking("top10/by_avg", top_avg)

def weighted_mse_loss(pred, target, w, eps=1e-12, normalize=False):
    """
    pred/target: [B, ...]
    w:           [B] or broadcastable to per-sample loss
    """
    # per-element squared error
    se = (pred - target) ** 2  # [B, ...]
    # per-sample MSE (average over non-batch dims)
    per_sample = se.flatten(1).mean(dim=1)  # [B]

    w = w.view(-1).to(per_sample.device)  # [B]

    if normalize:
        # stable weighted mean (keeps loss scale consistent)
        return (w * per_sample).sum() / (w.sum() + eps)
    else:
        # equivalent to mean if w already sums to B, etc.
        return (w * per_sample).mean()
    
    
def train(train_loader, model, criterion, optimizer, epoch, normalizer, train_duet=None, status_tracker=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    grad_norms = AverageMeter()
    if args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (batch_data, batch_idx, batch_w) in enumerate(train_loader):       
        input, target, _ = batch_data
        # measure data loading time
        data_time.update(time.time() - end)

        if args.cuda:
            input_var = (Variable(input[0].cuda(non_blocking=True)),
                         Variable(input[1].cuda(non_blocking=True)),
                         input[2].cuda(non_blocking=True),
                         [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
        else:
            input_var = (Variable(input[0]),
                         Variable(input[1]),
                         input[2],
                         input[3])
        # normalize target
        if args.task == 'regression':
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        if args.cuda:
            target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)
        loss = weighted_mse_loss(output, target_var, w=batch_w)
        # loss = criterion(output, target_var, weights=batch_data_score)

        # measure accuracy and record loss
        if args.task == 'regression':
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            losses.update(loss.data.cpu(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
        else:
            accuracy, precision, recall, fscore, auc_score = \
                class_eval(output.data.cpu(), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        grad_norm_values = []
        for param in model.parameters():
            if param.grad is None:
                continue
            grad_norm_values.append(param.grad.detach().norm(2).item())

        if grad_norm_values:
            batch_grad_norm = sum(grad_norm_values) / len(grad_norm_values)
        else:
            batch_grad_norm = 0.0
        grad_norms.update(batch_grad_norm)

        optimizer.step()
    
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        feat = batch_data[0][-1]          # or batch_data[0][4] if that's val_feat
        with torch.no_grad():
            train_duet.score_batch(
                batch_feat=feat,
                batch_indices=batch_idx,
                weight_phy=None,
                weight_gen=None,
                status=None if status_tracker is None else status_tracker.vector(),
            )

        
        if i % args.print_freq == 0:
            if args.task == 'regression':
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, mae_errors=mae_errors)
                )
                if args.use_wandb and wandb is not None:
                    try:
                        wandb.log({
                            'epoch': epoch,
                            'train/step': i,
                            'train/loss': float(losses.val),
                            'train/loss_avg': float(losses.avg),
                            'train/mae': float(mae_errors.val),
                            'train/mae_avg': float(mae_errors.avg),
                        })
                    except Exception:
                        pass
            else:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                      'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                      'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                      'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, accu=accuracies,
                    prec=precisions, recall=recalls, f1=fscores,
                    auc=auc_scores)
                )
                if args.use_wandb and wandb is not None:
                    try:
                        wandb.log({
                            'epoch': epoch,
                            'train/step': i,
                            'train/loss': float(losses.val),
                            'train/loss_avg': float(losses.avg),
                            'train/accuracy': float(accuracies.val),
                            'train/accuracy_avg': float(accuracies.avg),
                            'train/precision': float(precisions.val),
                            'train/recall': float(recalls.val),
                            'train/f1': float(fscores.val),
                            'train/auc': float(auc_scores.val),
                        })
                    except Exception:
                        pass
    avg_loss_value = float(losses.avg) if losses.count else float('nan')
    avg_grad_norm_value = float(grad_norms.avg) if grad_norms.count else 0.0
    return {
        'avg_loss': avg_loss_value,
        'avg_grad_norm': avg_grad_norm_value,
    }


def validate(val_loader, model, criterion, normalizer, test=False, val_type='val'):
    batch_time = AverageMeter()
    losses = AverageMeter()
    if args.task == 'regression':
        mae_errors = AverageMeter()
        # Collect predictions/targets for aggregate metrics (RMSE/NRMSE/R^2)
        all_targets = []
        all_preds = []
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()
    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        if args.cuda:
            with torch.no_grad():
                input_var = (Variable(input[0].cuda(non_blocking=True)),
                             Variable(input[1].cuda(non_blocking=True)),
                             input[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
        else:
            with torch.no_grad():
                input_var = (Variable(input[0]),
                             Variable(input[1]),
                             input[2],
                             input[3])
        if args.task == 'regression':
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        if args.cuda:
            with torch.no_grad():
                target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            with torch.no_grad():
                target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        if args.task == 'regression':
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            mae_error_value = float(mae_error.item()) if torch.is_tensor(mae_error) else float(mae_error)
            losses.update(loss.data.cpu().item(), target.size(0))
            mae_errors.update(mae_error_value, target.size(0))
            # Accumulate for aggregate metrics
            all_preds += normalizer.denorm(output.data.cpu()).view(-1).tolist()
            all_targets += target.view(-1).tolist()
            if test:
                test_pred = normalizer.denorm(output.data.cpu())
                test_target = target
                test_preds += test_pred.view(-1).tolist()
                test_targets += test_target.view(-1).tolist()
                test_cif_ids += batch_cif_ids
        else:
            accuracy, precision, recall, fscore, auc_score = \
                class_eval(output.data.cpu(), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))
            if test:
                test_pred = torch.exp(output.data.cpu())
                test_target = target
                assert test_pred.shape[1] == 2
                test_preds += test_pred[:, 1].tolist()
                test_targets += test_target.view(-1).tolist()
                test_cif_ids += batch_cif_ids

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if args.task == 'regression':
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    mae_errors=mae_errors))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                      'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                      'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                      'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    accu=accuracies, prec=precisions, recall=recalls,
                    f1=fscores, auc=auc_scores))
               

    if test:
        star_label = '**'
        import csv
        with open('test_results.csv', 'w') as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets,
                                            test_preds):
                writer.writerow((cif_id, target, pred))
    else:
        star_label = '*'
    if args.task == 'regression':
        # Compute additional metrics on full set
        preds_np = np.asarray(all_preds, dtype=np.float64)
        targets_np = np.asarray(all_targets, dtype=np.float64)
        mae_value = float(mae_errors.avg)
        if preds_np.size > 0 and targets_np.size == preds_np.size:
            rmse = float(np.sqrt(np.mean((preds_np - targets_np) ** 2)))
            # Normalized metrics by target standard deviation (dimensionless)
            std_y = float(np.std(targets_np))
            nrmse = float(rmse / (std_y + 1e-12))
            nmae = float(mae_value / (std_y + 1e-12))
            try:
                r2 = float(metrics.r2_score(targets_np, preds_np))
            except Exception:
                r2 = float('nan')
        else:
            rmse = float('nan')
            nrmse = float('nan')
            nmae = float('nan')
            r2 = float('nan')

        print(' {star} MAE {mae:.3f} | NMAE {nmae:.3f} | RMSE {rmse:.3f} | NRMSE {nrmse:.3f} | R^2 {r2:.3f}'.format(
            star=star_label, mae=mae_value, nmae=nmae, rmse=rmse, nrmse=nrmse, r2=r2))

        # Optional W&B logging per split
        if args.use_wandb and wandb is not None:
            try:
                prefix = 'test' if test else f'val_{val_type}'
                wandb.log({
                    f'{prefix}/mae': mae_value,
                    f'{prefix}/nmae': nmae,
                    f'{prefix}/rmse': rmse,
                    f'{prefix}/nrmse': nrmse,
                    f'{prefix}/r2': r2
                })
            except Exception:
                pass

        avg_loss_value = float(losses.avg) if losses.count else float('nan')
        val_stats = {
            'avg_loss': avg_loss_value,
            'avg_grad_norm': 0.0,
        }

        return {
            'mae': mae_value,
            'nmae': nmae,
            'rmse': rmse,
            'nrmse': nrmse,
            'r2': r2
        }, val_stats
    else:
        print(' {star} AUC {auc.avg:.3f}'.format(star=star_label,
                                                 auc=auc_scores))
        avg_loss_value = float(losses.avg) if losses.count else float('nan')
        val_stats = {
            'avg_loss': avg_loss_value,
            'avg_grad_norm': 0.0,
        }
        return float(auc_scores.avg), val_stats


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if torch.is_tensor(val):
            val = float(val.item())
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, k):
    """Sets the learning rate to the initial LR decayed by 10 every k epochs"""
    assert type(k) is int
    lr = args.lr * (0.1 ** (epoch // k))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
