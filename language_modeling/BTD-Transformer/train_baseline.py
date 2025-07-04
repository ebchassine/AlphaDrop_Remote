# coding: utf-8
import os
import math
import matplotlib.pyplot as plt
import time
import argparse
import itertools
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim

from data_utils import get_lm_corpus
from models import *
from utils.exp_utils import create_exp_dir
from utils.data_parallel import BalancedDataParallel
from esd_utils import net_esd_estimator

# set manual seed
def set_seed(seed=42):
    print(f"=====> Set the random seed as {seed}")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKACE_CONFIG"] = ":16:8"

parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
parser.add_argument('--data',                     type=str,           default='data/ptb',           help='location of the data corpus')
parser.add_argument('--dataset',                  type=str,           default='ptb',
                    choices=['ptb'],                            help='dataset name')
parser.add_argument('--model',                    type=str,           default='tensorized',         help='transformer model used')
parser.add_argument('--n_layer',                  type=int,           default=12,                   help='number of total layers')
parser.add_argument('--n_head',                   type=int,           default=8,                    help='number of heads')
parser.add_argument('--d_head',                   type=int,           default=40,                   help='head dimension')
parser.add_argument('--d_embed',                  type=int,           default=-1,                   help='embedding dimension')
parser.add_argument('--d_model',                  type=int,           default=512,                  help='model dimension')
parser.add_argument('--d_inner',                  type=int,           default=2100,                 help='inner dimension in FF')
parser.add_argument('--dropout',                  type=float,         default=0.1,                  help='global dropout rate')
parser.add_argument('--dropatt',                  type=float,         default=0.0,                  help='attention probability dropout rate')
parser.add_argument('--init',                     type=str,           default='normal',              help='parameter initializer to use.')
parser.add_argument('--emb_init',                 type=str,           default='normal',             help='parameter initializer to use.')
parser.add_argument('--init_range',               type=float,         default=0.1,                  help='parameters initialized by U(-init_range, init_range)')
parser.add_argument('--emb_init_range',           type=float,         default=0.01,                 help='parameters initialized by U(-init_range, init_range)')
parser.add_argument('--init_std',                 type=float,         default=0.02,                 help='parameters initialized by N(0, init_std)')
parser.add_argument('--proj_init_std',            type=float,         default=0.01,                 help='parameters initialized by N(0, init_std)')
parser.add_argument('--optim',                    type=str,           default='adam', 
                    choices=['adam'],                       help='optimizer to use.')
parser.add_argument('--lr',                       type=float,         default=0.00025,              help='initial learning rate (0.00025|5 for adam|sgd)')
parser.add_argument('--mom',                      type=float,         default=0.0,                  help='momentum for sgd')
parser.add_argument('--scheduler',                type=str,           default='cosine', 
                    choices=['cosine', 'constant'],                         help='lr scheduler to use.')
parser.add_argument('--decay_rate',               type=float,         default=0.5,                  help='decay factor when ReduceLROnPlateau is used')
parser.add_argument('--lr_min',                   type=float,         default=0.0,                  help='minimum learning rate during annealing')
parser.add_argument('--clip',                     type=float,         default=0.25,                 help='gradient clipping')
parser.add_argument('--clip_nonemb',              action='store_true',                              help='only clip the gradient of non-embedding params')
parser.add_argument('--max_step',                 type=int,           default=200000,               help='upper step limit')
parser.add_argument('--max_epoch',                type=int,           default=100,                  help='upper epoch limit')
parser.add_argument('--batch_size',               type=int,           default=60,                   help='batch size')
parser.add_argument('--batch_chunk',              type=int,           default=1,                    help='split batch into chunks to save memory')
parser.add_argument('--tgt_len',                  type=int,           default=32,                   help='number of tokens to predict')
parser.add_argument('--eval_tgt_len',             type=int,           default=32,                   help='number of tokens to predict for evaluation')
parser.add_argument('--ext_len',                  type=int,           default=0,                    help='length of the extended context')
parser.add_argument('--mem_len',                  type=int,           default=32,                   help='length of the retained previous heads')
parser.add_argument('--not_tied',                 action='store_true',                              help='do not tie the word embedding and softmax weights')
parser.add_argument('--seed',                     type=int,           default=1111,                 help='random seed')
parser.add_argument('--cuda',                     action='store_true',                              help='use CUDA')
parser.add_argument('--div_val',                  type=int,           default=1,                    help='divident value for adapative input and softmax')
parser.add_argument('--pre_lnorm',                action='store_true',                              help='apply LayerNorm to the input instead of the output')
parser.add_argument('--varlen',                   action='store_true',                              help='use variable length')
parser.add_argument('--multi_gpu',                action='store_true',                              help='use multiple GPU')
parser.add_argument('--log-interval',             type=int,           default=200,                  help='report interval')
parser.add_argument('--esd-interval',             type=int,           default=200,                  help='esd interval')
parser.add_argument('--eval-interval',            type=int,           default=1000,                 help='evaluation interval')
parser.add_argument('--work_dir',                 default='LM-TFM',   type=str,                     help='experiment directory.')
parser.add_argument('--restart',                  action='store_true',                              help='restart training from the saved checkpoint')
parser.add_argument('--restart_dir',              type=str,           default='',                   help='restart dir')
parser.add_argument('--debug',                    action='store_true',                              help='run in debug mode (do not create exp dir)')
parser.add_argument('--same_length',              action='store_true',                              help='use the same attn length for all tokens')
parser.add_argument('--attn_type',                type=int,           default=0,                    help='attention type. 0 for ours, 1 for Shaw et al, 2 for Vaswani et al, 3 for Al Rfou et al.')
parser.add_argument('--clamp_len',                type=int,           default=-1,                   help='use the same pos embeddings after clamp_len')
parser.add_argument('--eta_min',                  type=float,         default=0.0,                  help='min learning rate for cosine scheduler')
parser.add_argument('--gpu0_bsz',                 type=int,           default=4,                    help='batch size on gpu 0')
parser.add_argument('--max_eval_steps',           type=int,           default=-1,                   help='max eval steps')
parser.add_argument('--sample_softmax',           type=int,           default=-1,                   help='number of samples in sampled softmax')
parser.add_argument('--patience',                 type=int,           default=0,                    help='patience')
parser.add_argument('--finetune_v2',              action='store_true',                              help='finetune v2')
parser.add_argument('--finetune_v3',              action='store_true',                              help='finetune v3')
parser.add_argument('--static-loss-scale',        type=float,         default=1,                    help='Static loss scale, positive power of 2 values can improve fp16 convergence.')
parser.add_argument('--dynamic-loss-scale',       action='store_true',                              help='Use dynamic loss scaling.  If supplied, this argument supersedes --static-loss-scale.')

# esd related parameters
parser.add_argument('--xmin-pos',                 type=float,         default=2,                    help='xmin_index = size of eigs // xmin_pos')
parser.add_argument('--pl-fitting',              type=str,           default='median',           help="")
parser.add_argument('--filter-zeros',             type=str,           default='False')
parser.add_argument('--conv-norm', type=float, default=0.5, help='scaling factor for convolution flattening')
# optimizer related parameters
parser.add_argument('--wdecay',                   type=float,         default=1.2e-6,               help='weight decay')
parser.add_argument('--eps',                      type=float,         default=1e-8)
parser.add_argument('--beta1',                    type=float,         default=0.9,                  help='beta1 value')
parser.add_argument('--beta2',                    type=float,         default=0.999,                help='bets2 value')
parser.add_argument('--block_length',             type=int,           default=4,                    help='block_length')

args = parser.parse_args()
args.tied = not args.not_tied

if args.d_embed < 0:
    args.d_embed = args.d_model

assert args.ext_len >= 0, 'extended context length must be non-negative'
assert args.batch_size % args.batch_chunk == 0

# modify work directory
args.work_dir = '{}/{}/{}/{}-{}'.format(args.work_dir, args.model, 'baseline', args.dataset, args.optim)
if args.optim.lower() == 'adam':
    args.work_dir = '{}/bs{}'.format(args.work_dir, args.batch_size)
    
args.work_dir = os.path.join(args.work_dir,  f'tensor_transformer_{args.n_layer}layer', 
                                    f'head_{args.n_head}', \
                                    f"max_step{args.max_step}_max_epoch{args.max_epoch}_log_interval{args.log_interval}", \
                                    f"{args.pl_fitting}_xmin_pos{args.xmin_pos}", \
                                    f"seed_{args.seed}_lr_{args.lr}" 
                                    )

logging = create_exp_dir(args.work_dir, debug=False) 


# Set the random seed manually for reproducibility.
set_seed(args.seed)
# torch.cuda.set_device(1)
if torch.cuda.is_available():
    if not args.cuda:
        print('WARNING: You have a CUDA device, so you should probably run with --cuda')
    else:
        torch.cuda.manual_seed_all(args.seed)

device = torch.device('cuda' if args.cuda else 'cpu')

###############################################################################
# Load data
###############################################################################
corpus = get_lm_corpus(args.data, args.dataset)
ntokens = len(corpus.vocab)
args.n_token = ntokens

eval_batch_size = 10
tr_iter = corpus.get_iterator('train', args.batch_size, args.tgt_len,
                              device=device, ext_len=args.ext_len)
va_iter = corpus.get_iterator('valid', eval_batch_size, args.eval_tgt_len,
                              device=device, ext_len=args.ext_len)
te_iter = corpus.get_iterator('test', eval_batch_size, args.eval_tgt_len,
                              device=device, ext_len=args.ext_len)

args.max_step = min(args.max_step,  tr_iter.n_batch * args.max_epoch)

print(f"--------------------> max_step: {args.max_step} <---------------------")

# adaptive softmax / embedding
cutoffs, tie_projs = [], [False]

###############################################################################
# Build the model
###############################################################################
def init_weight(weight):
    if args.init == 'uniform':
        nn.init.uniform_(weight, -args.init_range, args.init_range)
    elif args.init == 'normal':
        nn.init.normal_(weight, 0.0, args.init_std)


def init_bias(bias):
    nn.init.constant_(bias, 0.0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init_weight(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)

    # init
    elif classname.find('MultiHeadAttn') != -1:
        if hasattr(m, 'core_value'):
            for i in range(m.core_nums):
                nn.init.normal_(m.core_value[i], 0.0, args.proj_init_std)

    elif classname.find('AdaptiveEmbedding') != -1:
        if hasattr(m, 'emb_projs'):
            for i in range(len(m.emb_projs)):
                if m.emb_projs[i] is not None:
                    nn.init.normal_(m.emb_projs[i], 0.0, args.proj_init_std)
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            init_weight(m.weight)
    elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
        if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
            init_weight(m.cluster_weight)
        if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
            init_bias(m.cluster_bias)
        if hasattr(m, 'out_projs'):
            for i in range(len(m.out_projs)):
                if m.out_projs[i] is not None:
                    nn.init.normal_(m.out_projs[i], 0.0, args.proj_init_std)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, args.init_std)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('TransformerLM') != -1:
        if hasattr(m, 'r_emb'):
            init_weight(m.r_emb)
        if hasattr(m, 'r_w_bias'):
            init_weight(m.r_w_bias)
        if hasattr(m, 'r_r_bias'):
            init_weight(m.r_r_bias)
        if hasattr(m, 'r_bias'):
            init_bias(m.r_bias)


def update_dropout(m):
    classname = m.__class__.__name__
    if classname.find('Dropout') != -1:
        if hasattr(m, 'p'):
            m.p = args.dropout


def update_dropatt(m):
    if hasattr(m, 'dropatt'):
        m.dropatt.p = args.dropatt


if args.restart:
    with open(os.path.join(args.restart_dir, 'model.pt'), 'rb') as f:
        model = torch.load(f)
    model.apply(update_dropout)
    model.apply(update_dropatt)
else:
    if args.model == 'tensorized':
        model = TensorizedTransformerLM(ntokens, args.n_layer, args.n_head, args.d_model,
                                args.d_head, args.d_inner, args.dropout, args.dropatt,
                                tie_weight=args.tied, d_embed=args.d_embed, div_val=args.div_val,
                                tie_projs=tie_projs, pre_lnorm=args.pre_lnorm, tgt_len=args.tgt_len,
                                ext_len=args.ext_len, mem_len=args.mem_len, cutoffs=cutoffs,
                                same_length=args.same_length, attn_type=args.attn_type,
                                clamp_len=args.clamp_len, sample_softmax=args.sample_softmax)
        model.apply(weights_init)
        model.word_emb.apply(weights_init)  # ensure embedding init is not overridden by out_layer in case of weight sharing
args.n_all_param = sum([p.nelement() for p in model.parameters()])
args.n_nonemb_param = sum([p.nelement() for p in model.layers.parameters()])
self_attention_param = 0
for p in model.layers:
    for a in p.dec_attn.parameters():
        self_attention_param += a.nelement()

args.self_attention_param = self_attention_param

if args.multi_gpu:
    model = model.to(device)
    if args.gpu0_bsz >= 0:
        para_model = BalancedDataParallel(args.gpu0_bsz // args.batch_chunk,
                                          model, dim=1).to(device)
    else:
        para_model = nn.DataParallel(model, dim=1).to(device)
else:
    logging('Training on Single GPU......')
    para_model = model.to(device)

# record training stats
training_stats = {
    'train_loss': [],
    'train_ppl': [],
    'val_loss': [],
    'val_ppl': [],
    'test_loss': [],
    'test_ppl': [],
    'step': [],
    'lr': [],
}


# analyze the model using est estimator
print(model)
metrics = net_esd_estimator(model, 
                            EVALS_THRESH = 0.00001,
                            bins = 100,
                            pl_fitting=args.pl_fitting,
                            xmin_pos=args.xmin_pos, 
                            filter_zeros = args.filter_zeros=='True')

layer_stats=pd.DataFrame({key:metrics[key] for key in metrics if key!='eigs'})
layer_stats_origin = layer_stats.copy()
# create a new dir for stats
if not os.path.exists(os.path.join(args.work_dir, 'stats')):
    os.makedirs(os.path.join(args.work_dir, 'stats'))
layer_stats_origin.to_csv(os.path.join(args.work_dir, 'stats',  f"origin_layer_stats_epoch_0.csv"))
np.save(os.path.join(args.work_dir, 'stats', 'esd_epoch_0.npy'), metrics)


#### optimizer
if args.optim.lower() == 'adam':
    print('Using Adam optimizer')
    if args.sample_softmax > 0:
        dense_params, sparse_params = [], []
        for param in model.parameters():
            if param.size() == model.word_emb.weight.size():
                sparse_params.append(param)
            else:
                dense_params.append(param)
        optimizer_sparse = optim.SparseAdam(sparse_params, lr=args.lr)
        optimizer = optim.Adam(dense_params, lr=args.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
else:
    raise NotImplementedError

#### scheduler
if args.scheduler == 'cosine':
    # here we do not set eta_min to lr_min to be backward compatible
    # because in previous versions eta_min is default to 0
    # rather than the default value of lr_min 1e-6
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                        args.max_step, eta_min=args.eta_min)  # should use eta_min arg
    if args.sample_softmax > 0:
        scheduler_sparse = optim.lr_scheduler.CosineAnnealingLR(optimizer_sparse,
                                                                args.max_step,
                                                                eta_min=args.eta_min)  # should use eta_min arg
else:
    raise NotImplementedError

logging('=' * 100)
for k, v in args.__dict__.items():
    logging('    - {} : {}'.format(k, v))
logging('=' * 100)
logging('#params = {}'.format(args.n_all_param))
logging('#non emb params = {}'.format(args.n_nonemb_param))
logging('#self attention params = {}'.format(args.self_attention_param))

def cosine_decay(init, epoch, total_epoch):
    epoch = min(epoch, total_epoch)
    cosine_decay = 0.5 * (1 + math.cos(np.pi * epoch / total_epoch))
    
    return init * cosine_decay

###############################################################################
# Training and Evaluation code
###############################################################################

def evaluate(eval_iter):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    # If the model does not use memory at all, make the ext_len longer.
    # Otherwise, make the mem_len longer and keep the ext_len the same.
    if args.mem_len == 0:
        model.reset_length(args.eval_tgt_len,
                           args.ext_len + args.tgt_len - args.eval_tgt_len, args.mem_len)
    else:
        model.reset_length(args.eval_tgt_len,
                           args.ext_len, args.mem_len + args.tgt_len - args.eval_tgt_len)

    # Evaluation
    total_len, total_loss = 0, 0.
    with torch.no_grad():
        mems = tuple()
        for i, (data, target, seq_len) in enumerate(eval_iter):
            if args.max_eval_steps > 0 and i >= args.max_eval_steps:
                break
            ret = model(data, target, *mems)
            loss, mems = ret[0], ret[1:]
            loss = loss.mean()
            total_loss += seq_len * loss.float().item()
            total_len += seq_len

    logging(f'total_loss: {total_loss}, total_len: {total_len}')

    # Switch back to the training mode
    model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
    model.train()

    return total_loss / total_len

def getalpha(m):
    matrix = m.clone()
    eigs = torch.square(torch.linalg.svdvals(matrix).flatten())
    eigs, _ = torch.sort(eigs, descending=False)
    nz_eigs = eigs
    N = len(nz_eigs)
    log_nz_eigs  = torch.log(nz_eigs)
    i = int(len(nz_eigs) / 2)    
    n = float(N - i)
    final_alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i])
    final_alpha = final_alpha.item()
    return final_alpha

alphas = {i: [] for i in range(2 * args.n_layer)}

def train(epoch=0):
    # Turn on training mode which enables dropout.
    global train_step, train_loss, best_val_loss, eval_start_time, log_start_time,temps
    model.train()
    if args.batch_chunk > 1:
        mems = [tuple() for _ in range(args.batch_chunk)]
    else:
        mems = tuple()
    train_iter = tr_iter.get_varlen_iter() if args.varlen else tr_iter
    


    for batch, (data, target, seq_len) in enumerate(train_iter):
        
        # data, target = data.to(device), target.to(device)
######################################################################################################## 
################# alphadropout ###################
        if train_step % args.esd_interval == 0:
            alphalist=[-1] * (2 * args.n_layer)
            for i, decoderlayer in enumerate(model.layers): 
                linear1_weight = decoderlayer.pos_ff.linear1.weight
                alpha=getalpha(linear1_weight)
                alphalist[2*i]=alpha
                linear2_weight = decoderlayer.pos_ff.linear2.weight
                linear2_weight = linear2_weight.T
                alpha=getalpha(linear2_weight)
                alphalist[2*i+1]=alpha
            for i in range(2 * args.n_layer):
                alphas[i].append(alphalist[i])    
            n = len(alphalist)
            #for i in range(len(alphalist)):
            #    print(alphalist[i])
            untuned_rate=args.dropout
            temps = np.array([untuned_rate] * n)
            lr_range = [0.5 * untuned_rate, 1.5 * untuned_rate]
            
        
            score_range = [min(alphalist), max(alphalist)]
            temps = np.interp(alphalist, score_range, lr_range)
            print(temps)

########################################################################################################
        model.zero_grad()
        if args.batch_chunk > 1:
            data_chunks = torch.chunk(data, args.batch_chunk, 1)
            target_chunks = torch.chunk(target, args.batch_chunk, 1)
            for i in range(args.batch_chunk):
                data_i = data_chunks[i].contiguous()
                target_i = target_chunks[i].contiguous()
                ret = para_model(data_i, target_i, *mems[i],temps=temps)
                loss, mems[i] = ret[0], ret[1:]
                loss = loss.float().mean().type_as(loss) / args.batch_chunk
                loss.backward()
                train_loss += loss.float().item()
        else:
            ret = para_model(data, target, *mems,temps=temps)
            loss, mems = ret[0], ret[1:]
            loss = loss.float().mean().type_as(loss)
            loss.backward()
                    
            train_loss += loss.float().item()
            

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()
        if args.sample_softmax > 0:
            optimizer_sparse.step()

        # step-wise learning rate annealing
        train_step += 1

        if args.scheduler == 'cosine':
            scheduler.step(train_step)
            decayed_lr = cosine_decay(args.lr, train_step, args.max_step)
            assert decayed_lr == optimizer.param_groups[0]['lr'], f"lr: {decayed_lr}, optimizer lr: {optimizer.param_groups[0]['lr']} should be the same"
            if args.sample_softmax > 0:
                scheduler_sparse.step(train_step)
        else:
            raise NotImplementedError
        
        if train_step % args.log_interval == 0:
            # plot training results
            cur_loss = train_loss / args.log_interval
            elapsed = time.time() - log_start_time
            log_str = '| epoch {:3d} step {:>8d} | {:>6d} batches | lr {:.3g} ' \
                      '| ms/batch {:5.2f} | loss {:5.2f}'.format(
                epoch, train_step, batch + 1, optimizer.param_groups[0]['lr'],
                                   elapsed * 1000 / args.log_interval, cur_loss)
            training_stats['train_loss'].append(cur_loss)
            if args.dataset in ['enwik8', 'text8']:
                log_str += ' | bpc {:9.5f}'.format(cur_loss / math.log(2))
            else:
                log_str += ' | ppl {:9.3f}'.format(math.exp(cur_loss))
                training_stats['train_ppl'].append(math.exp(cur_loss))
            logging(log_str)
            train_loss = 0
            log_start_time = time.time()

            # evaluate the model in the validation set
            val_loss = evaluate(va_iter)
            test_loss = evaluate(te_iter)
            
            # ESD PER EPOCH Measurements 
            esd_dir = os.path.join(args.work_dir, 'stats')
            os.makedirs(esd_dir, exist_ok=True)

            metrics = net_esd_estimator(
                net=model,
                EVALS_THRESH=1e-5,
                bins=100,
                pl_fitting=args.pl_fitting,
                xmin_pos=args.xmin_pos,
                filter_zeros=(args.filter_zeros == 'True'),
                conv_norm=0.5 # epxose as arg later 
            )

            np.save(os.path.join(esd_dir, f'esd_epoch_{epoch}.npy'), metrics)


            logging('-' * 100)
            log_str = '| Eval {:3d} at step {:>8d} | time: {:5.2f}s ' \
                      '| valid loss {:5.2f}'.format(
                train_step // args.log_interval, train_step,
                (time.time() - eval_start_time), val_loss)
            if args.dataset in ['enwik8', 'text8']:
                log_str += ' | bpc {:9.5f}'.format(val_loss / math.log(2))
            else:
                log_str += ' | valid ppl {:9.3f}'.format(math.exp(val_loss))
            logging(log_str)
            logging('-' * 100)

            training_stats['val_loss'].append(val_loss)
            training_stats['val_ppl'].append(math.exp(val_loss))
            training_stats['test_loss'].append(test_loss)
            training_stats['test_ppl'].append(math.exp(test_loss))
            training_stats['step'].append(train_step)
            training_stats['lr'].append(optimizer.param_groups[0]['lr'])
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                PATH=os.path.join(args.work_dir, 'model.pt')
                torch.save({
                    'val_loss': val_loss,
                    'val_ppl': math.exp(val_loss),
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, PATH)
                np.save(os.path.join(args.work_dir, f'esd_best.npy'), metrics)
                logging('Best model and ESD saved.')
                best_val_loss = val_loss

            eval_start_time = time.time()
            
            np.save(os.path.join(args.work_dir, "training_stats.npy"), training_stats)

        if train_step == args.max_step:
            break

# Loop over epochs.
train_step = 0
train_loss = 0
best_val_loss = None

log_start_time = time.time()
eval_start_time = time.time()
total_start_time = time.time()

# test initialization
init_train_loss = evaluate(tr_iter)
init_test_loss = evaluate(te_iter)
init_val_loss = evaluate(va_iter)
logging('| Start of training | test loss {:5.2f} | test bpc {:9.5f}'.format(
        init_test_loss, math.exp(init_test_loss)))
training_stats['train_loss'].append(init_train_loss)
training_stats['val_loss'].append(init_val_loss)
training_stats['test_loss'].append(init_test_loss)
training_stats['train_ppl'].append(math.exp(init_train_loss))
training_stats['val_ppl'].append(math.exp(init_val_loss))
training_stats['test_ppl'].append(math.exp(init_test_loss))
training_stats['step'].append(0)

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in itertools.count(start=1):
        
        train(epoch)
        # # —— ESD per-epoch dump (copy of “epoch_0” logic) ——
        # esd_dir = os.path.join(args.work_dir, 'stats')  
        # os.makedirs(esd_dir, exist_ok=True)
        # metrics = net_esd_estimator(
        #     model,
        #     EVALS_THRESH = 0.00001,
        #     bins        = 100,
        #     pl_fitting  = args.pl_fitting,
        #     xmin_pos    = args.xmin_pos,
        #     filter_zeros= (args.filter_zeros == 'True')
        # )
        # np.save(os.path.join(esd_dir, f'esd_epoch_{epoch}.npy'), metrics)


        if train_step == args.max_step:
            logging('-' * 100)
            logging('End of training')
            break
except KeyboardInterrupt:
    logging('-' * 100)
    logging('Exiting from training early')
total_duration= time.time() - total_start_time
training_stats['total_duration'] = total_duration

# Load the best saved model.
with open(os.path.join(args.work_dir, 'model.pt'), 'rb') as f:
    state_dict = torch.load(f)
    model.load_state_dict(state_dict['model_state_dict'])
para_model = model.to(device)

# Run on test data.
test_loss = evaluate(te_iter)
logging('=' * 100)
if args.dataset in ['enwik8', 'text8']:
    logging('| End of training | test loss {:5.2f} | test bpc {:9.5f}'.format(
        test_loss, test_loss / math.log(2)))
else:
    logging('| End of training | test loss {:5.2f} | test ppl {:9.3f}'.format(
        test_loss, math.exp(test_loss)))
    training_stats['test_loss'].append(test_loss)
    training_stats['test_ppl'].append(math.exp(test_loss))
logging('=' * 100)

np.save(os.path.join(args.work_dir, "training_stats.npy"), training_stats)

np.save(os.path.join(args.work_dir, "alphas.npy"), alphas)