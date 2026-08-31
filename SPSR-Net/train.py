import json
import os
import warnings
import argparse
# from torchstat import stat
if 'p' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['p']
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['MKL_THREADING_LAYER'] = 'GNU'
warnings.filterwarnings('ignore')

import numpy as np
import torch
import random
# from thop import profile
from fastNLP.core.callbacks.topk_saver import TopkSaver
from fastNLP import cache_results, prepare_torch_dataloader
from fastNLP import print
from fastNLP import Trainer
from fastNLP import TorchGradClipCallback
from fastNLP import FitlogCallback, CheckpointCallback, TorchGradClipCallback, EarlyStopCallback
from fastNLP import SortedSampler, BucketedBatchSampler
from fastNLP import TorchWarmupCallback
import fitlog

# fitlog.debug()

from model.model import CNNNer
from model.metrics import NERMetric
from data.ner_pipe import SpanNerPipe
from data.padder import Torch3DMatrixPadder
from callbacks import AdversarialTrainingCallback

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=2e-5, type=float)
parser.add_argument('--encoder_lr', default=2e-5, type=float)
parser.add_argument('-b', '--batch_size', default=24, type=int)
parser.add_argument('-n', '--n_epochs', default=50, type=int)
parser.add_argument('--warmup', default=0.1, type=float)
parser.add_argument('-d', '--dataset_name', default='ace2004', type=str) # ace2005 genia ace2004
parser.add_argument('--model_name', default=None, type=str)
parser.add_argument('--ent_thres', default=0.5, type=float)
parser.add_argument('--cnn_depth', default=1, type=int)
parser.add_argument('--cnn_dim', default=120, type=int)
parser.add_argument('--num', default=1, type=int)
parser.add_argument('--logit_drop', default=0, type=float)
parser.add_argument('--biaffine_size', default=200, type=int)
parser.add_argument('--n_head', default=5, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--n_layer', default=1, type=int)
parser.add_argument('--accumulation_steps', default=1, type=int)
parser.add_argument('--separateness_rate', default=5, type=int)
parser.add_argument('--theta', default=1, type=float)
parser.add_argument('--loss_theta', default=1, type=float)
parser.add_argument('--loss_type', default='bce', choices=['bce', 'asl', 'balanced_asl'], type=str)
parser.add_argument('--asl_gamma_pos', default=0.0, type=float)
parser.add_argument('--asl_gamma_neg', default=3.0, type=float)
parser.add_argument('--asl_clip', default=0.05, type=float)
parser.add_argument('--balanced_neg_weight', default=1.0, type=float)
parser.add_argument('--label_weighting', default='none', choices=['none', 'sqrt_inv', 'inv'], type=str)
parser.add_argument('--label_weight_power', default=0.5, type=float)
parser.add_argument('--label_weight_min', default=0.5, type=float)
parser.add_argument('--label_weight_max', default=2.0, type=float)
parser.add_argument('--use_length_bias', action='store_true')
parser.add_argument('--length_bias_bins', default=6, type=int)
parser.add_argument('--sad_topk', default=2, type=int)
parser.add_argument('--sad_attn_dim', default=None, type=int)
parser.add_argument('--use_snsa', default=None, type=int)
parser.add_argument('--use_sad', default=1, type=int)
parser.add_argument('--use_hsr', default=1, type=int)
parser.add_argument('--sad_use_rel_bias', default=1, type=int)
parser.add_argument('--sad_gate', default=1, type=int)
parser.add_argument('--init_from_checkpoint', default='', type=str)
parser.add_argument('--head_type', default='linear', choices=['linear', 'residual_mlp', 'relation_aware', 'length_gated'], type=str)
parser.add_argument('--relation_dim', default=None, type=int,
                    help='Hidden size of the boundary-sharing relation-aware span classifier.')
parser.add_argument('--boundary_aux_weight', default=0.0, type=float,
                    help='Weight of training-only start/end boundary supervision.')
parser.add_argument('--boundary_align_weight', default=0.0, type=float,
                    help='Weight of training-only boundary/span consistency loss.')
parser.add_argument('--conflict_rank_weight', default=0.0, type=float,
                    help='Weight of training-only nested-aware conflict ranking loss.')
parser.add_argument('--conflict_margin', default=0.2, type=float,
                    help='Required logit margin between a gold span and its hardest conflict.')
parser.add_argument('--conflict_warmup_steps', default=0, type=int,
                    help='Number of main-task steps before activating conflict ranking.')
parser.add_argument('--length_gate_bins', default=8, type=int,
                    help='Number of length buckets used by the length-gated span classifier.')
parser.add_argument('--quality_aux_weight', default=0.0, type=float,
                    help='Weight of training-only span IoU-quality supervision on HSR features.')
parser.add_argument('--quality_min_iou', default=0.1, type=float,
                    help='Minimum IoU treated as a near-gold span for balanced quality loss.')
parser.add_argument('--hierarchy_aux_weight', default=0.0, type=float,
                    help='Weight of training-only contains/contained role supervision on HSR features.')
parser.add_argument('--difficulty_curriculum', action='store_true',
                    help='Gradually focus the main span loss on nested, long, and boundary-near spans.')
parser.add_argument('--curriculum_warmup_steps', default=0, type=int)
parser.add_argument('--curriculum_ramp_steps', default=1, type=int)
parser.add_argument('--curriculum_nested_boost', default=0.5, type=float)
parser.add_argument('--curriculum_long_boost', default=0.25, type=float)
parser.add_argument('--curriculum_hard_neg_boost', default=0.25, type=float)
parser.add_argument('--curriculum_long_span', default=4, type=int)
parser.add_argument('--adv_type', default='none', choices=['none', 'fgm', 'pgd'], type=str)
parser.add_argument('--adv_epsilon', default=1.0, type=float)
parser.add_argument('--adv_alpha', default=0.3, type=float)
parser.add_argument('--adv_k', default=3, type=int)
parser.add_argument('--adv_emb_name', default='word_embeddings', type=str)
parser.add_argument('--adv_loss_weight', default=1.0, type=float)
parser.add_argument('--adv_warmup_ratio', default=0.0, type=float)
parser.add_argument('--adv_every_n_steps', default=1, type=int)
parser.add_argument('--adv_random_start', action='store_true')
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--fp16_init_scale', default=1.0, type=float)
parser.add_argument('--evaluate_every', default=-1, type=int)
parser.add_argument('--num_train_batch_per_epoch', default=-1, type=int)
parser.add_argument('--num_workers', default=1, type=int)
parser.add_argument('--early_stop_patience', default=0, type=int)
parser.add_argument('--early_stop_monitor', default='f#f#dev', type=str)
parser.add_argument('--bhpc_weight', default=0.0, type=float)
parser.add_argument('--bhpc_dim', default=128, type=int)
parser.add_argument('--bhpc_temperature', default=0.1, type=float)
parser.add_argument('--bhpc_momentum', default=0.95, type=float)
parser.add_argument('--bhpc_margin', default=0.1, type=float)
parser.add_argument('--bhpc_warmup_steps', default=0, type=int)
parser.add_argument('--bhpc_class_balance', default='none', choices=['none', 'sqrt_inv'], type=str)
parser.add_argument('--bhpc_prototype_scale', default=1.0, type=float)
parser.add_argument('--bhpc_boundary_scale', default=1.0, type=float)
parser.add_argument('--checkpoint_monitor', default='f#f#test', type=str)

args = parser.parse_args()
if args.use_snsa is not None:
    args.use_sad = args.use_snsa
dataset_name = args.dataset_name
def _resolve_default_model_name(dataset_name):
    project_dir = os.path.dirname(os.path.abspath(__file__))
    model_root = os.path.join(project_dir, 'pretrained_models')
    model_candidates = []
    env_override = None
    hf_fallback = None

    if 'genia' in dataset_name:
        env_override = os.environ.get('DIFINET_GENIA_MODEL')
        model_candidates = [
            os.path.join(model_root, 'biobert-v1.1'),
            os.path.join(model_root, 'biomedbert-base-uncased-abstract-fulltext'),
            os.path.join(model_root, 'bert-base-cased'),
        ]
        hf_fallback = 'dmis-lab/biobert-v1.1'
    elif dataset_name == 'food':
        env_override = os.environ.get('DIFINET_FOOD_MODEL')
        model_candidates = [
            os.path.join(model_root, 'bert-base-chinese'),
            '/root/.cache/modelscope/hub/models/AI-ModelScope/bert-base-chinese',
        ]
        hf_fallback = 'bert-base-chinese'
    elif dataset_name == 'conll03':
        env_override = os.environ.get('DIFINET_CONLL_MODEL')
        model_candidates = [
            os.path.join(model_root, 'bert-large-cased'),
            os.path.join(model_root, 'bert-base-cased'),
        ]
        hf_fallback = 'bert-large-cased'
    elif dataset_name in ('ace2004', 'ace2005'):
        env_override = os.environ.get('DIFINET_ACE_MODEL')
        model_candidates = [os.path.join(model_root, 'roberta-base')]
        hf_fallback = 'roberta-base'
    else:
        raise RuntimeError(f'Unsupported dataset_name={dataset_name}')

    if env_override:
        return env_override
    for local_dir in model_candidates:
        if os.path.isdir(local_dir):
            return local_dir
    return hf_fallback


if args.model_name is None:
    args.model_name = _resolve_default_model_name(args.dataset_name)

model_name = args.model_name
n_head = args.n_head
######hyper
non_ptm_lr_ratio = 100
schedule = 'linear'
weight_decay = 1e-4
size_embed_dim = 25
ent_thres = args.ent_thres
kernel_size = 3
######hyper

fitlog.set_log_dir('logs/')

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
def seed_torch(seed=43):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False


seed = fitlog.set_rng_seed(rng_seed=args.seed)
seed_torch(args.seed)
os.environ['FASTNLP_GLOBAL_SEED'] = str(seed)
fitlog.add_hyper(args)
fitlog.add_hyper_in_file(__file__)

@cache_results('caches/ner_caches.pkl', _refresh=False)
def get_data(dataset_name, model_name):
    # 以下是我们自己的数据
    if dataset_name == 'ace2004':
        paths = 'preprocess/outputs/ace2004'
    elif dataset_name == 'ace2005':
        paths = 'preprocess/outputs/ace2005'
    elif dataset_name == 'genia':
        paths = 'preprocess/outputs/genia'
    elif dataset_name == 'food':
        paths = 'preprocess/outputs/food'
    elif dataset_name == 'conll03':
        paths = 'preprocess/outputs/conll03'
    else:
        raise RuntimeError("Does not support.")
    pipe = SpanNerPipe(model_name=model_name)
    dl = pipe.process_from_file(paths)

    return dl, pipe.matrix_segs


dl, matrix_segs = get_data(dataset_name, model_name)


def densify(x):
    x = x.todense().astype(np.float32)
    return x


dl.apply_field(densify, field_name='matrix', new_field_name='matrix', progress_bar='Densify')

print(dl)
label2idx = getattr(dl, 'ner_vocab') if hasattr(dl, 'ner_vocab') else getattr(dl, 'label2idx')
print(f"{len(label2idx)} labels: {label2idx}, matrix_segs:{matrix_segs}")


def build_label_loss_weights(data_bundle, label2idx, args):
    if args.label_weighting == 'none':
        return None
    num_labels = len(label2idx)
    counts = torch.zeros(num_labels, dtype=torch.float32)
    for ins in data_bundle.get_dataset('train'):
        for _, _, label_id in ins['ent_target']:
            counts[int(label_id)] += 1
    mean_count = counts[counts > 0].mean().clamp_min(1.0)
    power = 1.0 if args.label_weighting == 'inv' else args.label_weight_power
    weights = torch.pow(mean_count / counts.clamp_min(1.0), power)
    weights = weights.clamp(min=args.label_weight_min, max=args.label_weight_max)
    idx2label = {idx: label for label, idx in label2idx.items()}
    readable = {idx2label[i]: round(float(weights[i]), 4) for i in range(num_labels)}
    print(f"Label loss weights: {readable}, counts={counts.long().tolist()}")
    fitlog.add_other(value=json.dumps(readable, ensure_ascii=False), name='label_loss_weights')
    return weights


def build_bhpc_label_weights(data_bundle, label2idx, args):
    """Balance prototype/boundary losses without changing the main ASL loss.

    GENIA is highly skewed towards PROTEIN.  Inverse-square-root weighting keeps
    rare classes visible to the auxiliary branch while avoiding the instability
    of full inverse-frequency weights.
    """
    if args.bhpc_class_balance == 'none' or args.bhpc_weight <= 0:
        return None
    num_labels = len(label2idx)
    counts = torch.zeros(num_labels, dtype=torch.float32)
    for ins in data_bundle.get_dataset('train'):
        for _, _, label_id in ins['ent_target']:
            counts[int(label_id)] += 1
    mean_count = counts[counts > 0].mean().clamp_min(1.0)
    weights = torch.sqrt(mean_count / counts.clamp_min(1.0))
    # Preserve the global auxiliary-loss scale while limiting rare-class noise.
    weights = weights.clamp(min=0.5, max=3.0)
    weights = weights / weights.mean().clamp_min(1e-6)
    idx2label = {idx: label for label, idx in label2idx.items()}
    readable = {idx2label[i]: round(float(weights[i]), 4) for i in range(num_labels)}
    print(f"BHPC class weights: {readable}, counts={counts.long().tolist()}")
    fitlog.add_other(value=json.dumps(readable, ensure_ascii=False), name='bhpc_label_weights')
    return weights


label_loss_weights = build_label_loss_weights(dl, label2idx, args)
bhpc_label_weights = build_bhpc_label_weights(dl, label2idx, args)
dls = {}
for name, ds in dl.iter_datasets():
    ds.set_pad('matrix', pad_fn=Torch3DMatrixPadder(pad_val=ds.collator.input_fields['matrix']['pad_val'],
                                                    num_class=matrix_segs['ent'],
                                                    batch_size=args.batch_size))

    if name == 'train':
        _dl = prepare_torch_dataloader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                                       batch_sampler=BucketedBatchSampler(ds, 'input_ids',
                                                                          batch_size=args.batch_size,
                                                                          num_batch_per_bucket=30),
                                       pin_memory=True, shuffle=True)

    else:
        _dl = prepare_torch_dataloader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                                       sampler=SortedSampler(ds, 'input_ids'), pin_memory=True, shuffle=False)
    dls[name] = _dl

model = CNNNer(model_name, num_ner_tag=matrix_segs['ent'], cnn_dim=args.cnn_dim, biaffine_size=args.biaffine_size,
               size_embed_dim=size_embed_dim, logit_drop=args.logit_drop,n_layer=args.n_layer,
               kernel_size=kernel_size, n_head=n_head, cnn_depth=args.cnn_depth,
               separateness_rate=args.separateness_rate/100, theta=args.theta,
               sad_topk=args.sad_topk, sad_attn_dim=args.sad_attn_dim,
               use_sad=bool(args.use_sad),
               use_hsr=bool(args.use_hsr),
               sad_use_rel_bias=bool(args.sad_use_rel_bias), sad_gate=bool(args.sad_gate),
               head_type=args.head_type, loss_type=args.loss_type,
               asl_gamma_pos=args.asl_gamma_pos, asl_gamma_neg=args.asl_gamma_neg,
               asl_clip=args.asl_clip, balanced_neg_weight=args.balanced_neg_weight,
               label_loss_weights=label_loss_weights,
               use_length_bias=args.use_length_bias,
               length_bias_bins=args.length_bias_bins,
               bhpc_weight=args.bhpc_weight,
               bhpc_dim=args.bhpc_dim,
               bhpc_temperature=args.bhpc_temperature,
               bhpc_momentum=args.bhpc_momentum,
               bhpc_margin=args.bhpc_margin,
               bhpc_label_weights=bhpc_label_weights,
               bhpc_warmup_steps=args.bhpc_warmup_steps,
               bhpc_prototype_scale=args.bhpc_prototype_scale,
               bhpc_boundary_scale=args.bhpc_boundary_scale,
               relation_dim=args.relation_dim,
               boundary_aux_weight=args.boundary_aux_weight,
               boundary_align_weight=args.boundary_align_weight,
               conflict_rank_weight=args.conflict_rank_weight,
               conflict_margin=args.conflict_margin,
               conflict_warmup_steps=args.conflict_warmup_steps,
               length_gate_bins=args.length_gate_bins,
               quality_aux_weight=args.quality_aux_weight,
               quality_min_iou=args.quality_min_iou,
               hierarchy_aux_weight=args.hierarchy_aux_weight,
               difficulty_curriculum=args.difficulty_curriculum,
               curriculum_warmup_steps=args.curriculum_warmup_steps,
               curriculum_ramp_steps=args.curriculum_ramp_steps,
               curriculum_nested_boost=args.curriculum_nested_boost,
               curriculum_long_boost=args.curriculum_long_boost,
               curriculum_hard_neg_boost=args.curriculum_hard_neg_boost,
               curriculum_long_span=args.curriculum_long_span)

if args.init_from_checkpoint:
    if not os.path.exists(args.init_from_checkpoint):
        raise FileNotFoundError(f'Checkpoint not found: {args.init_from_checkpoint}')
    checkpoint_state = torch.load(args.init_from_checkpoint, map_location='cpu')
    if args.head_type in ('residual_mlp', 'relation_aware') and 'score_head.weight' in checkpoint_state:
        checkpoint_state = dict(checkpoint_state)
        checkpoint_state.setdefault('score_head.base.weight', checkpoint_state['score_head.weight'])
        if 'score_head.bias' in checkpoint_state:
            checkpoint_state.setdefault('score_head.base.bias', checkpoint_state['score_head.bias'])
    load_result = model.load_state_dict(checkpoint_state, strict=False)
    print(
        f"Loaded checkpoint from {args.init_from_checkpoint}, "
        f"missing_keys={len(load_result.missing_keys)}, "
        f"unexpected_keys={len(load_result.unexpected_keys)}"
    )

# optimizer
parameters = []
ln_params = []
non_ln_params = []
non_pretrain_params = []
non_pretrain_ln_params = []

import collections

counter = collections.Counter()
for name, param in model.named_parameters():
    counter[name.split('.')[0]] += torch.numel(param)
print(counter)
print("Total param ", sum(counter.values()))
fitlog.add_to_line(json.dumps(counter, indent=2))
fitlog.add_other(value=sum(counter.values()), name='total_param')
for name, param in model.named_parameters():
    name = name.lower()
    if param.requires_grad is False:
        continue
    if 'pretrain_model' in name:
        if 'norm' in name or 'bias' in name:
            ln_params.append(param)
        else:
            non_ln_params.append(param)
    else:
        if 'norm' in name or 'bias' in name:
            non_pretrain_ln_params.append(param)
        else:      
            non_pretrain_params.append(param)
optimizer = torch.optim.AdamW([{'params': non_ln_params, 'lr': args.encoder_lr, 'weight_decay': weight_decay},
                               {'params': ln_params, 'lr': args.encoder_lr, 'weight_decay': 0},
                               {'params': non_pretrain_ln_params, 'lr': args.lr * 
                                non_ptm_lr_ratio, 'weight_decay': 0},
                               {'params': non_pretrain_params, 'lr': args.lr * non_ptm_lr_ratio,
                                'weight_decay': weight_decay}])
# callbacks
callbacks = []
callbacks.append(FitlogCallback(log_loss_every=20))
callbacks.append(CheckpointCallback(monitor=args.checkpoint_monitor,save_evaluate_results=True, folder='_saved_models', topk=3))
callbacks.append(TorchGradClipCallback(clip_value=5))
callbacks.append(TorchWarmupCallback(warmup=args.warmup, schedule=schedule))
if args.early_stop_patience > 0:
    callbacks.append(EarlyStopCallback(monitor=args.early_stop_monitor, larger_better=True,
                                       patience=args.early_stop_patience))
if args.adv_type != 'none':
    callbacks.append(
        AdversarialTrainingCallback(
            adv_type=args.adv_type,
            epsilon=args.adv_epsilon,
            alpha=args.adv_alpha,
            k=args.adv_k,
            emb_name=args.adv_emb_name,
            adv_loss_weight=args.adv_loss_weight,
            adv_warmup_ratio=args.adv_warmup_ratio,
            adv_every_n_steps=args.adv_every_n_steps,
            pgd_random_start=args.adv_random_start,
        )
    )
    print(
        f'Enable adversarial training: type={args.adv_type}, epsilon={args.adv_epsilon}, '
        f'alpha={args.adv_alpha}, k={args.adv_k}, emb_name={args.adv_emb_name}, '
        f'loss_weight={args.adv_loss_weight}, warmup_ratio={args.adv_warmup_ratio}, '
        f'every_n_steps={args.adv_every_n_steps}, random_start={args.adv_random_start}'
    )
train_dls = {}
evaluate_dls = {}

if 'dev' in dls:
    evaluate_dls['dev'] = dls['dev']
if 'test' in dls:
    evaluate_dls['test'] = dls['test']
allow_nested = True
metrics = {'f': NERMetric(matrix_segs=matrix_segs, ent_thres=ent_thres, allow_nested=allow_nested)}
train_device = 0 if torch.cuda.is_available() else 'cpu'

trainer = Trainer(model=model,
                  driver='torch',
                  train_dataloader=dls.get('train'),
                  evaluate_dataloaders=evaluate_dls,
                  optimizers=optimizer,
                  callbacks=callbacks,
                  overfit_batches=0,
                  device=train_device,
                  n_epochs=args.n_epochs,
                  metrics=metrics,
                  monitor=args.checkpoint_monitor,
                  evaluate_every=args.evaluate_every,
                  evaluate_use_dist_sampler=True,
                  accumulation_steps=args.accumulation_steps,
                  fp16=args.fp16,
                  torch_kwargs=(
                      {'gradscaler_kwargs': {'init_scale': args.fp16_init_scale}}
                      if args.fp16 else None
                  ),
                  progress_bar='rich')

trainer.run(
    num_train_batch_per_epoch=args.num_train_batch_per_epoch,
    num_eval_batch_per_dl=-1,
    num_eval_sanity_batch=1,
)
fitlog.finish()
