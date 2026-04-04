import time
import argparse
import csv
from torch.autograd import Variable
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from utils import *
from apmeter import APMeter
import os
from torch.utils.tensorboard import SummaryWriter
import logging

# Import necessary modules from main_no_teacher.py
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

import models_MSTemba

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow (or joint for eval)')
parser.add_argument('-train', type=str, default='True', help='train or eval')
parser.add_argument('-backbone', type=str, default='i3d')
parser.add_argument('-comp_info', type=str)
parser.add_argument('-gpu', type=str, default='4')
parser.add_argument('-dataset', type=str, default='charades')
parser.add_argument('-rgb_root', type=str, default='/home/amax/ms_temba/MS-Temba-main/data/charades.json')
parser.add_argument('-flow_root', type=str, default='no_root')
parser.add_argument('-type', type=str, default='original')
# parser.add_argument('-lr', type=str, default='0.1')
parser.add_argument('-epochs', type=int, default=50)
parser.add_argument('-model', type=str, default='')
parser.add_argument('-load_model', type=str, default='False')
parser.add_argument('-batch_size', type=str, default='False')
parser.add_argument('-num_clips', type=str, default='False')
parser.add_argument('-skip', type=str, default='False')
parser.add_argument('-num_layer', type=str, default='False')
parser.add_argument('-unisize', type=str, default='False')
parser.add_argument('-alpha_l', type=float, default='1.0')
parser.add_argument('-beta_l', type=float, default='1.0')
parser.add_argument('-output_dir', type=str, default='./output', help='Directory to save output files')

# Add new arguments from main_no_teacher.py
parser.add_argument('--model', default='vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2', type=str, metavar='MODEL',
                    help='Name of model to train')
# parser.add_argument('--input-size', default=224, type=int, help='images input size')
# parser.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
# parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')
parser.add_argument('--model-ema', action='store_true')
parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
parser.set_defaults(model_ema=True)
parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-path', type=float, default=0.0, metavar='PCT',
                    help='Drop path rate (default: 0.0)')
# Optimizer parameters
parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "adamw"')
parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: 1e-8)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.01,
                    help='weight decay (default: 0.01)')
# Learning rate schedule parameters
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "cosine"')
parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                    help='learning rate (default: 5e-4)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                    help='warmup learning rate (default: 1e-6)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# ----- for test -----
parser.add_argument('-split_json', type=str, default='', help='Path to fold json file')
# ----- for test -----


args = parser.parse_args()

# set random seed
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print('Random_SEED:', SEED)

batch_size = int(args.batch_size)

from charades_dataloader import Charades as Dataset

def load_data(train_split, val_split, root):
    # Load Data
    print('load data', root)

    if len(train_split) > 0:
        dataset = Dataset(train_split, 'training', root, batch_size, classes, int(args.num_clips), int(args.skip))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                                 pin_memory=True, collate_fn=collate_fn)
        dataloader.root = root
    else:

        dataset = None
        dataloader = None

    val_dataset = Dataset(val_split, 'testing', root, batch_size, classes, int(args.num_clips), int(args.skip))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0,
                                                 pin_memory=True, collate_fn=collate_fn)
    val_dataloader.root = root
    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}
    
    return dataloaders, datasets


def run(models, criterion, num_epochs=50):
    since = time.time()
    Best_val_map = -1e9
    Best_sample_val_map = 0.
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard_logs'))
    
    for epoch in range(num_epochs):
        since1 = time.time()
        logging.info(f'Epoch {epoch}/{num_epochs - 1}')
        logging.info('-' * 10)
        for model, gpu, dataloader, optimizer, sched, model_file in models:
            train_map, train_loss = train_step(model, gpu, optimizer, dataloader['train'], epoch)
            logging.info(f'Epoch {epoch} - Train MAP: {train_map:.2f}, Train Loss: {train_loss:.4f}')
            
            # ----- for test -----
            prob_val, val_loss, val_map, precision, recall, f1, y_true_epoch, y_score_epoch = val_step(model, gpu, dataloader['val'], epoch)
            # ----- for test -----
            
            '''
            prob_val, val_loss, val_map, sample_val_map = val_step(model, gpu, dataloader['val'], epoch)
            '''
            logging.info(f'Epoch {epoch} - Val MAP: {val_map:.2f}, Val Loss: {val_loss:.4f}')
            
            # ----- for test -----
            logging.info(f'Epoch {epoch} - Val Precision: {precision:.2f}, Val Recall: {recall:.2f}, Val F1: {f1:.2f}')
            # ----- for test -----
            
            '''
            logging.info(f'Epoch {epoch} - Sampled Val MAP: {sample_val_map:.2f}')
            '''
            sched.step(val_loss)
            
            # Log metrics to TensorBoard
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('mAP/train', train_map, epoch)
            writer.add_scalar('mAP/val', val_map, epoch)
            
            # ----- for test -----
            writer.add_scalar('Metrics/val_precision', precision, epoch)
            writer.add_scalar('Metrics/val_recall', recall, epoch)
            writer.add_scalar('Metrics/val_f1', f1, epoch)
            # ----- for test -----
            
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            
            # Time
            epoch_time = time.time() - since1
            total_time = time.time() - since
            logging.info(f"Epoch {epoch}, Total_Time {total_time:.2f}, Epoch_time {epoch_time:.2f}")
            writer.add_scalar('Time/epoch', epoch_time, epoch)
            writer.add_scalar('Time/total', total_time, epoch)



            '''
            if Best_sample_val_map < sample_val_map:
                Best_sample_val_map = sample_val_map
                logging.info(f"Epoch {epoch}, Best Sampled Val Map Update {Best_sample_val_map:.4f}")
            '''
            # ----- for test -----
            if Best_val_map < val_map:
                Best_val_map = val_map
                logging.info(f"Epoch {epoch}, Best Val Map Update {Best_val_map:.4f}")
            # ----- for test -----
                pickle.dump(prob_val, open(os.path.join(args.output_dir, f'{epoch}.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)
                logging.info(f"Logit saved at: {args.output_dir}/{epoch}.pkl")
                
                # Save best model
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
                logging.info(f"Best model saved at: {args.output_dir}/best_model.pth")
                writer.add_scalar('Best_mAP/val', Best_val_map, epoch)
                np.savez(
                    os.path.join(args.output_dir, 'eval_arrays.npz'),
                    y_true=y_true_epoch.astype(np.float32),
                    y_score=y_score_epoch.astype(np.float32),
                )
            
            
            # if Best_val_map < val_map:
            #     Best_val_map = val_map
            #     logging.info(f"Epoch {epoch}, Best Val Map Update {Best_val_map:.4f}")
            #     pickle.dump(prob_val, open(os.path.join(args.output_dir, f'{epoch}.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)
            #     logging.info(f"Logit saved at: {args.output_dir}/{epoch}.pkl")
                
            #     # Save best model
            #     torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
            #     logging.info(f"Best model saved at: {args.output_dir}/best_model.pth")
            #     writer.add_scalar('Best_mAP/val', Best_val_map, epoch)
    
    writer.close()


def eval_model(model, dataloader, baseline=False):
    results = {}
    for data in dataloader:
        other = data[3]
        outputs, loss, probs, _ = run_network(model, data, 0, baseline)
        fps = outputs.size()[1] / other[1][0]

        results[other[0][0]] = (outputs.data.cpu().numpy()[0], probs.data.cpu().numpy()[0], data[2].numpy()[0], fps)
    return results


def run_network(model, data, gpu, epoch=0, baseline=False):
    # 
    inputs, mask, labels, other, hm = data
    # wrap them in Variable 
    inputs = Variable(inputs.cuda(gpu))
    mask = Variable(mask.cuda(gpu))
    labels = Variable(labels.cuda(gpu))
    hm = Variable(hm.cuda(gpu))

    inputs = inputs.squeeze(3).squeeze(3)

    outputs_final = model(inputs)
    # Logit
    probs_f = F.sigmoid(outputs_final) * mask.unsqueeze(2)

    # Loss
    loss_f = F.binary_cross_entropy_with_logits(outputs_final, labels, size_average=False)
    loss_f = torch.sum(loss_f) / torch.sum(mask)
    loss = args.alpha_l * loss_f
    corr = torch.sum(mask)
    tot = torch.sum(mask)

    return outputs_final, loss, probs_f, corr / tot


def train_step(model, gpu, optimizer, dataloader, epoch):
    model.train(True)
    tot_loss = 0.0
    error = 0.0
    num_iter = 0.
    apm = APMeter()
    for data in dataloader:
        optimizer.zero_grad()
        num_iter += 1
        outputs, loss, probs, err = run_network(model, data, gpu, epoch)
        apm.add(probs.data.cpu().numpy()[0], data[2].numpy()[0])
        error += err.data
        tot_loss += loss.data

        loss.backward()
        optimizer.step()

    train_map = 100 * apm.value().mean()
    logging.info(f'Epoch {epoch}, train-map: {train_map:.4f}')
    apm.reset()

    epoch_loss = tot_loss / num_iter

    return train_map, epoch_loss


def val_step(model, gpu, dataloader, epoch):
    
    # ----- for test -----
    model.eval()
    # ----- for test -----
    '''
    model.train(False)
    '''
    apm = APMeter()
    sampled_apm= APMeter()
    tot_loss = 0.0
    error = 0.0
    num_iter = 0.
    full_probs = {}
    
    # ----- for test -----
    true_positive = 0.0
    false_positive = 0.0
    false_negative = 0.0
    valid_total = 0
    valid_over_25 = 0
    valid_lengths = []
    y_true_all = []
    y_score_all = []
    
    
    valid_over_25 = 0
    valid_total = 0
    valid_lengths = []
    with torch.no_grad():
        for data in dataloader:
            num_iter += 1
            other = data[3]

            valid_t = int(sum(data[1].numpy()[0]))
            valid_total += 1
            valid_lengths.append(valid_t)

            if valid_t > 25:
                valid_over_25 += 1

            outputs, loss, probs, err = run_network(model, data, gpu, epoch)

            apm.add(probs.data.cpu().numpy()[0], data[2].numpy()[0])

            error += err.data
            tot_loss += loss.data

            probs_1 = np.asarray(mask_probs(probs.data.cpu().numpy()[0], data[1].numpy()[0])).reshape(-1)
            labels_1 = np.asarray(mask_probs(data[2].numpy()[0], data[1].numpy()[0])).reshape(-1)

            probs_binary = (probs_1 >= 0.5).astype(np.float32)
            labels_binary = (labels_1 >= 0.5).astype(np.float32)

            mask_np = data[1].numpy()[0].astype(bool)          # (T,)
            labels_np = data[2].numpy()[0].astype(np.float32)  # (T, C)
            scores_np = probs.data.cpu().numpy()[0].astype(np.float32)  # (T, C)

            labels_valid = labels_np[mask_np]   # (Tv, C)
            scores_valid = scores_np[mask_np]   # (Tv, C)

            y_true_all.append(labels_valid)
            y_score_all.append(scores_valid)



            true_positive += float(np.sum((probs_binary == 1) & (labels_binary == 1)))
            false_positive += float(np.sum((probs_binary == 1) & (labels_binary == 0)))
            false_negative += float(np.sum((probs_binary == 0) & (labels_binary == 1)))
            full_probs[other[0][0]] = probs_1.T
    logging.info(f'Val samples total: {valid_total}')
    logging.info(f'Val samples with valid_t > 25: {valid_over_25}')
    logging.info(f'Val valid_t min/max: {min(valid_lengths)}/{max(valid_lengths)}')
    logging.info(f'Val valid_t first 20: {valid_lengths[:20]}')
    epoch_loss = tot_loss / num_iter
    

    apm_values = apm.value()
    def safe_mean_ap(values):
        if not torch.is_tensor(values) or values.numel() == 0:
            return 0.0
        scaled = 100 * values
        nonzero = torch.nonzero(scaled)
        if nonzero.numel() == 0:
            return 0.0
        return (torch.sum(scaled) / nonzero.size(0)).item()
        
    val_map = safe_mean_ap(apm_values)

    precision = 100.0 * true_positive / max(true_positive + false_positive, 1.0)
    recall = 100.0 * true_positive / max(true_positive + false_negative, 1.0)
    f1 = 0.0
    if precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)

    logging.info(f'Val samples total: {valid_total}')
    logging.info(f'Val samples with valid_t > 25: {valid_over_25}')
    logging.info(f'Val valid_t min/max: {min(valid_lengths)}/{max(valid_lengths)}')
    logging.info(f'Val valid_t first 20: {valid_lengths[:20]}')

    logging.info(f'Epoch {epoch}, Full-val-map: {val_map:.4f}')
    
    logging.info(f'Epoch {epoch}, Val Precision: {precision:.4f}')
    logging.info(f'Epoch {epoch}, Val Recall: {recall:.4f}')
    logging.info(f'Epoch {epoch}, Val F1: {f1:.4f}')
    logging.info(f'Val samples total: {valid_total}')
    logging.info(f'Val samples with valid_t > 25: {valid_over_25}')
    logging.info(f'Val valid_t min/max: {min(valid_lengths)}/{max(valid_lengths)}')
    logging.info(f'Val valid_t first 20: {valid_lengths[:20]}')
    logging.info(f'apm.value() shape: {tuple(apm_values.shape)}')
    logging.info(f'apm.value() content: {apm_values}')
    logging.info(f'Val confusion stats - TP: {true_positive:.0f}, FP: {false_positive:.0f}, FN: {false_negative:.0f}')
    
    if len(y_true_all) > 0:
        y_true_epoch = np.concatenate(y_true_all, axis=0)   # (N, C)
        y_score_epoch = np.concatenate(y_score_all, axis=0) # (N, C)
    else:
        y_true_epoch = np.zeros((0, classes), dtype=np.float32)
        y_score_epoch = np.zeros((0, classes), dtype=np.float32)

    return full_probs, epoch_loss, val_map, precision, recall, f1, y_true_epoch, y_score_epoch
    
    # ----- for test -----
    
    
    # Iterate over data.
    '''
    for data in dataloader:
        num_iter += 1
        other = data[3]

        outputs, loss, probs, err = run_network(model, data, gpu, epoch)
        if sum(data[1].numpy()[0])>25:
            p1,l1=sampled_25(probs.data.cpu().numpy()[0],data[2].numpy()[0],data[1].numpy()[0])
            sampled_apm.add(p1,l1)

        apm.add(probs.data.cpu().numpy()[0], data[2].numpy()[0])

        error += err.data
        tot_loss += loss.data
        
        probs_1 = mask_probs(probs.data.cpu().numpy()[0],data[1].numpy()[0]).squeeze()

        full_probs[other[0][0]] = probs_1.T
    

    epoch_loss = tot_loss / num_iter
    #val_map = torch.sum(100 * apm.value()) / torch.nonzero(100 * apm.value()).size()[0]
    #sample_val_map = torch.sum(100 * sampled_apm.value()) / torch.nonzero(100 * sampled_apm.value()).size()[0]

    def safe_mean_ap(values):
        if not torch.is_tensor(values) or values.numel() == 0:
            return 0.0
        scaled = 100 * values
        nonzero = torch.nonzero(scaled)
        if nonzero.numel() == 0:
            return 0.0
        return (torch.sum(scaled) / nonzero.size(0)).item()

    val_map = safe_mean_ap(apm.value())
    sample_val_map = safe_mean_ap(sampled_apm.value())


    logging.info(f'Epoch {epoch}, Full-val-map: {val_map:.4f}')
    logging.info(f'Epoch {epoch}, sampled-val-map: {sample_val_map:.4f}')
    logging.info(f'Sampled AP values: {100 * sampled_apm.value()}')
    
    apm.reset()
    sampled_apm.reset()
    return full_probs, epoch_loss, val_map, sample_val_map
    '''
    # apm.reset()
    # return full_probs, epoch_loss, val_map, precision, recall, f1

def setup_logging(output_dir):
    log_file = os.path.join(output_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


if __name__ == '__main__':
    if str(args.unisize) == "True":
        print("uni-size padd all T to",args.num_clips)
        from charades_dataloader import collate_fn_unisize
        collate_fn_f = collate_fn_unisize(args.num_clips)
        collate_fn = collate_fn_f.charades_collate_fn_unisize
    else:
        from charades_dataloader import mt_collate_fn as collate_fn
    
    if args.dataset == 'charades':
        train_split = '/home/amax/ms_temba/MS-Temba-main/data/Charades_subset10_keepid.json'
        test_split = train_split
        rgb_root =  args.rgb_root 
        flow_root = '/flow_feat_path/' # optional
        classes = 157
        
    elif args.dataset == 'self_def':
        train_split = args.split_json if args.split_json else '/home/amax/ms_temba/data/reinforced_data_1/merged_json.json'
        test_split = train_split
        rgb_root = args.rgb_root
        flow_root = '/flow_feat_path/' # optional
        classes = 2
    
        
    elif args.dataset == 'tsu':
        train_split = '/path/to/smarthome.json'
        test_split = train_split
        rgb_root =  args.rgb_root 
        flow_root = '/flow_feat_path/' # optional
        classes = 51

    elif args.dataset == 'multithumos':
        train_split = '/path/to/multithumos.json'
        test_split = train_split
        rgb_root = args.rgb_root 
        flow_root = '/flow_feat_path/' # optional
        classes = 65

    if args.mode == 'flow':
        print('flow mode', flow_root)
        dataloaders, datasets = load_data(train_split, test_split, flow_root)
    elif args.mode == 'rgb':
        print('RGB mode', rgb_root)
        dataloaders, datasets = load_data(train_split, test_split, rgb_root)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    setup_logging(args.output_dir)
    logging.info(f"Arguments: {args}")

    if args.train:
        if args.backbone == 'i3d':
            in_feat_dim = 1024
        elif args.backbone == 'clip':
            in_feat_dim = 768

        # Create model using timm's create_model function
        model = create_model(
            args.model,
            pretrained=False,
            num_classes=classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            in_feat_dim=in_feat_dim
        )
        model.cuda()

        criterion = LabelSmoothingCrossEntropy()
 
        optimizer = create_optimizer(args, model)
        lr_scheduler, _ = create_scheduler(args, optimizer)

        if args.model_ema:
            model_ema = ModelEma(
                model,
                decay=args.model_ema_decay,
                device='cpu' if args.model_ema_force_cpu else '',
                resume=''
            )
        else:
            model_ema = None

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Number of parameters: {n_parameters}")

        run([(model, 0, dataloaders, optimizer, lr_scheduler, args.comp_info)], criterion, num_epochs=int(args.epochs))
