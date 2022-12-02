from torch_geometric.utils import structured_negative_sampling
import logging
import os
import sys
import random
import time

import numpy as np
import torch


def set_random_seed(seed=None):
    seed = 42 if seed==None else seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_free_device(args):
    def get_free_gpu():
        import gpustat
        stats = gpustat.GPUStatCollection.new_query()
        GPU_usage = {stats[ii].entry['index']: stats[ii].entry['memory.used'] for ii in range(len(stats))}
        bestGPU = sorted(GPU_usage.items(), key=lambda item: item[1])[0][0]
        print(f"setGPU: Setting GPU to: {bestGPU}, GPU_usage: {GPU_usage}")
        return bestGPU
    try:
        gid = args.gid
        if gid<0:
            args.gid = get_free_gpu()
    except:
        args.gid = get_free_gpu()
    return torch.device(f'cuda:{args.gid}') if torch.cuda.is_available() else torch.device('cpu')


def set_logger(args):
    format_str = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(format=format_str, level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    if args.log:
        formatter = logging.Formatter(format_str, "%Y-%m-%d %H:%M:%S")
        try:
            log_dir = args.log_dir
        except:
            log_dir = '../log'
        cur_time = time.strftime("%m.%d-%H:%M")
        log_file = os.path.join(log_dir, cur_time + f'_{args.log}.txt')
        
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    cmd_input = 'python ' + ' '.join(sys.argv)
    logger.info(cmd_input)
    logger.info(args)
    return logger


def local_neg_sample(pos_edges, num_nodes, num_neg, quick=False, neg_self_loops=False, random_src=False):
    num = pos_edges.size(1)
    if random_src:
        idx = torch.randint(0, 2, (num, ), dtype=torch.long)
        neg_src = pos_edges[idx, :]
    else:
        neg_src = pos_edges[0, :]
    
    if quick:
        neg_dst = torch.randint(0, num_nodes, (num_neg * num,)) # [n*k]
    else:
        neg_lst = []
        for i in range(num_neg):
            _, _, neg_tgt = structured_negative_sampling(pos_edges, num_nodes, contains_neg_self_loops=neg_self_loops)
            neg_lst.append(neg_tgt)
        neg_dst = torch.hstack(neg_lst) # [n*k]
    
    neg_src = neg_src.repeat_interleave(num_neg) # [n] -> [n*k]
    neg_edges = torch.stack((neg_src, neg_dst)) # [2, n*k]
    return neg_edges.reshape((2, -1, num_neg)) # [2, n, k]


def eval_mrr(y_pred_pos, y_pred_neg, k, cuts = None):
    y_pred_neg = y_pred_neg.view(y_pred_pos.shape[0], -1) # [b, n]
    mrr0 = evaluate_mrr(y_pred_pos, y_pred_neg, k)
    if cuts == None:
        return [round(mrr0, 4)]
    else:
        mrrs = []
        for i in range(len(cuts)-1):
            ss, ee = cuts[i], cuts[i+1]
            mrr = evaluate_mrr(y_pred_pos[ss:ee], y_pred_neg[ss:ee], k)
            mrrs.append(round(mrr, 4))
        mrrs.append(round(mrr0, 4))
        return mrrs


def evaluate_mrr(y_pred_pos, y_pred_neg, k):
    # num_pos = len(y_pred_pos)
    # if num_pos*k < len(y_pred_neg):
    #     y_pred_neg = y_pred_neg[0:num_pos*k]
    # elif num_pos*k > len(y_pred_neg):
    #     num = len(y_pred_neg) // k
    #     y_pred_pos = y_pred_pos[0:num]
    #     y_pred_neg = y_pred_neg[0:num*k]

    y_pred_neg = y_pred_neg.view(y_pred_pos.shape[0], -1) # [b, n]

    y_pred = torch.cat([y_pred_pos.view(-1,1), y_pred_neg], dim = 1) # [b, n+1]
    argsort = torch.argsort(y_pred, dim = 1, descending = True)
    ranking_list = torch.nonzero(argsort == 0, as_tuple=False)
    ranking_list = ranking_list[:, 1] + 1
    # hits1_list = (ranking_list <= 1).to(torch.float)
    # hits3_list = (ranking_list <= 3).to(torch.float)
    # hits10_list = (ranking_list <= 10).to(torch.float)
    mrr_list = 1./ranking_list.to(torch.float)
    mrr = mrr_list.mean().item()
    return round(mrr, 4)


def label_normorlization(labels):
    maximum = labels.max()
    minimum = labels.min()
    new_value = (labels-minimum)/(maximum-minimum)
    return new_value, (maximum,minimum)


def label_recover(values, scaler):
    label_max, label_min = scaler
    return values*(label_max-label_min)+label_min




def retrieve_name_ex(var):
    frame = sys._getframe(2)
    while(frame):
        for item in frame.f_locals.items():
            if (var is item[1]):
                return item[0]
        frame = frame.f_back
    return ""

def myout(*para, threshold=10):
    def get_mode(var):
        if isinstance(var, (list, dict, set)):
            return 'len'
        elif isinstance(var, (np.ndarray, torch.Tensor)):
            return 'shape'
        else: return ''

    for var in para:
        name = retrieve_name_ex(var)
        mode = get_mode(var)
        if mode=='len':
            len_var = len(var)
            if isinstance(var, list) and len_var>threshold and threshold>6:
                print(f'{name} : len={len_var}, list([{var[0]}, {var[1]}, {var[2]}, ..., {var[-3]}, {var[-2]}, {var[-1]}])')
            elif isinstance(var, set) and len_var>threshold and threshold>6:
                var = list(var)
                print(f'{name} : len={len_var}, set([{var[0]}, {var[1]}, {var[2]}, ..., {var[-3]}, {var[-2]}, {var[-1]}])')
            elif isinstance(var, dict) and len_var>threshold and threshold>6:
                tmp = []
                for kk, vv in var.items():
                    tmp.append(f'{kk}: {vv}')
                    if len(tmp) > threshold: break
                print(f'{name} : len={len_var}, dict([{tmp[0]}, {tmp[1]}, {tmp[2]}, {tmp[3]}, {tmp[4]}, {tmp[5]}, ...])')
            else:
                print(f'{name} : len={len_var}, {var}')
        elif mode=='shape':
            sp = var.shape
            if len(sp)<2:
                print(f'{name} : shape={sp}, {var}')
            else:
                print(f'{name} : shape={sp}')
                print(var)
        else:
            print(f"{name} = {var}")