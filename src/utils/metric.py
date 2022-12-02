import numpy as np
from scipy import stats
from sklearn.metrics import ndcg_score
import torch
from .utils import label_recover
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, ndcg_score


def calc_metric(mode, pos_preds, neg_preds, y_pred, y_gt, scaler, k=1):
    # Edge level
    pos_pred = torch.cat(pos_preds, dim=0) # [b, 1] -> [n, 1]
    neg_pred = torch.cat(neg_preds, dim=0).view(-1, k) # [b, 1] -> [n*k, 1] -> [n, k]
    scores = torch.cat([pos_pred.view(-1,1), neg_pred], dim = 1) # [n, k+1]
    mrr, hits1, hits3, hits10  = eval_mrr(scores) # input: [b, ], [b, k]
    if mode == 'test':
        ndcg, ndcg100 = eval_ndcg(scores)
    # Node level
    val_pred, val_gt = torch.vstack(y_pred).numpy(), torch.vstack(y_gt).numpy() # [n, 5], [n, 5]
    cor = similarity(val_pred, val_gt)
    if mode == 'test':
        male, rmsle = calc_male_rmsle(val_pred, val_gt, scaler)
    
    if mode == 'test':
        return {'mrr': mrr, 'hits@1,3,10': [hits1, hits3, hits10], 'ndcg,@100': [ndcg, ndcg100], \
            'male': male, 'rmsle': rmsle, 'cor': cor}
    else:
        return {'mrr': mrr, 'cor': cor}
    
    

def topk_prec(y_true, y_score, k):
    y1, y2 =  torch.from_numpy(y_true), torch.from_numpy(y_score)
    k = min(k, len(y1), len(y2))
    top = torch.topk(y2, k=k)[1].tolist()
    topk = torch.topk(y1, k=k)[1].tolist()

    cnt = 0
    for tt in top:
        if tt in topk:
            cnt += 1
    return cnt/k


def inverse_cum_log(pred, scaler=None):
    if scaler:
        pred = label_recover(pred, scaler)
    pred = np.clip(pred, 0, 100000)
    o_pred = np.exp(pred)-1
    
    pred[:,0:1] = o_pred[:,0:1]
    for yy in range(1, pred.shape[1]):
        pred[:, yy:yy+1] = o_pred[:, yy:yy+1] - o_pred[:, yy-1:yy]
    return pred


def calc_male_rmsle(cum_pred, cum_gt, scaler):
    pred = inverse_cum_log(cum_pred, scaler)
    gt = inverse_cum_log(cum_gt, scaler)
    
    pred = np.clip(pred, 0, 10000000)
    gt = np.clip(gt, 0, 10000000)
    
    log_pred = np.log(pred + 1) # log
    log_gt = np.log(gt + 1)
    
    male = mean_absolute_error(log_gt, log_pred)
    rmsle = np.sqrt(mean_squared_error(log_gt, log_pred))    
    return round(male, 4) , round(rmsle, 4) 


# def calc_male_rmsle(cum_pred, cum_gt, scaler):
#     pred = inverse_cum_log(cum_pred, scaler)
#     gt = inverse_cum_log(cum_gt, scaler)
    
#     pred = np.clip(pred, 0, 10000000)
#     gt = np.clip(gt, 0, 10000000)
    
#     log_pred = np.log(pred + 1) # log
#     log_gt = np.log(gt + 1)
    
#     males, rmsles = [], []
#     for yy in range(log_pred.shape[1]):
#         y_true, y_pred = log_gt[:, yy], log_pred[:, yy]
#         male = mean_absolute_error(y_true, y_pred)
#         males.append(male)
        
#         rmsle = np.sqrt(mean_squared_error(y_true, y_pred))
#         rmsles.append(rmsle)
        
#     males.append(mean_absolute_error(log_gt, log_pred))
#     rmsles.append(np.sqrt(mean_squared_error(log_gt, log_pred)))
    
#     males = [round(item, 3) for item in males]
#     rmsles = [round(item, 3) for item in rmsles]
#     return males, rmsles


def similarity(pred, gt):
    cor, _ = stats.spearmanr(pred.flatten(), gt.flatten())
    return round(cor, 4)


# def similarity(pred, gt, num_year = 5, is_show = False):
#     k = 50
#     # sims = {}
#     cors, maes, rmses = 0, 0, 0
#     for yy in range(num_year):
#         y_true, y_pred = gt[:, yy], pred[:, yy]
#         cor, pv = stats.spearmanr(y_true, y_pred)
#         # tau, p_value = stats.kendalltau(y_true, y_pred)
#         # prec = topk_prec(y_true, y_pred, k=k)
#         # ndcg = ndcg_score(y_true[None,:], y_pred[None, :], k=k)

#         mae = mean_absolute_error(y_true, y_pred)
#         rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
#         # if is_show:
#         #     print(f'year: {yy}, cor: {cor}, tau: {tau}, prec: {prec}, ndcg: {ndcg}')
#         cors += cor
#         maes += mae
#         rmses += rmse
#     return cors/num_year, maes/num_year, rmses/num_year


def split_cites(gt):
    aa = gt[:, 4:5]
    cut1 = np.percentile(aa, 33)
    cut2 = np.percentile(aa, 66)

    cite = pd.DataFrame(aa, columns=["cite"]) # [16200, 1]
    low = cite[cite.cite<=cut1].index.tolist() # list[41197]
    mid = cite[(cite.cite>cut1)&(cite.cite<=cut2)].index.tolist() # [12885]
    high =cite[cite.cite>cut2].index.tolist() # [15180]

    a, b, c = np.mean(gt[low, :]), np.mean(gt[mid, :]), np.mean(gt[high, :])
    return low, mid, high, np.round(np.array([a, b, c]), 3)


def eval_cites(y_pred, y_gt, scaler):
    pred = label_recover(y_pred, scaler)
    gt = label_recover(y_gt, scaler)

    low, mid, high, cites_mean = split_cites(gt)
    pred1, pred2, pred3 = np.mean(pred[low, :]), np.mean(pred[mid, :]), np.mean(pred[high, :])
    pred_mean = np.round(np.array([pred1, pred2, pred3]), 3)
    return list(cites_mean), list(pred_mean)


def eval_mrr(scores):
    '''
        scores is an array with shape (batch size, num_pos+num_neg).
    '''
    argsort = torch.argsort(scores, dim = 1, descending = True) # [b, k+1]
    ranking_list = torch.nonzero(argsort == 0, as_tuple=False) # [b, 2], 等于0的位置
    ranking_list = ranking_list[:, 1] + 1 # [b]
    hits1_list = (ranking_list <= 1).to(torch.float)
    hits3_list = (ranking_list <= 3).to(torch.float) # [b] 0 or 1
    hits10_list = (ranking_list <= 10).to(torch.float)
    mrr_list = 1./ranking_list.to(torch.float)
    return mrr_list.mean().item(), hits1_list.mean().item(), hits3_list.mean().item(), hits10_list.mean().item()


def eval_ndcg(scores):
    y_true = torch.zeros_like(scores)
    y_true[:, 0] = 1
    ndcg = ndcg_score(y_true, scores)
    ndcg_100 = ndcg_score(y_true, scores, k=100)
    return ndcg, ndcg_100
    

# def evaluate_mrr(y_pred_pos, y_pred_neg, k):
#     # num_pos = len(y_pred_pos)
#     # if num_pos*k < len(y_pred_neg):
#     #     y_pred_neg = y_pred_neg[0:num_pos*k]
#     # else:
#     #     num = len(y_pred_neg) // k
#     #     y_pred_pos = y_pred_pos[0:num]
#     #     y_pred_neg = y_pred_neg[0:num*k]

#     y_pred_neg = y_pred_neg.view(y_pred_pos.shape[0], -1) # [b, n]

#     y_pred = torch.cat([y_pred_pos.view(-1,1), y_pred_neg], dim = 1) # [b, n+1]
#     argsort = torch.argsort(y_pred, dim = 1, descending = True)
#     ranking_list = torch.nonzero(argsort == 0, as_tuple=False)
#     ranking_list = ranking_list[:, 1] + 1
#     # hits1_list = (ranking_list <= 1).to(torch.float)
#     # hits3_list = (ranking_list <= 3).to(torch.float)
#     # hits10_list = (ranking_list <= 10).to(torch.float)
#     mrr_list = 1./ranking_list.to(torch.float)
#     mrr = mrr_list.mean().item()
#     return mrr