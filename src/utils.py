import numpy as np
import random
import torch
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

def load_data(path):
    data_file = np.load(path)
    train = data_file['train_dict'].item()
    test = data_file['test_dict'].item()
    embed_dict = data_file['feature_dict'].item()
    embed_dims = {k: len(embed_dict[k]) for k in embed_dict}
    embed_keys = list(embed_dict.keys())
    for k in train:
        break
    example = train[k]
    float_keys = [k for k in example if k not in embed_keys]
    float_keys.remove('label')
    if 'duration' in float_keys:
        float_keys.remove('duration')
    if 'ret' in float_keys:
        float_keys.remove('ret')
    return train, test, embed_dict, embed_dims, embed_keys, float_keys

def fetch_data(data_dic, embed_keys, float_keys):
    label = int(data_dic['label'])
    duration = data_dic['duration']
    ret = data_dic['ret']
    X_float = [data_dic[k] for k in float_keys] 
    X_embed = [int(data_dic[k]) for k in embed_keys]
    return X_float, X_embed, label, duration, ret

def batch_iter(data, batch_size, embed_keys, float_keys, shuffle = False):
    start = -1 * batch_size
    keys = [k for k in data]
    num_samples = len(keys)
    indices = list(range(num_samples))
    if shuffle:
        random.shuffle(indices)
    while True:
        start += batch_size
        if start >= num_samples - batch_size:
            if shuffle:
                random.shuffle(indices)
            batch_idx = indices[:batch_size]
            start = batch_size
        else:
            batch_idx = indices[start: start + batch_size]
        batch_keys = [keys[i] for i in batch_idx]
        batch_data = [data[k] for k in batch_keys]
        batch_X_float, batch_X_embed, batch_label, batch_duration, batch_ret = [], [], [], [], [] 
        for i in range(len(batch_data)):
            X_float, X_embed, label, duration, ret = fetch_data(batch_data[i], embed_keys, float_keys)
            batch_label.append(label)
            batch_duration.append(duration)
            batch_ret.append(ret)
            batch_X_float.append(X_float)
            batch_X_embed.append(X_embed)
        batch_X_float = Variable(torch.FloatTensor(batch_X_float))
        batch_X_embed = Variable(torch.LongTensor(batch_X_embed))
        batch_ret = Variable(torch.FloatTensor(batch_ret))
        batch_label = Variable(torch.LongTensor(batch_label))
        batch_duration = Variable(torch.FloatTensor(batch_duration))
        yield set_cuda(batch_X_float), set_cuda(batch_X_embed), set_cuda(batch_label), set_cuda(batch_duration), set_cuda(batch_ret)

def set_cuda(var):
    if torch.cuda.is_available():
        return var.cuda()
    else:
        return var

def detach_cuda(var):
    if torch.cuda.is_available():
        return var.cpu()
    else:
        return var

def get_stats(pred_y, batch_y):
    abs_diff = torch.abs(pred_y.view(-1) - batch_y.view(-1))
    med_diff = torch.median(abs_diff).data[0]
    avg_diff = torch.mean(abs_diff).data[0]
    max_diff = torch.max(abs_diff).data[0]
    return med_diff, avg_diff, max_diff

def ret_strtgy(pred_y, batch_y, topk = 1000):
    pred_y = detach_cuda(pred_y).data.numpy().reshape(-1)
    batch_y = detach_cuda(batch_y).data.numpy().reshape(-1)
    indices = pred_y.argsort()[-topk:][::-1]
    rets = np.take(batch_y, indices)
    return np.mean(rets), np.median(rets), np.std(rets)
