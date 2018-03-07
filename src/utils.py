import numpy as np
import random
import torch
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support

def load_data(path):
    data_file = np.load(path)
    train = data_file['train_dict'].item()
    valid = data_file['valid_dict'].item()
    test = data_file['test_dict'].item()
    embed_dict = data_file['feature_dict'].item()
    embed_dims = {k: len(embed_dict[k]) for k in embed_dict}
    embed_keys = sorted(list(embed_dict.keys()))
    for k in train:
        break
    example = train[k]
    float_keys = sorted([k for k in example if k not in embed_keys])
    float_keys.remove('label')
    if 'duration' in float_keys:
        float_keys.remove('duration')
    if 'ret' in float_keys:
        float_keys.remove('ret')
    return train, valid, test, embed_dict, embed_dims, embed_keys, float_keys

def fetch_data(data_dic, embed_keys, float_keys):
    label = int(data_dic['label'])
    duration = data_dic['duration']
    ret = data_dic['ret']
    X_float = [data_dic[k] for k in float_keys] 
    X_embed = [int(data_dic[k]) for k in embed_keys]
    return X_float, X_embed, label, duration, ret

def batch_iter(data, batch_size, embed_keys, float_keys, shuffle = False):
    start = -1 * batch_size
    indices = sorted([k for k in data])
    num_samples = len(indices)
    while True:
        if shuffle:
            random.shuffle(indices)
        batch_X_float, batch_X_embed, batch_label, batch_duration, batch_ret = [], [], [], [], []
        for idx in indices:
            X_float, X_embed, label, duration, ret = fetch_data(data[idx], embed_keys, float_keys)
            batch_label.append(label)
            batch_duration.append(duration)
            batch_ret.append(ret)
            batch_X_float.append(X_float)
            batch_X_embed.append(X_embed)
            if len(batch_label) == batch_size:
                var_X_float = Variable(torch.FloatTensor(batch_X_float))
                var_X_embed = Variable(torch.LongTensor(batch_X_embed))
                var_ret = Variable(torch.FloatTensor(batch_ret))
                var_label = Variable(torch.LongTensor(batch_label))
                var_duration = Variable(torch.FloatTensor(batch_duration))
                batch_X_float, batch_X_embed, batch_label, batch_duration, batch_ret = [], [], [], [], []
                yield set_cuda(var_X_float), set_cuda(var_X_embed), set_cuda(var_label), set_cuda(var_duration), set_cuda(var_ret)
        if len(batch_label) > 0:
            var_X_float = Variable(torch.FloatTensor(batch_X_float))
            var_X_embed = Variable(torch.LongTensor(batch_X_embed))
            var_ret = Variable(torch.FloatTensor(batch_ret))
            var_label = Variable(torch.LongTensor(batch_label))
            var_duration = Variable(torch.FloatTensor(batch_duration))
            yield set_cuda(var_X_float), set_cuda(var_X_embed), set_cuda(var_label), set_cuda(var_duration), set_cuda(var_ret)

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

def get_clf_stats(prob, label):
    label = detach_cuda(label).data.numpy().reshape(-1)
    prob = detach_cuda(prob).data.numpy()
    auc = roc_auc_score(label, prob[:, 1])
    pred = np.argmax(prob, axis = 1)
    acc = accuracy_score(label, pred)
    return auc, acc

def ret_strtgy(pred_y, batch_y, topk = 1000):
    pred_y = detach_cuda(pred_y).data.numpy().reshape(-1)
    batch_y = detach_cuda(batch_y).data.numpy().reshape(-1)
    indices = pred_y.argsort()[-topk:][::-1]
    rets = np.take(batch_y, indices)
    return np.mean(rets), np.median(rets), np.std(rets)

def clf_strtgy(prob, true_ret, topk = 1000):
    prob = detach_cuda(prob).data.numpy().reshape(-1)
    true_ret = detach_cuda(true_ret).data.numpy().reshape(-1)
    indices = prob.argsort()[-topk:][::-1]
    rets = np.take(true_ret, indices)
    return np.mean(rets), np.median(rets), np.std(rets)

def con_regs_strtgy(prob, paid_ret, dft_ret, true_ret, topk = 1000):
    '''
    prob: default probability, num_data * 2 np array
    paid_ret: return if nondefault
    '''
    dft_ret = detach_cuda(dft_ret).data.numpy().reshape(-1)
    true_ret = detach_cuda(true_ret).data.numpy().reshape(-1)
    exp_ret = np.multiply(paid_ret, prob[:, 0]) + np.multiply(dft_ret, prob[:, 1])
    indices = exp_ret.argsort()[-topk:][::-1]
    rets = np.take(true_ret, indices)
    return np.mean(rets), np.median(rets), np.std(rets)
