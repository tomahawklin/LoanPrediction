import numpy as np
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
from utils import load_data, batch_iter, set_cuda, detach_cuda, get_stats, con_regs_strtgy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import time
import math

class MultiNet(nn.Module):
    def __init__(self, input_float_dim, embed_dims, embed_sizes, hidden_dim, embed_keys, num_class, lmbda = 0.1):
        super(MultiNet, self).__init__()
        self.linear1 = nn.Linear(input_float_dim, hidden_dim)
        # Embed_keys makes sure that order of layers are consistent
        embed_layers = [nn.Embedding(embed_dims[k], embed_sizes[k]) for k in embed_keys]
        self.embed = nn.ModuleList(embed_layers)
        self.linear2 = nn.Linear(hidden_dim + sum([embed_sizes[k] for k in embed_keys]), hidden_dim)
        sub_hidden_dim = int(hidden_dim / 2)
        self.linear3 = nn.Linear(hidden_dim, sub_hidden_dim)
        self.linear4_1 = nn.Linear(sub_hidden_dim, 1)
        self.rgrs_loss = nn.MSELoss()
    
    def forward(self, X_float, X_embed, batch_label, batch_ret):
        float_hid = self.linear1(X_float)
        float_hid = F.relu(float_hid)
        embed_hid = torch.cat([self.embed[i](X_embed[:, i]) for i in range(len(self.embed))], dim = 1)
        hid = torch.cat([float_hid, embed_hid], dim = 1)
        hid = self.linear2(hid)
        hid = F.relu(hid)
        hid = self.linear3(hid)
        hid = F.relu(hid)
        ret = self.linear4_1(hid)
        rgrs_loss = self.rgrs_loss(ret, batch_ret)
        return rgrs_loss, ret

def train(model, num_batch, train_batches, valid_batches, test_batches, opt, num_epochs, hidden_dim, verbose = True):
    epoch = 0
    step = 0
    best_epoch = 0
    best_val_ret = 0
    best_tst_ret = 0
    best_med_ret = 0
    best_std_ret = 0
    rpt_epoch = 5
    test_epoch = 10
    while epoch < num_epochs:
        batch_X_float, batch_X_embed, batch_label, batch_duration, batch_ret = next(train_batches)
        default_idx = (batch_label.data == 1).nonzero()
        batch_X_float = torch.squeeze(batch_X_float[default_idx,], 1)
        batch_X_embed = torch.squeeze(batch_X_embed[default_idx,], 1)
        batch_label = torch.squeeze(batch_label[default_idx,], 1)
        batch_ret = torch.squeeze(batch_ret[default_idx,], 1)
        opt.zero_grad()
        loss, pred_ret = model(batch_X_float, batch_X_embed, batch_label, batch_ret)
        loss.backward()
        #clip_grad_norm(model.parameters(), 1)
        opt.step()
        step += 1
        if step >= num_batch:
            epoch += 1
            step = 0
            if epoch % rpt_epoch == 0 and verbose:
                med_diff, avg_diff, max_diff = get_stats(pred_ret, batch_ret)
                print('Train: epoch: %d, avg loss: %.3f, median diff: %.3f, mean: %.3f, max: %.3f' % (epoch, loss.data[0], med_diff, avg_diff, max_diff))
            valid_X_float, valid_X_embed, valid_label, valid_duration, valid_ret = next(valid_batches)
            _, dft_ret = model(valid_X_float, valid_X_embed, valid_label, valid_ret)

            valid_avg_ret, valid_med_ret, valid_std_ret = con_regs_strtgy(prob_valid, paid_ret_valid, dft_ret, valid_ret)
            test_X_float, test_X_embed, test_label, test_duration, test_ret = next(test_batches)
            _, dft_ret = model(test_X_float, test_X_embed, test_label, test_ret)
            test_avg_ret, test_med_ret, test_std_ret = con_regs_strtgy(prob_test, paid_ret_test, dft_ret, test_ret)
            if valid_avg_ret > best_val_ret:
                best_epoch = epoch
                best_val_ret = valid_avg_ret
                best_tst_ret = test_avg_ret
                best_med_ret = test_med_ret
                best_std_ret = test_std_ret
                model_name = './models/rf_regs_mse_' + str(hidden_dim) + 'dim_model.pt'
                torch.save(model.state_dict(), model_name)
            if epoch % test_epoch == 0 and verbose:
                print('Test epoch: %d, avg return: %.3f, median: %.3f, std: %.3f' % (epoch, test_avg_ret, test_med_ret, test_std_ret))           
    return best_epoch, best_tst_ret, best_med_ret, best_std_ret

train_data, valid_data, test_data, embed_dict, embed_dims, embed_keys, float_keys = load_data('../data/final_data.npz')

embed_sizes = {'issue_month': 4, 'home_ownership': 2, 'verification_status': 2, 'emp_length': 4, 
               'initial_list_status': 1, 'addr_state': 10, 'early_month': 4, 'grade': 3, 
               'purpose': 5, 'sub_grade': 10, 'zip_code': 20, 'early_year': 6, 'term': 1, 'issue_year': 6}
num_samples = len(train_data)
batch_size = 1000
num_batch = math.ceil(num_samples / batch_size)
num_epochs = 200
learning_rate = 1e-3
weight_decay = 0.0001
num_class = len(set([train_data[t]['label'] for t in train_data]))

train_ids = sorted([k for k in train_data])
X_train = np.array([[train_data[d][k] for k in float_keys] + [train_data[d][k] for k in embed_keys] for d in train_ids])
label_train = np.array([train_data[k]['label'] for k in train_ids])
duration_train = np.array([train_data[k]['duration'] for k in train_ids])
ret_train = np.array([train_data[k]['ret'] for k in train_ids])
clf = RandomForestClassifier(max_depth = 10, random_state = 0)
clf.fit(X_train, label_train)

valid_ids = sorted([k for k in valid_data])
X_valid = np.array([[valid_data[d][k] for k in float_keys] + [valid_data[d][k] for k in embed_keys] for d in valid_ids])
ret_valid = np.array([valid_data[k]['ret'] for k in valid_ids])
prob_valid = clf.predict_proba(X_valid)
term = np.array([36 if valid_data[k]['term'] == 0 else 60 for k in valid_ids])
instl = np.array([valid_data[k]['installment'] for k in valid_ids])
amnt = np.array([valid_data[k]['funded_amnt'] for k in valid_ids])
paid_ret_valid = np.divide(np.multiply(term, instl) - amnt, amnt)

test_ids = sorted([k for k in test_data])
X_test = np.array([[test_data[d][k] for k in float_keys] + [test_data[d][k] for k in embed_keys] for d in test_ids])
label_test = np.array([test_data[k]['label'] for k in test_ids])
prob_test = clf.predict_proba(X_test)
term = np.array([36 if test_data[k]['term'] == 0 else 60 for k in test_ids])
instl = np.array([test_data[k]['installment'] for k in test_ids])
amnt = np.array([test_data[k]['funded_amnt'] for k in test_ids])
paid_ret_test = np.divide(np.multiply(term, instl) - amnt, amnt)
'''
label_pred = clf.predict(X_test)
pos_prob = clf.predict_proba(X_test)[:, 1]
roc_auc_score(label_test, pos_prob)
0.95615843482439034
sum([1 for i in range(len(test_data)) if label_pred[i] == label_test[i]]) / len(test_data)
0.9154688614978216
'''

for hidden_dim in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    print('Experimenting hidden dimension = %d' % hidden_dim)
    train_batches = batch_iter(train_data, batch_size, embed_keys, float_keys, shuffle = True)
    valid_batches = batch_iter(valid_data, len(valid_data), embed_keys, float_keys, shuffle = False)
    test_batches = batch_iter(test_data, len(test_data), embed_keys, float_keys, shuffle = False)
    
    model = MultiNet(len(float_keys), embed_dims, embed_sizes, hidden_dim, embed_keys, num_class)
    model = set_cuda(model)
    opt = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
    
    start = time.time()
    best_epoch, best_tst_ret, best_med_ret, best_std_ret = train(model, num_batch, train_batches, valid_batches, test_batches, opt, num_epochs, hidden_dim)
    
    print('Best test return: %.3f, median: %.3f, std: %.3f in epoch %d. Finished in %.3f seconds' % (best_tst_ret, best_med_ret, best_std_ret, best_epoch, time.time() - start))


print('Training finished')
#model.load_state_dict(torch.load('model.pt'))
#test_X_float, test_X_embed, test_label, test_duration, test_ret = next(test_batches)
#_, pred_ret = model(test_X_float, test_X_embed, test_label, test_ret)
#avg_ret, med_ret, std_ret = ret_strtgy(pred_ret, test_ret)
#print('Avg return: %.3f, median: %.3f, std: %.3f' % (avg_ret, med_ret, std_ret))

train_batches = batch_iter(train_data, len(train_data), embed_keys, float_keys, shuffle = False)