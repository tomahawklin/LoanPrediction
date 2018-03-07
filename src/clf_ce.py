import numpy as np
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
from utils import load_data, batch_iter, set_cuda, detach_cuda, get_clf_stats, clf_strtgy
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
        self.linear4 = nn.Linear(sub_hidden_dim, num_class)
        self.prob = nn.Softmax()
        self.clf_loss = nn.CrossEntropyLoss()
    
    def forward(self, X_float, X_embed, batch_label, batch_ret):
        float_hid = self.linear1(X_float)
        float_hid = F.relu(float_hid)
        embed_hid = torch.cat([self.embed[i](X_embed[:, i]) for i in range(len(self.embed))], dim = 1)
        hid = torch.cat([float_hid, embed_hid], dim = 1)
        hid = self.linear2(hid)
        hid = F.relu(hid)
        hid = self.linear3(hid)
        hid = F.relu(hid)
        hid = self.linear4(hid)
        prob = self.prob(hid)
        clf_loss = self.clf_loss(prob, batch_label)
        return clf_loss, prob

def train(model, num_batch, train_batches, valid_batches, test_batches, opt, num_epochs, hidden_dim, verbose = True):
    epoch = 0
    step = 0
    best_epoch = 0
    best_val_ret = 0
    best_tst_ret = 0
    best_med_ret = 0
    best_std_ret = 0
    rpt_epoch = 1
    test_epoch = 1
    while epoch < num_epochs:
        batch_X_float, batch_X_embed, batch_label, batch_duration, batch_ret = next(train_batches)
        opt.zero_grad()
        loss, prob = model(batch_X_float, batch_X_embed, batch_label, batch_ret)
        loss.backward()
        #clip_grad_norm(model.parameters(), 1)
        opt.step()
        step += 1
        if step >= num_batch:
            epoch += 1
            step = 0
            if epoch % rpt_epoch == 0:
                auc, acc = get_clf_stats(prob, batch_label)
                if verbose:
                    print('Train: epoch: %d, avg loss: %.3f, auc: %.3f, acc: %.3f' % (epoch, loss.data[0], auc, acc))
            valid_X_float, valid_X_embed, valid_label, valid_duration, valid_ret = next(valid_batches)
            _, prob = model(valid_X_float, valid_X_embed, valid_label, valid_ret)
            valid_avg_ret, valid_med_ret, valid_std_ret = clf_strtgy(prob[:, 1], valid_ret)
            test_X_float, test_X_embed, test_label, test_duration, test_ret = next(test_batches)
            _, prob = model(test_X_float, test_X_embed, test_label, test_ret)
            test_avg_ret, test_med_ret, test_std_ret = clf_strtgy(prob[:, 1], test_ret)
            if valid_avg_ret > best_val_ret:
                best_epoch = epoch
                best_val_ret = valid_avg_ret
                best_tst_ret = test_avg_ret
                best_med_ret = test_med_ret
                best_std_ret = test_std_ret
                model_name = './models/clf_ce_' + str(hidden_dim) + 'dim_model.pt'
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
