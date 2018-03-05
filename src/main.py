import numpy as np
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
from utils import load_data, batch_iter, set_cuda, detach_cuda, get_stats, ret_strtgy
import time

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
        #self.log_prob = nn.LogSoftmax()
        #self.linear4_2 = nn.Linear(sub_hidden_dim, num_class)
        #self.lmbda = lmbda
        self.rgrs_loss = nn.L1Loss()
        #self.clas_loss = nn.CrossEntropyLoss()
    
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
        #diff = torch.mean(ratio) - torch.mean(batch_ratio)
        #score = self.linear4_2(hid)
        #log_prob = self.log_prob(score)
        #clas_loss = self.clas_loss(log_prob, batch_label)
        return rgrs_loss, ret

def train(model, train_batches, test_batches, opt, num_epochs, verbose = True):
    epoch = 0
    step = 0
    best_epoch = 0
    best_ret = 0
    best_med_ret = 0
    best_std_ret = 0
    rpt_step = 10 * num_batch
    test_step = 20 * num_batch
    while epoch < num_epochs:
        batch_X_float, batch_X_embed, batch_label, batch_duration, batch_ret = next(train_batches)
        opt.zero_grad()
        loss, pred_ret = model(batch_X_float, batch_X_embed, batch_label, batch_ret)
        loss.backward()
        #clip_grad_norm(model.parameters(), 1)
        opt.step()
        step += 1
        if step >= num_batch:
            epoch += 1
            step = 0
        if (step + epoch * num_batch) % rpt_step == 0:
            med_diff, avg_diff, max_diff = get_stats(pred_ret, batch_ret)
            if verbose:
                print('Train: step: %d, avg loss: %.3f, median diff: %.3f, mean: %.3f, max: %.3f' % ((step + epoch * num_batch), loss.data[0], med_diff, avg_diff, max_diff))
        if (step + epoch * num_batch) % test_step == 0:
            test_X_float, test_X_embed, test_label, test_duration, test_ret = next(test_batches)
            _, ret = model(test_X_float, test_X_embed, test_label, test_ret)
            avg_ret, med_ret, std_ret = ret_strtgy(ret, test_ret)
            if verbose:
                print('Test step: %d, avg return: %.3f, median: %.3f, std: %.3f' % ((step + epoch * num_batch), avg_ret, med_ret, std_ret))
            if avg_ret > best_ret:
                best_ret = avg_ret
                best_med_ret = med_ret
                best_std_ret = std_ret
                torch.save(model.state_dict(), 'model.pt')
    return best_epoch, best_ret, best_med_ret, best_std_ret


train_data, test_data, embed_dict, embed_dims, embed_keys, float_keys = load_data('../data/data.npz')

embed_sizes = {'issue_month': 4, 'home_ownership': 2, 'verification_status': 2, 'emp_length': 4, 
               'initial_list_status': 1, 'addr_state': 10, 'early_month': 4, 'grade': 3, 
               'purpose': 5, 'sub_grade': 10, 'zip_code': 20, 'early_year': 6, 'term': 1, 'issue_year': 6}
num_samples = len(train_data)
batch_size = 1000
num_batch = int(num_samples / batch_size)
hidden_dim = 30
num_epochs = 50
learning_rate = 1e-3
num_class = len(set([train_data[t]['label'] for t in train_data]))

for hidden_dim in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    train_batches = batch_iter(train_data, batch_size, embed_keys, float_keys, shuffle = True)
    test_batches = batch_iter(test_data, len(test_data), embed_keys, float_keys, shuffle = False)
    
    model = MultiNet(len(float_keys), embed_dims, embed_sizes, hidden_dim, embed_keys, num_class)
    model = set_cuda(model)
    opt = optim.Adam(model.parameters(), lr = learning_rate)
    
    start = time.time()
    best_epoch, best_ret, best_med_ret, best_std_ret = train(model, train_batches, test_batches, opt, num_epochs)
    
    print('Best expected return: %.3f, median: %.3f, std: %.3f in epoch %d. Finished in %.3f seconds' % (best_ret, best_med_ret, best_std_ret, best_epoch, time.time() - start))

model.load_state_dict(torch.load('model.pt'))
test_X_float, test_X_embed, test_label, test_duration, test_ret = next(test_batches)
_, pred_ret = model(test_X_float, test_X_embed, test_label, test_ret)
avg_ret, med_ret, std_ret = ret_strtgy(pred_ret, test_ret)
print('Avg return: %.3f, median: %.3f, std: %.3f' % (avg_ret, med_ret, std_ret))