import os
from torch._C import dtype
from torch.utils import data
import yaml
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd 
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
# from metrics_eval import mrr
import matplotlib.pyplot as plt
import sys

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F 
import torch.optim as optim
import dgl

from rate_dataset import rateDataset
from user_dataset import userDataset
from model import DyHetGNN
from utils import user_id_to_idx, rate_id_to_idx, set_seed
from anomaly_generation_test import main_test
from anomaly_generation_all import main_all

class Logger(object):
   
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass


def nowdt():
    """
    get string representation of date and time of now()
    """
    from datetime import datetime

    now = datetime.now()
    return now.strftime("%d/%m/%Y %H:%M:%S")

def cprint(words: str):
    print(f"\033[0;30;43m{words}\033[0m")

def train(model, optimizer, cfg):
    # data: userID(MyID), userID(otherID), flag, time, rating(1, 0), labels(1, 0)
    print('Train......')
    if cfg['if_injected_train']:
        print('All Data Injected Anomaly ......')
        data = np.loadtxt(f'digg_{cfg["id"]}/injected_all_10%.csv', dtype=np.int64)
        knot1_user = int(data.shape[0] * cfg['train_ratio'])
        knot2_user = int(data.shape[0] * (cfg['train_ratio'] + cfg['val_ratio']))
        user_train_idx = torch.tensor(range(0, knot1_user))
        data = torch.tensor(data).to(cfg['device'])
        user_train_data = data[user_train_idx]
        user_val_idx = torch.tensor(range(knot1_user, knot2_user))
        user_val_data = data[user_val_idx]
        user_test_idx = torch.tensor(range(knot2_user, data.shape[0]))
        user_test_data = data[user_test_idx]
    else:
        print('Test Data Injected Anomaly ......')
        # user_test_data = np.loadtxt('digg_1/injected_test.csv', dtype=np.int64)
        # user_test_data = torch.tensor(user_test_data).to(device)
        user_train_data = data[user_train_idx]
        valAndtest = np.loadtxt(f'digg_{cfg["id"]}/injected_test_10%.csv', dtype=np.int64)
        valAndtest = torch.tensor(valAndtest).to(cfg['device'])
        user_val_idx = torch.tensor(range(0, int(valAndtest.shape[0] * cfg['val_ratio'])))
        user_test_idx = torch.tensor(range(int(valAndtest.shape[0] * cfg['val_ratio']), valAndtest.shape[0]))
        user_val_data = valAndtest[user_val_idx]
        user_test_data = valAndtest[user_test_idx]
    

    # time_rate = torch.unsqueeze(torch.tensor(time_rate), 1)
    best_loss = 100
    best_poch = 0

    # rate_data_loader = DataLoader(rate_train_data, cfg['batch_size'])
    user_data_loader = DataLoader(user_train_data, cfg['batch_size'])
    model.train()
    for epoch in tqdm(range(cfg['epochs'])):
        hg = dgl.heterograph({
            ('userID', 'rates', 'objectID'): ([], []),     #rates <==> votes
            ('userID', 'trusts', 'userID'): ([], [])       #trusts <==> friends
        })
        hg.add_nodes(cfg['n_userID'], {'time': torch.zeros(
            cfg['n_userID'], 1).long()}, ntype='userID')
        hg.add_nodes(cfg['n_objectID'], {'time': torch.zeros(
            cfg['n_objectID'], 1).long()}, ntype='objectID')
        # hg.add_edges(inters_write[:, 0], inters_write[:, 1], etype='writes')
        # hg.add_edges(inters_rate[:, 0], inters_rate[:, 1], etype = 'rates')
        # hg.nodes['objectID'].data['time'][inters_rate[:, 1]] = time_rate
        hg = hg.to(cfg['device'])
        
        

        tqdm.write('[Epoch %03d]' % (epoch + 1))
        for i, interactions in enumerate(tqdm(user_data_loader)):
            loss, train_acc, train_labels, train_preds = model.loss(hg, interactions)
            # train_auc = roc_auc_score(train_labels, train_preds)
            # loss = F.cross_entropy(logits[train_idx], labels[train_idx])
            #writer.add_scalar('train_loss', loss, i+1)
            if i % 10 == 0:
                # tqdm.write('%04d-th batch loss: %f | train_acc: %f | train_auc: %f' %
                #            (i+1, loss, train_acc, train_auc))
                tqdm.write('%04d-th batch loss: %f | train_acc: %f' %
                           (i+1, loss, train_acc))
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip']) 
            optimizer.step()
            # break

        # continue # no validation for quick debug 
        model.eval()
        val_loss, val_acc, val_labels, val_preds = model.loss(hg, user_val_data)
        val_auc = roc_auc_score(val_labels, val_preds)
        val_pr_auc = average_precision_score(val_labels, val_preds)
        tqdm.write('val loss: %f | val_acc: %f | val_auc: %f | val_pr_auc: %f' % (val_loss, val_acc, val_auc, val_pr_auc))
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = (epoch+1)
            torch.save(model.state_dict(), cfg['save_dir'])

        if epoch + 1 - best_epoch > 2:
            break

    with torch.no_grad():
        model.eval()
        # load_model = model.load_state_dict(torch.load(cfg['save_dir']))
        model.load_state_dict(torch.load(cfg['save_dir']))
        test_loss, test_acc, test_labels, test_preds = model.loss(hg, user_test_data)
        test_auc = roc_auc_score(test_labels, test_preds)
        p, r, _ = precision_recall_curve(test_labels, test_preds)
        test_pr_auc = average_precision_score(test_labels, test_preds)
        test_auc = roc_auc_score(test_labels, test_preds)
        fpr, tpr, _ = roc_curve(test_labels, test_preds)
        # mrr_value = mrr(test_labels, test_preds)
        plt.figure()
        plt.plot(r, p)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.savefig(f'digg_{cfg["id"]}/pr_10%.png')

        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC Curve')
        plt.savefig(f'digg_{cfg["id"]}/roc_10%.png')
        cprint(
            f'The Test Loss {test_loss} | The Test Accuracy {test_acc} | The Test AUC {test_auc} | The Test Pr_AUC {test_pr_auc}')
        # cprint(f'The Test MRR {mrr_value}')

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='digg', type=str)
    parser.add_argument('--id', type=int, default=1)

    parser.add_argument('--ab', type=str, default='node')
    parser.add_argument('--node',  action='store_false')
    parser.add_argument('--relation',  action='store_false')
    parser.add_argument('--att',  action='store_false')
    parser.add_argument('--interval',  action='store_false')

    parser.add_argument('--cuda', default=None, type=int)
    parser.add_argument('--clip', type=float, default=None) 
    parser.add_argument('--decay_w', type=float, default=None)  
    args = parser.parse_args()
    cfg = yaml.load(open('config.yaml'), yaml.SafeLoader)
    cfg['save_dir'] = f'{args.dataset}_{args.id}/run1__{args.ab}_digg_10%.pt'
    cfg['id'] = args.id
    cfg['use_ntype'] = args.node

    cfg['use_etype'] = args.relation
    cfg['use_att'] = args.att
    cfg['use_time_interval'] = args.interval
    
    if args.clip is not None: 
        cfg['clip'] = args.clip
    
    if args.decay_w is not None:  
        cfg['decay_w'] = args.decay_w
    
    log_file_path = f'{args.dataset}_{args.id}/ablation_{args.ab}_log.txt'
    sys.stdout = Logger(log_file_path)

    print(nowdt())
    print(args)
    print(cfg)

    print('train_ratio:', cfg['train_ratio'])
    if args.cuda is not None:
        assert torch.cuda.is_available()
        device = torch.device('cuda:' + str(args.cuda))
        print('Using cuda:', args.cuda)
        cfg['device'] = device
    else:
        device = torch.device('cpu')
        cfg['device'] = device

    set_seed(cfg['seed'])

    # timestamp, userID(voterID), objectID(storyID)
    dataset_rate = rateDataset(cfg)
    # rateing(1, 0), timestamp, userID, userID(friendID)
    dataset_user = userDataset(cfg)
    print('The length of users:', len(dataset_user.friends)) # 93696
    print('The length of objects:', len(dataset_rate.votes))  # 1110

    inters_user, user_userToidx = user_id_to_idx(dataset_user.friends[:, [2, 3]])
    inters_user = inters_user.to(device)
    inters_rate, rate_userToidx, rate_objectToidx = rate_id_to_idx(dataset_rate.votes[:, [1, 2]], user_userToidx)
    inters_rate = inters_rate.to(device)
    
    cfg['n_userID'] = np.max(list(rate_userToidx.values())) + 1  
    cfg['n_objectID'] = np.max(list(rate_objectToidx.values())) + 1  
    # print('user:', inters_user[:, :])
    # print('write:', write_inters[:, :])
    # print('rate:', rate_inters[:, :])
    print('user:', cfg['n_userID'])
    print('object:', cfg['n_objectID'])
    # print(rate_inters.shape[0])
    # print(rate_inters[:, 1])
    # quit()


    # print('Deal with time...')
    # if os.path.exists('digg_1/t_rate.csv'):
    #     time_rate = np.loadtxt('digg_1/t_rate.csv', dtype=np.int64)
    #     print('Load t_rate.csv ...')
    # else:
    #     time_rate_before = dataset_rate.votes[:, 0]
    #     time_rate_after = np.array(
    #         [dataset_rate.votes[:, 0][0]]).repeat(len(time_rate_before))
    #     time_rate = time_rate_before - time_rate_after
    #     np.savetxt('digg_1/t_rate.csv', time_rate, fmt='%d')
    

    # if os.path.exists('digg_1/t_user.csv'):
    #     time_user = np.loadtxt('digg_1/t_user.csv', dtype=np.int64)
    #     print('Load t_user.csv ...')
    # else:
    #     time_before = dataset_user.friends[:, 1]
    #     time_after = np.array(
    #             [dataset_user.friends[:, 1][0]]).repeat(len(time_before))
    #     time_user = time_before - time_after
    #     np.savetxt('digg_1/t_user.csv', time_user, fmt='%d')

    # print('Done!!!')

    # time_user = torch.unsqueeze(torch.tensor(dataset_user.friends[:, 1]), 1).to(device)
    # rating_user = torch.unsqueeze(torch.tensor(dataset_user.friends[:, 0].astype(np.int32)), 1).to(device)
    # labels_user = torch.unsqueeze(torch.tensor([1]).repeat(time_user.shape[0]), 1).to(device)
    # flag_user = torch.tensor([[1]]).repeat(time_user.shape[0], 1).to(device)
    # # userID(MyID), userID(otherID), flag(1), time, rating(1, 0), labels(1, 0)
    # data_user = torch.cat((inters_user, flag_user, time_user, rating_user, labels_user), dim=1)
    # # print(data_user[0:10, :])
    # # quit()
    # time_rate = torch.unsqueeze(torch.tensor(dataset_rate.votes[:, 0]), 1).to(device)
    # rating_rate = torch.tensor([[1]]).repeat(time_rate.shape[0], 1).to(device)
    # labels_rate = torch.tensor([[1]]).repeat(time_rate.shape[0], 1).to(device)
    # flag_rate = torch.tensor([[2]]).repeat(time_rate.shape[0], 1).to(device)
    # data_rate = torch.cat((inters_rate, flag_rate, time_rate,rating_rate, labels_rate), dim=1)

    # all_data = torch.cat((data_user, data_rate), 0)
    # all_data = torch.stack(sorted(all_data, key=lambda line: line[3]))
    # all_data[:, 3] = all_data[:, 3] - all_data[0, 3]  # make timestamp time_interval
    # if cfg['if_injected_train']:
    #     print('All Data Anomaly generation......')
    #     main_all(all_data, cfg)
    # else:
    #     print('Test Data Anomaly generation......')
    #     main_test(all_data, cfg)
    
    g = dgl.heterograph({
        ('userID', 'rates', 'objectID'): ([], []), 
        ('userID', 'trusts', 'userID'): ([], [])
    })
    g = g.to(device)
    model = DyHetGNN(g, cfg).to(device)
    # print('model:', next(model.parameters()).is_cuda)
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    # train(all_data, model, optimizer, cfg)
    # with torch.autograd.set_detect_anomaly(True): 
    train(model, optimizer, cfg)
