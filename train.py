import logging
import json
import operator
import random
import torch
import numpy as np
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import config
from data_helper import BatchManager
from ee_model import EEModel
from utils import get_decode, set_logger, calc_metric


def isRolesOverlap(data):
    """
    判断一条数据是否存在论元角色重叠的情况
    :param data:
    :return:
    """
    if len(data['event_list']) < 2:
        return False
    arg_dict = {}
    arg_cnt = 0
    tri_dict = []
    for event in data['event_list']:
        tri_dict.append(event['trigger'] + str(event['trigger_start_index']))
        for arg in event['arguments']:
            key = arg['argument'] + str(arg['argument_start_index'])
            if key not in arg_dict:
                arg_cnt += 1
                arg_dict[key] = arg['role']
            elif arg_dict[key] != arg['role']:
                return True
            else:
                arg_cnt += 1
    if len(arg_dict) != arg_cnt:
        if len(set(tri_dict)) == 1:
            return False
        else:
            arg_dict = sorted(arg_dict.items(), key=operator.itemgetter(0))
            for event in data['event_list']:
                temp_dict = {}
                for arg in event['arguments']:
                    temp_dict[arg['argument'] + str(arg['argument_start_index'])] = arg['role']
                temp_dict = sorted(temp_dict.items(), key=operator.itemgetter(0))
                if temp_dict != arg_dict:
                    tri_temp = []
                    for temp in data['event_list']:
                        tri_temp.append(temp['trigger'])
                    if len(set(tri_temp)) != len(tri_temp):
                        return False
                    return True
            return False
    return False

def load_data(filename):
    D = []
    with open(filename, "r") as f:
        datas = json.load(f)
    for data in datas:
        # if not isRolesOverlap(data):
        #     continue
        D.append((data['text'], data['event_list']))
    return D

def build_optimizer_and_scheduler(model, total_steps):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if n.startswith('bert') and not any(nd in n for nd in no_decay)],
         'lr': config.param.bert_learning_rate, 'weight_decay': config.param.bert_weight_decay},
        {'params': [p for n, p in model.named_parameters() if not n.startswith('bert') and not any(nd in n for nd in no_decay)],
         'lr': config.param.learning_rate, 'weight_decay': config.param.weight_decay},
        {'params': [p for n, p in model.named_parameters() if n.startswith('bert') and any(nd in n for nd in no_decay)],
         'lr': config.param.bert_learning_rate, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if not n.startswith('bert') and any(nd in n for nd in no_decay)],
         'lr': config.param.learning_rate, 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters)
    warmup_steps = int(total_steps * config.param.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    return optimizer, scheduler

# 一个epoch的训练
def train_epoch(model, train_manager, optimizer, scheduler):
    total_loss = []
    model.train()
    with tqdm(total=train_manager.len_data, desc='train batch') as pbar:
        for (batch_token_ids, batch_pos_ids, batch_dep_graph, batch_table_labels) in train_manager.iter_batch(shuffle=True):
            loss, _ = model(batch_token_ids, batch_pos_ids, batch_dep_graph, batch_table_labels)
            loss.backward()
            total_loss.append(loss.item())
            nn.utils.clip_grad_norm_(model.parameters(), config.param.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            pbar.update(1)
    train_loss = sum(total_loss) / len(total_loss)
    return train_loss

def dev_epoch(model, dev_manager):
    model.eval()
    results_all, labels_all, total_loss = [], [], []
    with tqdm(total=dev_manager.len_data, desc='dev batch') as pbar:
        for (batch_token_ids, batch_pos_ids, batch_dep_graph, batch_table_labels) in dev_manager.iter_batch(shuffle=False):
            with torch.no_grad():
                loss, results = model(batch_token_ids, batch_pos_ids, batch_dep_graph, batch_table_labels)
            total_loss.append(loss.item())
            for i in range(len(batch_table_labels)):
                results_all.append(get_decode(results[i]))
                labels_all.append(get_decode(batch_table_labels[i]))
            pbar.update(1)
    dev_loss = sum(total_loss) / len(total_loss)
    p, r, f = calc_metric(results_all, labels_all)
    return dev_loss, p, r, f

if __name__ == '__main__':
    set_logger(config.param.log_dir)
    random.seed(config.param.seed)
    np.random.seed(config.param.seed)
    torch.manual_seed(config.param.seed)

    # 读取数据
    train_data = load_data(config.param.train_path)
    dev_data = load_data(config.param.dev_path)

    train_manager = BatchManager(train_data, batch_size=config.param.train_batch_size)
    dev_manager = BatchManager(dev_data, batch_size=config.param.dev_batch_size)

    model = EEModel()
    model.to(config.param.device)

    total_steps = train_manager.len_data * config.param.num_epoch
    optimizer, scheduler = build_optimizer_and_scheduler(model, total_steps)

    trigger_classfication_best, role_classfication_best = -1, -1
    patience_counter = 0
    for epoch in range(0, config.param.num_epoch):
        logging.info('=============================== epoch:{} ==============================='.format(epoch + 1))
        train_loss = train_epoch(model, train_manager, optimizer, scheduler)
        dev_loss, p, r, f = dev_epoch(model, dev_manager)
        logging.info('train loss：{}  dev loss：{}'.format(train_loss, dev_loss))
        logging.info('edc_p：{}  edc_r：{}  edc_f：{}'.format(p[1], r[1], f[1]))
        logging.info('eaec_p：{}  eaec_r：{}  eaec_f：{}'.format(p[3], r[3], f[3]))
        patience_counter += 1
        if f[1] - trigger_classfication_best > config.param.patience:
            patience_counter = 0
            trigger_classfication_best = f[1]
            logging.info("Save trigger_classfication_best! at epoch：" + str(epoch + 1))
            torch.save(model, config.param.model_save_path + '/trigger_classfication_best.model')

        if f[3] - role_classfication_best > 1e-5:
            patience_counter = 0
            role_classfication_best = f[3]
            logging.info("Save role_classfication_best! at epoch：" + str(epoch + 1))
            torch.save(model, config.param.model_save_path + '/role_classfication_best.model')

        if patience_counter > config.param.patience_num:
            break

