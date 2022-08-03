# -*- coding: gbk -*-
import json
import os
import torch
import transformers


class param():
    base_path = os.path.abspath('.')
    train_path = os.path.join(base_path, 'data/DuEE/train.json')
    dev_path = os.path.join(base_path, 'data/DuEE/dev.json')
    test_path = os.path.join(base_path, 'data/DuEE/test.json')
    schema_path = os.path.join(base_path, 'data/DuEE/event_schema.json')
    dep_path = os.path.join(base_path, 'data/DuEE/dep.pkl')
    pos_path = os.path.join(base_path, 'data/DuEE/pos.pkl')
    model_save_path = os.path.join(base_path, 'checkpoint')
    log_dir = os.path.join(base_path, 'checkpoint/train.log')
    pre_model = "ERNIE_pretrain"
    tokenizer = transformers.BertTokenizer.from_pretrained(pre_model)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    seed = 104
    max_seq_len = 170
    train_batch_size = 6
    dev_batch_size = 8
    test_batch_size = 8
    num_epoch = 30
    patience = 1e-5
    patience_num = 10

    hidden_size = 800
    dropout = 0.05
    lstm_layers = 2

    gat_nhead = 4
    gat_nhidden = 200
    gat_layers = 3

    learning_rate = 1e-4
    weight_decay = 1e-3
    bert_learning_rate = 1e-5
    bert_weight_decay = 1e-5
    warmup_ratio = 0.1
    max_grad_norm = 1.0


# 词性标签
pos2id = {'B-nl': 1, 'B-a': 2, 'B-o': 3, 'B-nd': 4, 'I-p': 5, 'B-d': 6, 'B-q': 7, 'I-a': 8, 'B-wp': 9, 'I-q': 10,
'I-c': 11, 'B-m': 12, 'I-o': 13, 'B-v': 14, 'B-b': 15, 'B-nh': 16, 'B-j': 17, 'I-u': 18, 'I-j': 19,
'I-i': 20,'B-c': 21, 'B-p': 22, 'I-m': 23, 'B-ns': 24, 'I-ns': 25, 'B-i': 26, 'B-z': 27, 'I-v': 28, 'B-n': 29,
'B-nt': 30, 'B-r': 31, 'I-nz': 32, 'B-ni': 33, 'I-e': 34, 'B-ws': 35, 'I-wp': 36, 'I-nd': 37, 'I-z': 38,
'I-ni': 39, 'I-nl': 40, 'I-b': 41, 'B-h': 42, 'B-u': 43, 'B-k': 44, 'B-e': 45, 'I-n': 46, 'I-ws': 47,
'I-h': 48, 'I-nh': 49, 'I-d': 50, 'I-r': 51, 'B-nz': 52, 'I-nt': 53}

def load_schema():
    with open(param.schema_path) as f:
        event2id = {'O': 0}
        role2id = {'O': 0}
        for line in f:
            event = json.loads(line)
            if 'B-{}'.format(event['event_type']) not in event2id:
                event2id['B-{}'.format(event['event_type'])] = len(event2id)
                event2id['I-{}'.format(event['event_type'])] = len(event2id)
            for role in event["role_list"]:
                if role['role'] not in role2id:
                    role2id[role['role']] = len(role2id)
    return event2id, role2id
event2id, role2id = load_schema()


