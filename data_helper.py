# -*- coding: utf-8 -*-
import pickle
import random
import math
import numpy as np
import torch
import config

with open(config.param.dep_path, 'rb') as f:
    dep_dict = pickle.load(f)
with open(config.param.pos_path, 'rb') as f:
    pos_dict = pickle.load(f)


class BatchManager(object):
    def __init__(self, data, batch_size):
        self.batch_size = batch_size
        self.batch_data = self.gen_batch(data)
        self.len_data = len(self.batch_data)
        self.padding_length = 0

    def gen_batch(self, data):
        """
        构造batch数据
        :param data:
        :param mode:
        :return: batch_token_ids(字的id), batch_bio_id(BIO标签的id), batch_labels(要素类型的id)
        """
        num_batch = int(math.ceil(len(data) / self.batch_size))  # 总共有多少个batch
        batch_data = list()
        for i in range(num_batch):
            batch_data.append(
                self.pad_data(data[i * int(self.batch_size): (i + 1) * int(self.batch_size)]))
        return batch_data

    def pad_data(self, data):
        """
        :param data:
        :return: batch_token_ids(id of split word), batch_bio_id(id of ner bio label), batch_labels(id of element)
        """
        batch_token_ids, batch_pos_ids, batch_dep_graph, batch_table_labels = [], [], [], []
        for (text, event_list) in data:
            char_list = ['[CLS]']+[char for char in text[:config.param.max_seq_len]]
            token_ids = config.param.tokenizer.convert_tokens_to_ids(char_list)
            pos_ids = [config.pos2id[p] for p in pos_dict[text]]
            table_a_s = np.zeros([len(token_ids) - 1, len(token_ids) - 1], dtype=int)
            table_a_e = np.zeros([len(token_ids) - 1, len(token_ids) - 1], dtype=int)
            for event in event_list:
                event_type = event['event_type']
                event_start_pos = event['trigger_start_index']
                if event_start_pos + len(event['trigger']) < config.param.max_seq_len:
                    table_a_s[event_start_pos][event_start_pos] = config.event2id[f'B-{event_type}']
                    table_a_e[event_start_pos][event_start_pos] = config.event2id[f'B-{event_type}']
                    for pos in range(event_start_pos + 1, event_start_pos + len(event['trigger'])):
                        table_a_s[pos][pos] = config.event2id[f'I-{event_type}']
                        table_a_e[pos][pos] = config.event2id[f'I-{event_type}']
                    for argument in event['arguments']:
                        argument_role = argument['role']
                        arg_start_pos = argument['argument_start_index']
                        if arg_start_pos + len(argument['argument']) < config.param.max_seq_len:
                            for pos_i in range(event_start_pos, event_start_pos + len(event['trigger'])):
                                table_a_s[arg_start_pos][pos_i] = config.role2id[f'{argument_role}']
                                table_a_e[arg_start_pos + len(argument['argument']) - 1][pos_i] = config.role2id[f'{argument_role}']
                                table_a_s[pos_i][arg_start_pos] = config.role2id[f'{argument_role}']
                                table_a_e[pos_i][arg_start_pos + len(argument['argument']) - 1] = config.role2id[f'{argument_role}']
            batch_token_ids.append(token_ids)
            batch_pos_ids.append(pos_ids)
            batch_dep_graph.append(dep_dict[text])
            batch_table_labels.append((table_a_s, table_a_e))
        batch_token_ids = self.sequence_padding(batch_token_ids, 0, isToken=True)
        batch_pos_ids = self.sequence_padding(batch_pos_ids, 0)
        batch_token_ids = torch.tensor(batch_token_ids, dtype=torch.long).to(config.param.device)
        batch_pos_ids = torch.tensor(batch_pos_ids, dtype=torch.long).to(config.param.device)
        return [batch_token_ids, batch_pos_ids, batch_dep_graph, batch_table_labels]


    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]

    def sequence_padding(self, inputs, padding_num, isToken=False):
        """将序列padding到同一长度
        """
        res = []
        if isToken: 
            self.padding_length = max([len(x) for x in inputs])
            for x in inputs:
                x = x + [padding_num] * (self.padding_length - len(x))
                res.append(x)
        else:
            for x in inputs:
                if len(x) >= self.padding_length:
                    x = x[:self.padding_length-1]
                else:
                    x = x + [padding_num] * (self.padding_length-1 - len(x))
                res.append(x)
        return res
