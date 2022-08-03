# -*- coding: utf-8 -*-
import random
import numpy as np
import torch
from tqdm import tqdm
import config
from data_helper import BatchManager
from train import load_data
from utils import get_decode_1, set_logger, calc_metric, get_decode

if __name__ == '__main__':
    set_logger(config.param.log_dir)
    random.seed(config.param.seed)
    np.random.seed(config.param.seed)
    torch.manual_seed(config.param.seed)

    # 读取数据
    test_data = load_data(config.param.test_path)
    test_manager = BatchManager(test_data, batch_size=config.param.test_batch_size)

    # 加载模型
    model_t = torch.load(config.param.model_save_path + '/trigger_classfication_best.model')
    model_r = torch.load(config.param.model_save_path + '/role_classfication_best.model')
    model_t.eval()
    model_r.eval()

    results_all, labels_all = [], []
    with tqdm(total=test_manager.len_data, desc='test batch') as pbar:
        for (batch_token_ids, batch_pos_ids, batch_dep_graph, batch_table_labels) in test_manager.iter_batch():
            with torch.no_grad():
                _, results_tri = model_t(batch_token_ids, batch_pos_ids, batch_dep_graph, batch_table_labels)
                _, results_arg = model_r(batch_token_ids, batch_pos_ids, batch_dep_graph, batch_table_labels)

            for i in range(len(batch_table_labels)):
                results_all.append(get_decode_1(results_tri[i], results_arg[i]))
                # results_all.append(get_decode(results_arg[i]))
                labels_all.append(get_decode(batch_table_labels[i]))
            pbar.update(1)

    p, r, f = calc_metric(results_all, labels_all)
    print("trigger classfication f: {} p: {} r {}".format(f[1], p[1], r[1]))
    print("role classfication f: {} p: {} r {}".format(f[3], p[3], r[3]))