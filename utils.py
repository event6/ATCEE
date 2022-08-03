# -*- coding: utf-8 -*-
import logging
import ujson as json

def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

def get_decode(table):
    """
    获取触发词抽取三元组 （触发词开始位置，触发词结束位置，事件类型）
    和论元抽取六元组  （触发词开始位置，触发词结束位置，事件类型，论元开始位置，触发词结束位置，论元角色类型）
    :param table:
    :return:
    """
    n = table[0].shape[0]
    i = 0
    trigger_list = []
    while i < n:
        if table[0][i][i] % 2 == 1:
            trigger_class = table[0][i][i]
            trigger_start_index = i
            while i + 1 < n and table[0][i + 1][i + 1] == trigger_class + 1:
                i += 1
            trigger_end_index = i
            trigger_list.append((trigger_start_index, trigger_end_index, trigger_class))
        i += 1
    role_list = []

    for i in range(len(trigger_list)):
        trigger_start_index, trigger_end_index, trigger_class = trigger_list[i]
        start_role, end_role = [], [] # 保存候选论元的开始位置或结束位置
        # 遍历两个表格中触发词所在的行序列
        for ii in range(trigger_start_index):
            start_temp, end_temp = [], [] # 保存触发词所在的行上某一列论元的分类结果
            for jj in range(trigger_start_index, trigger_end_index + 1):
                start_temp.append(table[0][ii][jj])
                end_temp.append(table[1][ii][jj])

            # 如果某一列取值相同且不为0, 则为候选论元的开始位置或结束位置,取值即为论元的分类类别
            if len(set(start_temp)) == 1 and start_temp[0] != 0:
                start_role.append(start_temp[0])
            else:
                start_role.append(0)
            if len(set(end_temp)) == 1 and end_temp[0] != 0:
                end_role.append(end_temp[0])
            else:
                end_role.append(0)

        for ii in range(trigger_start_index, trigger_end_index + 1):
            start_role.append(0)
            end_role.append(0)

        for ii in range(trigger_end_index + 1, n):
            start_temp, end_temp = [], []  # 保存触发词所在的行上某一列论元的分类结果
            for jj in range(trigger_start_index, trigger_end_index + 1):
                start_temp.append(table[0][ii][jj])
                end_temp.append(table[1][ii][jj])

            # 如果某一列取值相同且不为0, 则为候选论元的开始位置或结束位置,取值即为论元的分类类别
            if len(set(start_temp)) == 1 and start_temp[0] != 0:
                start_role.append(start_temp[0])
            else:
                start_role.append(0)
            if len(set(end_temp)) == 1 and end_temp[0] != 0:
                end_role.append(end_temp[0])
            else:
                end_role.append(0)

        k = 0
        while k < n:
            if start_role[k] != 0:
                role_class = start_role[k]
                for j in range(k, n):
                    if end_role[j] == role_class:
                        role_start_index = k
                        role_end_index = j
                        role_list.append((trigger_start_index, trigger_end_index, trigger_class, role_start_index, role_end_index, role_class))
                        break
            k += 1
    return trigger_list, role_list

def get_decode_1(table_tri, table_arg):
    """
    获取事件检测三元组 （触发词开始位置，触发词结束位置，事件类型）
    和论元抽取六元组  （触发词开始位置，触发词结束位置，事件类型，论元开始位置，触发词结束位置，论元角色类型）
    :param table_tri: trigger_classfication_best.model 预测结果
    :param table_arg: role_classfication_best.model 预测结果
    :return:
    """
    n = table_tri[0].shape[0]
    i = 0
    trigger_list = []
    while i < n:
        if table_tri[0][i][i] % 2 == 1:
            trigger_class = table_tri[0][i][i]
            trigger_start_index = i
            while i + 1 < n and table_tri[0][i + 1][i + 1] == trigger_class + 1:
                i += 1
            trigger_end_index = i
            trigger_list.append((trigger_start_index, trigger_end_index, trigger_class))
        i += 1
    role_list = []

    for i in range(len(trigger_list)):
        trigger_start_index, trigger_end_index, trigger_class = trigger_list[i]
        start_role, end_role = [], []  # 保存候选论元的开始位置或结束位置
        # 遍历两个表格中触发词所在的行序列
        for ii in range(trigger_start_index):
            start_temp, end_temp = [], []  # 保存触发词所在的行上某一列论元的分类结果
            for jj in range(trigger_start_index, trigger_end_index + 1):
                start_temp.append(table_arg[0][ii][jj])
                end_temp.append(table_arg[1][ii][jj])

            # 如果某一列取值相同且不为0, 则为候选论元的开始位置或结束位置,取值即为论元的分类类别
            if len(set(start_temp)) == 1 and start_temp[0] != 0:
                start_role.append(start_temp[0])
            else:
                start_role.append(0)
            if len(set(end_temp)) == 1 and end_temp[0] != 0:
                end_role.append(end_temp[0])
            else:
                end_role.append(0)

        for ii in range(trigger_start_index, trigger_end_index + 1):
            start_role.append(0)
            end_role.append(0)

        for ii in range(trigger_end_index + 1, n):
            start_temp, end_temp = [], []  # 保存触发词所在的行上某一列论元的分类结果
            for jj in range(trigger_start_index, trigger_end_index + 1):
                start_temp.append(table_arg[0][ii][jj])
                end_temp.append(table_arg[1][ii][jj])

            # 如果某一列取值相同且不为0, 则为候选论元的开始位置或结束位置,取值即为论元的分类类别
            if len(set(start_temp)) == 1 and start_temp[0] != 0:
                start_role.append(start_temp[0])
            else:
                start_role.append(0)
            if len(set(end_temp)) == 1 and end_temp[0] != 0:
                end_role.append(end_temp[0])
            else:
                end_role.append(0)

        k = 0
        while k < n:
            if start_role[k] != 0:
                role_class = start_role[k]
                for j in range(k, n):
                    if end_role[j] == role_class:
                        role_start_index = k
                        role_end_index = j
                        role_list.append((trigger_start_index, trigger_end_index, trigger_class, role_start_index,
                                          role_end_index, role_class))
                        break
            k += 1
    return trigger_list, role_list

def calc_metric(results_all, labels_all):
    """

    :param results_all:
    :param labels_all:
    :return: 返回触发词、论元的识别和分类的p,r,f值
    """
    edid_correct, ed_correct, ed_predict, ed_label = 0, 0, 0, 0
    eaeid_correct, eae_correct, eae_predict, eae_label = 0, 0, 0, 0

    for i in range(len(results_all)):
        result, label = results_all[i], labels_all[i]

        # 触发词分类
        result_ed, label_ed = result[0], label[0]
        for res in result_ed:
            if res in label_ed:
                ed_correct += 1
        ed_predict += len(result_ed)
        ed_label += len(label_ed)

        # 触发词识别
        result_edid, label_edid = [], []
        for j in range(len(result_ed)):
            result_edid.append((result_ed[j][0], result_ed[j][1]))
        for j in range(len(label_ed)):
            label_edid.append((label_ed[j][0], label_ed[j][1]))
        for res in result_edid:
            if res in label_edid:
                edid_correct += 1

        # 论元分类
        result_eae, label_eae = result[1], label[1]
        for res in result_eae:
            if res in label_eae:
                eae_correct += 1
        eae_predict += len(result_eae)
        eae_label += len(label_eae)

        # 论元识别
        result_eaeid, label_eaeid = [], []
        for j in range(len(result_eae)):
            result_eaeid.append(
                (result_eae[j][0], result_eae[j][1], result_eae[j][2], result_eae[j][3], result_eae[j][4]))
        for j in range(len(label_eae)):
            label_eaeid.append((label_eae[j][0], label_eae[j][1], label_eae[j][2], label_eae[j][3], label_eae[j][4]))
        for res in result_eaeid:
            if res in label_eaeid:
                eaeid_correct += 1

    f = [0.0] * 4
    p = [edid_correct / ed_predict if ed_predict != 0 else 1, ed_correct / ed_predict if ed_predict != 0 else 1,
         eaeid_correct / eae_predict if eae_predict != 0 else 1, eae_correct / eae_predict if eae_predict != 0 else 1]
    r = [edid_correct / ed_label, ed_correct / ed_label, eaeid_correct / eae_label, eae_correct / eae_label]
    for i in range(4):
        f[i] = 2 * p[i] * r[i] / (p[i] + r[i]) if p[i] + r[i] != 0 else 0
    return p, r, f




