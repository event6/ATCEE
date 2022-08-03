import transformers
import config
import json
from ltp import LTP
from tqdm import tqdm

import pickle
ltp = LTP(device='cpu')

def get_word_mapping(seg, text):
    token = [w for w in text]
    def search(token, word, start_index):
        index = []
        w_i = 0
        for i in range(len(token)):
            if w_i >= len(word):
                break
            if token[i] == word[w_i]:
                index.append(i + start_index)
                w_i += 1
        if len(index) > 1:
            assert all(y-x==1 for x, y in zip(index, index[1:])), (text)
        return index
    # seg词到token的映射
    word_mapping = {}
    start_index = 0
    for i in range(len(seg)):
        index = search(token[start_index:], [w for w in seg[i]], start_index)
        # 找到了对应token，则加入字典
        if len(index) > 0:
            start_index = index[-1] + 1
            word_mapping[i + 1] = index
    return word_mapping

def get_dep(text):
    text = text[:config.param.max_seq_len]
    text = text.replace(' ', '-').replace('\n', ',').replace('\u3000', '-').replace('\xa0', ',').replace('\ue627', ',')
    seg, hidden = ltp.seg([text])
    word_mapping = get_word_mapping(seg[0], text)
    assert word_mapping[len(word_mapping)][-1] == len(text.rstrip())-1, (text)
    dep = ltp.dep(hidden)
    dep_list = []
    for tup in dep[0]:
        if tup[0] in word_mapping.keys() and tup[1] in word_mapping.keys():
            dep_list.append((word_mapping[tup[0]], word_mapping[tup[1]], tup[2]))
    return dep_list

def get_pos(text):
    text = text[:config.param.max_seq_len]
    text = text.replace(' ', '-').replace('\n', ',').replace('\u3000', '-').replace('\xa0', ',').replace('\ue627', ',')
    seg, hidden = ltp.seg([text])
    token = seg[0]
    word_mapping = get_word_mapping(token, text)
    pos = ltp.pos(hidden)[0]
    assert word_mapping[len(word_mapping)][-1] == len(text) - 1, (text)

    pos_list = []
    for i in range(len(pos)):
        pos_list.extend(['B-' + pos[i]] + ['I-' + pos[i]] * (len(word_mapping[i + 1]) - 1))
    return pos_list

def gen_pkl():
    """
    产生pos.pkl和dep.pkl文件
    pos.pkl中存储的是每个事件文本对应的词性标注结果
    dep.pkl中存储的是每个事件文本对应的依存句法分析结果
    :return:
    """
    dep_out = {}
    pos_out = {}
    for data_path in [config.param.train_path, config.param.dev_path, config.param.test_path]:
        with open(data_path) as f:
            datas = json.load(f)
            for data in tqdm(datas):
                dep_out[data['text']] = get_dep(data['text'])
                pos_out[data['text']] = get_pos(data['text'])
    with open(config.param.dep_path, 'wb') as handle:
        pickle.dump(dep_out, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(config.param.pos_path, 'wb') as handle:
        pickle.dump(pos_out, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # gen_pkl()

    text = "华特股份董事长石平湘先生荣膺“FIT粤”科创先锋大赛领军人物奖"
    print(get_dep(text))
    print(get_pos(text))



