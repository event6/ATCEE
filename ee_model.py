import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from transformers import BertModel
import config


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='none', ignore_index=-100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output, target):
        c = output.size()[-1]
        log_pred = torch.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_pred.sum()
        else:
            loss = -log_pred.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()

        return loss * self.eps / c + (1 - self.eps) * torch.nn.functional.nll_loss(log_pred, target,
                                                                                   reduction=self.reduction,
                                                                                   ignore_index=self.ignore_index)


class EEModel(nn.Module):
    def __init__(self):
        super(EEModel, self).__init__()
        self.bert_module = BertModel.from_pretrained(config.param.pre_model, hidden_dropout_prob=0.1)
        self.pos_embedding = nn.Embedding(len(config.pos2id)+1, 32)

        self.lstm_encoder = nn.LSTM(input_size=config.param.hidden_size,
                                    hidden_size=config.param.hidden_size,
                                    num_layers=config.param.lstm_layers,
                                    batch_first=True,
                                    dropout=config.param.dropout,
                                    bidirectional=True)
        self.lstm_fc = nn.Linear(2 * config.param.hidden_size, config.param.hidden_size)

        self.gat = GATConv(config.param.hidden_size, config.param.gat_nhidden, config.param.gat_nhead)

        self.table_fc = nn.Linear(2 * config.param.hidden_size, config.param.hidden_size)

        self.trigger_fc = nn.Linear(config.param.hidden_size, len(config.event2id))
        self.role_start_fc = nn.Linear(config.param.hidden_size, len(config.role2id))
        self.role_end_fc = nn.Linear(config.param.hidden_size, len(config.role2id))

        self.loss_fct = LabelSmoothingCrossEntropy(reduction='none')
        self.relu = nn.ReLU()

        init_blocks = [self.pos_embedding, self.lstm_fc, self.table_fc, self.role_start_fc, self.role_start_fc, self.role_end_fc]
        self._init_weights(init_blocks, initializer_range=self.bert_module.config.initializer_range)

    def _init_weights(self, blocks, **kwargs):
        """
        参数初始化，将 Linear / Embedding与 Bert 进行一样的初始化
        """
        for block in blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0, std=kwargs.pop('initializer_range', 0.02))
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

    def get_gat_tensor(self, dep, enc):
        """
        获取句子依存的图表示，考虑词内部信息和词与词信息
        dep: [([0, 1], [2, 3], 'FEAT'), ([2, 3], [5, 6], 'PAT')]
        enc: 字的嵌入
        """
        if len(dep) == 0:
            return enc
        src = []
        target = []
        temp = list()
        for d in dep:
            # 词内部的关系
            if str(d[0]) not in temp:
                for i in range(len(d[0]) - 1, 0, -1):
                    if d[0][i] < config.param.max_seq_len:
                        src.append(d[0][i])
                        target.append(d[0][i - 1])
            temp.append(str(d[0]))
            if str(d[1]) not in temp:
                for i in range(len(d[1]) - 1, 0, -1):
                    if d[1][i] < config.param.max_seq_len:
                        src.append(d[1][i])
                        target.append(d[1][i - 1])
            temp.append(str(d[1]))
            # 词与词的关系
            if d[0][-1] < config.param.max_seq_len and d[1][-1] < config.param.max_seq_len:
                src.append(d[1][-1])
                target.append(d[0][-1])
        edge_index = torch.tensor([src, target], dtype=torch.long).to(enc.device)
        data = Data(x=enc, edge_index=edge_index)
        x, edge_index = data.x, data.edge_index
        for i in range(config.param.gat_layers):
            # 使用残差网络结构
            x = x + F.dropout(self.relu(self.gat(x, edge_index)), p=config.param.dropout, training=self.training)
        return x

    def forward(self, token_ids, pos_ids, dep_graphs, table_labels):
        # 1. 获取预训练字向量
        token_mask = token_ids.gt(0)
        outputs = self.bert_module(input_ids=token_ids, attention_mask=token_mask)[0]  # [batch_size, seq_len+1, 768]

        # 2. 融合词性标注向量
        pos_emb = self.pos_embedding(pos_ids) # [batch_size, seq_len, 32]
        pos_emb_0 = torch.zeros([pos_emb.shape[0], 1, pos_emb.shape[-1]],device=pos_ids.device)  # [batch_size, 1, 32]
        pos_emb = torch.cat((pos_emb_0, pos_emb), 1) # [batch_size, seq_len+1, 32]
        outputs = torch.cat((outputs, pos_emb), -1) # [batch_size, seq_len+1, 800]

        # 3. LSTM获取强化语义特征
        lstm_outputs = self.lstm_encoder(outputs)[0] # [batch_size, seq_len+1, 800]
        # 使用残差网络结构
        outputs = outputs + self.relu(self.lstm_fc(lstm_outputs)) # [batch_size, seq_len+1, 800]
        # 剔除非token序列
        outputs = outputs[:, 1:, :] # [batch_size, seq_len, 800]

        # 4. GAT字依存编码
        gnn_out = [self.get_gat_tensor(dep_graphs[i], outputs[i]).to(outputs.device) for i in
                   range(len(outputs))]
        outputs = torch.stack(gnn_out) # [batch_size, seq_len, 800]

        # 5. 表填充融合模块
        outputs1, outputs2 = torch.broadcast_tensors(outputs[:, :, None], outputs[:, None])
        s = F.dropout(F.gelu(self.table_fc(torch.cat([outputs1, outputs2], dim=-1))),p=config.param.dropout, training=self.training)

        # 6. 联合解码层
        total_loss = 0
        results = []
        for k in range(len(table_labels)):
            n = table_labels[k][0].shape[0]
            # 表特征
            table_outputs = s[k, 0:n, 0:n]  # [n, n, 800]
            # 论元开始位置表格标注
            start_labels = torch.tensor(table_labels[k][0]).to(token_ids.device)
            # 论元结束位置表格标注
            end_labels = torch.tensor(table_labels[k][1]).to(token_ids.device)

            # 触发词预测
            trigger_logits = F.dropout(self.trigger_fc(table_outputs), p=config.param.dropout, training=self.training)
            trigger_mask = torch.eye(n).int().to(token_ids.device) # 只处理主对角线元素
            loss1 = self.loss_fct(trigger_logits.flatten(0, 1), (start_labels*trigger_mask).flatten())
            total_loss += torch.masked_select(loss1, trigger_mask.bool().flatten()).mean()

            # 论元开始位置预测
            argument_logits1 = F.dropout(self.role_start_fc(s[k, 0:n, 0:n]), p=config.param.dropout, training=self.training)
            argument_mask = (torch.ones(n, n) - torch.eye(n)).int().to(token_ids.device) # 只处理非主对角线元素
            loss2 = self.loss_fct(argument_logits1.flatten(0, 1), (start_labels * argument_mask).flatten())
            total_loss += torch.masked_select(loss2, argument_mask.bool().flatten()).mean()

            # 论元结束位置预测
            argument_logits2 = F.dropout(self.role_end_fc(s[k, 0:n, 0:n]), p=config.param.dropout, training=self.training)
            loss3 = self.loss_fct(argument_logits2.flatten(0, 1), (end_labels * argument_mask).flatten())
            total_loss += torch.masked_select(loss3, argument_mask.bool().flatten()).mean()

            # 取每个token预测概率最大值对应的索引作为预测的触发词标签标识和论元标签标示
            trigger_logits = torch.argmax(trigger_logits, dim=2)
            # 论元开始位置
            argument_logits1 = torch.argmax(argument_logits1 + argument_logits1.transpose(0, 1), dim=2)
            results1 = (trigger_logits *trigger_mask+ argument_logits1 * argument_mask).to('cpu').numpy()
            # 论元结束位置
            argument_logits2 = torch.argmax(argument_logits2 + argument_logits2.transpose(0, 1), dim=2)
            results2 = (trigger_logits *trigger_mask+ argument_logits2 * argument_mask).to('cpu').numpy()

            results.append((results1, results2))
        return total_loss, results
