
## Requirements
* Python	3.7.12
* CUDA	11.3
* PyTorch	1.8.0
* Transformers	4.15.0
* torch-geometric	2.0.4
* torch-scatter	2.0.6
* torch-sparse	0.6.9
* torch-cluster	1.5.9 
* torch-spline-conv	1.2.1

## 文件目录说明
```shell
paperCode
├── data                                    # 数据文件夹
│   ├── DuEE                                
│   │   ├── train.json                      # 训练集
│   │   ├── dev.json                        # 验证集
│   │   ├── test.json                       # 测试集
│   │   ├── dep.pkl                         # 经LTP工具处理得到的事件文本对应的依存信息
│   │   ├── pos.pkl                         # 经LTP工具处理得到的事件文本对应的词性标注信息
│   │   └── pos.pkl                         # 经LTP工具处理得到的事件文本对应的词性标注信息
│   └── ACE                                  
│
├── ERNIE_pretrain                          # 存放百度ERNIE预训练模型
│
├── train.py                                # 模型训练  python train.py
│
├── test.py                                 # 模型测试  python test.py
│
├── ee_model.py                             # 模型类
│
├── config.py                               # 模型超参数配置类
│
├── utils.py                                # 工具类
│
├── data_helper.py                          # 构造batch数据
│
└── data_process.py                         # LTP工具对事件文本进行预处理生成dep.pkl和pos.pkl
```



