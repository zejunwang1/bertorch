# bertorch: 基于 pytorch 的 bert 实现和下游任务微调

bertorch 是一个基于 pytorch 进行 bert 实现和下游任务微调的工具，支持常用的自然语言处理任务，包括文本分类、文本匹配、语义理解和序列标注等。

## 1. 依赖环境

- Python >= 3.6
  
- torch >= 1.1
  
- argparse
  
- json
  
- loguru
  
- numpy
  
- packaging
  
- re
  

## 2. 文本分类

本项目展示了以 BERT 为代表的预训练模型如何 Finetune 完成文本分类任务。我们以中文情感分类公开数据集 ChnSentiCorp 为例，运行如下的命令，基于 DistributedDataParallel 进行单机多卡分布式训练，在训练集 (train.tsv) 上进行模型训练，并在验证集 (dev.tsv) 上进行评估：

```shell
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 run_classifier.py --train_data_file ./data/ChnSentiCorp/train.tsv --dev_data_file ./data/ChnSentiCorp/dev.tsv --label_file ./data/ChnSentiCorp/labels.txt --save_best_model --epochs 3 --batch_size 32
```

可支持的配置参数：

```
usage: run_classifier.py [-h] [--local_rank LOCAL_RANK]
                         [--pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH]
                         [--init_from_ckpt INIT_FROM_CKPT] --train_data_file
                         TRAIN_DATA_FILE [--dev_data_file DEV_DATA_FILE]
                         --label_file LABEL_FILE [--batch_size BATCH_SIZE]
                         [--scheduler {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}]
                         [--learning_rate LEARNING_RATE]
                         [--warmup_proportion WARMUP_PROPORTION] [--seed SEED]
                         [--save_steps SAVE_STEPS]
                         [--logging_steps LOGGING_STEPS]
                         [--weight_decay WEIGHT_DECAY] [--epochs EPOCHS]
                         [--max_seq_length MAX_SEQ_LENGTH]
                         [--saved_dir SAVED_DIR]
                         [--max_grad_norm MAX_GRAD_NORM] [--save_best_model]
                         [--is_text_pair]
```

- local_rank: 可选，分布式训练的节点编号，默认为 -1。
  
- pretrained_model_name_or_path: 可选，huggingface 中的预训练模型名称或路径，默认为 bert-base-chinese。
  
- train_data_file: 必选，训练集数据文件路径。
  
- dev_data_file: 可选，验证集数据文件路径，默认为 None。
  
- label_file: 必选，类别标签文件路径。
  
- batch_size: 可选，批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数。默认为 32。
  
- init_from_ckpt: 可选，要加载的模型参数路径，热启动模型训练。默认为None。
  
- scheduler: 可选，优化器学习率变化策略，默认为 linear。
  
- learning_rate: 可选，优化器的最大学习率，默认为 5e-5。
  
- warmup_proportion: 可选，学习率 warmup 策略的比例，如果为 0.1，则学习率会在前 10% 训练 step 的过程中从 0 慢慢增长到 learning_rate，而后再缓慢衰减。默认为 0。
  
- weight_decay: 可选，控制正则项力度的参数，用于防止过拟合，默认为 0.0。
  
- seed: 可选，随机种子，默认为1000。
  
- logging_steps: 可选，日志打印的间隔 steps，默认为 20。
  
- save_steps: 可选，保存模型参数的间隔 steps，默认为 100。
  
- epochs: 可选，训练轮次，默认为 3。
  
- max_seq_length: 可选，输入到预训练模型中的最大序列长度，最大不能超过 512，默认为 128。
  
- saved_dir: 可选，保存训练模型的文件夹路径，默认保存在当前目录的 checkpoint 文件夹下。
  
- max_grad_norm: 可选，训练过程中梯度裁剪的 max_norm 参数，默认为 1.0。
  
- save_best_model: 可选，是否在最佳验证集指标上保存模型，当训练命令中加入
  
  --save_best_model 时，save_best_model 为 True，否则为 False。
  
- is_text_pair: 可选，是否进行文本对分类，当训练命令中加入 --is_text_pair 时，进行文本对的分类，否则进行普通文本分类。
  

模型训练的中间日志如下：

```python
2022-05-25 07:22:29.403 | INFO     | __main__:train:301 - global step: 20, epoch: 1, batch: 20, loss: 0.23227, accuracy: 0.87500, speed: 2.12 step/s
2022-05-25 07:22:39.131 | INFO     | __main__:train:301 - global step: 40, epoch: 1, batch: 40, loss: 0.30054, accuracy: 0.87500, speed: 2.06 step/s
2022-05-25 07:22:49.010 | INFO     | __main__:train:301 - global step: 60, epoch: 1, batch: 60, loss: 0.23514, accuracy: 0.93750, speed: 2.02 step/s
2022-05-25 07:22:58.909 | INFO     | __main__:train:301 - global step: 80, epoch: 1, batch: 80, loss: 0.12026, accuracy: 0.96875, speed: 2.02 step/s
2022-05-25 07:23:08.804 | INFO     | __main__:train:301 - global step: 100, epoch: 1, batch: 100, loss: 0.21955, accuracy: 0.90625, speed: 2.02 step/s
2022-05-25 07:23:13.534 | INFO     | __main__:train:307 - eval loss: 0.22564, accuracy: 0.91750
2022-05-25 07:23:25.222 | INFO     | __main__:train:301 - global step: 120, epoch: 1, batch: 120, loss: 0.32157, accuracy: 0.90625, speed: 2.03 step/s
2022-05-25 07:23:35.104 | INFO     | __main__:train:301 - global step: 140, epoch: 1, batch: 140, loss: 0.20107, accuracy: 0.87500, speed: 2.02 step/s
2022-05-25 07:23:44.978 | INFO     | __main__:train:301 - global step: 160, epoch: 2, batch: 10, loss: 0.08750, accuracy: 0.96875, speed: 2.03 step/s
2022-05-25 07:23:54.869 | INFO     | __main__:train:301 - global step: 180, epoch: 2, batch: 30, loss: 0.08308, accuracy: 1.00000, speed: 2.02 step/s
2022-05-25 07:24:04.754 | INFO     | __main__:train:301 - global step: 200, epoch: 2, batch: 50, loss: 0.10256, accuracy: 0.93750, speed: 2.02 step/s
2022-05-25 07:24:09.480 | INFO     | __main__:train:307 - eval loss: 0.22497, accuracy: 0.93083
2022-05-25 07:24:21.020 | INFO     | __main__:train:301 - global step: 220, epoch: 2, batch: 70, loss: 0.23989, accuracy: 0.93750, speed: 2.03 step/s
2022-05-25 07:24:30.919 | INFO     | __main__:train:301 - global step: 240, epoch: 2, batch: 90, loss: 0.00897, accuracy: 1.00000, speed: 2.02 step/s
2022-05-25 07:24:40.777 | INFO     | __main__:train:301 - global step: 260, epoch: 2, batch: 110, loss: 0.13605, accuracy: 0.93750, speed: 2.03 step/s
2022-05-25 07:24:50.640 | INFO     | __main__:train:301 - global step: 280, epoch: 2, batch: 130, loss: 0.14508, accuracy: 0.93750, speed: 2.03 step/s
2022-05-25 07:25:00.529 | INFO     | __main__:train:301 - global step: 300, epoch: 2, batch: 150, loss: 0.04770, accuracy: 0.96875, speed: 2.02 step/s
2022-05-25 07:25:05.256 | INFO     | __main__:train:307 - eval loss: 0.23039, accuracy: 0.93500
2022-05-25 07:25:16.818 | INFO     | __main__:train:301 - global step: 320, epoch: 3, batch: 20, loss: 0.04312, accuracy: 0.96875, speed: 2.04 step/s
2022-05-25 07:25:26.700 | INFO     | __main__:train:301 - global step: 340, epoch: 3, batch: 40, loss: 0.05103, accuracy: 0.96875, speed: 2.02 step/s
2022-05-25 07:25:36.588 | INFO     | __main__:train:301 - global step: 360, epoch: 3, batch: 60, loss: 0.12114, accuracy: 0.87500, speed: 2.02 step/s
2022-05-25 07:25:46.443 | INFO     | __main__:train:301 - global step: 380, epoch: 3, batch: 80, loss: 0.01080, accuracy: 1.00000, speed: 2.03 step/s
2022-05-25 07:25:56.228 | INFO     | __main__:train:301 - global step: 400, epoch: 3, batch: 100, loss: 0.14839, accuracy: 0.96875, speed: 2.04 step/s
2022-05-25 07:26:00.953 | INFO     | __main__:train:307 - eval loss: 0.22589, accuracy: 0.94083
2022-05-25 07:26:12.483 | INFO     | __main__:train:301 - global step: 420, epoch: 3, batch: 120, loss: 0.14986, accuracy: 0.96875, speed: 2.05 step/s
2022-05-25 07:26:22.289 | INFO     | __main__:train:301 - global step: 440, epoch: 3, batch: 140, loss: 0.00687, accuracy: 1.00000, speed: 2.04 step/s
```

当需要进行文本对分类时，仅需设置 is_text_pair 为 True。以 CLUEbenchmark 中的 AFQMC 蚂蚁金融语义相似度数据集为例，可以运行如下的命令进行训练：

```shell
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 run_classifier.py --train_data_file ./data/AFQMC/train.txt --dev_data_file ./data/AFQMC/dev.txt --label_file ./data/AFQMC/labels.txt --is_text_pair --save_best_model --epochs 3 --batch_size 32
```

在不同数据集上进行训练，验证集上的效果如下：

| Task | ChnSentiCorp | AFQMC | TNEWS |
| --- | --- | --- | --- |
| dev-acc | 0.94083 | 0.74305 | 0.56990 |

TNEWS 为 CLUEbenchmark 中的今日头条新闻分类数据集。

CLUEbenchmark 数据集链接： https://github.com/CLUEbenchmark/CLUE

## 3. 文本匹配

本项目展示了如何基于 Sentence-BERT 结构 Finetune 完成中文文本匹配任务。Sentence BERT 采用了双塔 (Siamese) 的网络结构。Query 和 Title 分别输入到两个共享参数的 bert encoder 中，得到各自的 token embedding 特征。然后对 token embedding 进行 pooling (论文中使用 mean pooling 操作)，输出分别记作 u 和 v。最后将三个向量 (u,v,|u-v|) 拼接起来输入到线性分类器中进行分类。

更多关于 Sentence-BERT 的信息可以参考论文： https://arxiv.org/abs/1908.10084

我们以中文文本匹配数据集 LCQMC 为例，运行下面的命令，基于 DistributedDataParallel 进行单机多卡分布式训练，在训练集上进行模型训练，在验证集上进行评估：

```shell
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 run_sentencebert.py --train_data_file ./data/LCQMC/train.txt --dev_data_file ./data/LCQMC/dev.txt --save_best_model --epochs 3 --batch_size 32
```

可支持的配置参数：

```
usage: run_sentencebert.py [-h] [--local_rank LOCAL_RANK]
                           [--pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH]
                           [--init_from_ckpt INIT_FROM_CKPT] --train_data_file
                           TRAIN_DATA_FILE [--dev_data_file DEV_DATA_FILE]
                           [--label_file LABEL_FILE] [--batch_size BATCH_SIZE]
                           [--scheduler {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}]
                           [--learning_rate LEARNING_RATE]
                           [--warmup_proportion WARMUP_PROPORTION]
                           [--seed SEED] [--save_steps SAVE_STEPS]
                           [--logging_steps LOGGING_STEPS]
                           [--weight_decay WEIGHT_DECAY] [--epochs EPOCHS]
                           [--max_seq_length MAX_SEQ_LENGTH]
                           [--saved_dir SAVED_DIR]
                           [--max_grad_norm MAX_GRAD_NORM] [--save_best_model]
                           [--is_nli] [--pooling_mode {linear,cls,mean}]
                           [--concat_multiply]
                           [--output_emb_size OUTPUT_EMB_SIZE]
```

其中大部分参数与文本分类中介绍的相同，如下为特有的参数：

- is_nli: 可选，当训练命令中加入 --is_nli 时，使用 NLI 自然语言推断数据集进行模型训练。
  
- pooling_mode: 可选，当为 linear 时，使用 cls 向量经过 linear pooler 后的输出作为 encoder 编码的句子向量；当为 cls 时，使用 cls 向量作为 encoder 编码的句子向量；当为 mean 时，使用所有 token 向量的平均值作为 encoder 编码的句子向量。默认为 linear。
  
- concat_multiply: 可选，当训练命令中加入 --concat_multiply 时，使用 (u, v, |u-v|, u*v) 作为分类器的输入特征；否则使用 (u, v, |u-v|) 作为分类器的输入特征。
  
- output_emb_size: 可选，encoder 输出的句子向量维度，当为 None 时，输出句子向量的维度为 encoder 的 hidden_size。默认为 None。
  

模型训练的部分中间日志如下：

```python
......
2022-05-24 17:07:26.672 | INFO     | __main__:train:308 - global step: 9620, epoch: 3, batch: 2158, loss: 0.16183, accuracy: 0.90625, speed: 3.38 step/s
2022-05-24 17:07:32.407 | INFO     | __main__:train:308 - global step: 9640, epoch: 3, batch: 2178, loss: 0.09866, accuracy: 0.96875, speed: 3.49 step/s
2022-05-24 17:07:38.177 | INFO     | __main__:train:308 - global step: 9660, epoch: 3, batch: 2198, loss: 0.38715, accuracy: 0.90625, speed: 3.47 step/s
2022-05-24 17:07:43.796 | INFO     | __main__:train:308 - global step: 9680, epoch: 3, batch: 2218, loss: 0.12515, accuracy: 0.93750, speed: 3.56 step/s
2022-05-24 17:07:49.740 | INFO     | __main__:train:308 - global step: 9700, epoch: 3, batch: 2238, loss: 0.03231, accuracy: 1.00000, speed: 3.37 step/s
2022-05-24 17:08:04.752 | INFO     | __main__:train:314 - eval loss: 0.38621, accuracy: 0.86549
2022-05-24 17:08:12.245 | INFO     | __main__:train:308 - global step: 9720, epoch: 3, batch: 2258, loss: 0.08337, accuracy: 0.96875, speed: 3.45 step/s
2022-05-24 17:08:18.112 | INFO     | __main__:train:308 - global step: 9740, epoch: 3, batch: 2278, loss: 0.15085, accuracy: 0.93750, speed: 3.41 step/s
2022-05-24 17:08:23.895 | INFO     | __main__:train:308 - global step: 9760, epoch: 3, batch: 2298, loss: 0.11466, accuracy: 0.93750, speed: 3.46 step/s
2022-05-24 17:08:29.703 | INFO     | __main__:train:308 - global step: 9780, epoch: 3, batch: 2318, loss: 0.04269, accuracy: 1.00000, speed: 3.44 step/s
2022-05-24 17:08:35.658 | INFO     | __main__:train:308 - global step: 9800, epoch: 3, batch: 2338, loss: 0.28312, accuracy: 0.90625, speed: 3.36 step/s
2022-05-24 17:08:50.674 | INFO     | __main__:train:314 - eval loss: 0.39262, accuracy: 0.86424
2022-05-24 17:08:56.609 | INFO     | __main__:train:308 - global step: 9820, epoch: 3, batch: 2358, loss: 0.13456, accuracy: 0.96875, speed: 3.37 step/s
2022-05-24 17:09:02.259 | INFO     | __main__:train:308 - global step: 9840, epoch: 3, batch: 2378, loss: 0.06361, accuracy: 1.00000, speed: 3.54 step/s
2022-05-24 17:09:08.120 | INFO     | __main__:train:308 - global step: 9860, epoch: 3, batch: 2398, loss: 0.09087, accuracy: 0.96875, speed: 3.41 step/s
2022-05-24 17:09:13.834 | INFO     | __main__:train:308 - global step: 9880, epoch: 3, batch: 2418, loss: 0.19537, accuracy: 0.90625, speed: 3.50 step/s
2022-05-24 17:09:19.531 | INFO     | __main__:train:308 - global step: 9900, epoch: 3, batch: 2438, loss: 0.05254, accuracy: 1.00000, speed: 3.51 step/s
2022-05-24 17:09:34.531 | INFO     | __main__:train:314 - eval loss: 0.39561, accuracy: 0.86560
2022-05-24 17:09:42.084 | INFO     | __main__:train:308 - global step: 9920, epoch: 3, batch: 2458, loss: 0.05342, accuracy: 1.00000, speed: 3.41 step/s
2022-05-24 17:09:47.781 | INFO     | __main__:train:308 - global step: 9940, epoch: 3, batch: 2478, loss: 0.22660, accuracy: 0.87500, speed: 3.51 step/s
2022-05-24 17:09:53.496 | INFO     | __main__:train:308 - global step: 9960, epoch: 3, batch: 2498, loss: 0.14745, accuracy: 0.93750, speed: 3.50 step/s
2022-05-24 17:09:59.350 | INFO     | __main__:train:308 - global step: 9980, epoch: 3, batch: 2518, loss: 0.06218, accuracy: 0.96875, speed: 3.42 step/s
2022-05-24 17:10:05.157 | INFO     | __main__:train:308 - global step: 10000, epoch: 3, batch: 2538, loss: 0.15225, accuracy: 0.96875, speed: 3.44 step/s
2022-05-24 17:10:20.159 | INFO     | __main__:train:314 - eval loss: 0.39152, accuracy: 0.86730
......
```

当使用 NLI 数据进行训练时，需要加入 --is_nli 选项和 --label_file LABEL_FILE，训练命令如下：

```shell
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 run_sentencebert.py --train_data_file ./data/CMNLI/train.txt --dev_data_file ./data/CMNLI/dev.txt --label_file ./data/CMNLI/labels.txt --is_nli --save_best_model --epochs 3 --batch_size 32
```

在不同数据集上进行训练，验证集上的效果如下：

| Task | LCQMC | Chinese-MNLI | Chinese-SNLI |
| --- | --- | --- | --- |
| dev-acc | 0.86730 | 0.71105 | 0.80567 |

Chinese-MNLI 和 Chinese-SNLI 链接： https://github.com/zejunwang1/CSTS

## 4. 语义理解

### 4.1 SimCSE

SimCSE 模型适合缺乏监督数据，但是又有大量无监督数据的匹配和检索场景。本项目实现了 SimCSE 无监督方法，并在中文维基百科句子数据上进行句向量表示模型的训练。

更多关于 SimCSE 的信息可以参考论文： https://arxiv.org/abs/2104.08821

从中文维基百科中抽取 15 万条句子数据，保存于 data/zhwiki/ 文件夹下的 wiki_sents.txt 文件中，运行下面的命令，基于腾讯 uer 开源的预训练语言模型 uer/chinese_roberta_L-6_H-128 (https://huggingface.co/uer/chinese_roberta_L-6_H-128) ，使用 SimCSE 无监督方法进行训练，并在 Chinese-STS-B 验证集 ( https://github.com/zejunwang1/CSTS ) 上进行评估：

```shell
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 run_simcse.py --pretrained_model_name_or_path uer/chinese_roberta_L-6_H-128 --train_data_file ./data/zhwiki/wiki_sents.txt --dev_data_file ./data/STS-B/sts-b-dev.txt --learning_rate 5e-5 --epochs 1 --dropout 0.1 --margin 0.2 --scale 20 --batch_size 32
```

可支持的配置参数：

```
usage: run_simcse.py [-h] [--local_rank LOCAL_RANK]
                     [--pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH]
                     [--init_from_ckpt INIT_FROM_CKPT] --train_data_file
                     TRAIN_DATA_FILE [--dev_data_file DEV_DATA_FILE]
                     [--batch_size BATCH_SIZE]
                     [--scheduler {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}]
                     [--learning_rate LEARNING_RATE]
                     [--warmup_proportion WARMUP_PROPORTION] [--seed SEED]
                     [--save_steps SAVE_STEPS] [--logging_steps LOGGING_STEPS]
                     [--weight_decay WEIGHT_DECAY] [--epochs EPOCHS]
                     [--max_seq_length MAX_SEQ_LENGTH] [--saved_dir SAVED_DIR]
                     [--max_grad_norm MAX_GRAD_NORM] [--save_best_model]
                     [--margin MARGIN] [--scale SCALE] [--dropout DROPOUT]
                     [--pooling_mode {linear,cls,mean}]
                     [--output_emb_size OUTPUT_EMB_SIZE]
```

其中大部分参数与文本分类中介绍的相同，如下为特有的参数：

- margin: 可选，正样本相似度与负样本之间的目标 Gap，默认为 0.2。
  
- dropout: 可选，SimCSE 网络中 encoder 部分使用的 dropout 取值，默认为 0.1。
  
- scale: 可选，在计算交叉熵损失之前，对余弦相似度进行缩放的因子，默认为 20。
  
- pooling_mode: 可选，当为 linear 时，使用 cls 向量经过 linear pooler 后的输出作为 encoder 编码的句子向量；当为 cls 时，使用 cls 向量作为 encoder 编码的句子向量；当为 mean 时，使用所有 token 向量的平均值作为 encoder 编码的句子向量。默认为 linear。
  
- output_emb_size: 可选，encoder 输出的句子向量维度，当为 None 时，输出句子向量的维度为 encoder 的 hidden_size。默认为 None。
  

模型训练的部分中间日志如下：

```python
2022-05-27 09:14:58.471 | INFO     | __main__:train:315 - global step: 20, epoch: 1, batch: 20, loss: 1.04241, speed: 8.45 step/s
2022-05-27 09:15:01.063 | INFO     | __main__:train:315 - global step: 40, epoch: 1, batch: 40, loss: 0.15792, speed: 7.72 step/s
2022-05-27 09:15:03.700 | INFO     | __main__:train:315 - global step: 60, epoch: 1, batch: 60, loss: 0.18357, speed: 7.58 step/s
2022-05-27 09:15:06.365 | INFO     | __main__:train:315 - global step: 80, epoch: 1, batch: 80, loss: 0.13284, speed: 7.51 step/s
2022-05-27 09:15:09.000 | INFO     | __main__:train:315 - global step: 100, epoch: 1, batch: 100, loss: 0.14146, speed: 7.59 step/s
2022-05-27 09:15:09.847 | INFO     | __main__:train:321 - spearman corr: 0.6048, pearson corr: 0.5870
2022-05-27 09:15:12.507 | INFO     | __main__:train:315 - global step: 120, epoch: 1, batch: 120, loss: 0.03073, speed: 7.74 step/s
2022-05-27 09:15:15.110 | INFO     | __main__:train:315 - global step: 140, epoch: 1, batch: 140, loss: 0.09425, speed: 7.69 step/s
2022-05-27 09:15:17.749 | INFO     | __main__:train:315 - global step: 160, epoch: 1, batch: 160, loss: 0.08629, speed: 7.58 step/s
2022-05-27 09:15:20.386 | INFO     | __main__:train:315 - global step: 180, epoch: 1, batch: 180, loss: 0.03206, speed: 7.59 step/s
2022-05-27 09:15:23.052 | INFO     | __main__:train:315 - global step: 200, epoch: 1, batch: 200, loss: 0.11463, speed: 7.50 step/s
2022-05-27 09:15:24.023 | INFO     | __main__:train:321 - spearman corr: 0.5954, pearson corr: 0.5807
......
```

隐藏层数 num_hidden_layers=6，维度 hidden_size=128 的 SimCSE 句向量预训练模型 simcse_tiny_chinese_wiki 可以从如下链接获取：

| model_name | link |
| --- | --- |
| WangZeJun/simcse-tiny-chinese-wiki | https://huggingface.co/WangZeJun/simcse-tiny-chinese-wiki |

### 4.2 In-Batch Negatives

从哈工大 LCQMC 数据集、谷歌 PAWS-X 数据集、北大文本复述 PKU-Paraphrase-Bank 数据集 (https://github.com/zejunwang1/CSTS) 中抽取出所有语义相似的文本 Pair 作为训练集，保存于：data/batchneg/paraphrase_lcqmc_semantic_pairs.txt

运行下面的命令，基于腾讯 uer 开源的预训练语言模型 uer/chinese_roberta_L-6_H-128，采用 In-batch negatives 策略，在 GPU 0,1,2,3 四张卡上训练句向量表示模型，并在 Chinese-STS-B 验证集上进行评估：

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 run_batchneg.py --pretrained_model_name_or_path uer/chinese_roberta_L-6_H-128 --train_data_file ./data/batchneg/paraphrase_lcqmc_semantic_pairs.txt --dev_data_file ./data/STS-B/sts-b-dev.txt --learning_rate 5e-5 --epochs 3 --margin 0.2 --scale 20 --batch_size 64 --mean_loss
```

可支持的配置参数：

```
usage: run_batchneg.py [-h] [--local_rank LOCAL_RANK]
                       [--pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH]
                       [--init_from_ckpt INIT_FROM_CKPT] --train_data_file
                       TRAIN_DATA_FILE [--dev_data_file DEV_DATA_FILE]
                       [--batch_size BATCH_SIZE]
                       [--scheduler {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}]
                       [--learning_rate LEARNING_RATE]
                       [--warmup_proportion WARMUP_PROPORTION] [--seed SEED]
                       [--save_steps SAVE_STEPS]
                       [--logging_steps LOGGING_STEPS]
                       [--weight_decay WEIGHT_DECAY] [--epochs EPOCHS]
                       [--max_seq_length MAX_SEQ_LENGTH]
                       [--saved_dir SAVED_DIR] [--max_grad_norm MAX_GRAD_NORM]
                       [--save_best_model] [--margin MARGIN] [--scale SCALE]
                       [--pooling_mode {linear,cls,mean}]
                       [--output_emb_size OUTPUT_EMB_SIZE] [--mean_loss]
```

各参数的介绍与 SimCSE 中相同，模型训练的部分中间日志如下：

```python
......
2022-05-27 13:20:48.428 | INFO     | __main__:train:318 - global step: 7220, epoch: 3, batch: 1888, loss: 0.73655, speed: 6.70 step/s
2022-05-27 13:20:51.454 | INFO     | __main__:train:318 - global step: 7240, epoch: 3, batch: 1908, loss: 0.70207, speed: 6.61 step/s
2022-05-27 13:20:54.308 | INFO     | __main__:train:318 - global step: 7260, epoch: 3, batch: 1928, loss: 1.10231, speed: 7.01 step/s
2022-05-27 13:20:57.107 | INFO     | __main__:train:318 - global step: 7280, epoch: 3, batch: 1948, loss: 0.94975, speed: 7.15 step/s
2022-05-27 13:20:59.898 | INFO     | __main__:train:318 - global step: 7300, epoch: 3, batch: 1968, loss: 0.34252, speed: 7.17 step/s
2022-05-27 13:21:00.322 | INFO     | __main__:train:324 - spearman corr: 0.6950, pearson corr: 0.6801
2022-05-27 13:21:03.168 | INFO     | __main__:train:318 - global step: 7320, epoch: 3, batch: 1988, loss: 1.10022, speed: 7.20 step/s
2022-05-27 13:21:05.929 | INFO     | __main__:train:318 - global step: 7340, epoch: 3, batch: 2008, loss: 1.00207, speed: 7.25 step/s
2022-05-27 13:21:08.687 | INFO     | __main__:train:318 - global step: 7360, epoch: 3, batch: 2028, loss: 0.72985, speed: 7.25 step/s
2022-05-27 13:21:11.372 | INFO     | __main__:train:318 - global step: 7380, epoch: 3, batch: 2048, loss: 0.88964, speed: 7.45 step/s
2022-05-27 13:21:14.090 | INFO     | __main__:train:318 - global step: 7400, epoch: 3, batch: 2068, loss: 0.70836, speed: 7.36 step/s
2022-05-27 13:21:14.520 | INFO     | __main__:train:324 - spearman corr: 0.6922, pearson corr: 0.6764
......
```

以上面得到的模型为热启，在科研句子数据集 data/batchneg/domain_finetune.txt 上继续进行 In-batch negatives 训练：

```shell
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 run_batchneg.py --pretrained_model_name_or_path uer/chinese_roberta_L-6_H-128 --init_from_ckpt ./checkpoint/pytorch_model.bin --train_data_file ./data/batchneg/domain_finetune.txt --dev_data_file ./data/STS-B/sts-b-dev.txt --learning_rate 1e-5 --epochs 1 --margin 0.2 --scale 20 --batch_size 32 --mean_loss
```

可以得到隐藏层数 num_hidden_layers=6，维度 hidden_size=128 的句向量预训练模型：

| model_name | link |
| --- | --- |
| WangZeJun/batchneg-tiny-chinese | https://huggingface.co/WangZeJun/batchneg-tiny-chinese |

## 5. 序列标注

本项目展示了以 BERT 为代表的预训练模型如何 Finetune 完成序列标注任务。以中文命名实体识别任务为例，分别在 msra、ontonote4、resume 和 weibo 四个数据集上进行训练和测试。每个数据集的训练集和验证集均被预处理为如下的格式，每一行为文本和标签组成的 json 字符串。

```json
{"text": ["我", "们", "的", "藏", "品", "中", "有", "几", "十", "册", "为", "北", "京", "图", "书", "馆", "等", "国", "家", "级", "藏", "馆", "所", "未", "藏", "。"], "label": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "B-NS", "I-NS", "I-NS", "I-NS", "I-NS", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]}
{"text": ["由", "于", "这", "一", "时", "期", "战", "争", "频", "繁", "，", "条", "件", "艰", "苦", "，", "又", "遭", "国", "民", "党", "毁", "禁", "，", "传", "世", "量", "稀", "少", "，", "购", "藏", "不", "易", "。"], "label": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "B-NT", "I-NT", "I-NT", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]}
```

运行下面的命令，在 msra 数据集上使用 BERT+Linear 结构进行单机多卡分布式训练，并在验证集上进行评估：

```shell
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 run_ner.py --train_data_file ./data/ner/msra/train.json --dev_data_file ./data/ner/msra/dev.json --label_file ./data/ner/msra/labels.txt --tag bios --learning_rate 5e-5 --save_best_model --batch_size 32
```

可支持的配置参数：

```
usage: run_ner.py [-h] [--local_rank LOCAL_RANK]
                  [--pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH]
                  [--init_from_ckpt INIT_FROM_CKPT] --train_data_file
                  TRAIN_DATA_FILE [--dev_data_file DEV_DATA_FILE] --label_file
                  LABEL_FILE [--tag {bios,bio}] [--batch_size BATCH_SIZE]
                  [--scheduler {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}]
                  [--learning_rate LEARNING_RATE]
                  [--crf_learning_rate CRF_LEARNING_RATE]
                  [--warmup_proportion WARMUP_PROPORTION] [--seed SEED]
                  [--save_steps SAVE_STEPS] [--logging_steps LOGGING_STEPS]
                  [--weight_decay WEIGHT_DECAY] [--epochs EPOCHS]
                  [--max_seq_length MAX_SEQ_LENGTH] [--saved_dir SAVED_DIR]
                  [--max_grad_norm MAX_GRAD_NORM] [--save_best_model]
                  [--use_crf]
```

大部分参数与文本分类中介绍的相同，如下为特有的参数：

- tag: 可选，实体标记方法，支持 bios 和 bio 的标注方法，默认为 bios。
  
- use_crf: 可选，是否使用 CRF 结构，当训练命令中加入 --use_crf 时，使用 BERT+CRF 模型结构；否则使用 BERT+Linear 模型结构。
  
- crf_learning_rate: 可选，CRF 模型参数的初始学习率，默认为 5e-5。
  

模型训练的部分中间日志如下：

```python
2022-05-27 15:56:59.043 | INFO     | __main__:train:355 - global step: 20, epoch: 1, batch: 20, loss: 0.20780, speed: 2.10 step/s
2022-05-27 15:57:08.723 | INFO     | __main__:train:355 - global step: 40, epoch: 1, batch: 40, loss: 0.09440, speed: 2.07 step/s
2022-05-27 15:57:18.001 | INFO     | __main__:train:355 - global step: 60, epoch: 1, batch: 60, loss: 0.05570, speed: 2.16 step/s
2022-05-27 15:57:27.357 | INFO     | __main__:train:355 - global step: 80, epoch: 1, batch: 80, loss: 0.02468, speed: 2.14 step/s
2022-05-27 15:57:36.994 | INFO     | __main__:train:355 - global step: 100, epoch: 1, batch: 100, loss: 0.05032, speed: 2.08 step/s
2022-05-27 15:57:53.299 | INFO     | __main__:train:362 - eval loss: 0.03203, F1: 0.86481
2022-05-27 15:58:03.264 | INFO     | __main__:train:355 - global step: 120, epoch: 1, batch: 120, loss: 0.04150, speed: 2.16 step/s
2022-05-27 15:58:12.712 | INFO     | __main__:train:355 - global step: 140, epoch: 1, batch: 140, loss: 0.04907, speed: 2.12 step/s
2022-05-27 15:58:21.959 | INFO     | __main__:train:355 - global step: 160, epoch: 1, batch: 160, loss: 0.01224, speed: 2.16 step/s
2022-05-27 15:58:31.039 | INFO     | __main__:train:355 - global step: 180, epoch: 1, batch: 180, loss: 0.01846, speed: 2.20 step/s
2022-05-27 15:58:40.542 | INFO     | __main__:train:355 - global step: 200, epoch: 1, batch: 200, loss: 0.06604, speed: 2.10 step/s
2022-05-27 15:58:56.831 | INFO     | __main__:train:362 - eval loss: 0.02589, F1: 0.89128
2022-05-27 15:59:07.813 | INFO     | __main__:train:355 - global step: 220, epoch: 1, batch: 220, loss: 0.07066, speed: 2.15 step/s
2022-05-27 15:59:16.857 | INFO     | __main__:train:355 - global step: 240, epoch: 1, batch: 240, loss: 0.03061, speed: 2.21 step/s
2022-05-27 15:59:26.240 | INFO     | __main__:train:355 - global step: 260, epoch: 1, batch: 260, loss: 0.01680, speed: 2.13 step/s
2022-05-27 15:59:35.568 | INFO     | __main__:train:355 - global step: 280, epoch: 1, batch: 280, loss: 0.01245, speed: 2.14 step/s
2022-05-27 15:59:44.684 | INFO     | __main__:train:355 - global step: 300, epoch: 1, batch: 300, loss: 0.02699, speed: 2.19 step/s
2022-05-27 16:00:00.977 | INFO     | __main__:train:362 - eval loss: 0.01928, F1: 0.92157
```

当使用 BERT+CRF 结构进行训练时，运行下面的命令：

```shell
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 run_ner.py --train_data_file ./data/ner/msra/train.json --dev_data_file ./data/ner/msra/dev.json --label_file ./data/ner/msra/labels.txt --tag bios --learning_rate 5e-5 --save_best_model --batch_size 32 --use_crf --crf_learning_rate 1e-4
```

模型在不同验证集上的 F1 指标：

| 模型  | Msra | Resume | Ontonote | Weibo |
| --- | --- | --- | --- | --- |
| BERT+Linear | 0.94179 | 0.95643 | 0.80206 | 0.70588 |
| BERT+CRF | 0.94265 | 0.95818 | 0.80257 | 0.72215 |

其中 Msra、Resume 和 Ontonote 训练了 3 个 epochs，Weibo 训练了 5 个 epochs，Resume、Ontonote 和 Weibo 的 logging_steps 和 save_steps 均设置为 10，所有数据集的 BERT 参数初始学习率设置为 5e-5，CRF 参数初始学习率设置为 1e-4，batch_size 设置为 32。

## 6. Contact

邮箱： wangzejunscut@126.com

微信：autonlp
