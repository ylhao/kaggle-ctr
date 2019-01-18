# coding: utf-8
import os
import sys
import collections
import random
import time
import numpy as np


categorial_features = range(1, 22)  # 定义了非连续型特征对应的列号范围，起始列号为 0


class CategoryDictGenerator:
    """
    非连续型特征生成器
    """

    def __init__(self, num_feature):
        """
        @param num_feature: 非连续型特征的数量
        """
        self.dicts = []
        self.num_feature = num_feature
        for i in range(0, num_feature):
            # class collections.defaultdict([default_factory[, ...]])，将 default_factory 设置成 int 使 defaultdict 对于计数非常有用 
            self.dicts.append(collections.defaultdict(int))


    def build(self, datafile, categorial_features, cutoff=0, sep=','):
        """
        该函数的作用就是从每个非连续特征中筛选部分取值，这部分取值保留，当出现其它取值的时候，都作为一个统一的未知值处理。
        这样做，我的理解是，首先达到了降维的目的，然后还增强了模型的鲁棒性，允许模型预测过程中出现异常值。
        但是个人认为每个特征都应该有自己的 cutoff 取值，这里统一取了 200。

        @param datafile: 数据文件
        @param categorial_features: 非连续特征对应的列号范围
        @param cutoff: 特征出现次数下限（如果某个特征取值出现次数小于 cutoff，那么就删除这个特征取值）
        """
        with open(datafile, 'r') as f:
            for line_num, line in enumerate(f):
                if line_num % 100000 == 0:
                    print(line_num)
                features = line.rstrip('\n').split(sep)  # 获取每一行的所有特征
                # 遍历每行的每一个非连续型特征，每个特征的所有取值计数
                for i in range(0, self.num_feature):
                    if features[categorial_features[i]] != '':
                        self.dicts[i][features[categorial_features[i]]] += 1
        
        for i in range(0, self.num_feature):
            # filter() 函数用于过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新列表。
            self.dicts[i] = filter(lambda x: x[1] >= cutoff, self.dicts[i].items())
            if i == 0:
                print(self.dicts[i])  # <filter object at 0x7fa4009a22b0>
            self.dicts[i] = sorted(self.dicts[i], key=lambda x: (-x[1], x[0]))  # 按每个特征出现的次数对特征进行排序，如果两个特征的取值一样，则按特征取值进行排序
            if i == 0:
                print(self.dicts[i])  # [('1005', 37140632), ('1002', 2220812), ('1010', 903457), ('1012', 113512), ('1007', 35304), ('1001', 9463), ('1008', 5787)]
            vocabs, _ = list(zip(*self.dicts[i]))
            if i == 0:
                print(vocabs)  # ('1005', '1002', '1010', '1012', '1007', '1001', '1008')
            self.dicts[i] = dict(zip(vocabs, range(1, len(vocabs) + 1)))
            if i == 0:
                print(self.dicts[i])  # {'1001': 6, '1012': 4, '1010': 3, '1008': 7, '1007': 5, '1002': 2, '1005': 1}
            self.dicts[i]['<unk>'] = 0
            if i == 0:
                print(self.dicts[i])  # {'1001': 6, '1012': 4, '1010': 3, '1008': 7, '1007': 5, '1002': 2, '1005': 1, '<unk>': 0}
                # 特征编号为 0，特征的取值有 '1001', '1012', '1010', '1008', '1007', '1002', '1005', <unk>
                # dicts = [{'1001': 6, '1012': 4, '1010': 3, '1008': 7, '1007': 5, '1002': 2, '1005': 1, '<unk>': 0}]


    def gen(self, idx, key):
        """
        @param idx: 特征编号
        @param key: 特征取值
        @return: res
        @rtype res: int
        """
        # 判断特征取值在不在字典里，不在字典里，返回 <unk> 对应的值，例如 {'1001': 6, '1012': 4, '1010': 3, '1008': 7, '1007': 5, '1002': 2, '1005': 1, '<unk>': 0}，特征不在字典里，返回 0。
        if key not in self.dicts[idx]:
            res = self.dicts[idx]['<unk>']
        else:
            res = self.dicts[idx][key]
        return res


    def dicts_sizes(self):
        """
        计算每个特征的长度（取值个数）
        """
        return list(map(len, self.dicts))


def preprocess(datadir, outdir, train_data_file, test_data_file, sep=','):
    """
    @param datadir: 数据所在文件夹
    @param outdir: 将文件写到哪个文件夹下
    """

    dicts = CategoryDictGenerator(len(categorial_features))
    dicts.build(os.path.join(datadir, train_data_file), categorial_features, cutoff=200)
    dict_sizes = dicts.dicts_sizes()

    # categorial_feature_offset 中存的就是每个特征的起始位置
    categorial_feature_offset = [0]  # 第一个特征的起始位置为 0
    for i in range(1, len(categorial_features)):
        offset = categorial_feature_offset[i - 1] + dict_sizes[i - 1]
        categorial_feature_offset.append(offset)

    """
    构造输入，划分数据集
    """
    train_ffm = open(os.path.join(outdir, 'train_ffm.txt'), 'w') # FFM 训练集
    valid_ffm = open(os.path.join(outdir, 'valid_ffm.txt'), 'w') # FFM 验证集
    train_lgb = open(os.path.join(outdir, 'train_lgb.txt'), 'w') # LightGBM 训练集
    valid_lgb = open(os.path.join(outdir, 'valid_lgb.txt'), 'w') # LightGBM 验证集
    train_txt = open(os.path.join(outdir, 'train.txt'), 'w')
    valid_txt = open(os.path.join(outdir, 'valid.txt'), 'w')

    start = time.time()      
    with open(os.path.join(datadir, train_data_file), 'r') as f:
        for line_num, line in enumerate(f):
            if line_num % 100000 == 0:
                print(line_num)
                end=time.time()
                print('Running time: {} min'.format((end - start)/ 60))
            features = line.rstrip('\n').split(sep)  # 得到该行的所有特征
            categorial_vals = []
            categorial_lgb_vals = []
            for i in range(0, len(categorial_features)):  # 遍历该行的所有非连续型特征
                """
                最终的目的是将每行数据转成一个长的一维向量。
                每个非连续型特征对应的是一个 one-hot 向量。
                val = dicts.gen(i, features[categorial_features[i]]) + categorial_feature_offset[i]
                这个 val 的值其实就是整个长的一维向量哪个位置应该设为 1。
                """
                pos = dicts.gen(i, features[categorial_features[i]])
                offset = categorial_feature_offset[i]
                val =  pos + offset
                val_lgb = pos
                categorial_vals.append(str(val))
                categorial_lgb_vals.append(str(val_lgb))
            categorial_vals = ','.join(categorial_vals)
            label = features[0]  # 提取标签
            if random.random() < 0.9:  # 90% 的数据写入训练集
                # train_txt
                train_txt.write(','.join([categorial_vals, label]) + '\n')
                # train_ffm
                # label\t{}:{}:1\t{}:{}:1...
                train_ffm.write('\t'.join(label) + '\t')
                train_ffm.write('\t'.join(['{}:{}:1'.format(ii, str(np.int32(val))) for ii, val in enumerate(categorial_vals.split(','))]) + '\n')
                # train_lgb
                train_lgb.write('\t'.join(label) + '\t')
                train_lgb.write('\t'.join(categorial_lgb_vals) + '\n')
            else:  # 10% 的数据写入测试集
                # valid_txt
                valid_txt.write(','.join([categorial_vals, label]) + '\n')
                # valid_ffm
                valid_ffm.write('\t'.join(label) + '\t')
                valid_ffm.write('\t'.join(['{}:{}:1'.format(ii, str(np.int32(val))) for ii, val in enumerate(categorial_vals.split(','))]) + '\n')
                # valid_lgb
                valid_lgb.write('\t'.join(label) + '\t')
                valid_lgb.write('\t'.join(categorial_lgb_vals) + '\n')
    
    train_txt.close()
    valid_txt.close()
    train_ffm.close()
    valid_ffm.close()
    train_lgb.close()
    valid_lgb.close()

    test_txt = open(os.path.join(outdir, 'test.txt'), 'w')
    test_ffm = open(os.path.join(outdir, 'test_ffm.txt'), 'w')
    test_lgb = open(os.path.join(outdir, 'test_lgb.txt'), 'w')

    with open(os.path.join(datadir, test_data_file), 'r') as f:
        for line_num, line in enumerate(f):
            if line_num % 100000 == 0:
                print(line_num)
                end=time.time()
                print('Running time: {} min'.format((end - start)/ 60))
            features = line.rstrip('\n').split(sep)
            categorial_vals = []
            categorial_lgb_vals = []
            for i in range(0, len(categorial_features)):
                pos = dicts.gen(i, features[categorial_features[i] - 1])
                offset = categorial_feature_offset[i]
                val = pos + offset
                categorial_vals.append(str(val))
                val_lgb = dicts.gen(i, pos)
                categorial_lgb_vals.append(str(val_lgb))
            categorial_vals = ','.join(categorial_vals)
            # test_txt
            test_txt.write(categorial_vals + '\n')
            # test_ffm
            test_ffm.write('\t'.join(['{}:{}:1'.format(ii, str(np.int32(val))) for ii, val in enumerate(categorial_vals.split(','))]) + '\n')
            # test_lgb
            test_lgb.write('\t'.join(categorial_lgb_vals) + '\n')
    
    test_txt.close()
    test_ffm.close()
    test_lgb.close()

preprocess('./data', './data', 'train_drop_id_hour.csv', 'test_drop_id_hour.csv', sep=',')
