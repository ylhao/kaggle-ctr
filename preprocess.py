# coding: utf-8
import os
import sys
import collections
import random


# continous_features = range(1, 14)  # 定义了连续型特征对应的列号范围，起始列号为 0
categorial_features = range(1, 22)  # 定义了非连续型特征对应的列号范围，起始列号为 0


"""
Clip integer features. The clip point for each integer feature is derived from the 95% quantile of the total values in each feature.
共有 13 个连续型数值特征，以下列表对应的是每个特征的 95% 分位数的值。
"""
# continous_clip = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]


# class ContinuousFeatureGenerator:
#     """
#     连续型特征生成器
#     Normalize the integer features to [0, 1] by min-max normalization.
#     最大最小缩放，将每列特征的值缩放到 [0, 1] 之间
#     """

#     def __init__(self, num_feature):
#         """
#         @param num_feature: 连续型特征的数量
#         """
#         self.num_feature = num_feature
#         self.min = [sys.maxsize] * num_feature
#         self.max = [-sys.maxsize] * num_feature

#     def build(self, datafile, continous_features, sep=','):
#         """
#         @param datafile: 数据文件
#         @continous_features: 连续型数值特征的对应的列号范围
#         """
#         with open(datafile, 'r') as f:
#             # 遍历数据文件的每一个行
#             for line in f:
#                 features = line.rstrip('\n').split(sep)
#                 # 遍历每行的每个连续型特征取值
#                 for i in range(0, self.num_feature):
#                     val = features[continous_features[i]]
#                     if val != '':
#                         val = int(val)
#                         # 裁剪（大于 95% 分位数的，用 95% 分位数替代）
#                         if val > continous_clip[i]:
#                             val = continous_clip[i]
#                         # 更新最大最小值
#                         self.min[i] = min(self.min[i], val)
#                         self.max[i] = max(self.max[i], val)

#     def gen(self, idx, val):
#         if val == '':
#             return 0.0
#         val = float(val)
#         return (val - self.min[idx]) / (self.max[idx] - self.min[idx])


class CategoryDictGenerator:
    """
    Generate dictionary for each of the categorical features
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
            # 遍历数据文件的每一行
            for line in f:
                features = line.rstrip('\n').split(sep)
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

    # dists = ContinuousFeatureGenerator(len(continous_features))
    # dists.build(os.path.join(datadir, 'train.txt'), continous_features)

    dicts = CategoryDictGenerator(len(categorial_features))
    dicts.build(os.path.join(datadir, 'train_drop_id_hour.csv'), categorial_features, cutoff=200)
    dict_sizes = dicts.dicts_sizes()

    # categorial_feature_offset 中存的就是每个特征的起始位置
    categorial_feature_offset = [0]  # 第一个特征的起始位置为 0
    for i in range(1, len(categorial_features)):
        offset = categorial_feature_offset[i - 1] + dict_sizes[i - 1]
        categorial_feature_offset.append(offset)


    """
    划分数据集
    90% of the data are used for training, and 10% of the data are used for validation.
    """
    train_ffm = open(os.path.join(outdir, 'train_ffm.txt'), 'w') # FFM 训练集
    valid_ffm = open(os.path.join(outdir, 'valid_ffm.txt'), 'w') # FFM 验证集
    train_lgb = open(os.path.join(outdir, 'train_lgb.txt'), 'w') # LightGBM 训练集
    valid_lgb = open(os.path.join(outdir, 'valid_lgb.txt'), 'w') # LightGBM 验证集

    with open(os.path.join(outdir, 'train.txt'), 'w') as out_train:
        with open(os.path.join(outdir, 'valid.txt'), 'w') as out_valid:
            with open(os.path.join(datadir, train_data_file), 'r') as f:
                for line in f:
                    features = line.rstrip('\n').split(sep)
                    # continous_feats = []
                    # continous_vals = []
                    # for i in range(0, len(continous_features)):
                    #     val = dists.gen(i, features[continous_features[i]])
                    #     continous_vals.append("{0:.6f}".format(val).rstrip('0').rstrip('.'))
                    #     continous_feats.append("{0:.6f}".format(val).rstrip('0').rstrip('.'))
                    categorial_vals = []
                    categorial_lgb_vals = []
                    # 遍历该行的所有非连续型特征
                    for i in range(0, len(categorial_features)):
                        #  categorial_feature_offset[i] 是对应的特征的起始位置，比如第 1 个特征的起始位置是 0
                        val = dicts.gen(i, features[categorial_features[i]]) + categorial_feature_offset[i]
                        categorial_vals.append(str(val))
                        val_lgb = dicts.gen(i, features[categorial_features[i]])
                        categorial_lgb_vals.append(str(val_lgb))

                    # continous_vals = ','.join(continous_vals)
                    categorial_vals = ','.join(categorial_vals)
                    label = features[0]
                    if random.randint(0, 9999) % 10 != 0:
                        
                        # out_train.write(','.join([continous_vals, categorial_vals, label]) + '\n')
                        out_train.write(','.join([categorial_vals, label]) + '\n')
                        

                        train_ffm.write('\t'.join(label) + '\t')
                        # train_ffm.write('\t'.join(['{}:{}:{}'.format(ii, ii, val) for ii,val in enumerate(continous_vals.split(','))]) + '\t')
                        train_ffm.write('\t'.join(['{}:{}:1'.format(ii + 13, str(np.int32(val) + 13)) for ii, val in enumerate(categorial_vals.split(','))]) + '\n')
                        
                        train_lgb.write('\t'.join(label) + '\t')
                        # train_lgb.write('\t'.join(continous_feats) + '\t')
                        train_lgb.write('\t'.join(categorial_lgb_vals) + '\n')

                    else:
                        
                        # out_valid.write(','.join([continous_vals, categorial_vals, label]) + '\n')
                        out_valid.write(','.join([categorial_vals, label]) + '\n')
                        
                        valid_ffm.write('\t'.join(label) + '\t')
                        # valid_ffm.write('\t'.join(['{}:{}:{}'.format(ii, ii, val) for ii,val in enumerate(continous_vals.split(','))]) + '\t')
                        valid_ffm.write('\t'.join(['{}:{}:1'.format(ii + 13, str(np.int32(val) + 13)) for ii, val in enumerate(categorial_vals.split(','))]) + '\n')
                                                
                        valid_lgb.write('\t'.join(label) + '\t')
                        # valid_lgb.write('\t'.join(continous_feats) + '\t')
                        valid_lgb.write('\t'.join(categorial_lgb_vals) + '\n')
                        
    train_ffm.close()
    valid_ffm.close()
    train_lgb.close()
    valid_lgb.close()

    test_ffm = open(os.path.join(outdir, 'test_ffm.txt'), 'w')
    test_lgb = open(os.path.join(outdir, 'test_lgb.txt'), 'w')

    with open(os.path.join(outdir, 'test.txt'), 'w') as out:
        with open(os.path.join(datadir, test_data_file), 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t')
                # continous_feats = []
                # continous_vals = []
                # for i in range(0, len(continous_features)):
                #     val = dists.gen(i, features[continous_features[i] - 1])
                #     continous_vals.append("{0:.6f}".format(val).rstrip('0').rstrip('.'))
                #     continous_feats.append("{0:.6f}".format(val).rstrip('0').rstrip('.'))
                categorial_vals = []
                categorial_lgb_vals = []
                for i in range(0, len(categorial_features)):
                    val = dicts.gen(i, features[categorial_features[i] - 1]) + categorial_feature_offset[i]
                    categorial_vals.append(str(val))
                    val_lgb = dicts.gen(i, features[categorial_features[i] - 1])
                    categorial_lgb_vals.append(str(val_lgb))

                # continous_vals = ','.join(continous_vals)
                categorial_vals = ','.join(categorial_vals)

                # out.write(','.join([continous_vals, categorial_vals]) + '\n')
                out.write(','.join([categorial_vals]) + '\n')
                
                # test_ffm.write('\t'.join(['{}:{}:{}'.format(ii, ii, val) for ii,val in enumerate(continous_vals.split(','))]) + '\t')
                test_ffm.write('\t'.join(['{}:{}:1'.format(ii + 13, str(np.int32(val) + 13)) for ii, val in enumerate(categorial_vals.split(','))]) + '\n')
                # test_lgb.write('\t'.join(continous_feats) + '\t')
                test_lgb.write('\t'.join(categorial_lgb_vals) + '\n')

    test_ffm.close()
    test_lgb.close()
    return dict_sizes

preprocess('./data', './data', 'train_drop_id_hour.csv', 'test_drop_id_hour.csv', sep=',')
