import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
import joblib
# import feather
import torch


class Dataset_regression():
    '''
    daily 原始特征
        - 数据处理：前值填充，robust
        - 不在此处划分train val，对除了test之外的数据做截断，删除异常值
    '''

    def __init__(self, args, data_path='/data/daily_label_research_v1_extend_factors.csv', flag='train',
                 size=None, train_start_year='2010', train_end_year='2018', test_year='2014', val_start_year='2019',
                 ticker_type=0):
        if size == None:
            self.seq_len = 0
        else:
            self.seq_len = size[0]

        self.flag = flag
        assert flag in ['train', 'test']
        type_map = {'train': 0, 'test': 1}
        self.set_type = type_map[flag]
        self.train_start_year = train_start_year
        self.train_end_year = train_end_year
        self.val_start_year = val_start_year
        self.test_year =test_year
        # if test_year == '2025':
        #     self.test_year = train_end_year
        self.data_path = data_path
        self.ticker_type = ticker_type
        self.args = args
        self.pred_task = 'y' + str(self.args.pred_task)
        self.__read_data__()

    def __get_data__(self):
        df = pd.read_hdf(self.data_path)
        # df = df.iloc[:, :20]
        df = df.replace([-np.inf, np.inf], np.nan)
        df = df.rename(columns={'date': 'CalcDate','ticker':'Code'})
        df['CalcDate'] = pd.to_datetime(df['CalcDate'].astype(str), format='%Y%m%d')
        # df = df.drop(columns='assetClass')
        # columns = df.columns
        feature_cols = ['ret_slp', 'ret', 'close_adj', 'high_adj', 'low_adj', 'open_adj',
                   'tr_ret', 'volume_adj', 'capvol0','vwap_adj']
        # feature_cols = columns[3:]
        # 删异常值
        # df.loc[df['mask_data'] == 0, feature_cols+[self.pred_task]] = np.nan
        df.loc[df[self.pred_task] > 50, 'y10'] = np.nan
        columns = ['CalcDate', 'Code', self.pred_task, 'ret_slp', 'ret', 'close_adj', 'high_adj', 'low_adj', 'open_adj',
                   'tr_ret', 'volume_adj', 'capvol0','vwap_adj']
        df = df[columns]
        if self.data_path == '/data/stock_daily_2005_2021.feather':
            df['ret'] = df.pop('ret')

        # cols = ['main_ret_slp', 'close_adj', 'high_adj', 'low_adj', 'open_adj', 'tr', 'capvol0']
        grouped_df = df.groupby('Code')
        # 填充规则 (保持不变，用于指导 FFILL/0 填充)
        fill_rules = {
            **{c: 0 for c in feature_cols if c.startswith(('ret', 'volume'))},
            **{c: 'ffill' for c in feature_cols if not c.startswith(('ret', 'volume'))}
        }

        # 对所有特征进行序列填充
        for col, rule in fill_rules.items():
            if rule == 'ffill':
                df[col] = grouped_df[col].ffill()
            else:
                df[col] = df[col].fillna(rule)
        # df[feature_cols] = df.groupby('CalcDate')[feature_cols].transform(
        #     lambda x: (x - x.mean()) / x.std())
        df[self.pred_task] = df.groupby('CalcDate')[self.pred_task].transform(
            lambda x: (x - x.mean()) / x.std())

        # def robust_zscore(x):
        #     median = x.median()
        #     mad = (x - median).abs().median()
        #     # 避免除以 0, 如果 mad 为 0 则用 1 代替 (即只去中心化)
        #     return (x - median) / (mad+1e-8)
        # df[feature_cols] = df.groupby('CalcDate')[feature_cols].transform(robust_zscore)
        train_set = pd.DataFrame()
        test_set = pd.DataFrame()
        grouped = df.groupby('Code')
        for _, group in grouped:
            # 删除第一行nan
            group.dropna(subset=group.columns[3:], inplace=True, how='any')

            data = group[
                (group['CalcDate'] >= str(self.train_start_year + '-01-01')) & (
                        group['CalcDate'] <= str(self.train_end_year + '-12-31'))]
            test = group[
                (group['CalcDate'] >= str(self.test_year + '-01-01')) & (
                            group['CalcDate'] <= str(self.test_year + '-12-31'))]

            if len(data) <= self.args.pred_task + self.seq_len - 1:  # 当训练数据不足以生成完整的序列，这里多加了predtask
                if self.seq_len > 1:
                    test = pd.concat([data.iloc[-(self.seq_len - 1):], test])
                    if len(test) <= self.seq_len - 1:  # 如果训练数据不足生成完整序列就打算把他放到test中生成新的test数据，但是需要防止新的test不能生成完整序列，这里如果加入了不完整的数据会导致加载数据错位，这个地方是不是不应该加predtask
                        continue
                    test_set = pd.concat([test_set, test])  # 短数据拼接到test，生成新的test数据
                else:
                    if len(test) <= self.seq_len - 1:
                        continue
                    test_set = pd.concat([test_set, test])
                continue
            # 选取当前 Code 的数据
            Code_data = data.copy()
            if len(Code_data.iloc[:-(self.args.pred_task)]) < self.seq_len:
                if len(test) <= self.seq_len - 1:
                    continue
                else:
                    if self.seq_len > 1:
                        test = pd.concat([data.iloc[-(self.seq_len - 1):], test])  # 为了预测test起始数据
                        test_set = pd.concat([test_set, test])
                    else:
                        test_set = pd.concat([test_set, test])
            else:
                train_data = Code_data.iloc[:-(self.args.pred_task)]
                train_set = pd.concat([train_set, train_data])
                if self.seq_len > 1:
                    test = pd.concat([data.iloc[-(self.seq_len - 1):], test])
                    if len(test) <= self.seq_len - 1:
                        continue
                    test_set = pd.concat([test_set, test])
                else:
                    if len(test) <= self.seq_len - 1:
                        continue
                    test_set = pd.concat([test_set, test])

        nowcast_set = pd.DataFrame()
        nowcast_grouped = train_set.groupby('Code')
        for _, group in nowcast_grouped:
            data = group[
                (group['CalcDate'] >= str(self.train_start_year + '-01-01')) & (
                        group['CalcDate'] <= str(str(int(self.train_end_year) - 1) + '-12-31'))]
            vali = group[
                (group['CalcDate'] >= str(self.train_end_year + '-01-01')) & (
                        group['CalcDate'] <= str(self.train_end_year + '-12-31'))]

            if len(data) <= self.args.pred_task + self.seq_len - 1:  # 当训练数据不足以生成完整的序列，这里多加了predtask
                if self.seq_len > 1:
                    vali = pd.concat([data.iloc[-(self.seq_len - 1):], vali])
                    if len(vali) <= self.seq_len - 1:  # 如果训练数据不足生成完整序列就打算把他放到test中生成新的test数据，但是需要防止新的test不能生成完整序列，这里如果加入了不完整的数据会导致加载数据错位，这个地方是不是不应该加predtask
                        continue
                    nowcast_set = pd.concat([nowcast_set, vali])  # 短数据拼接到test，生成新的test数据
                else:
                    if len(vali) <= self.seq_len - 1:
                        continue
                    nowcast_set = pd.concat([nowcast_set, vali])
                continue
            # 选取当前 Code 的数据
            Code_data = data.copy()
            if len(Code_data.iloc[:-(self.args.pred_task)]) < self.seq_len:
                if len(vali) <= self.seq_len - 1:
                    continue
                else:
                    if self.seq_len > 1:
                        vali = pd.concat([data.iloc[-(self.seq_len - 1):], vali])  # 为了预测test起始数据
                        nowcast_set = pd.concat([nowcast_set, vali])
                    else:
                        nowcast_set = pd.concat([nowcast_set, vali])
            else:
                if self.seq_len > 1:
                    vali = pd.concat([data.iloc[-(self.seq_len - 1):], vali])
                    if len(vali) <= self.seq_len - 1:
                        continue
                    nowcast_set = pd.concat([nowcast_set, vali])
                else:
                    if len(vali) <= self.seq_len - 1:
                        continue
                    nowcast_set = pd.concat([nowcast_set, vali])

        # 对异常值做截断，排除y
        quantiles_label = train_set[self.pred_task].quantile([0.01, 0.99])
        # 提取分位数值（直接通过索引访问）
        q1 = quantiles_label[0.01]  # 0.01 分位数
        q99 = quantiles_label[0.99]  # 0.99 分位数
        # 截断异常值
        train_set[self.pred_task] = np.clip(train_set[self.pred_task], q1, q99)

        quantiles = train_set[feature_cols].quantile([0.05, 0.95])
        quantiles = quantiles.transpose()
        quantiles.columns = ['q5', 'q95']
        for col in feature_cols:
            train_set[col] = np.clip(train_set[col], quantiles.loc[col, 'q5'], quantiles.loc[col, 'q95'])

        scaler = RobustScaler(quantile_range=(5, 95))
        #
        train_set[feature_cols] = scaler.fit_transform(train_set[feature_cols])
        nowcast_set[feature_cols] = scaler.transform(nowcast_set[feature_cols])
        test_set[feature_cols] = scaler.transform(test_set[feature_cols])
        joblib.dump(scaler, f'{self.args.save_path}/robust_scaler.pkl')

        # label_scaler = RobustScaler(quantile_range=(5, 95))
        # train_set[self.pred_task] = label_scaler.fit_transform(train_set[[self.pred_task]])
        # nowcast_set[self.pred_task] = label_scaler.transform(nowcast_set[[self.pred_task]])
        # test_set[self.pred_task] = label_scaler.transform(test_set[[self.pred_task]])
        # joblib.dump(label_scaler, f'{self.args.save_path}/label_robust_scaler.pkl')
        # 计算每个通道的l2
        self.args.c_norms = []
        for c in range(3, 3 + self.args.enc_in):
            self.args.c_norms.append(np.linalg.norm(train_set.iloc[:, c], ord=2))

        return train_set, nowcast_set, test_set

    def __read_data__(self):
        train_data, nowcast_data, test_data = self.__get_data__()
        '''
        df_raw.columns: ['tk', 'y_5', 'y_20',...(features), ...(time_mark)
        '''
        if self.set_type == 0:
            self.train_Code = train_data.iloc[:, 1:2]
            self.train_stamp = train_data.iloc[:, 0:1]
            self.train_set = torch.tensor(train_data.iloc[:, 3:].values)
            #
            # if self.args.pred_task == 10:
            #     self.train_label = torch.tensor(train_data[self.pred_task].values)  # y10
            # elif self.args.pred_task == 20:
            #     self.train_label = torch.tensor(train_data[self.pred_task].values)  # y20
            # elif self.args.pred_task == 5:
            self.train_label = torch.tensor(train_data[self.pred_task].values)  # y5
            self.nowcast_Code = nowcast_data.iloc[:, 1:2]
            self.nowcast_stamp = nowcast_data.iloc[:, 0:1]
            self.nowcast_set = torch.tensor(nowcast_data.iloc[:, 3:].values)
            self.nowcast_label = torch.tensor(nowcast_data[self.pred_task].values)

        else:
            self.test_Code = test_data.iloc[:, 1:2]
            self.test_stamp = test_data.iloc[:, 0:1]
            self.test_set = torch.tensor(test_data.iloc[:, 3:].values)
            # if self.args.pred_task == 10:
            #     self.test_label = torch.tensor(test_data['y10'].values)
            # elif self.args.pred_task == 20:
            #     self.test_label = torch.tensor(test_data['y20'].values)
            # elif self.args.pred_task == 5:
            self.test_label = torch.tensor(test_data[self.pred_task].values)

class Dataset_regression_dataset(Dataset):
    def __init__(self, data_x, data_y, Codes, data_stamp, seq_len,stride=1):
        self.data_x = data_x
        self.data_y = data_y
        self.Codes = Codes
        self.seq_len = seq_len
        self.data_stamp = data_stamp
        self.stride = stride

        # 创建 Code 与其在数据中索引的映射
        self.start_indices = []
        self.Code_indices = {Code: [] for Code in self.Codes['Code'].unique()}
        for i, Code in enumerate(self.Codes['Code']):
            self.Code_indices[Code].append(i)
        for i, (Code, indices) in enumerate(self.Code_indices.items()):
            self.start_indices.append(indices[0])

        self.total_windows = 0
        self.data_length = [0]
        for Code, indices in self.Code_indices.items():
            data_len = len(indices)

            # 计算有效滑动窗口数。
            # 窗口起始点的索引 k 必须满足 k + seq_len <= data_len。
            # 从 0 开始，步长为 stride。
            if data_len >= seq_len:
                # 最后一个窗口起始点的索引是 data_len - seq_len
                # 窗口总数 = floor((data_len - seq_len) / stride) + 1
                num_windows = (data_len - self.seq_len) // self.stride + 1
            else:
                num_windows = 0

            if num_windows > 0:
                self.total_windows += num_windows
            self.data_length.append(self.total_windows)  # 累计窗口数

    def __getitem__(self, index):
        time_gra = [0, 0, 0, 0, 1]
        for i in range(len(self.data_length)):
            if index < self.data_length[i]:
                rel_idx = index - self.data_length[i - 1]  # 该票里的第几个窗口
                s_begin = self.start_indices[i - 1] + rel_idx*self.stride  # ←
                # s_begin = index - self.data_length[i - 1] + self.start_indices[i - 1]  # 前面是算在Code中的相对位置，后面是算在所有数据的位置
                s_end = s_begin + self.seq_len
                seq_x = self.data_x[s_begin:s_end]
                seq_y = self.data_y[s_end - 1]

                time_gra = {'Code': str(self.Codes.iloc[s_end - 1:s_end].values[0]),
                            'CalcDate': str(self.data_stamp['CalcDate'].iloc[s_end - 1:s_end].values[0])}

                return seq_x, seq_y, time_gra

    def __len__(self):
        return self.total_windows

def Dataset_regression_train_val(args):
    dataset = Dataset_regression(args, data_path=args.data_path,
                                 flag='train', size=args.size,
                                 train_start_year=args.train_start_year,
                                 train_end_year=args.train_end_year,
                                 test_year=args.test_year,
                                 )
    seq_len = args.size[0]
    train_dataset = Dataset_regression_dataset(dataset.train_set, dataset.train_label, dataset.train_Code,
                                               dataset.train_stamp, seq_len,stride=2)
    nowcast_dataset = Dataset_regression_dataset(dataset.nowcast_set, dataset.nowcast_label, dataset.nowcast_Code,
                                                 dataset.nowcast_stamp, seq_len)
    nowcast_loader = DataLoader(dataset=nowcast_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                drop_last=False, num_workers=10)
    return train_dataset, nowcast_loader

def Dataset_regression_test(args):
    dataset = Dataset_regression(args, data_path=args.data_path,
                                 flag='test', size=args.size,
                                 train_start_year=args.train_start_year,
                                 train_end_year=args.train_end_year,
                                 test_year=args.test_year,
                                 )
    seq_len = args.size[0]
    test_dataset = Dataset_regression_dataset(dataset.test_set, dataset.test_label, dataset.test_Code,
                                              dataset.test_stamp, seq_len)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                             drop_last=False, num_workers=10)
    return test_dataset, test_loader

