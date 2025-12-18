import sys
sys.path.append('/cpfs/dss/dev/lxjie/hy_stock')
from hy_daily.src.tools import EarlyStopping, adjust_learning_rate
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
from sklearn.preprocessing import RobustScaler
import joblib
import numpy as np
# from torchsummary import summary
from datetime import datetime
from src.metrics import metric
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import hy_daily.data_provider.data_loader_heiyi_daily as data_loader_heiyi_daily
import utils.plt_heiyi as plt_heiyi
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader, ConcatDataset
from exp.exp_basic import Exp_Basic
from scipy.stats import pearsonr
from hy_daily.data_provider.cluter_filter import Clutter_Filter
import ast

warnings.filterwarnings('ignore')


class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.use_phi = loss_params.get('use_phi')

    def get_V_P(self, batch_size, device):
        """
        引入谓词,构造V和P矩阵
        问题：p矩阵和V矩阵是否需要归一化？
        """
        # phi=1
        V = torch.eye(batch_size, device=device)  # 单位矩阵 (batch_size x batch_size)
        # P = torch.ones((batch_size, batch_size), device=device)  # 全1矩阵
        phi = torch.ones((batch_size, 1), device=device)
        phi = phi / torch.linalg.norm(phi)
        P = torch.mm(phi, phi.T)
        return V, P

    def get_x_V_P(self, batch_x, batch_size, device):
        """
        引入谓词,构造V和P矩阵
        问题：p矩阵和V矩阵是否需要归一化？
        """
        # phi=1
        V = torch.eye(batch_size, device=device)  # 单位矩阵 (batch_size x batch_size)
        # P = torch.zeros((batch_size, batch_size), device=device)
        # for m in range(len(c_norms)):
        #     temp_phi = batch_x[:, -1, m]
        #     # temp_phi = batch_x[:, :, m].mean(dim=1)
        #     temp_phi = (temp_phi / c_norms[m]).unsqueeze(0) # 全局归一化
        #     temp_phi = temp_phi / torch.linalg.norm(temp_phi, ord=2)
        #     temp_P = torch.mm(temp_phi.T, temp_phi)
        #     P = P + temp_P

        #phi-x
        # phi = batch_x[:,-1,:]
        # phi = phi / torch.linalg.norm(phi)
        # P = torch.mm(phi,phi.T)

        #phi-ret_mean
        if batch_x.shape[-1] in [4,8,15]:
            phi = batch_x[:, :, 0].mean(dim=1)
        else:
            phi = batch_x[:,:,1].mean(dim=1)
        phi = (phi / torch.linalg.norm(phi)).unsqueeze(0)
        P = torch.mm( phi.T,phi)
        return V, P

    def forward(self, batch_x, outputs, targets, tau_hat=None, tau=None, c_norms=None):
        batch_size = outputs.shape[0]
        device = outputs.device
        # V, P = self.get_ones_V_P(batch_size, device)
        V, P = self.get_x_V_P(batch_x, batch_size, device)
        # P = P + P_ones
        error = outputs - targets
        if tau_hat is None or tau is None:
            weighted_error = error
            return {
                # 'total':  torch.nn.functional.mse_loss(outputs, targets),
                'total': torch.mean(weighted_error ** 2),
                'mse': torch.mean(weighted_error ** 2),
                'V_loss': torch.mean(torch.matmul(V, error) ** 2),
                'P_loss': torch.mean(torch.matmul(P, error) ** 2),
            }
        else:
            weight_matrix = tau_hat * V + tau * P
            weighted_error = torch.matmul(weight_matrix, error)
            return {
                'total': torch.mean(weighted_error ** 2),
                'mse': torch.mean(weighted_error ** 2),
                'V_loss': torch.mean(torch.matmul(tau_hat * V, error) ** 2),
                'P_loss': torch.mean(torch.matmul(tau * P, error) ** 2),
            }


def print_model_parameters(model, only_num=True):
    print('*************************Model Total Parameter************************')
    if not only_num:
        for name, param in model.named_parameters():
            print(name, param.shape, param.requires_grad)
    total_num = sum([param.nelement() for param in model.parameters()])
    print('Total params num: {}'.format(total_num))
    print('************************Finish Parameter************************')


class CCC(nn.Module):
    def __init__(self):
        super(CCC, self).__init__()
        self.cos = nn.CosineSimilarity(dim=0, eps=1e-6)

    def forward(self, y_true, y_pred):
        loss = 1 - self.cos(y_pred - y_pred.mean(dim=0, keepdim=True), y_true - y_true.mean(dim=0, keepdim=True))
        return loss  # 返回 1 减去 CCC 的值作为损失函数


def cosine_similarity(x, y, dim=2, eps=1e-6):
    """
    计算两个张量之间的余弦相似度。

    参数:
        x (Tensor): 形状为 [batch_size, seq_len, dim] 的张量。
        y (Tensor): 形状为 [batch_size, seq_len, dim] 的张量。
        dim (int): 沿着哪个维度计算余弦相似度（默认为最后一个维度）。
        eps (float): 为了防止除以零而添加的小常数。

    返回:
        Tensor: 形状为 [batch_size, seq_len] 的张量，包含每对向量的余弦相似度。
    """
    x_norm = torch.norm(x, dim=dim, keepdim=True)
    y_norm = torch.norm(y, dim=dim, keepdim=True)
    product = (x * y).sum(dim=dim, keepdim=True)
    cosine = product / (x_norm * y_norm + eps)
    return cosine.squeeze(dim=dim)


def interval_loss(y_true, y_pred):
    """
    区间损失函数，包括区间覆盖损失和区间宽度损失
    y_pred[:, 0] 是区间下界，y_pred[:, 1] 是区间上界
    """
    # 区间覆盖损失：如果真实值不在预测区间内，则惩罚
    coverage_loss = torch.where(
        (y_true >= y_pred[:, 0]) & (y_true <= y_pred[:, 1]),
        torch.tensor(0.0, device=y_true.device),
        torch.min((y_true - y_pred[:, 0]) ** 2, (y_true - y_pred[:, 1]) ** 2)
    ).mean()

    # 区间宽度损失：惩罚过宽的区间
    width_loss = ((y_pred[:, 1] - y_pred[:, 0]) ** 2).mean()

    # 总损失是覆盖损失和宽度损失的加权和
    total_loss = coverage_loss + 0.5 * width_loss
    return total_loss


def cosine_loss(x, y, dim=1, reduction='mean'):
    """
    计算余弦损失函数。

    参数:
        x (Tensor): 形状为 [batch_size, seq_len, dim] 的张量。
        y (Tensor): 形状为 [batch_size, seq_len, dim] 的张量。
        dim (int): 沿着哪个维度计算余弦相似度（默认为最后一个维度）。
        reduction (str): 指定如何减少损失：'none' | 'mean' | 'sum'。

    返回:
        Tensor: 如果 reduction 不是 'none'，则返回一个标量损失值；否则返回与输入形状相同的张量。
    """
    x = x[:, -1]
    y = y[:, -1]
    cosine_sim = cosine_similarity(x, y, dim)
    if reduction == 'none':
        return 1 - cosine_sim
    elif reduction == 'mean':
        return 1 - cosine_sim.mean()
    elif reduction == 'sum':
        return 1 - cosine_sim.sum()
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")


def correlation_loss(x, y):
    # x = x[:,-1:]
    # y = y[:,-1:]
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    numerator = torch.sum((x - x_mean) * (y - y_mean))
    denominator = torch.sqrt(torch.sum((x - x_mean) ** 2) * torch.sum((y - y_mean) ** 2))
    correlation_coefficient = numerator / denominator
    loss = 1 - correlation_coefficient
    return loss

class TDataset(Dataset):

    def __init__(self, X, y, time_gra):
        self.X = X
        self.time_gra = time_gra
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.time_gra[idx]

class Exp_Multiple_Regression_Fold(Exp_Basic):
    def __init__(self, args, single_fold=None):
        super(Exp_Multiple_Regression_Fold, self).__init__(args)
        self.all_test_preds = np.array([])
        self.single_fold = single_fold  # 如果指定则只训练单个fold
        # self.all_test_trues = []
        # self.args = args
        # self.device = self._acquire_device()
        # self.model = model.to(self.device)

    def _build_model(self):
        # model init
        self.model = self.model_dict[self.args.model].Model(self.args).float().to(self.device)
        if self.args.use_multi_gpu and self.args.use_gpu:
            # self.model = nn.DataParallel(self.model, device_ids=[3,0,1,2])
            self.model = nn.DataParallel(self.model, device_ids=self.args.device_ids)
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data.shape)
        # sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # print("Number of Parameters:", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        return self.model

    def _acquire_device(self):
        if self.args.use_gpu:
            # os.environ["CUDA_VISIBLE_DEVICES"] = str(
            #     self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def __init_normal_(self, model, init_type='norm '):
        if init_type == 'norm':
            for m in model.parameters():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight.data)
                    nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.LSTM):
                    nn.init.xavier_normal_(m.weight_ih_l0)
                    nn.init.orthogonal_(m.weight_hh_l0)
                    nn.init.constant_(m.bias_ih_l0, 0)
                    nn.init.constant_(m.bias_hh_l0, 0)
        elif init_type == 'unif':
            for p in model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.uniform_(p)

    def _get_data(self, flag):
        self.args.size = [self.args.seq_len]
        if self.args.data_type == 'daily':
            if flag == 'train':
                train_dataset,nowcast_dataset = data_loader_heiyi_daily.Dataset_regression_train_val(self.args)
                return train_dataset,nowcast_dataset
            elif flag == 'test':
                test_dataset, test_loader = data_loader_heiyi_daily.Dataset_regression_test(self.args)
                return test_dataset, test_loader

    def _select_optimizer(self):
        optim_type = self.args.optim_type
        if optim_type == 'SGD':
            model_optim = optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=0.9,
                                    weight_decay=self.args.weight_decay)
        elif optim_type == 'Adam':
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                                     weight_decay=self.args.weight_decay)
        else:
            raise ValueError("can't find your optimizer! please defined a optimizer!")
        scheduler = None
        if self.args.lradj == 'cos':
            scheduler = lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.train_epochs // 2)
        elif self.args.lradj == 'steplr':
            scheduler = lr_scheduler.StepLR(model_optim, step_size=2, gamma=0.5)
        return model_optim, scheduler

    def _select_criterion(self):
        loss_func = self.args.loss
        if loss_func == 'MSE':
            criterion = nn.MSELoss(reduction='none')
        elif loss_func == 'MAE':
            criterion = nn.L1Loss()
        elif loss_func == 'SmoothL1Loss':
            criterion = nn.SmoothL1Loss()
        elif loss_func == 'ccc':
            criterion = CCC()
        elif loss_func == 'Huber':
            criterion = torch.nn.HuberLoss(reduction='mean', delta=self.args.delta)
        elif loss_func == 'MSE_with_weak':
            criterion = WeightedMSELoss()
        else:
            raise ValueError("can't find your loss function! please defined it!")
        return criterion

    def mean_std_loss(self,outputs, batch_y, mask, lambda_std=1.0):
        product = outputs[~mask] * batch_y[~mask]
        mean_term = -torch.mean(product)  # 最大化均值
        std_term = torch.std(product) + 1e-8  # 最小化标准差
        return mean_term + lambda_std * std_term

    def prepared_dataset(self, train_data):
        data_x = []
        data_y = []
        dates = []
        tickers = []
        temp_loader = DataLoader(dataset=train_data, batch_size=self.args.batch_size, shuffle=False, pin_memory=True,
                                 drop_last=False, num_workers=4)
        for i, (batch_x, batch_y, time_gra) in enumerate(temp_loader):
            data_x.append(batch_x)
            data_y.append(batch_y)
            d = [datetime.strptime(t[:26], '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y-%m-%d%H:%M:%S') if isinstance(t,
                                                                                                              str) else t
                 for t in time_gra['CalcDate']]
            ticker = [t.strip("[]' ") for t in time_gra['Code']]
            dates.append(d)
            tickers.append(ticker)
        data_x = torch.cat(data_x, dim=0)
        data_y = torch.cat(data_y, dim=0)
        dates = np.concatenate(dates)
        tickers = np.concatenate(tickers)

        # 获取所有唯一日期并排序
        unique_dates = np.unique(dates)
        sorted_dates = np.sort(unique_dates)
        num_dates = len(sorted_dates)
        return num_dates, sorted_dates, data_x, data_y, dates

    def load_dataset(self, num_dates, fold, sorted_dates, data_x, data_y, dates):
        # 计算当前fold的验证集日期范围
        fold_size = num_dates // self.args.num_fold
        start_idx = fold * fold_size
        end_idx = (fold + 1) * fold_size if fold != self.args.num_fold - 1 else num_dates
        val_dates = sorted_dates[start_idx:end_idx]
        # train_dates = np.concatenate([sorted_dates[:start_idx], sorted_dates[end_idx:]])
        if (fold + 1) == 1:
            train_dates = np.concatenate([sorted_dates[:start_idx], sorted_dates[end_idx + self.args.pred_task:]])
        elif (fold + 1) > 1 and (fold + 1) < self.args.num_fold:
            train_dates = np.concatenate(
                [sorted_dates[:start_idx - self.args.pred_task], sorted_dates[end_idx + self.args.pred_task:]])
        elif (fold + 1) == self.args.num_fold:
            train_dates = np.concatenate([sorted_dates[:start_idx - self.args.pred_task], sorted_dates[end_idx:]])
        # 创建mask
        val_mask = np.isin(dates, val_dates)
        train_mask = np.isin(dates, train_dates)
        # 分割数据
        train_set_x = data_x[train_mask]
        train_set_y = data_y[train_mask]
        val_set_x = data_x[val_mask]
        val_set_y = data_y[val_mask]
        dates_train_x = dates[train_mask]
        dates_val_x = dates[val_mask]

        # 创建数据集和数据加载器
        train_dataset = TDataset(train_set_x, train_set_y, dates_train_x)
        val_dataset = TDataset(val_set_x, val_set_y, dates_val_x)
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, pin_memory=True,
                                  drop_last=False, num_workers=4)
        vali_loader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False, pin_memory=True,
                                 drop_last=False, num_workers=4)
        return train_dataset, train_loader, val_dataset, vali_loader

    def load_dataset_time(self, num_dates, fold, sorted_dates, data_x, data_y, dates):
        val_size = int(0.2 * num_dates)
        each_train_size = (num_dates - val_size) // self.args.num_fold
        start_idx = fold * each_train_size  # train idx
        end_idx = num_dates - val_size
        # 去掉val前面的
        val_dates = sorted_dates[end_idx + self.args.pred_task:]
        train_dates = sorted_dates[start_idx:end_idx]

        # 创建mask
        val_mask = np.isin(dates, val_dates)
        train_mask = np.isin(dates, train_dates)
        # 分割数据
        train_set_x = data_x[train_mask]
        train_set_y = data_y[train_mask]
        val_set_x = data_x[val_mask]
        val_set_y = data_y[val_mask]
        dates_train_x = dates[train_mask]
        dates_val_x = dates[val_mask]

        # 创建数据集和数据加载器
        train_dataset = TDataset(train_set_x, train_set_y, dates_train_x)
        val_dataset = TDataset(val_set_x, val_set_y, dates_val_x)
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, pin_memory=True,
                                  drop_last=False, num_workers=4)
        vali_loader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False, pin_memory=True,
                                 drop_last=False, num_workers=4)
        return train_dataset, train_loader, val_dataset, vali_loader

    def load_dataset_last(self, num_dates, fold, sorted_dates, data_x, data_y, dates):
        # 计算当前fold的验证集日期范围 最后20%
        fold_size = int(0.2 * num_dates)
        start_idx = 0
        end_idx = num_dates - fold_size  # train idx
        train_dates = sorted_dates[start_idx:end_idx - self.args.pred_task]
        val_dates = np.concatenate([sorted_dates[:start_idx], sorted_dates[end_idx:]])
        # 创建mask
        train_mask = np.isin(dates, train_dates)
        val_mask = np.isin(dates, val_dates)
        # 分割数据
        train_set_x = data_x[train_mask]
        train_set_y = data_y[train_mask]
        val_set_x = data_x[val_mask]
        val_set_y = data_y[val_mask]
        dates_train_x = dates[train_mask]
        dates_val_x = dates[val_mask]

        # 创建数据集和数据加载器
        train_dataset = TDataset(train_set_x, train_set_y, dates_train_x)
        val_dataset = TDataset(val_set_x, val_set_y, dates_val_x)
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, pin_memory=True,
                                  drop_last=False, num_workers=4)
        vali_loader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False, pin_memory=True,
                                 drop_last=False, num_workers=4)
        return train_dataset, train_loader, val_dataset, vali_loader

    def train(self, setting):
        train_data,nowcast_loader = self._get_data(flag='train')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        val_ratio = 1.0 / self.args.num_fold
        print(f'val_ratio:{val_ratio}\n')
        print_model_parameters(self.model)

        num_dates, sorted_dates, data_x, data_y, dates = self.prepared_dataset(train_data)

        print('__________ Start training !____________')
        start_training_time = time.time()

        best_train_corr_list = []
        best_val_losses = []
        best_val_corr_list = []
        best_val_metric_list = []
        best_val_sr_list = []
        nowcast_corr_list = []

        # 如果指定了single_fold，只训练该fold
        fold_range = [self.single_fold] if self.single_fold is not None else range(self.args.num_fold)
        
        for fold in fold_range:
            start_fold_time = time.time()
            print(f"Training fold {fold + 1}/{self.args.num_fold}")

            if self.args.num_fold == 1:
                train_dataset, train_loader, val_dataset, vali_loader = self.load_dataset_last(num_dates, fold,
                                                                                               sorted_dates, data_x,
                                                                                               data_y, dates)
            else:
                train_dataset, train_loader, val_dataset, vali_loader = self.load_dataset(num_dates, fold, sorted_dates,
                                                                                          data_x, data_y, dates)
            train_steps = len(train_loader)
            self.model = self._build_model()  # 每个折叠重新初始化模型
            model_optim, scheduler = self._select_optimizer()  # 每次初始化模型后也要重新初始化优化器和调度器
            criterion = self._select_criterion()
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=self.args.pct_start,
                                                epochs=self.args.train_epochs,
                                                max_lr=self.args.learning_rate)

            if self.args.use_amp:
                scaler = torch.cuda.amp.GradScaler()

            early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

            best_epoch = 0
            best_train_corr = -1
            best_val_metric = -999
            best_val_corr = -1
            best_val_sr = -1
            best_train_mse = 999
            best_val_mse = 999
            # Logging setup for this fold
            log_file_path = f'{self.args.save_path}/training_logs_fold_{fold + 1}.txt'
            with open(log_file_path, 'w') as file:
                file.write('Item\tTrain Loss\tBatch Correlation\n')

            with open(f'{self.args.save_path}/_result_of_multiple_regression.txt', 'a') as file:
                file.write(f'training fold {fold + 1}\n')

            for epoch in range(self.args.train_epochs):

                iter_count = 0
                train_loss_list = []
                # batch_corr = torch.tensor([], device=self.device)
                preds_list = []
                trues_list = []
                # corr_loss = []
                # mse_loss_list = []
                self.model.train()
                epoch_time = time.time()

                for i, (batch_x, batch_y, time_gra) in enumerate(train_loader):
                    if i == 0: print(batch_x.shape, batch_y.shape)
                    if batch_x.shape[0] <= 1:
                        continue
                    iter_count += 1
                    model_optim.zero_grad()
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    if self.args.model == 'AMD' or 'Path' in self.args.model:
                        if self.args.enc_in in [3,7,14]:
                            outputs, moe_loss = self.model(batch_x[:, :, 1:])
                        else:
                            outputs, moe_loss = self.model(batch_x)
                    else:
                        if self.args.enc_in in [3,7,14]:
                            outputs = self.model(batch_x[:, :, 1:])
                        else:
                            outputs = self.model(batch_x)

                    if self.args.task_name == 'Long_term_forecasting':
                        f_dim = 1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim]

                    outputs = outputs.squeeze(-1)
                    # batch_y = batch_y[:,label_index]
                    batch_y = batch_y.squeeze(-1).squeeze(-1)
                    if i == 0: print('output and batch_y', outputs.shape, batch_y.shape)
                    # true_trend = batch_y[:, 1:] - batch_y[:, :-1]
                    # # 预测序列的趋势变化
                    # pred_trend = outputs[:, 1:] - outputs[:, :-1]
                    #
                    # # 使用符号函数来判断趋势方向是否一致
                    # trend_consistency = torch.sign(pred_trend) * torch.sign(true_trend)
                    # # 计算趋势一致性损失
                    # trend_loss = torch.mean(1 - trend_consistency)
                    mask = torch.zeros(batch_y.shape, dtype=torch.bool)
                    if batch_y.isnan().sum() > 0:
                        mask = torch.isnan(batch_y)
                    if mask.all():
                        continue
                    # sr_loss = self.mean_std_loss(outputs, batch_y,mask)
                    if self.args.loss == 'MSE_with_weak':
                        tau_hat = torch.sigmoid(self.model.alpha)
                        tau = 1 - tau_hat
                        loss_dict = criterion(batch_x[~mask], outputs[~mask], batch_y[~mask], tau_hat, tau,
                                              self.args.c_norms)
                        mse = loss_dict['total']
                    else:
                        # loss = interval_loss(batch_y[~mask],outputs[~mask])
                        mse = torch.mean(criterion(outputs[~mask], batch_y[~mask]))
                    if self.args.model == 'AMD' or 'Path' in self.args.model:
                        loss = moe_loss + mse
                    else:
                        loss = mse
                    train_loss_list.append(torch.tensor([loss.item()], device=self.device))

                    if self.args.task_name == 'Long_term_forecasting':
                        labels = torch.sum(batch_y, dim=1)
                        preds_labels = torch.sum(outputs, dim=1)
                        corr = torch.corrcoef(torch.stack([preds_labels, labels]))[0, 1]
                        preds_list.append(preds_labels.detach())
                        trues_list.append(labels.detach())
                    elif self.args.task_name == 'classification':
                        _, temp = torch.max(outputs, dim=1)
                        corr = ((batch_y[~mask] == temp[~mask]).sum()) / (batch_y.size(0) - mask.sum())  # nan值不参与计算
                    else:
                        corr = \
                            torch.corrcoef(torch.stack([outputs[~mask].reshape(-1), batch_y[~mask].reshape(-1)]))[
                                0, 1]

                    # mse_loss_list.append(torch.tensor([mse.item()], device=self.device))

                    with open(f'{self.args.save_path}/training_logs.txt', 'a') as file:
                        file.write(f'{i}\t{loss:.4f}\t{corr:.4f}\n')
                    if (i == 0) or ((i + 1) % 1000 == 0):
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f} | corr: {3:.8f}".format(i + 1, epoch + 1,
                                                                                                loss.item(), corr))

                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        # sr_loss.backward(retain_graph=True)
                        # model_optim.step()
                        # model_optim.zero_grad()

                        loss.backward()
                        model_optim.step()

                        if self.args.grad_norm:
                            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_value)  # 进行梯度裁剪
                        scheduler.step()

                # Epoch end statistics
                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                train_loss, corr, sr = self.vali(train_loader,criterion,
                                                 fold)  # 保持和val一致，每个epoch模型固定后train的corr
                vali_loss, vali_corr, vali_sr = self.vali(vali_loader,criterion, fold)
                # test_loss, test_corr = self.vali(test_loader, criterion,epoch,fold)
                # train_loss = torch.mean(torch.cat(train_loss_list).to(self.device))
                mse_loss = train_loss  # torch.mean(torch.cat(mse_loss_list).to(self.device)) # amd模型和path模型还有moe loss，没有加在里面

                # if vali_corr > best_val_corr and early_triggered == False: # 当还没有早停保存最优模型，早停触发后不再保存模型
                #     best_epoch = epoch + 1
                #     best_val_corr = vali_corr
                #     best_model_path = f'{path}/best_model_fold_{fold+1}.pth'
                #     torch.save(self.model.state_dict(), best_model_path)

                print(
                    f"Epoch {epoch + 1} | Train Loss: {train_loss:.7f} | mse:{mse_loss:.8f} | Train Corr: {corr:.8f} "
                    f"| Val Loss: {vali_loss:.8f} | Val Corr: {vali_corr:.8f} | Val Sr: {vali_sr:.8f}")

                with open(f'{self.args.save_path}/_result_of_multiple_regression.txt', 'a') as file:
                    file.write(f"Epoch {epoch + 1} | Train Loss: {train_loss:.7f} | Train Corr: {corr:.8f} "
                               f"| Val Loss: {vali_loss:.8f} | Val Corr: {vali_corr:.8f} | Val Sr: {vali_sr:.8f}\n")
                # Early stopping

                # if 0.01 < (best_val_mse-vali_loss)/best_val_mse or (vali_corr+vali_sr)>best_val_metric:  # 当还没有早停保存最优模型，早停触发后不再保存模型
                #     best_epoch = epoch + 1
                #     best_train_corr = corr
                #     best_val_corr = vali_corr
                #     best_val_sr = vali_sr
                #     best_train_mse = train_loss
                #     best_val_mse = vali_loss
                #     best_val_metric = vali_corr+vali_sr

                if vali_loss < best_val_mse:  # 当还没有早停保存最优模型，早停触发后不再保存模型
                    best_epoch = epoch + 1
                    best_train_corr = corr
                    best_val_corr = vali_corr
                    best_val_sr = vali_sr
                    best_train_mse = train_loss
                    best_val_mse = vali_loss
                    best_val_metric = vali_corr + vali_sr

                if self.args.task_name == 'Long_term_forecasting':
                    early_stopping(-vali_loss, vali_sr + vali_corr, self.model, path)
                else:
                    early_stopping(vali_loss, self.model,
                                   f'{path}/best_model_fold_{fold + 1}.pth')  # vali_corr vali_loss

                if early_stopping.early_stop:
                    print("Early stopping triggered.")
                    break
                    # early_triggered = True
                if self.args.lradj != 'not':
                    adjust_learning_rate(model_optim, epoch + 1, self.args)
                # print(f'save...{epoch+1}...model...')
                # torch.save(self.model.state_dict(), f'{path}/{fold+1}_epoch{epoch+1}_model.pth')

                # Fold summary
                # print(f"Best validation correlation for fold {fold+1}: {best_val_corr} at epoch {best_epoch}")
                # best_val_corrs = torch.cat([best_val_corrs, best_val_corr.unsqueeze(0)])
                # fold_time = (time.time() - start_fold_time) / 60
                # with open(f'{self.args.save_path}/_result_of_multiple_regression.txt', 'a') as file:
                #         file.write(f'Best validation correlation for fold {fold+1}: {best_val_corr} at epoch {best_epoch}\nfold{fold+1} training time: {fold_time:.2f} minutes\n')
                # if best_val_corr > best_val_corr_overall:
                #     best_val_corr_overall = best_val_corr

                # print(f"Best validation loss for fold {fold+1}: {best_val_mse} at epoch {best_epoch}")
            best_val_losses.append(best_val_mse.cpu().numpy())
            best_train_corr_list.append(best_train_corr.cpu().numpy())
            best_val_corr_list.append(best_val_corr.cpu().numpy())
            best_val_metric_list.append(best_val_metric.cpu().numpy())
            best_val_sr_list.append(best_val_sr.cpu().numpy())
            fold_time = (time.time() - start_fold_time) / 60
            _, nowcast_corr, vali_sr = self.vali(nowcast_loader, criterion, fold)
            nowcast_corr_list.append(nowcast_corr.cpu().numpy())
            print(
                f"best train corr: {best_train_corr:.6f} | best train mse: {best_train_mse} | nowcast corr: {nowcast_corr:.6f} | best val sr: {best_val_sr:.6f} | best val corr: {best_val_corr:.6f} | best val mse: {best_val_mse}")

            with open(f'{self.args.save_path}/_result_of_multiple_regression.txt', 'a') as file:
                file.write(
                    f'best train corr: {best_train_corr:.6f}\n Best validation metric for fold {fold + 1}: {best_val_sr} at epoch {best_epoch}\nfold{fold + 1} training time: {fold_time:.2f} minutes\n')

            # Final training summary
        total_time = (time.time() - start_training_time) / 60

        # 保存单折训练结果
        if self.single_fold is not None:
            result_path = f'{self.args.save_path}/fold_{self.single_fold + 1}_results.npy'
            np.save(result_path, {
                'best_train_corr': best_train_corr_list[0] if best_train_corr_list else None,
                'best_val_loss': best_val_losses[0] if best_val_losses else None,
                'best_val_corr': best_val_corr_list[0] if best_val_corr_list else None,
                'best_val_metric': best_val_metric_list[0] if best_val_metric_list else None,
                'best_val_sr': best_val_sr_list[0] if best_val_sr_list else None,
                'nowcast_corr': nowcast_corr_list[0] if nowcast_corr_list else None,
            })
            print(f"Fold {self.single_fold + 1} training completed in {total_time:.2f} minutes")
        else:
            # 多折训练汇总
            with open(f'{self.args.save_path}/_result_of_multiple_regression.txt', 'a') as f:
                f.write(
                    f'Total training time: {total_time:.2f} minutes\n best train corr: {np.mean(best_train_corr_list)}\n best val loss:{np.mean(best_val_losses)} best val corr:{np.mean(best_val_corr_list):.6f} best val metric: {np.mean(best_val_metric_list):.6f}\n')
            print(f"Total training time: {total_time:.2f} minutes")
            print(f"average best train corr: {np.mean(best_train_corr_list):.6f}")
            print(f"average best val loss: {np.mean(best_val_losses):.6f}")
            print(f"average best val corr: {np.mean(best_val_corr_list):.6f}")
            print(f"average best val sr: {np.mean(best_val_sr_list):.6f}")
            print(f"average best val metric: {np.mean(best_val_metric_list):.6f}")
            print(f"average nowcast corr: {np.mean(nowcast_corr_list):.6f}")
        # Load the best model after training
        # self.model.load_state_dict(torch.load(best_model_path))

        return self.model
    

    def vali(self, vali_loader, criterion, fold):
        total_loss_list = []
        self.model.eval()
        preds_list = []
        trues_list = []
        y_times = []
        # batch_corr = []
        with torch.no_grad():
            for i, (batch_x, batch_y, time_gra) in enumerate(vali_loader):
                if batch_x.shape[0] <= 1:
                    continue
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if self.args.model == 'AMD' or 'Path' in self.args.model:
                    if self.args.enc_in in [3, 7, 14]:
                        outputs, moe_loss = self.model(batch_x[:, :, 1:])
                    else:
                        outputs, moe_loss = self.model(batch_x)
                else:
                    if self.args.enc_in in [3, 7, 14]:
                        outputs = self.model(batch_x[:, :, 1:])
                    else:
                        outputs = self.model(batch_x)

                if self.args.task_name == 'Long_term_forecasting':
                    f_dim = 1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim]
                outputs = outputs.squeeze(-1)
                batch_y = batch_y.squeeze(-1).squeeze(-1)
                mask = torch.zeros(batch_y.shape, dtype=torch.bool)
                if batch_y.isnan().sum() > 0:
                    mask = torch.isnan(batch_y)
                if mask.all():
                    continue
                # loss = criterion(outputs[~mask], batch_y[~mask])
                if self.args.loss == 'MSE_with_weak':
                    tau_hat = torch.sigmoid(self.model.alpha)
                    tau = 1 - tau_hat
                    loss_dict = criterion(batch_x[~mask], outputs[~mask], batch_y[~mask], tau_hat, tau,
                                          self.args.c_norms)
                    mse = loss_dict['total']
                else:
                    # loss = interval_loss(batch_y[~mask],outputs[~mask])
                    mse = torch.mean(criterion(outputs[~mask], batch_y[~mask]))
                if self.args.model == 'AMD' or 'Path' in self.args.model:
                    loss = moe_loss + mse
                else:
                    loss = mse
                pred = outputs.detach()
                true = batch_y.detach()
                total_loss_list.append(torch.tensor([loss.item()]).to(self.device))
                pred = pred.squeeze(-1)
                true = true.squeeze(-1)
                if self.args.task_name == 'Long_term_forecasting':
                    pred = torch.sum(pred, dim=1)
                    true = torch.sum(true, dim=1)
                if true.dim() == 0:
                    pred = pred.unsqueeze(0)  # 确保至少是一维
                    true = true.unsqueeze(0)
                # pred = pred[:,-1]
                # true = true[:,-1]
                preds_list.append(pred)
                trues_list.append(true)

            # golbel_scaler = joblib.load(f'{self.args.save_path}/{fold}_robust_scaler.pkl')
            total_loss = torch.mean(torch.cat(total_loss_list))
            preds = torch.cat(preds_list).to(self.device)
            trues = torch.cat(trues_list).to(self.device)
            # cos_loss = torch.cat(cos_loss_list)
            # preds = preds * golbel_scaler.scale_[0] + golbel_scaler.center_[0]
            # trues = trues * golbel_scaler.scale_[0] + golbel_scaler.center_[0]
            valid_mask = ~torch.isnan(preds) & ~torch.isnan(trues)
            # 过滤掉 NaN 值
            preds_filtered = preds[valid_mask]
            trues_filtered = trues[valid_mask]
            # total_loss = criterion(preds_filtered, trues_filtered).item()
            # cos_loss = torch.mean(cos_loss)
            # print('vali shape:', preds.shape, trues.shape)
            vali_corr = torch.corrcoef(torch.stack([preds_filtered.reshape(-1), trues_filtered.reshape(-1)]))[0, 1]
            vali_sr = (preds_filtered * trues_filtered).mean() / (trues_filtered * preds_filtered).std()
            self.model.train()
            return total_loss, vali_corr, vali_sr

    def epoch_test(self, test_loader, criterion, epoch, fold):
        total_loss_list = []
        preds = []
        trues = []
        y_tickers = np.array([], dtype=str)
        y_times = np.array([], dtype=str)

        mse_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, time_gra) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if self.args.model == 'AMD' or 'Path' in self.args.model:
                    if self.args.enc_in in [3, 7, 14]:
                        outputs, moe_loss = self.model(batch_x[:, :, 1:])
                    else:
                        outputs, moe_loss = self.model(batch_x)
                else:
                    if self.args.enc_in in [3, 7, 14]:
                        outputs = self.model(batch_x[:, :, 1:])
                    else:
                        outputs = self.model(batch_x)

                if self.args.task_name == 'Long_term_forecasting':
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                outputs = torch.mean(outputs, dim=-1)
                outputs = outputs.squeeze(-1)
                batch_y = batch_y.squeeze(-1)
                if i == 0: print('output and batch_y', outputs.shape, batch_y.shape)

                mask = torch.zeros(batch_y.shape, dtype=torch.bool)
                if batch_y.isnan().sum() > 0:
                    mask = torch.isnan(batch_y)

                # outputs = outputs * scalers.scale_[0] + scalers.center_[0]
                # batch_y = batch_y * scalers.scale_[0] + scalers.center_[0]
                mse_loss.append(criterion(outputs[~mask], batch_y[~mask]))

                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                preds = np.append(preds, pred)
                trues = np.append(trues, true)

                y_ticker = time_gra['ticker']
                y_time = time_gra['time']
                # 优化日期格式转换
                y_time = [
                    datetime.strptime(t[:26], '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y-%m-%d%H:%M:%S') if isinstance(t,
                                                                                                                 str) else t
                    for t in y_time]

                # 优化字符串处理
                y_ticker = [t.strip("[]' ") for t in y_ticker]

                # 扩展列表
                y_tickers = np.concatenate([y_tickers, y_ticker])
                y_times = np.concatenate([y_times, y_time])

            y_tickers = np.array(y_tickers)
            y_times = np.array(y_times)

            mse_loss_cpu = [loss.cpu().numpy() for loss in mse_loss]
            # np.save('mse_loss.npy', mse_loss_cpu)
            # np.save(f'{self.args.save_path}/true', trues)
            # np.save(f'{self.args.save_path}/pred', preds)

            mse = np.average(mse_loss_cpu)
            # print('test data mse: ', mse)

            mask = np.isnan(trues)
            corr = np.corrcoef(preds[~mask], trues[~mask])[0, 1]  # 所有test的corr（拼接完一起）1折的
            # print('the  test corr result is {}'.format(corr))

            if self.all_test_preds.size == 0:
                # 将 self.all_test_preds 初始化为二维数组
                self.all_test_preds = preds.reshape(1, -1)
            else:
                self.all_test_preds = np.concatenate((self.all_test_preds, preds.reshape(1, -1)))

            data = {'ticker': y_tickers, 'date': y_times, 'True Values': trues, 'Predicted Values': preds}
            df = pd.DataFrame(data)

            # Define the path where you want to store the CSV file
            csv_file_path = self.args.save_path + '/' + self.args.model + self.args.task_name + self.args.test_year + f'predicted_true_values_{fold + 1}_epoch{epoch + 1}.csv'

            # Write the DataFrame to a CSV file
            # if os.path.exists(csv_file_path):
            #     # Load the existing data from the CSV file
            #     existing_df = pd.read_csv(csv_file_path)
            #
            #     # Combine the existing data with the new data
            #     combined_df = pd.concat([existing_df, df], ignore_index=True)
            #
            #     # Write the combined DataFrame back to the CSV file
            #     combined_df.to_csv(csv_file_path, index=False)
            #
            #     print("New data appended to the existing file.")
            # else:
            #     # Write the DataFrame to a new CSV file
            df.to_csv(csv_file_path, index=False)

            print("created with the true and predicted values.")

            print("True and predicted values have been saved to:", csv_file_path)

        return mse, corr

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        print("\n开始测试所有5折模型...\n")
        for fold in range(self.args.num_fold):
            # print(f'cycle:{cycle}-epoch:{epoch}')
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join(self.args.checkpoints + '/' + setting, f'best_model_fold_{fold + 1}.pth')),
                strict=True)
            # scalers = joblib.load(f'{self.args.save_path}/robust_scaler.pkl')
            # self.model.load_state_dict(torch.load(f'results/y10/PatchTST_NNN_5_task_namemultiple_regression_ticker_type2PatchTST_test_year2020_feaFalse_kfoldTrue_seq120_pred1_freqd_ep50_bs64_early5_lr0.0001_wd0_block1024_nl6_nh12_ne_768_era_dp0.1_0.1_0.1_M_invFalse_dmo128_dff256_pl8_sr4/NNN_5_task_namemultiple_regression_ticker_type2PatchTST_test_year2020_feaFalse_kfoldTrue_seq120_pred1_freqd_ep50_bs64_early5_lr0.0001_wd0_block1024_nl6_nh12_ne_768_era_dp0.1_0.1_0.1_M_invFalse_dmo128_dff256_pl8_sr4/best_model_fold_{fold+1}.pth'))
            criterion = self._select_criterion()
            folder_path = self.args.save_path + '/multi_reg_results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            preds = []
            trues = []
            y_tickers = np.array([], dtype=str)
            y_times = np.array([], dtype=str)

            mse_loss = []
            self.model.eval()
            with torch.no_grad():
                for i, (batch_x, batch_y, time_gra) in enumerate(test_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    if self.args.model == 'AMD' or 'Path' in self.args.model:
                        if self.args.enc_in in [3,7,14]:
                            outputs, moe_loss = self.model(batch_x[:, :, 1:])
                        else:
                            outputs, moe_loss = self.model(batch_x)
                    else:
                        if self.args.enc_in in [3,7,14]:
                            outputs = self.model(batch_x[:, :, 1:])
                        else:
                            outputs = self.model(batch_x)

                    if self.args.task_name == 'Long_term_forecasting':
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                    outputs = outputs.squeeze(-1)
                    batch_y = batch_y.squeeze(-1)
                    if i == 0: print('output and batch_y', outputs.shape, batch_y.shape)

                    mask = torch.zeros(batch_y.shape, dtype=torch.bool)
                    if batch_y.isnan().sum() > 0:
                        mask = torch.isnan(batch_y)

                    # outputs = outputs * scalers.scale_[0] + scalers.center_[0]
                    # batch_y = batch_y * scalers.scale_[0] + scalers.center_[0]
                    # mse_loss.append(criterion(outputs[~mask], batch_y[~mask]))

                    pred = outputs.detach().cpu().numpy()
                    true = batch_y.detach().cpu().numpy()
                    # pred = pred[:,-1]
                    # true = true[:,-1]
                    preds = np.append(preds, pred)
                    trues = np.append(trues, true)

                    y_ticker = time_gra['Code']
                    y_time = time_gra['CalcDate']
                    # 优化日期格式转换
                    y_time = [
                        datetime.strptime(t[:26], '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y-%m-%d') if isinstance(t,
                                                                                                             str) else t
                        for t in y_time]

                    # 优化字符串处理
                    y_ticker = [t.strip("[]' ") for t in y_ticker]

                    # 扩展列表
                    y_tickers = np.concatenate([y_tickers, y_ticker])
                    y_times = np.concatenate([y_times, y_time])

                y_tickers = np.array(y_tickers)
                y_times = np.array(y_times)
                # golbel_scaler = joblib.load(f'{self.args.save_path}/label_robust_scaler.pkl')
                # mse_loss_cpu = [loss.cpu().numpy() for loss in mse_loss]
                # preds = preds * golbel_scaler.scale_[0] + golbel_scaler.center_[0]
                # trues = trues * golbel_scaler.scale_[0] + golbel_scaler.center_[0]
                # 计算每个ticker的相关系数
                # test_corr,pred_list,true_list = every_ticker_norr(self.args, test_path = self.args.data_path,test_year=self.args.test_year,
                #                                                 preds=preds,trues=trues,seq_len=self.args.seq_len,
                #                                                 ticker_type=self.args.ticker_type)
                # valid_indices = ~np.isnan(trues)
                # # 提取有效的 preds 和 trues
                # valid_preds = preds[valid_indices]
                # valid_trues = trues[valid_indices]
                # corr = np.corrcoef(preds, trues)[0, 1] # 所有test的corr（拼接完一起）1折的
                valid_mask = ~np.isnan(preds) & ~np.isnan(trues)
                # 过滤掉 NaN 值
                preds_filtered = preds[valid_mask]
                trues_filtered = trues[valid_mask]
                total_loss = np.mean(np.square(preds_filtered - trues_filtered)).item()
                # np.save('mse_loss.npy', mse_loss_cpu)
                np.save(f'{self.args.save_path}/true', trues)
                np.save(f'{self.args.save_path}/pred', preds)
                # mse = np.average(mse_loss_cpu)
                print('test data mse: ', total_loss)
                corr, _ = pearsonr(preds_filtered, trues_filtered)
                # corr = np.corrcoef(pred_list, true_list)[0, 1] # 所有test的corr（拼接完一起）1折的
                if self.all_test_preds.size == 0:
                    # 将 self.all_test_preds 初始化为二维数组
                    self.all_test_preds = preds.reshape(1, -1)
                else:
                    self.all_test_preds = np.concatenate((self.all_test_preds, preds.reshape(1, -1)))

                data = {'Code': y_tickers, 'CalcDate': y_times, 'True Values': trues, 'Predicted Values': preds}
                df = pd.DataFrame(data)

                def daily_ic(sub):
                    mask = np.isnan(sub['True Values'])
                    corr = np.corrcoef(sub['True Values'][~mask], sub['Predicted Values'][~mask])[0, 1]
                    return corr

                ic_series = (df.groupby('CalcDate')
                             .apply(daily_ic)
                             .rename('IC'))
                sr = np.sqrt(len(ic_series)) * ic_series.mean() / ic_series.std()
                print('the  test corr result is {} ;sr is {}'.format(corr, sr))
                # Define the path where you want to store the CSV file
                csv_file_path = self.args.save_path + '/' + self.args.model + self.args.task_name + self.args.test_year + f'predicted_true_values_{fold + 1}.csv'

                df.to_csv(csv_file_path, index=False)

                print("created with the true and predicted values.")

                print("True and predicted values have been saved to:", csv_file_path)

                mae, mse, rmse, mape, _ = metric(pred=preds, true=trues)
                # corr_df = pd.DataFrame(list(test_corr.items()), columns=['ticker', 'correlation'])
                # 保存 DataFrame 到 CSV 文件
                directory = os.path.join(
                    f'{self.args.save_path}/each_ticker_corr/' + self.args.task_name + self.args.model + self.args.test_year)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                # corr_df.to_csv(directory+f'/corr_dict_fold{fold+1}.csv', index=False) # 只test
                # plt_heiyi.plt_each_ticker_corr(self.args, directory+f'/corr_dict_fold{fold+1}.csv') # 柱状图

                current_time = datetime.now().strftime('%Y%m%d%H%M%S')
                f = open(f'{self.args.save_path}/_result_of_multiple_regression.txt', 'a')
                f.write(setting + "\n" + current_time + " ")
                f.write(': the corr valued {} ;'.format(
                    corr) + ' the results of total horizon mae {}, mse {}, rmse {}, mape {}'.format(mae, mse, rmse,
                                                                                                    mape))
                f.write('\n')
                f.write('\n')
                f.close()
                plt_heiyi.plt_epoch_train_val_trend_fold(self.args,
                                                         f'{self.args.save_path}/_result_of_multiple_regression.txt')

        all_test_mean_preds = np.mean(self.all_test_preds, axis=0)

        df_data = {'Code': y_tickers, 'CalcDate': y_times, 'True Values': trues,
                   'mean Predicted Values': all_test_mean_preds}
        test_mean_csv_file_path = self.args.save_path + '/' + self.args.model + self.args.task_name + self.args.test_year + f'predicted_true_values_mean.csv'
        mean_df = pd.DataFrame(df_data)
        mean_df.to_csv(test_mean_csv_file_path, index=False)

        mask = np.isnan(trues)
        all_test_corr = np.corrcoef(all_test_mean_preds[~mask], trues[~mask])[0, 1]
        def daily_ic(sub):
            mask = np.isnan(sub['True Values'])
            corr = np.corrcoef(sub['True Values'][~mask], sub['mean Predicted Values'][~mask])[0, 1]
            return corr

        ic_series = (mean_df.groupby('CalcDate')
                     .apply(daily_ic)
                     .rename('IC'))
        all_test_sr = np.sqrt(len(ic_series)) * ic_series.mean() / ic_series.std()
        mse = np.mean(np.square(all_test_mean_preds[~mask] - trues[~mask]))
        print(f'the average mse value of {mse}')
        print(f'the average corr value of {all_test_corr}')
        print(f'the average sr value of {all_test_sr}')
        with open(f'{self.args.save_path}/_result_of_multiple_regression.txt', 'a') as f:
            f.write(f'the average mse value of {mse}\n'
                    f'the average corr value of {all_test_corr}\n'
                    f'the average sr value of {all_test_sr}\n'
                    )

        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return