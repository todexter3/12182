import warnings
warnings.filterwarnings(
    "ignore",
    "The pynvml package is deprecated. Please install nvidia-ml-py instead.",
    FutureWarning
)
import argparse
import os
import torch
from exp.exp_multiple_regression_fold_time import Exp_Multiple_Regression_Fold
import random
import numpy as np
# from src.model_phi_heiyi import phi  # 加载phi
import joblib
import os
# 使用多进程并行训练5折
from multiprocessing import Process, set_start_method
import torch.multiprocessing
import time
import subprocess
import sys



os.environ["KMP_AFFINITY"] = "noverbose"

parser = argparse.ArgumentParser(description='phi2')

# basic config
parser.add_argument('--task_name', type=str, default='multiple_regression',
                    help='task name, options:[Long_term_forecasting, anomaly_detection, predict_feature,multiple_regression, LGB]')
parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--model_id', type=str, default='test', help='model id')
parser.add_argument('--model', type=str, default='FC_MLP',
                    help='model name, options: [GPT2TS, ]') # PatchTST_multi_scale

# data loader
parser.add_argument('--dataset', type=str, default='ETTh1',
                    help='[ETTh1, ETTh2, ETTm1, ETTm2, weather, psm, smap]')
parser.add_argument('--prompt',type=str, default='Etth1')
parser.add_argument('--root_path', type=str, default='/home/liangxijie1/phi-2/dataset/',
                    help='root path of the data file:feature_1419_5, d1')
parser.add_argument('--data_path', type=str, default='LongtermForecast/ETT-small/',
                    help='data file, options: [ETT-small, electricity, exchange_rate, illness, traffic, weather]')
parser.add_argument('--freq', type=str, default='t',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--features', type=str, default='S',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--checkpoints', type=str, default='./checkpoints_heiyi/', help='location of model checkpoints')

parser.add_argument('--drop_ratio', type=float, default=0.2, help='Set a dropping ratio for feature_selection')
parser.add_argument('--train_data_start_year', type=int, default=2010)
parser.add_argument('--test_data_start_year', type=int, default=2021)
parser.add_argument('--feature_selection',type = bool, default=False, help='whether to use feature selection')
parser.add_argument('--extra_input',type = bool, default=False, help='whether to add tikcter')

# Forecast task
parser.add_argument('--seq_len', type=int, default=64, help='input sequence length')
parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')

# phi-2
parser.add_argument('--block_size', type=int, default=1024)
parser.add_argument('--n_layer', type=int, default=6)
parser.add_argument('--n_head', type=int, default=12)
parser.add_argument('--n_embd', type=int, default=768)
parser.add_argument('--embd_pdrop', type=float, default=0.1)
parser.add_argument('--resid_pdrop', type=float, default=0.1)
parser.add_argument('--attn_pdrop', type=float, default=0.1)
parser.add_argument('--patch_len', type=int, default=32)
parser.add_argument('--stride', type=int, default=4)
parser.add_argument('--individual', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--r', type=int, default=8)

# model define
parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=1, help='output size')
parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--channel_independence', type=int, default=0,
                    help='0: channel dependence 1: channel independence for FreTS model')
parser.add_argument('--decomp_method', type=str, default='moving_avg',
                    help='method of series decompsition, only support moving_avg or dft_decomp')
parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
# parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
# parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
# parser.add_argument('--down_sampling_method', type=str, default=None,
#                     help='down sampling method, only support avg, max, conv')
parser.add_argument('--seg_len', type=int, default=48,
                    help='the length of segmen-wise iteration of SegRNN')
# LGB
parser.add_argument('--feature_path', type=str, default='/home/dmz-ai/liruoling/heiy/results/fea/PatchTST', help='npy')

# MLP
parser.add_argument('--MLP_hidden', type=int, default=32,
                    help='The middle tier scale of fc MLPn in ecoder')
parser.add_argument('--MLP_layers', type=int, default=2, help='layers of MLP')
parser.add_argument('--kernel_size', type=int, default=7, help='kernel size of fc conv')
parser.add_argument('--max_depth', type=int, default=2, help='kernel size of fc conv')
parser.add_argument('--weight_std', type=float, default=0.01, help='weight initializes standard deviation')

# timeMixer
parser.add_argument('--down_sampling_layers', type=int, default=3, help='num of down sampling layers')
parser.add_argument('--down_sampling_window', type=int, default=2, help='down sampling window size')
parser.add_argument('--down_sampling_method', type=str, default='avg',
                    help='down sampling method, only support avg, max, conv')
# Client
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--w_lin', type=float, default=1.0, help='initial weight of the linear model')
# Fredformer
parser.add_argument('--cf_dim',         type=int, default=640)   #feature dimension
parser.add_argument('--cf_drop',        type=float, default=0.2)#dropout
parser.add_argument('--cf_depth',       type=int, default=3)    #Transformer layer
parser.add_argument('--cf_heads',       type=int, default=8)    #number of multi-heads
#parser.add_argument('--cf_patch_len',  type=int, default=16)   #patch length
parser.add_argument('--cf_mlp',         type=int, default=640)  #ff dimension
parser.add_argument('--cf_head_dim',    type=int, default=32)   #dimension for single head
parser.add_argument('--cf_weight_decay',type=float, default=0)  #weight_decay
parser.add_argument('--cf_p',           type=int, default=1)    #patch_type
parser.add_argument('--use_nys',           type=int, default=1)    #use nystrom
parser.add_argument('--mlp_drop',           type=float, default=0.3)    #output type
parser.add_argument('--ablation',       type=int, default=0)    #ablation study 012.
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
# parser.add_argument('--mlp_hidden', type=int, default=64, help='hidden layer dimension of model')
# CycleNet.
parser.add_argument('--cycle', type=int, default=24, help='cycle length')
parser.add_argument('--model_type', type=str, default='mlp', help='model type, options: [linear, mlp]')
# optimization
parser.add_argument('--num_workers', type=int, default=8, help='data loader num workers')
parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--early_open', type=bool, default=True)
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.005, help='optimizer learning rate')
parser.add_argument('--optim_type', type=str, default='Adam', help='select optimizer type, optional[SGD, Adam]')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay value')
parser.add_argument('--loss', type=str, default='MSE_with_weak', help='loss function, optional[ MSE, MAE, CCC]')
parser.add_argument('--lradj', type=str, default='type1',
                    help='adjust learning rate, optional:[type1, type2, not, cos, steplr]')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--clip_value', type=float, default=0.5, help='clip grad')
parser.add_argument('--pct_start', type=int, default=0.6)
# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--dataset_num', type=str, default='0', help='AIOps have 29 dataset,number:0-28')

# FITS
parser.add_argument('--train_mode', type=int,default=0)
parser.add_argument('--cut_freq', type=int,default=0)
parser.add_argument('--base_T', type=int,default=24)
parser.add_argument('--H_order', type=int,default=2)

# tsAMD
parser.add_argument('--n_block', type=int,default=1)
parser.add_argument('--mix_layer_num', type=int,default=2)
parser.add_argument('--mix_layer_scale', type=int,default=2)
parser.add_argument('--alpha', type=float,default=0.0)

# pathformer
parser.add_argument('--num_nodes', type=int, default=7)
parser.add_argument('--layer_nums', type=int, default=3)
parser.add_argument('--k', type=int, default=2, help='choose the Top K patch size at the every layer ')
parser.add_argument('--num_experts_list', type=list, default=[4, 4, 4])
parser.add_argument('--patch_size_list', nargs='+', type=int, default=[16,12,8,32,12,8,6,4,8,6,4,2])
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
# parser.add_argument('--revin', type=int, default=1, help='whether to apply RevIN')
parser.add_argument('--drop', type=float, default=0.1, help='dropout ratio')
# parser.add_argument('--embed', type=str, default='timeF',
#                     help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--residual_connection', type=int, default=1)
parser.add_argument('--batch_norm', type=int, default=0)

# heiyi
parser.add_argument('--save_path', type=str, default='/data/lrlresults/multiscale_patch', help='train start year')
# parser.add_argument('--is_training', type=int, default=1)
parser.add_argument('--train_start_year', type=str, default='2010', help='train start year')
parser.add_argument('--train_end_year', type=str, default='2019', help='train end year')
parser.add_argument('--val_start_year', type=str, default='2014', help='vali start year')
parser.add_argument('--use_original_feature', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--kfold', action='store_true', help='use kfold', default=False)
parser.add_argument('--per20', action='store_true', help='use foldper20', default=False)
parser.add_argument('--num_fold', type=int, default=5, help='')
parser.add_argument('--pred_task', type=int, default=10, help='y5,y10,y20')
parser.add_argument('--lgb', action='store_true', help='use lgb regressor', default=False)
parser.add_argument('--output_channels', type=int,default=1)
parser.add_argument('--label_type', type=str,default='raw')

parser.add_argument('--seed', type=int, default=42, help='seed')
parser.add_argument('--single_fold', type=int, default=None, help='train single fold for parallel execution')
parser.add_argument('--fold_start', type=int, default=3, help='fold_start')
parser.add_argument('--fold_end', type=int, default=5, help='fold_end')
parser.add_argument('--gpu_list', type=str, default='6,7', help='GPU list for 5-fold parallel training, separated by comma')
parser.add_argument('--test_only', action='store_true', help='only run testing', default=False)

# 并行训练函数
def train_single_fold(fold_id, args_dict, setting):
    """单个fold的训练函数，用于多进程"""
    import torch
    import random
    import numpy as np
    from exp.exp_multiple_regression_fold_time import Exp_Multiple_Regression_Fold
    import os

    # --- 关键修改：设置 CUDA 隔离 ---
    # 获取分配给该进程的物理 GPU ID
    assigned_gpu = args_dict['gpus'][fold_id-args_dict['fold_start']]  # 注意：这里要从字典里取 gpus
    # 限制该进程只能看到这一块 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu)

    # 重建args对象
    class Args:
        pass

    log_file_path = os.path.join(args_dict['train_log_dir'], f'fold_{fold_id + 1}_training.log')
    log_file = open(log_file_path, 'a', buffering=1)  # buffering=1 表示行缓冲，实时写入
    original_stdout = sys.stdout

    # 将当前进程的所有 print() 输出指向文件
    sys.stdout = log_file
    # 如果希望错误信息也进文件，取消下面这行的注释；如果希望报错在屏幕显示，则保留注释
    # sys.stderr = log_file
    args = Args()
    for key, value in args_dict.items():
        setattr(args, key, value)

    # --- 关键修改：重置内部 GPU ID 为 0 ---
    # 因为设置了 CUDA_VISIBLE_DEVICES，现在这就变成了该进程的第 0 号设备
    args.gpu = 0
    args.device = torch.device("cuda:0")
    torch.set_num_threads(8)
    # 打印调试信息
    print(f'>>>>>>> Fold {fold_id + 1}: PID {os.getpid()} using Physical GPU {assigned_gpu} (Logical cuda:0) >>>>>>>')
    print(f"日志文件路径: {log_file_path}")
    # 设置随机种子
    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    try:
        # 创建实验并训练
        exp = Exp_Multiple_Regression_Fold(args, single_fold=fold_id)
        exp.train(setting)
        print(f'>>>>>>> Fold {fold_id + 1} Finished Successfully <<<<<<<')
    except Exception as e:
        import traceback
        traceback.print_exc(file=log_file)
        sys.stderr.write(f"\n!!!! Fold {fold_id + 1} Error !!!! 查看日志: {log_file_path}\n")
        traceback.print_exc()
    finally:
        # 关闭文件，虽然进程结束会自动关闭，但显式关闭是好习惯
        log_file.close()

    return fold_id

def check_fold_complete(log_file, fold_id):
    """
    检查指定fold的日志是否包含完成标志（固定格式）
    :param log_file: 日志文件绝对路径
    :param fold_id: 要检查的fold索引（3/4）
    :return: True=完成，False=未完成/日志不存在
    """
    if not os.path.exists(log_file):
        return False

    # 匹配的核心标志（必须和日志输出完全一致）
    fold_num = fold_id + 1  # fold3→Fold4，fold4→Fold5
    complete_flag = f">>>>>>> Fold {fold_num} Finished Successfully <<<<<<<"

    count = 0
    try:
        # 逐行读取统计，避免内存问题
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if complete_flag in line:
                    count += 1
    except Exception as e:
        print(f"⚠️ 读取日志文件失败 {log_file}: {str(e)}")
        return False
    return count

def wait_serverB_folds(serverB_log_dir, wait_interval=300):
    """
    轮询等待服务器B的fold3/4训练完成
    :param serverB_log_dir: 服务器B日志所在目录（共享存储路径）
    :param wait_interval: 轮询间隔（秒，建议5分钟=300秒）
    """
    target_folds = [fold for fold in range(args.num_fold)]  # 服务器B负责的fold索引
    completed_folds = set()

    print(f"\n========== 开始监控服务器B训练进度 ==========")
    print(f"监控目录：{serverB_log_dir}")
    print(f"待监控fold：{[f + 1 for f in target_folds]}")
    base_log_file = os.path.join(serverB_log_dir, 'fold_1_training.log')
    base_count = check_fold_complete(base_log_file, 0)
    print(f"基准计数（Fold 1日志完成次数）：{base_count}")
    while len(completed_folds) < len(target_folds):
        # 检查每个fold的日志
        for fold_id in target_folds:
            if fold_id in completed_folds:
                continue

            log_file = os.path.join(serverB_log_dir, f'fold_{fold_id + 1}_training.log')
            current_count = check_fold_complete(log_file, fold_id)
            if current_count > 0 and (fold_id not in completed_folds):
                print(f"  Fold {fold_id + 1}: 当前完成次数={current_count}，基准={base_count}")
                # 判断条件：当前次数 >= 基准次数
            if current_count >= base_count:
                completed_folds.add(fold_id)
                print(f"✅ Fold {fold_id + 1} 达成同步条件（{current_count}/{base_count}）")

        # 计算未完成的fold
        remaining = [f + 1 for f in target_folds if f not in completed_folds]
        if remaining:
            print(f"⏳ 未完成fold：{remaining}，{wait_interval / 60:.1f}分钟后重试...")
            time.sleep(wait_interval)

    print(f"🎉 服务器B所有fold训练完成！")
    return True

def summarize_fold_results(args, setting):
    """
    汇总所有fold的训练结果
    """
    print(f"\n汇总训练结果: {args.save_path}")

    results = {}
    missing_folds = []
    missing_models = []

    # 读取各fold的结果
    for fold in range(args.num_fold):
        result_file = f'{args.save_path}/fold_{fold + 1}_results.npy'
        model_file = os.path.join(args.checkpoints + '/' + setting, f'best_model_fold_{fold + 1}.pth')

        # 检查结果文件
        if os.path.exists(result_file):
            try:
                fold_result = np.load(result_file, allow_pickle=True).item()
                results[fold] = fold_result
                print(f"\n✓ Fold {fold + 1} 结果:")
                print(f"  - Best Train Corr: {fold_result.get('best_train_corr', 'N/A'):.6f}")
                print(f"  - Best Val Loss:   {fold_result.get('best_val_loss', 'N/A'):.6f}")
                print(f"  - Best Val Corr:   {fold_result.get('best_val_corr', 'N/A'):.6f}")
                print(f"  - Best Val SR:     {fold_result.get('best_val_sr', 'N/A'):.6f}")
                print(f"  - Best Val Metric: {fold_result.get('best_val_metric', 'N/A'):.6f}")
                print(f"  - Nowcast Corr:    {fold_result.get('nowcast_corr', 'N/A'):.6f}")
            except Exception as e:
                print(f"\n× Fold {fold + 1} 结果文件读取失败: {e}")
                missing_folds.append(fold + 1)
        else:
            print(f"\n× Fold {fold + 1} 结果文件未找到")
            missing_folds.append(fold + 1)

        # 检查模型文件
        if not os.path.exists(model_file):
            print(f"  × 模型文件未找到: {model_file}")
            missing_models.append(fold + 1)
        else:
            file_size = os.path.getsize(model_file) / (1024 * 1024)  # MB
            print(f"  ✓ 模型文件: {file_size:.2f} MB")

    # 计算平均值
    if results:
        print("\n" + "=" * 60)
        print("平均结果汇总:")
        print("=" * 60)

        metrics = ['best_train_corr', 'best_val_loss', 'best_val_corr',
                   'best_val_sr', 'best_val_metric', 'nowcast_corr']

        avg_results = {}
        for metric in metrics:
            values = [r.get(metric) for r in results.values() if r.get(metric) is not None]
            if values:
                values = [v.item() if hasattr(v, 'item') else v for v in values]
                mean_val = np.mean(values)
                std_val = np.std(values)
                avg_results[metric] = {'mean': mean_val, 'std': std_val}
                print(f"{metric:20s}: {mean_val:.6f} ± {std_val:.6f}")

        # 保存汇总结果
        with open(f'{args.save_path}/_result_of_multiple_regression.txt', 'a') as f:
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"5折交叉验证汇总结果\n")
            f.write("=" * 60 + "\n\n")

            for fold, result in results.items():
                f.write(f"Fold {fold + 1}:\n")
                for metric in metrics:
                    val = result.get(metric, 'N/A')
                    if val != 'N/A':
                        val = val.item() if hasattr(val, 'item') else val
                        f.write(f"  {metric:20s}: {val:.6f}\n")
                f.write("\n")

            f.write("=" * 60 + "\n")
            f.write("平均结果:\n")
            f.write("=" * 60 + "\n")
            for metric in metrics:
                if metric in avg_results:
                    f.write(f"{metric:20s}: {avg_results[metric]['mean']:.6f}\n")

    # 检查是否可以开始测试
    print("\n" + "=" * 60)
    if missing_folds or missing_models:
        if missing_folds:
            print(f"⚠ 警告: 以下fold缺少结果文件: {missing_folds}")
        if missing_models:
            print(f"⚠ 警告: 以下fold缺少模型文件: {missing_models}")
        print("建议等待所有fold训练完成后再进行测试")
        print("=" * 60)
        return False
    else:
        print("✓ 所有fold训练已完成，模型文件完整，可以开始测试")
        print("=" * 60)
        return True

if __name__ == '__main__':
    pip_path = sys.executable.replace("python3.11", "pip")
    result = subprocess.run([pip_path, 'freeze'], capture_output=True, text=True)
    dependencies = result.stdout
    with open('requirements.txt', 'w') as file:
        file.write(dependencies)
    print("已生成安装包列表：requirements.txt")
    # 检查是否在守护进程中运行，避免"daemonic processes are not allowed to have children"错误
    try:
        # 使用 torch.multiprocessing.set_start_method 更安全
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # 如果已经设置，跳过
        pass

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False


    # args.use_multi_gpu=1

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[-1]



    # args.is_training = 0
    args.data_new = '5'
    # args.ticker_type = 2  #0,1,2(all)

    args.weight_decay = 1e-5
    args.drop_ratio = 0.1
    args.pct_start = 0.6
    args.label_type = 'res'
    args.feature_selection = False
    args.train_epochs = 60
    args.patience = 10
    # args.individual = True
    args.n_splits = 3
    args.dataset = 'heiyi'  # [ETTh1, ETTh2, ETTm1, ETTm2, weather, public, elc, traffic,AIOps]
    args.lradj = 'not'
    args.random_zero_prob = 0.0
    args.random_mask_prob = 0.0

    "/cpfs/dss/dev/fxi/project/flap01/cta/daily/daily_label5_20_all_data_202312.feather"
    '/cpfs/dss/dev/lxjie/lxj_results/daily_10_p9_price_Basis_202312.feather'
    #/cpfs/dss/dev/fxi/project/flap01/stock/daily/daily_label10_data_addMask_202312.h5
    args.data_path = '/cpfs/dss/dev/fxi/project/flap01/stock/daily/daily_label10_data_addMask_202312.h5' # daily 因子集和原始特征
    # args.data_path = '/data/stock_daily_2005_2021.feather'
    args.data_type = 'daily'
    args.freq = 'd'
    args.learning_rate = 1e-5
    # args.data_path = '/data/downsample_data3/15min_label36_320_data.feather' # min 原始特征
    # args.data_path = '/data/downsample_data3/15min_label320_factors.feather' # min15 因子集
    # args.data_type = 'min15'
    pred_task = 10

    args.grad_norm = False
    args.dropout = args.drop_ratio
    args.tau_hat_init = 0.0
    args.MLP_layers = 3
    args.MLP_hidden = 128
    # args.seq_len = 120
    #
    args.train_start_year = '2017'
    # args.train_end_year = '2022'
    # args.gpu = 0
    args.test_year = str(int(args.train_end_year)+1)
    args.device = torch.device(f"cuda:{args.gpu}")
    args.features = 'M' # long MS
    args.task_name = 'multiple_regression'  # [Long_term_forecasting, multiple_regression, predict_feature, classification]
    args.model = 'PatchTST'  # MHPatchTST, FC_MLP_layer, PatchTST, FC_MLP, PatchTST_C_group, FC_Conv, FITS,MTPatchTST,MTMLP,LSTM
    args.fold_type = 'time_fold' # k_fold,time_fold
    args.val = True
    args.enc_in = 10
    args.num_fold = 5
    args.epsilon = 2
    # args.delta=0.3
    save_path = f'/cpfs/dss/dev/lxjie/lxj_results/stock/fold/{args.fold_type}/phi_ret_mean/{args.data_type}_price_{args.label_type}_cross_section_sample/'

    if args.cut_freq == 0:
        args.cut_freq = int(args.seq_len // args.base_T + 1) * args.H_order + 10

    '''
        每个ticker的val取20%做kfold, 因子集和原始特征
    '''
    epsilon = 2
    args.is_training = 1
    i=0
    args.patch_size_list = np.array(args.patch_size_list).reshape(args.layer_nums, -1).tolist()
    # def set_seed(seed=42):
    #     random.seed(seed)
    #     np.random.seed(seed)
    #     torch.manual_seed(seed)
    #     if torch.cuda.is_available():
    #         torch.cuda.manual_seed_all(seed)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False

    if args.is_training:
        for args.batch_size in [512]:
            for args.tau_hat_init in [0.0,1.0,2.0,3.0,4.0,4.5]:
                    # for args.seq_len in [120]:
                        # for args.kernel_size in [[3,5,7],[3,7,15]]:
                        # for args.seq_len in [90,120]:
                # for args.seq_len in range(180, 210+30,30): # min15 720
                    # --mlp LSTM
                    # for args.MLP_layers in [3,4]:
                    #     for args.MLP_hidden in [64,128,256]:
                    # for args.MLP_layers in [5,6,7]:
                    #     for args.MLP_hidden in [256,512,1024]:
                    # for args.MLP_layers in [4]:
                    #     for args.MLP_hidden in [64]:
                        # --patchtst
                        #     for args.d_model in [128]:
                                args.d_ff = args.d_model*2
                        # for args.d_ff in [32,64,128]:
                                for args.patch_len in [16,8]:
                                    # i+=1
                                    # if i<=1:
                                    #     continue
                                    args.stride = args.patch_len//2
                                        # if args.patch_len==16 and args.stride in [8,12]:
                                        #     continue
                                    args.e_layers = 3
                                    print('Args in experiment:')
                                    print(args)
                                    if args.data_type == 'daily':
                                        if args.task_name == 'Long_term_forecasting':
                                            args.pred_task = pred_task
                                            args.pred_len = args.pred_task
                                        elif args.task_name == 'multiple_regression':
                                            args.pred_task = pred_task
                                            args.pred_len = 1
                                        elif args.task_name == 'predict_feature':
                                            args.pred_task = pred_task
                                            args.pred_len = 1
                                    elif args.data_type == 'min15':
                                        if args.task_name == 'Long_term_forecasting':
                                            args.pred_task = pred_task
                                            args.pred_len = args.pred_task
                                        elif args.task_name == 'multiple_regression' or args.task_name == 'classification':
                                            args.pred_task = pred_task
                                            args.pred_len = 1
                                        elif args.task_name == 'predict_feature':
                                            args.pred_task = pred_task
                                            args.pred_len = 1
                                # for args.pred_len in [1]:# [96, 192, 336, 720]
                                    # if args.model == 'TimesNet':
                                    #     args.pred_len = 0

                                    fix_seed = args.seed
                                    # fix_seed = 42
                                    random.seed(fix_seed)
                                    torch.manual_seed(fix_seed)
                                    np.random.seed(fix_seed)
                                    args.size = [args.seq_len, args.pred_len]
                                    if args.loss == 'MSE_with_weak':
                                        train_des = f"{args.model}_test_year{args.test_year}_tau_x{args.tau_hat_init}_kfold{args.kfold}_seq{args.seq_len}_pred{args.pred_len}_ep{args.train_epochs}_bs{args.batch_size}_early{args.patience}_lr{args.learning_rate}_wd{args.weight_decay}_"
                                    else:
                                        train_des = f"{args.model}_test_year{args.test_year}_kfold{args.kfold}_seq{args.seq_len}_pred{args.pred_len}_ep{args.train_epochs}_bs{args.batch_size}_early{args.patience}_lr{args.learning_rate}_wd{args.weight_decay}_"
                                    # model = Model(args)
                                    # train_des_pretrain = f"NNN_{args.data_new}_task_name{args.task_name}_ticker_type{0}{args.model}_test_year{args.test_year}_seq{args.seq_len}_pred{args.pred_len}_freq{args.freq}_ep{args.train_epochs}_bs{128}_early{args.patience}_lr{args.learning_rate}_wd{args.weight_decay}_"
                                    if args.model == 'FITS':
                                        model_des = f"nl{args.n_layer}_nh{args.n_head}_ne_{args.n_embd}_era_dp{args.drop_ratio}_{args.features}_inv{args.individual}_dmo{args.d_model}_dff{args.d_ff}_horder{args.H_order}"
                                    else:
                                        model_des = f"eps{args.epsilon}_nl{args.n_layer}_nh{args.n_head}_ne_{args.n_embd}_era_dp{args.drop_ratio}_{args.features}_inv{args.individual}_dmo{args.d_model}_dff{args.d_ff}"
                                    patching_des = f'_pl{args.patch_len}_sr{args.stride}_val{args.val}'
                                    setting = train_des + model_des + patching_des
                                    # if args.task_name == 'multiple_regression':
                                    args.save_path = os.path.join(save_path, f'y{args.pred_task}/{args.model}_{setting}')
                                    args.checkpoints = args.save_path
                                    args.logs_dir = args.save_path + f'/logs'
                                    args.train_log_dir = f'/cpfs/dss/dev/lxjie/hy_stock/hy_daily/logs/{args.loss}_sample_cross_section_{args.test_year}_dm{args.d_model}_sq{args.seq_len}'
                                    if not os.path.exists( args.train_log_dir):
                                        os.makedirs( args.train_log_dir, exist_ok=True)
                                    if not os.path.exists(args.save_path):
                                        os.makedirs(args.save_path)
                                    with open(f'{args.save_path}/_result_of_multiple_regression.txt', 'a') as file:
                                        file.write('Args in experiment:\n' + f'{args}\n\n')

                                    # 判断是否是单折训练模式（通过--single_fold参数控制）
                                    if args.single_fold is not None:
                                        # 单折模式：直接训练指定的fold
                                        print(f'>>>>>>>start training fold {args.single_fold + 1} : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
                                        Exp = Exp_Multiple_Regression_Fold
                                        exp = Exp(args, single_fold=args.single_fold)
                                        exp.train(setting)
                                        print(f'>>>>>>>fold {args.single_fold + 1} training completed<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                                    else:
                                        args.gpus = [int(gpu.strip()) for gpu in args.gpu_list.split(',')]
                                        if len(args.gpus) != args.fold_end-args.fold_start:
                                            raise ValueError(
                                                f"GPU list size ({len(args.gpus)}) must match num_fold ({args.num_fold}).")

                                        start_time = time.time()
                                        processes = []

                                        # 1. 启动训练进程
                                        for fold_id in range(args.fold_start,args.fold_end):
                                            # 为当前 fold 创建参数副本
                                            current_args = vars(args).copy()

                                            # 为子进程分配 GPU
                                            current_args['gpu'] = args.gpus[fold_id-args.fold_start]

                                            # 创建 Process 实例
                                            p = Process(
                                                target=train_single_fold,
                                                args=(fold_id, current_args, setting)  # 传递参数
                                            )
                                            # Process 默认是非守护进程，可以启动 DataLoader 的子进程
                                            p.start()
                                            processes.append(p)

                                        print(
                                            f"✅ All {args.num_fold} folds started. Waiting for training to complete...")

                                        # 2. 等待所有进程完成
                                        for p in processes:
                                            p.join()  # 阻塞主进程，直到所有子进程结束
                                        if args.fold_start == 0:
                                            wait_serverB_folds(args.train_log_dir, wait_interval=300)

                                            end_time = time.time()
                                            print(f"Total time for all folds: {end_time - start_time:.2f} seconds.")

                                            # 3. 结果汇总和自动测试 (在所有进程完成后执行)
                                            print(
                                                '\n================== Folds Finished. Summarizing Results... ==================\n')
                                            all_ready = summarize_fold_results(args, setting)

                                            if all_ready and not args.test_only:
                                                print(f'Training of {setting} finished. Auto-testing...')
                                                exp = Exp_Multiple_Regression_Fold(args)
                                                exp.test(setting)
                                        # 清理显存
                                        torch.cuda.empty_cache()
    else:
        # 测试模式：确保参数与训练时一致
        for args.batch_size in [512]:
            # 测试时不遍历tau_hat_init，使用默认值或训练时的值
            # for args.tau_hat_init in [0.0, 1.0, 2.0, 3.0, 4.0, 4.5]:
                # for args.learning_rate in [1e-5]:
                    # for args.kernel_size in [[3,5,7],[3,7,15]]:
                    # for args.seq_len in [90,120]:
                    # for args.seq_len in range(180, 210+30,30): # min15 720
                    # --mlp LSTM
                    # for args.MLP_layers in [3,4]:
                    #     for args.MLP_hidden in [64,128,256]:
                    # for args.MLP_layers in [5,6,7]:
                    #     for args.MLP_hidden in [256,512,1024]:
                    # for args.MLP_layers in [4]:
                    #     for args.MLP_hidden in [64]:
                    # --patchtst
                    #     for args.d_model in [64,128]:
            args.d_ff = args.d_model * 2
            # for args.d_ff in [32,64,128]:
            for args.patch_len in [16, 32]:
                # i+=1
                # if i<=2:
                #     continue
                args.stride = args.patch_len // 2
                # if args.patch_len==16 and args.stride in [8,12]:
                #     continue
                args.e_layers = 3
                print('Args in experiment:')
                print(args)
                if args.data_type == 'daily':
                    if args.task_name == 'Long_term_forecasting':
                        args.pred_task = pred_task
                        args.pred_len = args.pred_task
                    elif args.task_name == 'multiple_regression':
                        args.pred_task = pred_task
                        args.pred_len = 1
                    elif args.task_name == 'predict_feature':
                        args.pred_task = pred_task
                        args.pred_len = 1
                elif args.data_type == 'min15':
                    if args.task_name == 'Long_term_forecasting':
                        args.pred_task = pred_task
                        args.pred_len = args.pred_task
                    elif args.task_name == 'multiple_regression' or args.task_name == 'classification':
                        args.pred_task = pred_task
                        args.pred_len = 1
                    elif args.task_name == 'predict_feature':
                        args.pred_task = pred_task
                        args.pred_len = 1
                # for args.pred_len in [1]:# [96, 192, 336, 720]
                # if args.model == 'TimesNet':
                #     args.pred_len = 0

                fix_seed = args.seed
                # fix_seed = 42
                random.seed(fix_seed)
                torch.manual_seed(fix_seed)
                np.random.seed(fix_seed)
                args.size = [args.seq_len, args.pred_len]
                # 测试时使用与训练相同的路径命名规则
                if args.loss == 'MSE_with_weak':
                    train_des = f"{args.model}_test_year{args.test_year}_tau_x{args.tau_hat_init}_kfold{args.kfold}_seq{args.seq_len}_pred{args.pred_len}_ep{args.train_epochs}_bs{args.batch_size}_early{args.patience}_lr{args.learning_rate}_wd{args.weight_decay}_"
                else:
                    train_des = f"{args.model}_test_year{args.test_year}_kfold{args.kfold}_seq{args.seq_len}_pred{args.pred_len}_ep{args.train_epochs}_bs{args.batch_size}_early{args.patience}_lr{args.learning_rate}_wd{args.weight_decay}_"
                # model = Model(args)
                # train_des_pretrain = f"NNN_{args.data_new}_task_name{args.task_name}_ticker_type{0}{args.model}_test_year{args.test_year}_seq{args.seq_len}_pred{args.pred_len}_freq{args.freq}_ep{args.train_epochs}_bs{128}_early{args.patience}_lr{args.learning_rate}_wd{args.weight_decay}_"
                if args.model == 'FITS':
                    model_des = f"nl{args.n_layer}_nh{args.n_head}_ne_{args.n_embd}_era_dp{args.drop_ratio}_{args.features}_inv{args.individual}_dmo{args.d_model}_dff{args.d_ff}_horder{args.H_order}"
                else:
                    model_des = f"eps{args.epsilon}_nl{args.n_layer}_nh{args.n_head}_ne_{args.n_embd}_era_dp{args.drop_ratio}_{args.features}_inv{args.individual}_dmo{args.d_model}_dff{args.d_ff}"
                patching_des = f'_pl{args.patch_len}_sr{args.stride}_val{args.val}'
                setting = train_des + model_des + patching_des
                # if args.task_name == 'multiple_regression':
                args.save_path = os.path.join(save_path, f'y{args.pred_task}/{args.model}_{setting}')
                args.checkpoints = args.save_path
                args.logs_dir = args.save_path + f'/logs'

                # 测试前检查路径是否存在
                if not os.path.exists(args.save_path):
                    print(f"错误: 找不到训练结果路径: {args.save_path}")
                    print("请确保已完成训练，或检查超参数设置是否与训练时一致")
                    continue

                with open(f'{args.save_path}/_result_of_multiple_regression.txt', 'a') as file:
                    file.write('\n' + '='*60 + '\n')
                    file.write('Testing with Args:\n' + f'{args}\n\n')

                Exp = Exp_Multiple_Regression_Fold
                exp = Exp(args)  # set experiments
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)
                torch.cuda.empty_cache()
