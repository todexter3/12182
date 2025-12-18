#!/bin/bash
# 循环执行实验：遍历年份和d_model，自动分配GPU（从2开始）
# 用法：./run_loop.sh

# 配置参数列表
TRAIN_END_YEARS=("2022")  # 年份列表
SEQ_LEN=(120)                    # d_model列表
D_MODEL=(128)
gpu_list=('6,7')                       # 起始GPU编号
FOLD_START=3
FOLD_END=5
# Python脚本路径（替换为你的实际路径）
PYTHON_SCRIPT="/cpfs/dss/dev/lxjie/hy_stock/hy_daily/run_heiyi_fold_patchTST.py"

# 当前使用的GPU编号（从START_GPU开始）
#current_gpu=$START_GPU
# 循环执行所有组合
for ((i=0; i<${#TRAIN_END_YEARS[@]}; i++)); do
    # 1. 绑定当前年份与对应的GPU（核心：每个i对应一组固定的「年份-GPU」）
    year=${TRAIN_END_YEARS[$i]}
    for seq_len in "${SEQ_LEN[@]}"; do
        for d_model in "${D_MODEL[@]}"; do
          test_year=$((year + 1))  # "2019" → 2019 + 1 = 2020（自动类型转换）
#          if [[ $test_year -eq 2020 && $seq_len -eq 180 ]] || \
#               [[ $test_year -eq 2022 && $seq_len -eq 180 ]]; then
#                echo "【跳过】year=${test_year} seq_len=${seq_len} d_model=${d_model}"
#                continue   # 实验跳过，GPU 也不占
#            fi
          # 日志文件名（包含所有关键参数）
          log_file="y10_PatchTST_stock_sample_cross_section_${test_year}_sq${seq_len}_dm${d_model}_fold_start${FOLD_START}_end${FOLD_END}_rgf004.out"

          # 执行实验
          echo "启动实验：年份=$test_year, seq_len=$seq_len, GPU=$gpu_list, 日志=$log_file"
          nohup python $PYTHON_SCRIPT \
              --train_end_year $year \
              --gpu_list $gpu_list \
              --seq_len $seq_len \
              --d_model $d_model \
              --fold_start $FOLD_START \
              --fold_end $FOLD_END \
              --is_training 1 \
              > $log_file 2>&1 &

          # 输出进程ID
          echo "  进程ID: $!"

          # GPU编号+1，为下一个实验准备
  #        current_gpu=$((current_gpu + 1))

          # 可选：等待10秒再启动下一个实验，避免同时启动过多进程导致资源竞争
          sleep 2
        done
    done
done

echo "所有实验已启动！"
echo "可用以下命令查看日志："
echo "  tail -f patchtst_<年份>_gpu<编号>_seq<值>.out"
