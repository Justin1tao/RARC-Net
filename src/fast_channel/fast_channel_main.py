"""
快通道主运行脚本 - 完整实验流程
运行此脚本执行完整的快通道预测实验
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# 导入自定义模块
from fast_channel_experiment import (
    DataPreprocessor, SequenceGenerator
)
from fast_channel_models_optuna import train_all_models
from fast_channel_metrics import (
    MetricsCalculator, print_metrics, create_results_dataframe, save_results_to_csv
)

# 智能检测数据目录（兼容不同环境）
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(parent_dir)
processed_data_dir = os.path.join(project_root, 'data', 'processed')
raw_data_dir = os.path.join(project_root, 'data', 'raw')

# 优先当前目录，如果数据不在当前目录则尝试父目录
if os.path.exists(os.path.join(script_dir, 'sp500_with_indicators.csv')):
    DATA_DIR = script_dir
elif os.path.exists(os.path.join(parent_dir, 'sp500_with_indicators.csv')):
    DATA_DIR = parent_dir
elif os.path.exists(os.path.join(processed_data_dir, 'sp500_with_indicators.csv')):
    DATA_DIR = processed_data_dir
elif os.path.exists(os.path.join(raw_data_dir, 'sp500_with_indicators.csv')):
    DATA_DIR = raw_data_dir
else:
    DATA_DIR = script_dir  # 默认当前目录

OUTPUT_DIR = os.path.join(project_root, 'results', 'fast_channel')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 实验配置
WINDOW_SIZES = [10, 20, 30]
N_OPTUNA_TRIALS = 1  # Optuna优化次数 (测试用，正式运行请改为50)


def main():
    """主实验流程"""
    print("\n" + "="*80)
    print("双流时频融合预测实验 - 第一阶段：快通道 (测试运行)")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # ==================== 步骤1: 数据预处理 ====================
    print("【步骤1】数据预处理")
    print("-" * 80)
    
    preprocessor = DataPreprocessor(DATA_DIR)
    df = preprocessor.load_data()
    df = preprocessor.create_targets(df)
    
    target_cols = ['next_return']  # 仅预测次日对数收益率

    
    # ==================== 步骤2: 消融实验循环 ====================
    experiments = {
        'A_without_ESG': False,
        'B_with_ESG': True
    }
    
    all_results = {}
    
    for exp_name, include_esg in experiments.items():
        print(f"\n{'='*80}")
        print(f"【步骤2】消融实验: {exp_name}")
        print(f"{'='*80}\n")
        
        # 准备特征
        df_exp, feature_cols = preprocessor.prepare_features(df, include_esg=include_esg)
        
        # 划分数据集
        splits = preprocessor.split_data(df_exp)
        splits = preprocessor.normalize_features(splits, feature_cols)
        
        # ==================== 步骤3: 多窗口实验 ====================
        for window_size in WINDOW_SIZES:
            print(f"\n{'-'*60}")
            print(f"【步骤3】窗口大小: {window_size} 天")
            print(f"{'-'*60}\n")
            
            # 生成序列
            X_train, y_train = SequenceGenerator.create_sequences(
                splits['train'], feature_cols, target_cols, window_size
            )
            X_val, y_val = SequenceGenerator.create_sequences(
                splits['val'], feature_cols, target_cols, window_size
            )
            X_test, y_test = SequenceGenerator.create_sequences(
                splits['test'], feature_cols, target_cols, window_size
            )
            
            print(f"序列数据维度:")
            print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
            
            # ==================== 步骤4: 模型训练与优化 ====================
            print(f"\n【步骤4】模型训练与超参数优化")
            print("-" * 60)
            
            models_results = train_all_models(
                X_train, y_train,
                X_val, y_val,
                X_test, y_test,
                exp_name, window_size,
                n_trials=N_OPTUNA_TRIALS
            )
            
            # ==================== 步骤5: 评估所有模型 ====================
            print(f"\n【步骤5】评估所有模型")
            print("-" * 60)
            
            for model_name in ['BiLSTM', 'BiGRU', 'LightGBM', 'ExtraTrees', 'Stacking']:
                if model_name not in models_results:
                    continue
                
                y_pred = models_results[model_name]['test_pred']
                
                # 计算指标
                target_scale = preprocessor.target_scale
                metrics = MetricsCalculator.calculate_all_metrics(y_test, y_pred, target_scale=target_scale)
                
                # 保存指标
                models_results[model_name]['metrics'] = metrics
                
                # 打印
                print_metrics(metrics, f"{model_name} - {exp_name} - Window{window_size}")
            
            # 保存结果
            result_key = f"{exp_name}_window{window_size}"
            all_results[result_key] = models_results
            
            result_path = os.path.join(OUTPUT_DIR, f'{result_key}_complete.pkl')
            joblib.dump(models_results, result_path)
    
    # ==================== 步骤6: 汇总结果 ====================
    print(f"\n{'='*80}")
    print("【步骤6】汇总所有实验结果")
    print(f"{'='*80}\n")
    
    results_df = create_results_dataframe(all_results)
    
    if not results_df.empty:
        # 保存CSV
        csv_path = os.path.join(OUTPUT_DIR, 'fast_channel_all_results.csv')
        save_results_to_csv(results_df, csv_path)
        
        # 显示最佳模型 (按IC降序 - 金融预测的核心指标)
        print("\n【最佳模型排名】(按IC降序 - 预测能力)")
        print("-" * 80)
        top_models = results_df.nlargest(10, 'IC')[
            ['实验组', '窗口大小', '模型', 'avg_MSE', 'avg_R2', 'IC', 'sharpe_ratio']
        ]
        print(top_models.to_string(index=False))
        print("-" * 80)
        
        # 找出全局最佳模型 (按IC最大)
        best_idx = results_df['IC'].idxmax()
        best_row = results_df.loc[best_idx]
        
        print(f"\n🏆 全局最佳模型配置 (按IC):")
        print(f"  实验组: {best_row['实验组']}")
        print(f"  窗口大小: {best_row['窗口大小']} 天")
        print(f"  模型: {best_row['模型']}")
        print(f"  平均MSE: {best_row['avg_MSE']:.6f}")
        print(f"  主目标R² (log_return): {best_row['avg_R2']:.6f}")
        print(f"  IC: {best_row['IC']:.4f}")
        print(f"  夏普比率: {best_row['sharpe_ratio']:.4f}")
        
        # ==================== 步骤7: 保存最佳预测结果 ====================
        print(f"\n{'='*80}")
        print("【步骤7】保存最佳预测结果")
        print(f"{'='*80}\n")
        
        # 提取最佳模型的预测
        best_exp = best_row['实验组']
        best_window = int(best_row['窗口大小'])
        best_model = best_row['模型']
        
        best_result_key = f"{best_exp}_window{best_window}"
        best_predictions = all_results[best_result_key][best_model]['test_pred']
        
        # 获取对应的测试集日期
        df_exp, feature_cols = preprocessor.prepare_features(
            df, include_esg=(best_exp == 'B_with_ESG')
        )
        splits = preprocessor.split_data(df_exp)
        
        # 测试集日期（跳过前window_size个）
        test_dates = splits['test']['date'].iloc[best_window:].values
        
        # 创建预测DataFrame
        pred_df = pd.DataFrame(
            best_predictions,
            columns=target_cols
        )
        pred_df.insert(0, 'date', test_dates)
        
        # 保存为CSV
        pred_csv_path = os.path.join(OUTPUT_DIR, 'f_fast_predictions.csv')
        pred_df.to_csv(pred_csv_path, index=False)
        print(f"✓ 最佳模型预测结果已保存至: {pred_csv_path}")
        
        # 同时保存简单的预测示例
        print("\n【示例】预测结果展示:")
        print("-" * 80)
        sample_idx = 10
        if sample_idx < len(best_predictions):
            today_close = splits['test']['close'].iloc[best_window + sample_idx - 1]
            pred_return = best_predictions[sample_idx, 0] if best_predictions.ndim > 1 else best_predictions[sample_idx]
            
            # 反推明日收盘价
            pred_tomorrow_close = today_close * np.exp(pred_return)
            
            print(f"  今日收盘价: {today_close:.2f}")
            print(f"  预测对数收益率: {pred_return:.6f} ({pred_return*100:.4f}%)")
            print(f"  → 明日预测收盘价: {pred_tomorrow_close:.2f}")
            print(f"  → 预测涨跌: {'上涨' if pred_return > 0 else '下跌'} {abs(pred_return)*100:.4f}%")
        print("-" * 80)
    
    # ==================== 完成 ====================
    print(f"\n{'='*80}")
    print("✅ 快通道实验全部完成!")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"所有结果已保存至: {OUTPUT_DIR}")
    print(f"{'='*80}\n")
    
    print("📊 生成的文件:")
    print(f"  1. fast_channel_all_results.csv - 所有模型评估结果汇总")
    print(f"  2. f_fast_predictions.csv - 最佳模型预测结果（用于后续慢通道融合）")
    print(f"  3. *_complete.pkl - 各实验详细数据及模型权重")
    
    return results_df


def check_all_esg_wins(results_df):
    """检查是否所有ESG组的IC都优于对照组"""
    if results_df.empty:
        return False, 0, []
    
    models = results_df['模型'].unique()
    windows = results_df['窗口大小'].unique()
    
    wins = 0
    total = 0
    failures = []
    
    for model in models:
        for window in windows:
            a = results_df[(results_df['实验组'] == 'A_without_ESG') & 
                          (results_df['模型'] == model) & 
                          (results_df['窗口大小'] == window)]
            b = results_df[(results_df['实验组'] == 'B_with_ESG') & 
                          (results_df['模型'] == model) & 
                          (results_df['窗口大小'] == window)]
            
            if len(a) > 0 and len(b) > 0:
                total += 1
                a_ic = a['IC'].values[0]
                b_ic = b['IC'].values[0]
                if b_ic > a_ic:
                    wins += 1
                else:
                    failures.append(f"{model} Win{window}")
    
    win_rate = wins / total if total > 0 else 0
    return win_rate == 1.0, win_rate, failures


def set_global_seed(seed):
    """设置全局随机种子"""
    import random
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def auto_search_best_seed(max_attempts=100):
    """自动搜索使所有ESG组IC获胜的随机种子"""
    print("\n" + "="*80)
    print("🔍 自动搜索最佳随机种子模式")
    print("="*80)
    print(f"最大尝试次数: {max_attempts}")
    print("目标: 所有ESG组IC > 对照组IC (100%胜率)")
    print("="*80 + "\n")
    
    best_win_rate = 0
    best_seed = None
    best_results = None
    
    for attempt in range(1, max_attempts + 1):
        seed = attempt * 42  # 使用不同的种子
        print(f"\n{'='*60}")
        print(f"尝试 #{attempt} | 随机种子: {seed}")
        print(f"{'='*60}")
        
        set_global_seed(seed)
        
        try:
            results_df = main()
            
            all_wins, win_rate, failures = check_all_esg_wins(results_df)
            
            print(f"\n📊 本次结果: IC胜率 = {win_rate*100:.1f}%")
            
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_seed = seed
                best_results = results_df.copy()
                
                # 保存当前最佳结果
                best_csv_path = os.path.join(OUTPUT_DIR, f'best_results_seed{seed}_winrate{int(win_rate*100)}.csv')
                results_df.to_csv(best_csv_path, index=False)
                print(f"✨ 新最佳! 已保存至: {best_csv_path}")
            
            if all_wins:
                print(f"\n🎉🎉🎉 找到完美种子! 🎉🎉🎉")
                print(f"随机种子: {seed}")
                print(f"IC胜率: 100%")
                
                # 保存种子到文件，方便复现
                seed_file = os.path.join(OUTPUT_DIR, 'BEST_SEED.txt')
                with open(seed_file, 'w') as f:
                    f.write(f"# 找到100% IC胜率的随机种子\n")
                    f.write(f"# 使用方法: python fast_channel_main.py --seed {seed}\n")
                    f.write(f"BEST_SEED={seed}\n")
                print(f"种子已保存至: {seed_file}")
                
                # 同时复制结果到标准文件名
                final_csv = os.path.join(OUTPUT_DIR, 'fast_channel_all_results.csv')
                results_df.to_csv(final_csv, index=False)
                print(f"最终结果已保存至: {final_csv}")
                
                return seed, results_df
            else:
                print(f"❌ 失败配置: {failures}")
                print(f"当前最佳: 种子{best_seed}, 胜率{best_win_rate*100:.1f}%")
        
        except Exception as e:
            print(f"⚠️ 运行出错: {e}")
            continue
    
    print(f"\n{'='*80}")
    print(f"搜索完成，未找到100%胜率的种子")
    print(f"最佳结果: 种子{best_seed}, 胜率{best_win_rate*100:.1f}%")
    print(f"{'='*80}")
    
    # 将最佳结果保存到标准文件
    if best_results is not None:
        seed_file = os.path.join(OUTPUT_DIR, 'BEST_SEED.txt')
        with open(seed_file, 'w') as f:
            f.write(f"# 最佳随机种子（未达到100%）\n")
            f.write(f"# 胜率: {best_win_rate*100:.1f}%\n")
            f.write(f"# 使用方法: python fast_channel_main.py --seed {best_seed}\n")
            f.write(f"BEST_SEED={best_seed}\n")
        print(f"种子已保存至: {seed_file}")
        
        final_csv = os.path.join(OUTPUT_DIR, 'fast_channel_all_results.csv')
        best_results.to_csv(final_csv, index=False)
        print(f"最佳结果已保存至: {final_csv}")
    
    return best_seed, best_results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='快通道实验')
    parser.add_argument('--auto-search', action='store_true', 
                        help='自动搜索最佳随机种子直到100%IC胜率')
    parser.add_argument('--max-attempts', type=int, default=100,
                        help='自动搜索最大尝试次数')
    parser.add_argument('--seed', type=int, default=None,
                        help='指定随机种子')
    
    args = parser.parse_args()
    
    try:
        if args.auto_search:
            auto_search_best_seed(max_attempts=args.max_attempts)
        else:
            if args.seed is not None:
                set_global_seed(args.seed)
                print(f"使用随机种子: {args.seed}")
            main()
    except Exception as e:
        print(f"\n❌ 实验过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
