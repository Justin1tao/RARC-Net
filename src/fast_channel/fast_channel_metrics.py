"""
快通道评估指标计算模块
包含: MSE, RMSE, MAE, R², IC, 方向准确率, 夏普比率、Sortino比率、最大回撤、盈亏比
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from typing import Dict, Tuple


class MetricsCalculator:
    """评估指标计算器"""
    
    @staticmethod
    def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_scale: float = 0.01) -> Dict[str, float]:
        """
        计算回归指标
        y_true, y_pred: (n_samples, n_targets) 或 (n_samples,)
        target_scale: 目标变量的缩放因子，用于还原MSE到原始范围
        """
        metrics = {}
        
        # 确保是2D数组
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
            y_pred = y_pred.reshape(-1, 1)
        
        # 对每个目标计算指标
        n_targets = y_true.shape[1]
        target_names = ['next_return']  # 现在只有一个目标
        
        for i in range(n_targets):
            target_name = target_names[i] if i < len(target_names) else f'target_{i}'
            
            mse = mean_squared_error(y_true[:, i], y_pred[:, i])
            # 还原MSE到原始收益率范围（乘以缩放因子的平方）
            mse_original = mse * (target_scale ** 2)
            
            rmse = np.sqrt(mse_original)
            mae = mean_absolute_error(y_true[:, i], y_pred[:, i]) * target_scale
            r2 = r2_score(y_true[:, i], y_pred[:, i])
            
            metrics[f'{target_name}_MSE'] = mse_original
            metrics[f'{target_name}_RMSE'] = rmse
            metrics[f'{target_name}_MAE'] = mae
            metrics[f'{target_name}_R2'] = r2
        
        # 计算平均指标（对于单目标就是本身）
        metrics['avg_MSE'] = np.mean([metrics[f'{tn}_MSE'] for tn in target_names[:n_targets]])
        metrics['avg_RMSE'] = np.mean([metrics[f'{tn}_RMSE'] for tn in target_names[:n_targets]])
        metrics['avg_MAE'] = np.mean([metrics[f'{tn}_MAE'] for tn in target_names[:n_targets]])
        metrics['avg_R2'] = metrics['next_return_R2']
        
        return metrics
    
    @staticmethod
    def calculate_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算信息系数 (Information Coefficient)
        IC = 预测收益率与真实收益率的相关系数
        """
        # 使用log_return（第0列）计算IC
        ic, _ = pearsonr(y_true[:, 0], y_pred[:, 0])
        return ic
    
    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """
        计算夏普比率
        Sharpe = (E[R] - Rf) / std(R)
        年化: 乘以 sqrt(252)
        """
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        daily_rf = risk_free_rate / 252
        sharpe = (mean_return - daily_rf) / std_return
        
        # 年化
        sharpe_annual = sharpe * np.sqrt(252)
        
        return sharpe_annual
    
    @staticmethod
    def calculate_max_drawdown(cumulative_returns: np.ndarray) -> float:
        """
        计算最大回撤
        MaxDD = max((peak - trough) / peak)
        """
        cummax = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cummax - cumulative_returns) / cummax
        max_dd = np.max(drawdowns)
        
        return max_dd
    
    @staticmethod
    def calculate_profit_loss_ratio(returns: np.ndarray) -> float:
        """
        计算盈亏比
        PLR = avg(positive_returns) / abs(avg(negative_returns))
        """
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        if len(positive_returns) == 0 or len(negative_returns) == 0:
            return 0.0
        
        avg_profit = np.mean(positive_returns)
        avg_loss = np.abs(np.mean(negative_returns))
        
        if avg_loss == 0:
            return 0.0
        
        plr = avg_profit / avg_loss
        
        return plr
    
    @staticmethod
    def calculate_directional_accuracy(y_true_returns: np.ndarray, y_pred_returns: np.ndarray) -> Dict[str, float]:
        """
        计算方向预测准确率 - 大盘预测的核心指标
        """
        # 实际方向和预测方向
        true_direction = np.sign(y_true_returns)
        pred_direction = np.sign(y_pred_returns)
        
        # 方向准确率 (Directional Accuracy)
        correct = (true_direction == pred_direction)
        directional_accuracy = np.mean(correct)
        
        # 上涨日和下跌日的分离统计
        up_days = y_true_returns > 0
        down_days = y_true_returns < 0
        
        # 上涨日预测准确率 (预测为涨且实际为涨)
        if np.sum(up_days) > 0:
            up_accuracy = np.mean(pred_direction[up_days] > 0)
        else:
            up_accuracy = 0.0
        
        # 下跌日预测准确率 (预测为跌且实际为跌)
        if np.sum(down_days) > 0:
            down_accuracy = np.mean(pred_direction[down_days] < 0)
        else:
            down_accuracy = 0.0
        
        # 精确率和召回率 (拿上涨作为正类)
        pred_up = pred_direction > 0
        true_up = true_direction > 0
        
        # 上涨精确率: 预测为涨中实际为涨的比例
        if np.sum(pred_up) > 0:
            precision_up = np.sum(pred_up & true_up) / np.sum(pred_up)
        else:
            precision_up = 0.0
        
        # 上涨召回率: 实际为涨中预测为涨的比例
        if np.sum(true_up) > 0:
            recall_up = np.sum(pred_up & true_up) / np.sum(true_up)
        else:
            recall_up = 0.0
        
        # F1
        if precision_up + recall_up > 0:
            f1_up = 2 * precision_up * recall_up / (precision_up + recall_up)
        else:
            f1_up = 0.0
        
        return {
            'directional_accuracy': directional_accuracy,  # 方向准确率 (!!!核心指标!!!)
            'up_day_accuracy': up_accuracy,    # 上涨日预测正确率
            'down_day_accuracy': down_accuracy,  # 下跌日预测正确率
            'precision_up': precision_up,  # 上涨精确率
            'recall_up': recall_up,        # 上涨召回率
            'f1_up': f1_up                 # 上涨F1
        }
    
    @staticmethod
    def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """
        计算Sortino比率 - 只考虑下行风险
        Sortino = (E[R] - Rf) / std(negative_returns)
        """
        mean_return = np.mean(returns)
        daily_rf = risk_free_rate / 252
        
        # 只计算负收益的标准差
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return 10.0  # 没有负收益，给一个很高的值
        
        downside_std = np.std(negative_returns)
        if downside_std == 0:
            return 0.0
        
        sortino = (mean_return - daily_rf) / downside_std * np.sqrt(252)
        return sortino
    
    @staticmethod
    def calculate_calmar_ratio(returns: np.ndarray, max_drawdown: float) -> float:
        """
        计算Calmar比率 - 年化收益/最大回撤
        衡量风险调整后的收益
        """
        if max_drawdown == 0:
            return 0.0
        
        # 年化收益
        annual_return = np.mean(returns) * 252
        calmar = annual_return / max_drawdown
        return calmar
    
    @staticmethod
    def calculate_consecutive_stats(correct_predictions: np.ndarray) -> Dict[str, float]:
        """
        计算连续预测统计 - 衡量预测稳定性
        """
        # 计算连续正确/错误的最大次数
        max_consecutive_correct = 0
        max_consecutive_wrong = 0
        current_correct = 0
        current_wrong = 0
        
        for pred in correct_predictions:
            if pred:
                current_correct += 1
                current_wrong = 0
                max_consecutive_correct = max(max_consecutive_correct, current_correct)
            else:
                current_wrong += 1
                current_correct = 0
                max_consecutive_wrong = max(max_consecutive_wrong, current_wrong)
        
        return {
            'max_consecutive_correct': max_consecutive_correct,
            'max_consecutive_wrong': max_consecutive_wrong
        }
    

    @staticmethod
    def calculate_trading_metrics(y_true_scaled: np.ndarray, y_pred_scaled: np.ndarray, target_scale: float = 0.01) -> Dict[str, float]:
        """
        计算交易指标
        
        y_true_scaled, y_pred_scaled: 标准化后的收益率 (Z-score)
        target_scale: 用于还原真实收益率计算总收益
        
        策略：预测>阈值时做多，否则持有现金
        - Sharpe/Sortino用策略信号(Z-score)计算，反映策略一致性
        - Total Return用真实收益率计算，反映实际盈亏
        """
        # 1. 计算预测阈值（忽略最弱的30%预测）
        threshold = np.percentile(np.abs(y_pred_scaled), 30)
        threshold = max(threshold, 0.0005)  # 最小阈值0.05%
        
        # 2. 交易策略：预测>阈值时做多，否则持有现金
        # 用Z-score的策略收益（用于Sharpe/Sortino等相对指标）
        strategy_returns_zscore = np.where(
            y_pred_scaled > threshold,   # 预测明确看涨
            y_true_scaled,               # 做多
            0                            # 不交易
        )
        
        # 3. 真实收益率（用于累计收益计算）
        real_log_returns = y_true_scaled * target_scale
        real_simple_returns = np.expm1(real_log_returns)  # exp(r) - 1
        
        strategy_returns_real = np.where(
            y_pred_scaled > threshold,
            real_simple_returns,
            0
        )
        
        # 4. 累计收益（用真实收益率）
        cumulative_returns = np.cumprod(1 + strategy_returns_real)
        
        # 5. 最大回撤（用真实累计曲线）
        max_dd = MetricsCalculator.calculate_max_drawdown(cumulative_returns)
        
        # 6. 方向准确率（用Z-score判断方向）
        directional_metrics = MetricsCalculator.calculate_directional_accuracy(y_true_scaled, y_pred_scaled)
        
        # 7. 连续预测统计
        correct_predictions = np.sign(y_true_scaled) == np.sign(y_pred_scaled)
        consecutive_stats = MetricsCalculator.calculate_consecutive_stats(correct_predictions)
        
        # 8. 统计指标（用Z-score的策略收益，反映预测能力）
        metrics = {
            'sharpe_ratio': MetricsCalculator.calculate_sharpe_ratio(strategy_returns_zscore),
            'sortino_ratio': MetricsCalculator.calculate_sortino_ratio(strategy_returns_zscore),
            'calmar_ratio': MetricsCalculator.calculate_calmar_ratio(strategy_returns_real, max_dd),
            'max_drawdown': max_dd,
            'profit_loss_ratio': MetricsCalculator.calculate_profit_loss_ratio(strategy_returns_real),
            'total_return': cumulative_returns[-1] - 1,  # 真实收益
            'win_rate': np.sum(strategy_returns_real > 0) / max(np.sum(strategy_returns_real != 0), 1),  # 交易日胜率
            **directional_metrics,
            **consecutive_stats
        }
        
        return metrics
    
    @classmethod
    def calculate_all_metrics(cls, y_true: np.ndarray, y_pred: np.ndarray, target_scale: float = 0.01) -> Dict[str, float]:
        """计算所有评估指标"""
        all_metrics = {}
        
        # 回归指标
        regression_metrics = cls.calculate_regression_metrics(y_true, y_pred, target_scale=target_scale)
        all_metrics.update(regression_metrics)
        
        # IC
        ic = cls.calculate_ic(y_true, y_pred)
        all_metrics['IC'] = ic
        
        # 交易指标（基于log_return，传入缩放因子以还原真实收益）
        trading_metrics = cls.calculate_trading_metrics(y_true[:, 0], y_pred[:, 0], target_scale=target_scale)
        all_metrics.update(trading_metrics)
        
        return all_metrics


def print_metrics(metrics: Dict[str, float], title: str = "评估指标"):
    """格式化打印指标"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    
    # 分类打印
    print("\n【回归指标】")
    print(f"  平均 MSE:  {metrics.get('avg_MSE', 0):.6f}")
    print(f"  平均 RMSE: {metrics.get('avg_RMSE', 0):.6f}")
    print(f"  平均 MAE:  {metrics.get('avg_MAE', 0):.6f}")
    print(f"  预测目标 R² (next_return): {metrics.get('avg_R2', 0):.6f}")
    
    print("\n【★ 方向预测指标 (核心) ★】")
    print(f"  方向准确率:         {metrics.get('directional_accuracy', 0):.4f} ({metrics.get('directional_accuracy', 0)*100:.2f}%)")
    print(f"  上涨日预测准确率:   {metrics.get('up_day_accuracy', 0):.4f} ({metrics.get('up_day_accuracy', 0)*100:.2f}%)")
    print(f"  下跌日预测准确率:   {metrics.get('down_day_accuracy', 0):.4f} ({metrics.get('down_day_accuracy', 0)*100:.2f}%)")
    print(f"  上涨精确率:         {metrics.get('precision_up', 0):.4f}")
    print(f"  上涨召回率:         {metrics.get('recall_up', 0):.4f}")
    print(f"  上涨F1:             {metrics.get('f1_up', 0):.4f}")
    
    print("\n【金融指标】")
    print(f"  IC (信息系数):      {metrics.get('IC', 0):.4f}")
    print(f"  夏普比率 (年化):    {metrics.get('sharpe_ratio', 0):.4f}")
    print(f"  Sortino比率 (年化): {metrics.get('sortino_ratio', 0):.4f}")
    print(f"  Calmar比率:         {metrics.get('calmar_ratio', 0):.4f}")
    print(f"  最大回撤:           {metrics.get('max_drawdown', 0):.4f}")
    print(f"  盈亏比:             {metrics.get('profit_loss_ratio', 0):.4f}")
    print(f"  总收益率:           {metrics.get('total_return', 0):.4f} ({metrics.get('total_return', 0)*100:.2f}%)")
    print(f"  胜率:               {metrics.get('win_rate', 0):.4f} ({metrics.get('win_rate', 0)*100:.2f}%)")
    
    print("\n【预测稳定性】")
    print(f"  最大连续正确:       {int(metrics.get('max_consecutive_correct', 0))} 天")
    print(f"  最大连续错误:       {int(metrics.get('max_consecutive_wrong', 0))} 天")
    
    print(f"{'='*80}\n")


def create_results_dataframe(all_results: Dict) -> pd.DataFrame:
    """创建结果汇总DataFrame"""
    rows = []
    
    for exp_key, exp_data in all_results.items():
        if '_window' not in exp_key:
            continue
        
        exp_name, window_info = exp_key.split('_window')
        window_size = window_info.replace('models', '').replace('.pkl', '')
        
        for model_name, model_data in exp_data.items():
            if 'metrics' in model_data:
                row = {
                    '实验组': exp_name,
                    '窗口大小': int(window_size),
                    '模型': model_name,
                    **model_data['metrics']
                }
                rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # 排序
    if not df.empty:
        df = df.sort_values(['实验组', '窗口大小', 'avg_MSE'])
    
    return df


def save_results_to_csv(results_df: pd.DataFrame, output_path: str):
    """保存结果到CSV"""
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✓ 结果已保存至: {output_path}")


if __name__ == '__main__':
    print("评估指标模块已加载")
    
    # 测试示例
    y_true = np.random.randn(100, 4)
    y_pred = np.random.randn(100, 4)
    
    metrics = MetricsCalculator.calculate_all_metrics(y_true, y_pred)
    print_metrics(metrics, "测试指标")
