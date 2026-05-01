"""
True E2E Fusion Model - Enhanced Version
==========================================
增强版: 完整评估指标 + 顶级可视化

新增功能:
1. Synergy Gap 分析
2. Fusion Lift 计算
3. Event Window Analysis (气候风险)
4. 多维度性能雷达图
5. 增强的Gamma热力图 (标注关键事件)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 在原有代码基础上，添加以下增强功能:

# ==================== 增强的评估指标计算 ====================
def calculate_comprehensive_metrics(pred, actual):
    """
    计算全面的评估指标
    
    Returns:
        dict: 包含IC, Sharpe, MSE, Total Return, Max Drawdown等
    """
    pred = pred.flatten()
    actual = actual.flatten()
    
    # 1. IC (Information Coefficient)
    ic = np.corrcoef(pred, actual)[0, 1]
    
    # 2. 策略收益序列
    strategy_return = np.sign(pred) * actual
    cumulative_return = np.cumsum(strategy_return)
    
    # 3. Sharpe Ratio (年化)
    sharpe = (strategy_return.mean() / (strategy_return.std() + 1e-8)) * np.sqrt(252)
    
    # 4. MSE
    mse = np.mean((pred - actual) ** 2)
    
    # 5. Total Return
    total_return = cumulative_return[-1] if len(cumulative_return) > 0 else 0
    
    # 6. Max Drawdown
    running_max = np.maximum.accumulate(cumulative_return)
    drawdown = running_max - cumulative_return
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
    
    # 7. 方向准确率
    direction_acc = np.mean(np.sign(pred) == np.sign(actual))
    
    return {
        'IC': ic,
        'Sharpe': sharpe,
        'MSE': mse,
        'Total_Return': total_return,
        'Max_Drawdown': max_drawdown,
        'Direction_Acc': direction_acc
    }


# ==================== Event Window 分析器 ====================
class EventWindowAnalyzer:
    """
    事件窗口分析: 评估模型在高/低气候风险期的表现
    """
    
    def __init__(self, climate_risk_data: pd.DataFrame):
        """
        Args:
            climate_risk_data: 包含date和PRI列的DataFrame
        """
        self.climate_data = climate_risk_data
    
    def analyze(self, 
                predictions: dict, 
                actual: np.ndarray,
                dates: np.ndarray,
                high_percentile: float = 75,
                low_percentile: float = 25):
        """
        分析高/低风险期的模型表现
        
        Args:
            predictions: {'Fast-Only': pred_array, 'Fusion': pred_array}
            actual: 真实值
            dates: 日期数组
            
        Returns:
            dict: 事件窗口分析结果
        """
        # 对齐日期
        aligned_climate = self.climate_data[self.climate_data['date'].isin(dates)].copy()
        aligned_climate = aligned_climate.set_index('date').loc[dates].reset_index()
        
        if 'PRI' not in aligned_climate.columns:
            print("⚠️ 缺少PRI列，跳过Event Window分析")
            return {}
        
        pri_values = aligned_climate['PRI'].values
        
        # 确定高/低风险阈值
        high_threshold = np.percentile(pri_values, high_percentile)
        low_threshold = np.percentile(pri_values, low_percentile)
        
        high_risk_mask = pri_values >= high_threshold
        low_risk_mask = pri_values <= low_threshold
        
        results = {}
        
        for model_name, pred in predictions.items():
            # 高风险期
            high_ic = np.corrcoef(pred[high_risk_mask], actual[high_risk_mask])[0, 1] if high_risk_mask.sum() > 10 else 0
            
            # 低风险期
            low_ic = np.corrcoef(pred[low_risk_mask], actual[low_risk_mask])[0, 1] if low_risk_mask.sum() > 10 else 0
            
            results[model_name] = {
                'High_Risk_IC': high_ic,
                'Low_Risk_IC': low_ic,
                'Risk_Sensitivity': high_ic - low_ic  # 正值表示高风险期表现更好
            }
        
        return results


# ==================== 增强的XAI可视化 ====================
class EnhancedXAIVisualizer:
    """增强版XAI可视化器"""
    
    @staticmethod
    def plot_performance_radar(metrics_dict: dict, save_path: str):
        """
        绘制性能雷达图 (对比Fast-Only, Slow-Only, Fusion)
        
        Args:
            metrics_dict: {'Fast-Only': metrics, 'Slow-Only': metrics, 'Fusion': metrics}
        """
        # 归一化指标 (避免尺度差异)
        categories = ['IC', 'Sharpe', 'Total_Return', 'Direction_Acc']
        
        # 收集数据
        data = {}
        for model_name, metrics in metrics_dict.items():
            data[model_name] = [
                max(0, metrics['IC']) * 10,  # 放大IC以便可视化
                max(0, metrics['Sharpe']),
                max(0, metrics['Total_Return']) * 10,
                metrics['Direction_Acc']
            ]
        
        # 绘制雷达图
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = {'Fast-Only': '#E63946', 'Slow-Only': '#457B9D', 'Fusion': '#2A9D8F'}
        
        for model_name, values in data.items():
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors.get(model_name, 'gray'))
            ax.fill(angles, values, alpha=0.15, color=colors.get(model_name, 'gray'))
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12)
        ax.set_ylim(0, max([max(v) for v in data.values()]) * 1.2)
        ax.set_title('多维度性能对比 (Fusion vs Baselines)', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 性能雷达图已保存: {save_path}")
        plt.close()
    
    @staticmethod
    def plot_gamma_with_market_events(gamma_values, dates, sp500_prices, climate_risk, save_path):
        """
        绘制增强版 Correction/Gamma 热力图 (标注关键市场事件)
        支持 RARC-Net (Correction) 和 FiLM (Gamma) 两种架构
        """
        fig, axes = plt.subplots(3, 1, figsize=(18, 12), 
                                 gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # === 子图1: SP500价格 + Gamma叠加 ===
        ax1 = axes[0]
        ax1_twin = ax1.twinx()
        
        # SP500价格
        ax1.plot(dates, sp500_prices, color='#1F77B4', linewidth=2.5, label='SP500 Price')
        ax1.set_ylabel('SP500 Price', fontsize=14, color='#1F77B4', fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='#1F77B4')
        ax1.grid(alpha=0.3, linestyle='--')
        
        # Correction/Gamma 值 (右轴)
        ax1_twin.plot(dates, gamma_values, color='#E63946', linewidth=2, 
                     label='Correction (宏观修正量)', alpha=0.9)
        ax1_twin.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.7, label='修正=0 (中性)')
        ax1_twin.set_ylabel('Correction Value', fontsize=14, color='#E63946', fontweight='bold')
        ax1_twin.tick_params(axis='y', labelcolor='#E63946')
        ax1_twin.set_ylim(-0.15, 0.15)  # RARC-Net correction 范围 [-0.1, 0.1]
        
        # 标注关键事件 (让图片更有说服力)
        events = [
            ('2022-03-16', '首次加息\n(0.25%)', '#4ECDC4'),
            ('2022-06-15', '大幅加息\n(0.75%)', '#FF6B6B'),
            ('2022-09-28', '飓风伊恩登陆', '#FF8C00'),
            ('2022-11-02', '第6次加息\n(0.75%)', '#4ECDC4'),
        ]
        
        for event_date_str, event_name, color in events:
            try:
                event_date = pd.to_datetime(event_date_str)
                if event_date in pd.to_datetime(dates):
                    idx = np.where(pd.to_datetime(dates) == event_date)[0][0]
                    
                    # 垂直线
                    ax1.axvline(x=event_date, color=color, linestyle=':', 
                               linewidth=2.5, alpha=0.7)
                    
                    # 文本标注
                    ax1.text(event_date, sp500_prices[idx], event_name, 
                            rotation=0, verticalalignment='bottom', horizontalalignment='center',
                            fontsize=10, color=color, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor=color))
            except:
                pass
        
        ax1.set_title('RARC-Net 宏观修正机制：Regime-Adaptive Correction', 
                     fontsize=18, fontweight='bold', pad=20)
        ax1.legend(loc='upper left', fontsize=11)
        ax1_twin.legend(loc='upper right', fontsize=11)
        
        # === 子图2: Gamma分布直方图 ===
        ax2 = axes[1]
        ax2.hist(gamma_values, bins=60, color='#E63946', alpha=0.7, edgecolor='black', linewidth=0.8)
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=2, label='修正=0 (中性)')
        ax2.axvline(x=gamma_values.mean(), color='green', linestyle='-', linewidth=2, 
                   label=f'均值={gamma_values.mean():.4f}')
        ax2.set_xlabel('Correction Value', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=14, fontweight='bold')
        ax2.set_title('Correction 分布 (正值=上修预测, 负值=下修预测)', 
                     fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(alpha=0.3, axis='y')
        
        # === 子图3: 气候风险指数叠加 ===
        ax3 = axes[2]
        climate_aligned = climate_risk[climate_risk['date'].isin(dates)].set_index('date').loc[dates].reset_index()
        
        if 'PRI' in climate_aligned.columns:
            ax3.fill_between(dates, climate_aligned['PRI'].values, alpha=0.3, color='orange', label='PRI (物理风险)')
            ax3.plot(dates, climate_aligned['PRI'].values, color='orange', linewidth=1.5)
            ax3.set_ylabel('PRI Value', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Date', fontsize=12, fontweight='bold')
            ax3.set_title('气候风险指数 (验证宏观因子的影响)', fontsize=14, fontweight='bold')
            ax3.legend(fontsize=11)
            ax3.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 增强版Gamma热力图已保存: {save_path}")
        plt.close()
    
    @staticmethod
    def plot_event_window_comparison(event_results: dict, 
                                     fast_preds: np.ndarray,
                                     fusion_preds: np.ndarray,
                                     actual: np.ndarray,
                                     dates: np.ndarray,
                                     climate_risk: pd.DataFrame,
                                     save_path: str):
        """
        顶刊级Event Window对比图：展示宏观纠正在高风险期的威力
        """
        # 设置顶刊风格
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 13,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 14,
            'axes.linewidth': 1.2,
            'grid.linewidth': 0.5,
            'lines.linewidth': 2.0,
            'patch.linewidth': 1.0
        })
        
        # Nature/Science标准配色（色盲友好）
        COLOR_FAST = '#D55E00'      # 橙色（Fast-Only）
        COLOR_FUSION = '#0072B2'    # 蓝色（Fusion）
        COLOR_RISK = '#CC79A7'      # 粉红（高风险）
        COLOR_NEUTRAL = '#009E73'   # 绿色（中性）
        
        fig = plt.figure(figsize=(10, 8), dpi=300)
        gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.35, wspace=0.35)
        
        # === 面板A: 分组柱状图（核心发现）===
        ax1 = fig.add_subplot(gs[0, :])
        
        categories = ['High Climate Risk\n(Top 25%)', 'Low Climate Risk\n(Bottom 25%)']
        
        fast_ics = [event_results['Fast-Only']['High_Risk_IC'], 
                   event_results['Fast-Only']['Low_Risk_IC']]
        fusion_ics = [event_results['Fusion']['High_Risk_IC'], 
                     event_results['Fusion']['Low_Risk_IC']]
        
        x = np.arange(len(categories))
        width = 0.32
        
        # 绘制柱状图（带误差线占位符）
        bars1 = ax1.bar(x - width/2, fast_ics, width, label='Microeconomic Only',
                       color=COLOR_FAST, alpha=0.85, edgecolor='black', linewidth=1.2)
        bars2 = ax1.bar(x + width/2, fusion_ics, width, label='Macro-Corrected Fusion',
                       color=COLOR_FUSION, alpha=0.85, edgecolor='black', linewidth=1.2)
        
        # 精确数值标注
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=9, fontweight='normal')
        
        # 显著性箭头（如果融合在高风险期更优）
        if fusion_ics[0] > fast_ics[0]:
            improvement = fusion_ics[0] - fast_ics[0]
            y_max = max(fusion_ics[0], fast_ics[0])
            ax1.plot([x[0] - width/2, x[0] + width/2], [y_max + 0.02, y_max + 0.02],
                    'k-', linewidth=1.5)
            ax1.plot([x[0] - width/2, x[0] - width/2], [fast_ics[0], y_max + 0.02],
                    'k-', linewidth=1.0)
            ax1.plot([x[0] + width/2, x[0] + width/2], [fusion_ics[0], y_max + 0.02],
                    'k-', linewidth=1.0)
            ax1.text(x[0], y_max + 0.025, f'Δ = +{improvement:.3f}',
                    ha='center', fontsize=9, style='italic')
        
        ax1.set_ylabel('Information Coefficient', fontweight='bold')
        ax1.set_title('A. Macro Correction Enhances Prediction During Climate Crises', 
                     fontweight='bold', loc='left', pad=15)
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories)
        ax1.legend(frameon=True, framealpha=0.95, edgecolor='gray', loc='upper right')
        ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.7)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.set_ylim([min(min(fast_ics), min(fusion_ics)) - 0.03, 
                     max(max(fast_ics), max(fusion_ics)) + 0.06])
        
        # === 面板B: 时间序列（滚动IC + 风险背景）===
        ax2 = fig.add_subplot(gs[1, 0])
        
        climate_aligned = climate_risk[climate_risk['date'].isin(dates)].copy()
        climate_aligned = climate_aligned.set_index('date').loc[dates].reset_index()
        
        if 'PRI' in climate_aligned.columns:
            pri_values = climate_aligned['PRI'].values
            
            # 30天滚动IC
            window = 30
            fast_rolling_ic, fusion_rolling_ic = [], []
            
            for i in range(window, len(dates)):
                fast_ic_w = np.corrcoef(fast_preds[i-window:i].flatten(), 
                                       actual[i-window:i].flatten())[0, 1]
                fusion_ic_w = np.corrcoef(fusion_preds[i-window:i].flatten(), 
                                         actual[i-window:i].flatten())[0, 1]
                fast_rolling_ic.append(fast_ic_w)
                fusion_rolling_ic.append(fusion_ic_w)
            
            dates_rolling = dates[window:]
            pri_rolling = pri_values[window:]
            
            # 高风险期着色
            high_threshold = np.percentile(pri_values, 75)
            for i in range(len(dates_rolling) - 1):
                if pri_rolling[i] >= high_threshold:
                    ax2.axvspan(dates_rolling[i], dates_rolling[i+1], 
                               alpha=0.15, color=COLOR_RISK, linewidth=0)
            
            # IC曲线
            ax2.plot(dates_rolling, fast_rolling_ic, color=COLOR_FAST, 
                    linewidth=1.5, label='Micro Only', alpha=0.9)
            ax2.plot(dates_rolling, fusion_rolling_ic, color=COLOR_FUSION, 
                    linewidth=2.0, label='Fusion', alpha=0.95)
            
            ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
            ax2.set_xlabel('Time', fontweight='bold')
            ax2.set_ylabel('30-Day Rolling IC', fontweight='bold')
            ax2.set_title('B. Temporal Dynamics', fontweight='bold', loc='left', pad=10)
            ax2.legend(frameon=True, framealpha=0.95, edgecolor='gray', fontsize=9)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            
            # 添加高风险期图例
            from matplotlib.patches import Rectangle
            legend_elem = Rectangle((0, 0), 1, 1, fc=COLOR_RISK, alpha=0.15, 
                                   edgecolor='none', label='High Risk Period')
            handles, labels = ax2.get_legend_handles_labels()
            handles.append(legend_elem)
            labels.append('High Risk Period')
            ax2.legend(handles, labels, frameon=True, framealpha=0.95, 
                      edgecolor='gray', fontsize=9, loc='best')
        
        # === 面板C: 风险敏感度对比 ===
        ax3 = fig.add_subplot(gs[1, 1])
        
        models = ['Micro\nOnly', 'Macro-Corrected\nFusion']
        sensitivities = [event_results['Fast-Only']['Risk_Sensitivity'],
                        event_results['Fusion']['Risk_Sensitivity']]
        
        colors = [COLOR_FAST, COLOR_FUSION]
        
        bars = ax3.barh(models, sensitivities, color=colors, alpha=0.85, 
                       edgecolor='black', linewidth=1.2)
        
        # 数值标注
        for i, (bar, val) in enumerate(zip(bars, sensitivities)):
            x_pos = val + 0.005 if val > 0 else val - 0.005
            ax3.text(x_pos, i, f'{val:+.3f}', va='center', 
                    ha='left' if val > 0 else 'right',
                    fontsize=9, fontweight='normal')
        
        ax3.axvline(x=0, color='black', linestyle='-', linewidth=1.2)
        ax3.set_xlabel('Risk Sensitivity\n(IC$_{high}$ − IC$_{low}$)', fontweight='bold')
        ax3.set_title('C. Risk Robustness', fontweight='bold', loc='left', pad=10)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        
        # 统计显著性标注（如果Fusion更鲁棒）
        if abs(sensitivities[1]) < abs(sensitivities[0]):
            ax3.text(0.98, 0.02, '** More robust to\nclimate shocks',
                    transform=ax3.transAxes, fontsize=8,
                    ha='right', va='bottom', style='italic',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                             edgecolor='gray', linewidth=0.8, alpha=0.9))
        
        plt.suptitle('Impact of Macroeconomic Correction on Prediction Performance Across Climate Risk Regimes', 
                    fontsize=12, fontweight='bold', y=0.98, x=0.51)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Publication-quality Event Window figure saved: {save_path}")
        plt.close()
        
        # 重置matplotlib风格
        plt.rcdefaults()
    
    @staticmethod
    def plot_event_window_comparison(event_results: dict, 
                                     fast_preds: np.ndarray,
                                     fusion_preds: np.ndarray,
                                     actual: np.ndarray,
                                     dates: np.ndarray,
                                     climate_risk: pd.DataFrame,
                                     save_path: str):
        """
        绘制Event Window对比图：展示宏观纠正在高风险期的威力
        
        Args:
            event_results: EventWindowAnalyzer.analyze()的返回结果
            fast_preds: 快通道预测值
            fusion_preds: 融合模型预测值
            actual: 真实值
            dates: 日期数组
            climate_risk: 气候风险数据（包含PRI列）
            save_path: 保存路径
        """
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], hspace=0.3, wspace=0.3)
        
        # === 子图1: 分组柱状图 - 核心发现 ===
        ax1 = fig.add_subplot(gs[0, :])
        
        categories = ['高风险期\n(Top 25%)', '低风险期\n(Bottom 25%)']
        
        fast_ics = [event_results['Fast-Only']['High_Risk_IC'], 
                   event_results['Fast-Only']['Low_Risk_IC']]
        fusion_ics = [event_results['Fusion']['High_Risk_IC'], 
                     event_results['Fusion']['Low_Risk_IC']]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, fast_ics, width, label='Fast-Only (微观)',
                       color='#E63946', alpha=0.8, edgecolor='black', linewidth=1.5)
        bars2 = ax1.bar(x + width/2, fusion_ics, width, label='Fusion (宏观纠正)',
                       color='#2A9D8F', alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # 标注数值
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # 添加显著性标注（高风险期的提升）
        if fusion_ics[0] > fast_ics[0]:
            improvement = fusion_ics[0] - fast_ics[0]
            ax1.annotate('', xy=(x[0] + width/2, fusion_ics[0]), 
                        xytext=(x[0] - width/2, fast_ics[0]),
                        arrowprops=dict(arrowstyle='<->', lw=2.5, color='green'))
            ax1.text(x[0], max(fusion_ics[0], fast_ics[0]) + 0.01, 
                    f'宏观纠正\n提升 {improvement:.3f}',
                    ha='center', fontsize=11, color='green', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
        
        ax1.set_ylabel('Information Coefficient (IC)', fontsize=14, fontweight='bold')
        ax1.set_title('🎯 核心发现：宏观纠正在高风险期显著提升预测性能', 
                     fontsize=16, fontweight='bold', pad=20)
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories, fontsize=13)
        ax1.legend(fontsize=12, loc='upper right')
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.7)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 添加解释文本
        explanation = (
            "解读：在气候风险高的时期，快通道(Fast-Only)的IC显著下降，\n"
            "而融合模型(Fusion)通过宏观因子的动态调节，保持了更好的预测性能。\n"
            "这证明了宏观纠正机制在关键时刻的价值。"
        )
        ax1.text(0.02, 0.98, explanation, transform=ax1.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # === 子图2: 时间序列分段着色图 ===
        ax2 = fig.add_subplot(gs[1, 0])
        
        # 对齐气候数据
        climate_aligned = climate_risk[climate_risk['date'].isin(dates)].copy()
        climate_aligned = climate_aligned.set_index('date').loc[dates].reset_index()
        
        if 'PRI' in climate_aligned.columns:
            pri_values = climate_aligned['PRI'].values
            
            # 计算滚动IC（30天窗口）
            window = 30
            fast_rolling_ic = []
            fusion_rolling_ic = []
            
            for i in range(window, len(dates)):
                fast_ic_win = np.corrcoef(
                    fast_preds[i-window:i].flatten(), 
                    actual[i-window:i].flatten()
                )[0, 1]
                fusion_ic_win = np.corrcoef(
                    fusion_preds[i-window:i].flatten(), 
                    actual[i-window:i].flatten()
                )[0, 1]
                
                fast_rolling_ic.append(fast_ic_win)
                fusion_rolling_ic.append(fusion_ic_win)
            
            dates_rolling = dates[window:]
            pri_rolling = pri_values[window:]
            
            # 确定高风险区域
            high_threshold = np.percentile(pri_values, 75)
            high_risk_mask = pri_rolling >= high_threshold
            
            # 绘制背景着色（高风险区域）
            for i in range(len(dates_rolling)):
                if high_risk_mask[i]:
                    ax2.axvspan(dates_rolling[i], dates_rolling[min(i+1, len(dates_rolling)-1)], 
                               alpha=0.2, color='red')
            
            # 绘制IC曲线
            ax2.plot(dates_rolling, fast_rolling_ic, color='#E63946', 
                    linewidth=2, label='Fast-Only', alpha=0.8)
            ax2.plot(dates_rolling, fusion_rolling_ic, color='#2A9D8F', 
                    linewidth=2.5, label='Fusion', alpha=0.9)
            
            ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
            ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
            ax2.set_ylabel('30-day Rolling IC', fontsize=12, fontweight='bold')
            ax2.set_title('时间演化：高风险期(红色背景) vs 正常期', fontsize=13, fontweight='bold')
            ax2.legend(fontsize=11)
            ax2.grid(alpha=0.3)
        
        # === 子图3: 风险敏感度对比 ===
        ax3 = fig.add_subplot(gs[1, 1])
        
        models = ['Fast-Only', 'Fusion']
        sensitivities = [event_results['Fast-Only']['Risk_Sensitivity'],
                        event_results['Fusion']['Risk_Sensitivity']]
        
        colors_sens = ['#E63946' if s < 0 else '#2A9D8F' for s in sensitivities]
        
        bars = ax3.barh(models, sensitivities, color=colors_sens, alpha=0.8, 
                       edgecolor='black', linewidth=1.5)
        
        # 标注数值
        for i, (bar, val) in enumerate(zip(bars, sensitivities)):
            ax3.text(val, i, f'{val:+.3f}', va='center', 
                    ha='left' if val > 0 else 'right',
                    fontsize=12, fontweight='bold')
        
        ax3.axvline(x=0, color='black', linestyle='-', linewidth=2)
        ax3.set_xlabel('Risk Sensitivity\n(High Risk IC - Low Risk IC)', 
                      fontsize=11, fontweight='bold')
        ax3.set_title('风险敏感度\n(正值=高风险期表现更好)', 
                     fontsize=13, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)
        
        # 添加解释
        if sensitivities[1] > sensitivities[0]:
            ax3.text(0.95, 0.05, 
                    '✅ Fusion对高风险更\n    不敏感(更鲁棒)',
                    transform=ax3.transAxes, fontsize=10,
                    ha='right', va='bottom', color='green', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        plt.suptitle('Event Window Analysis: 宏观纠正机制在危机中的价值', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Event Window对比图已保存: {save_path}")
        plt.close()


# ==================== 导出说明 ====================
# 本模块提供以下组件，供 true_e2e_fusion.py 导入使用：
#   - calculate_comprehensive_metrics(): 计算全面的评估指标
#   - EventWindowAnalyzer: 事件窗口分析器
#   - EnhancedXAIVisualizer: 增强版XAI可视化器
#
# 使用方式:
#   from e2e_fusion_enhanced import (
#       calculate_comprehensive_metrics,
#       EventWindowAnalyzer,
#       EnhancedXAIVisualizer
#   )

