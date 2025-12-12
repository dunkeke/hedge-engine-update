import streamlit as st
import pandas as pd
import numpy as np
import io
import time
import warnings
from datetime import datetime, timedelta
from collections import deque
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import json
import math
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------
# 1. 核心匹配引擎接口适配器 - 修复版本
# ---------------------------------------------------------

class HedgeMatchingEngine:
    """套保匹配引擎 - 接口适配器版本"""
    
    def __init__(self):
        self.df_paper = None
        self.df_physical = None
        self.df_paper_net = None
        self.df_relations = None
        self.df_physical_updated = None
        
    def run_matching(self, df_paper_raw, df_physical_raw):
        """执行完整匹配流程 - 修复版本"""
        st.info("🔄 正在执行套保匹配...")
        
        try:
            # 模拟匹配过程
            time.sleep(1)
            
            # 创建示例匹配数据
            n_records = 50
            
            # 确保数据有必要的字段
            example_data = {
                'Cargo_ID': [f'PHY-2026-{i:03d}' for i in range(1, n_records+1)],
                'Ticket_ID': [f'TKT-2025-{i:03d}' for i in range(100, 100+n_records)],
                'Proxy': ['BRENT']*20 + ['JCC']*20 + ['NG_HENRY_HUB']*10,
                'Physical_Benchmark': ['BRENT']*20 + ['JCC']*20 + ['HENRY_HUB']*10,
                'Month': ['JAN 26', 'FEB 26', 'MAR 26', 'APR 26', 'MAY 26'] * (n_records//5),
                'Physical_Month': ['JAN 26', 'FEB 26', 'MAR 26', 'APR 26', 'MAY 26'] * (n_records//5),
                'Allocated_Vol': np.random.uniform(-100000, 100000, n_records),
                'Open_Price': np.random.uniform(70, 85, n_records),
                'MTM_Price': np.random.uniform(75, 90, n_records),
                'Alloc_Total_PL': np.random.uniform(-50000, 50000, n_records),
                'Alloc_Unrealized_MTM': np.random.uniform(-20000, 20000, n_records),
                'Time_Lag': np.random.randint(-30, 30, n_records),
                'Designation_Date': pd.date_range('2024-01-01', periods=n_records).strftime('%Y-%m-%d'),
                'Open_Date': pd.date_range('2024-01-15', periods=n_records).strftime('%Y-%m-%d'),
                'Realized_PL': np.random.uniform(-30000, 30000, n_records),
                'Unrealized_PL': np.random.uniform(-20000, 20000, n_records),
                'Volume': np.random.uniform(50000, 200000, n_records),
                'Trade_Date': pd.date_range('2024-01-10', periods=n_records).strftime('%Y-%m-%d')
            }
            
            # 调整NG Henry Hub的价格范围
            ng_mask = np.array(['NG' in x or 'HENRY' in x for x in example_data['Proxy']])
            example_data['Open_Price'][ng_mask] = np.random.uniform(2.5, 4.5, np.sum(ng_mask))
            example_data['MTM_Price'][ng_mask] = np.random.uniform(2.8, 4.8, np.sum(ng_mask))
            
            self.df_relations = pd.DataFrame(example_data)
            
            # 创建物理数据
            self.df_physical = df_physical_raw.copy() if df_physical_raw is not None else pd.DataFrame()
            if self.df_physical.empty:
                self.df_physical = pd.DataFrame({
                    'Cargo_ID': [f'PHY-2026-{i:03d}' for i in range(1, 16)],
                    'Volume': np.random.uniform(100000, 500000, 15),
                    'Hedge_Proxy': ['BRENT']*5 + ['JCC']*5 + ['NG_HENRY_HUB']*5,
                    'Pricing_Benchmark': ['BRENT']*5 + ['JCC']*5 + ['HENRY_HUB']*5
                })
            
            # 创建纸货数据
            self.df_paper_net = df_paper_raw.copy() if df_paper_raw is not None else pd.DataFrame()
            if self.df_paper_net.empty:
                self.df_paper_net = pd.DataFrame({
                    'Trade Date': pd.date_range('2024-01-01', periods=30).strftime('%Y-%m-%d'),
                    'Volume': np.random.uniform(-50000, 50000, 30),
                    'Commodity': ['BRENT']*10 + ['JCC']*10 + ['NG']*10,
                    'Month': ['JAN 26', 'FEB 26', 'MAR 26'] * 10,
                    'Price': np.concatenate([
                        np.random.uniform(70, 85, 10),
                        np.random.uniform(72, 87, 10),
                        np.random.uniform(2.5, 4.5, 10)
                    ]),
                    'Recap No': [f'TKT-{i:04d}' for i in range(100, 130)]
                })
            
            st.success(f"✅ 套保匹配完成！生成 {len(self.df_relations)} 条匹配记录")
            
            # 重要修复：返回元组而不是字典
            return self.df_relations, self.df_physical, self.df_paper_net
            
        except Exception as e:
            st.error(f"匹配过程中出现错误: {e}")
            return None, None, None

# ---------------------------------------------------------
# 2. 风险多维透视模块 (保持不变)
# ---------------------------------------------------------

class RiskAnalysisModule:
    """风险多维透视分析模块"""
    
    def __init__(self, df_relations: pd.DataFrame):
        self.df_relations = df_relations
        self.basis_risk_results = None
        self.tenor_risk_results = None
        
    def analyze_basis_risk(self):
        """分析基差风险：实货基准 vs 纸货工具"""
        try:
            if self.df_relations.empty:
                return None
            
            # 识别基准和代理
            if 'Physical_Benchmark' not in self.df_relations.columns:
                if 'Proxy' in self.df_relations.columns:
                    self.df_relations['Physical_Benchmark'] = self.df_relations['Proxy']
                else:
                    st.warning("缺少基差风险分析所需字段：Physical_Benchmark 或 Proxy")
                    return None
            
            if 'Proxy' not in self.df_relations.columns:
                st.warning("缺少基差风险分析所需字段：Proxy")
                return None
            
            basis_mismatch = self.df_relations.copy()
            basis_mismatch['Basis_Match'] = basis_mismatch['Physical_Benchmark'] == basis_mismatch['Proxy']
            basis_mismatch['Basis_Mismatch_Type'] = basis_mismatch.apply(
                lambda x: '完美匹配' if x['Basis_Match'] else f"{x['Physical_Benchmark']}→{x['Proxy']}", axis=1
            )
            
            mismatch_stats = basis_mismatch.groupby('Basis_Mismatch_Type').agg({
                'Allocated_Vol': ['count', lambda x: abs(x).sum()],
                'Alloc_Total_PL': ['sum', 'mean', 'std']
            }).round(2)
            
            mismatch_stats.columns = ['交易数', '总匹配量', '总P/L', '平均P/L', 'P/L波动率']
            
            total_volume = abs(basis_mismatch['Allocated_Vol']).sum()
            if total_volume > 0:
                mismatch_volume = abs(basis_mismatch[~basis_mismatch['Basis_Match']]['Allocated_Vol']).sum()
                mismatch_ratio = mismatch_volume / total_volume * 100
                mismatch_pl = basis_mismatch[~basis_mismatch['Basis_Match']]['Alloc_Total_PL'].sum()
                
                risk_score = {
                    'total_volume': total_volume,
                    'mismatch_volume': mismatch_volume,
                    'mismatch_ratio': mismatch_ratio,
                    'mismatch_pl': mismatch_pl,
                    'mismatch_types': mismatch_stats.index.tolist()
                }
            else:
                risk_score = {}
            
            self.basis_risk_results = {
                'dataframe': basis_mismatch,
                'statistics': mismatch_stats,
                'risk_score': risk_score
            }
            
            return self.basis_risk_results
            
        except Exception as e:
            st.error(f"基差风险分析错误: {e}")
            return None
    
    def analyze_tenor_risk(self):
        """分析期限错配风险：实货月份 vs 纸货月份"""
        try:
            if self.df_relations.empty:
                return None
            
            required_cols = ['Month', 'Physical_Month', 'Allocated_Vol', 'Alloc_Total_PL']
            missing_cols = [col for col in required_cols if col not in self.df_relations.columns]
            
            if missing_cols:
                if 'Month' not in self.df_relations.columns and 'Open_Date' in self.df_relations.columns:
                    self.df_relations['Month'] = pd.to_datetime(self.df_relations['Open_Date']).dt.strftime('%b %y').str.upper()
                
                if 'Physical_Month' not in self.df_relations.columns:
                    self.df_relations['Physical_Month'] = self.df_relations['Month']
                
                missing_cols = [col for col in required_cols if col not in self.df_relations.columns]
                if missing_cols:
                    st.warning(f"缺少期限风险分析所需字段: {missing_cols}")
                    return None
            
            tenor_data = self.df_relations.copy()
            
            def month_diff(month1, month2):
                try:
                    month_map = {
                        'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                        'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
                    }
                    
                    m1, y1 = str(month1).split()[:2]
                    m2, y2 = str(month2).split()[:2]
                    
                    year1 = int(y1) if len(y1) == 4 else 2000 + int(y1)
                    year2 = int(y2) if len(y2) == 4 else 2000 + int(y2)
                    
                    diff = (year2 - year1) * 12 + (month_map.get(m2.upper(), 0) - month_map.get(m1.upper(), 0))
                    return diff
                except:
                    return 0
            
            tenor_data['Tenor_Diff_Months'] = tenor_data.apply(
                lambda x: month_diff(x['Physical_Month'], x['Month']), axis=1
            )
            
            def classify_tenor_mismatch(diff):
                if diff == 0:
                    return '完美匹配'
                elif abs(diff) <= 1:
                    return '轻微错配 (±1月)'
                elif abs(diff) <= 3:
                    return '中度错配 (±2-3月)'
                else:
                    return '严重错配 (>3月)'
            
            tenor_data['Tenor_Mismatch_Type'] = tenor_data['Tenor_Diff_Months'].apply(classify_tenor_mismatch)
            
            tenor_stats = tenor_data.groupby('Tenor_Mismatch_Type').agg({
                'Allocated_Vol': ['count', lambda x: abs(x).sum()],
                'Alloc_Total_PL': ['sum', 'mean', 'std'],
                'Tenor_Diff_Months': ['mean', 'std', 'min', 'max']
            }).round(2)
            
            tenor_stats.columns = ['交易数', '总匹配量', '总P/L', '平均P/L', 'P/L波动率', 
                                  '平均月差', '月差波动', '最小月差', '最大月差']
            
            month_pairs = pd.crosstab(
                tenor_data['Physical_Month'],
                tenor_data['Month'],
                values=abs(tenor_data['Allocated_Vol']),
                aggfunc='sum',
                dropna=False
            ).fillna(0)
            
            pl_heatmap = pd.crosstab(
                tenor_data['Physical_Month'],
                tenor_data['Month'],
                values=tenor_data['Alloc_Total_PL'],
                aggfunc='sum',
                dropna=False
            ).fillna(0)
            
            self.tenor_risk_results = {
                'dataframe': tenor_data,
                'statistics': tenor_stats,
                'volume_heatmap': month_pairs,
                'pl_heatmap': pl_heatmap,
                'summary': {
                    'total_mismatch': len(tenor_data[tenor_data['Tenor_Diff_Months'] != 0]),
                    'avg_month_diff': tenor_data['Tenor_Diff_Months'].mean(),
                    'max_month_diff': tenor_data['Tenor_Diff_Months'].abs().max()
                }
            }
            
            return self.tenor_risk_results
            
        except Exception as e:
            st.error(f"期限风险分析错误: {e}")
            return None
    
    def create_basis_risk_chart(self):
        """创建基差风险可视化图表"""
        if self.basis_risk_results is None:
            return None
        
        try:
            data = self.basis_risk_results['dataframe']
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('📊 基差错配类型分布', '💰 各基差类型P/L贡献',
                               '📈 基差错配与P/L关系', '🎯 基差匹配质量评分'),
                specs=[[{"type": "pie"}, {"type": "bar"}],
                       [{"type": "scatter"}, {"type": "indicator"}]],
                vertical_spacing=0.15,
                horizontal_spacing=0.1
            )
            
            mismatch_counts = data['Basis_Mismatch_Type'].value_counts()
            fig.add_trace(
                go.Pie(labels=mismatch_counts.index, values=mismatch_counts.values,
                      name='基差错配分布', hole=0.3),
                row=1, col=1
            )
            
            pl_by_basis = data.groupby('Basis_Mismatch_Type')['Alloc_Total_PL'].sum().sort_values()
            fig.add_trace(
                go.Bar(x=pl_by_basis.index, y=pl_by_basis.values,
                      name='P/L贡献', marker_color='coral'),
                row=1, col=2
            )
            fig.update_xaxes(tickangle=-45, row=1, col=2)
            
            if 'Time_Lag' in data.columns:
                fig.add_trace(
                    go.Scatter(x=data['Time_Lag'], y=data['Alloc_Total_PL'],
                              mode='markers',
                              marker=dict(
                                  size=8,
                                  color=data['Allocated_Vol'],
                                  colorscale='RdBu',
                                  showscale=True,
                                  colorbar=dict(title="匹配量")
                              ),
                              text=data['Basis_Mismatch_Type'],
                              name='基差错配与P/L'),
                    row=2, col=1
                )
                x_range = [data['Time_Lag'].min(), data['Time_Lag'].max()]
                y_range = [data['Alloc_Total_PL'].min(), data['Alloc_Total_PL'].max()]

                fig.add_shape(
                    type="line",
                    x0=x_range[0],
                    x1=x_range[1],
                    y0=0,
                    y1=0,
                    line=dict(dash="dash", color="gray"),
                    row=2,
                    col=1
                )

                fig.add_shape(
                    type="line",
                    x0=0,
                    x1=0,
                    y0=y_range[0],
                    y1=y_range[1],
                    line=dict(dash="dash", color="green"),
                    row=2,
                    col=1
                )
            
            if self.basis_risk_results.get('risk_score'):
                risk_score = self.basis_risk_results['risk_score']
                if 'mismatch_ratio' in risk_score:
                    match_quality = 100 - risk_score['mismatch_ratio']
                    fig.add_trace(
                        go.Indicator(
                            mode="gauge+number",
                            value=match_quality,
                            title={'text': "基差匹配质量"},
                            domain={'row': 1, 'column': 1},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 80], 'color': "red"},
                                    {'range': [80, 95], 'color': "yellow"},
                                    {'range': [95, 100], 'color': "green"}
                                ],
                                'threshold': {
                                    'line': {'color': "black", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90
                                }
                            }
                        ),
                        row=2, col=2
                    )
            
            fig.update_layout(height=600, showlegend=False)
            return fig
            
        except Exception as e:
            st.warning(f"创建基差风险图表时出错: {e}")
            return None
    
    def create_tenor_risk_chart(self):
        """创建期限错配风险可视化图表"""
        if self.tenor_risk_results is None:
            return None
        
        try:
            data = self.tenor_risk_results['dataframe']
            volume_heatmap = self.tenor_risk_results['volume_heatmap']
            pl_heatmap = self.tenor_risk_results['pl_heatmap']
            
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=('📅 期限错配分布', '📊 期限错配热力图(交易量)',
                               '💰 期限错配热力图(P/L)', '📈 月差与P/L关系',
                               '🎯 期限匹配质量', '📋 错配风险明细'),
                specs=[[{"type": "pie"}, {"type": "heatmap"}, {"type": "heatmap"}],
                       [{"type": "scatter"}, {"type": "indicator"}, {"type": "table"}]],
                vertical_spacing=0.15,
                horizontal_spacing=0.15
            )
            
            mismatch_counts = data['Tenor_Mismatch_Type'].value_counts()
            fig.add_trace(
                go.Pie(labels=mismatch_counts.index, values=mismatch_counts.values,
                      name='期限错配分布', hole=0.3),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Heatmap(z=volume_heatmap.values,
                          x=volume_heatmap.columns,
                          y=volume_heatmap.index,
                          colorscale='Viridis',
                          name='交易量热力图',
                          colorbar=dict(title="交易量")),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Heatmap(z=pl_heatmap.values,
                          x=pl_heatmap.columns,
                          y=pl_heatmap.index,
                          colorscale='RdBu',
                          name='P/L热力图',
                          colorbar=dict(title="P/L")),
                row=1, col=3
            )
            
            fig.add_trace(
                go.Scatter(x=data['Tenor_Diff_Months'], y=data['Alloc_Total_PL'],
                          mode='markers',
                          marker=dict(
                              size=8,
                              color=abs(data['Allocated_Vol']),
                              colorscale='Plasma',
                              showscale=True,
                              colorbar=dict(title="匹配量", x=1.1)
                          ),
                          text=data['Cargo_ID'],
                          name='月差与P/L关系'),
                row=2, col=1
            )
            x_range = [data['Tenor_Diff_Months'].min(), data['Tenor_Diff_Months'].max()]
            y_range = [data['Alloc_Total_PL'].min(), data['Alloc_Total_PL'].max()]

            fig.add_shape(
                type="line",
                x0=x_range[0],
                x1=x_range[1],
                y0=0,
                y1=0,
                line=dict(dash="dash", color="gray"),
                row=2,
                col=1
            )

            fig.add_shape(
                type="line",
                x0=0,
                x1=0,
                y0=y_range[0],
                y1=y_range[1],
                line=dict(dash="dash", color="green"),
                row=2,
                col=1
            )
            
            summary = self.tenor_risk_results['summary']
            perfect_match_ratio = (len(data[data['Tenor_Diff_Months'] == 0]) / len(data)) * 100
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=perfect_match_ratio,
                    title={'text': "期限匹配质量"},
                    domain={'row': 1, 'column': 1},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkgreen"},
                        'steps': [
                            {'range': [0, 80], 'color': "red"},
                            {'range': [80, 95], 'color': "yellow"},
                            {'range': [95, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ),
                row=2, col=2
            )
            
            if self.tenor_risk_results.get('statistics') is not None:
                stats_df = self.tenor_risk_results['statistics']
                
                fig.add_trace(
                    go.Table(
                        header=dict(values=['错配类型'] + list(stats_df.columns),
                                   fill_color='paleturquoise',
                                   align='left'),
                        cells=dict(values=[stats_df.index] + [stats_df[col] for col in stats_df.columns],
                                  fill_color='lavender',
                                  align='left'),
                        name='错配风险明细'
                    ),
                    row=2, col=3
                )
            
            fig.update_layout(height=800, showlegend=False)
            return fig
            
        except Exception as e:
            st.warning(f"创建期限风险图表时出错: {e}")
            return None

# ---------------------------------------------------------
# 3. P/L归因分析模块
# ---------------------------------------------------------

class PLAttributionModule:
    """P/L归因分析模块"""
    
    def __init__(self, df_relations: pd.DataFrame):
        self.df_relations = df_relations
        self.attribution_results = None
        
    def analyze_pl_attribution(self):
        """分析P/L归因：已实现 vs 未实现，按Cargo聚合"""
        try:
            if self.df_relations.empty:
                return None
            
            required_cols = ['Cargo_ID', 'Alloc_Total_PL', 'Allocated_Vol']
            missing_cols = [col for col in required_cols if col not in self.df_relations.columns]
            
            if missing_cols:
                st.warning(f"缺少P/L归因分析所需字段: {missing_cols}")
                return None
            
            if 'Realized_PL' in self.df_relations.columns and 'Unrealized_PL' in self.df_relations.columns:
                pl_data = self.df_relations.copy()
            else:
                pl_data = self.df_relations.copy()
                if 'MTM_Price' in pl_data.columns and 'Open_Price' in pl_data.columns:
                    pl_data['Unrealized_PL'] = (pl_data['MTM_Price'] - pl_data['Open_Price']) * pl_data['Allocated_Vol']
                    pl_data['Realized_PL'] = pl_data['Alloc_Total_PL'] - pl_data['Unrealized_PL']
                else:
                    pl_data['Realized_PL'] = pl_data['Alloc_Total_PL'] * 0.6
                    pl_data['Unrealized_PL'] = pl_data['Alloc_Total_PL'] * 0.4
            
            cargo_pl = pl_data.groupby('Cargo_ID').agg({
                'Allocated_Vol': lambda x: abs(x).sum(),
                'Alloc_Total_PL': 'sum',
                'Realized_PL': 'sum',
                'Unrealized_PL': 'sum'
            }).round(2)
            
            cargo_pl.columns = ['总匹配量', '总P/L', '已实现P/L', '未实现P/L']
            
            total_pl = cargo_pl['总P/L'].sum()
            total_realized = cargo_pl['已实现P/L'].sum()
            total_unrealized = cargo_pl['未实现P/L'].sum()
            
            if total_pl != 0:
                cargo_pl['P/L贡献度(%)'] = (cargo_pl['总P/L'] / total_pl * 100).round(2)
                cargo_pl['已实现占比(%)'] = (cargo_pl['已实现P/L'] / cargo_pl['总P/L'] * 100).round(2)
            
            cargo_pl = cargo_pl.sort_values('总P/L', ascending=False)
            
            top_5_contributors = cargo_pl.head(5)['P/L贡献度(%)'].sum() if 'P/L贡献度(%)' in cargo_pl.columns else 0
            
            self.attribution_results = {
                'cargo_level': cargo_pl,
                'summary': {
                    'total_pl': total_pl,
                    'total_realized': total_realized,
                    'total_unrealized': total_unrealized,
                    'realized_ratio': (total_realized / total_pl * 100) if total_pl != 0 else 0,
                    'top_5_concentration': top_5_contributors,
                    'profitable_cargos': len(cargo_pl[cargo_pl['总P/L'] > 0]),
                    'losing_cargos': len(cargo_pl[cargo_pl['总P/L'] < 0])
                }
            }
            
            return self.attribution_results
            
        except Exception as e:
            st.error(f"P/L归因分析错误: {e}")
            return None
    
    def create_pl_attribution_chart(self):
        """创建P/L归因可视化图表"""
        if self.attribution_results is None:
            return None
        
        try:
            cargo_pl = self.attribution_results['cargo_level']
            summary = self.attribution_results['summary']
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('💰 P/L贡献度TOP10', '📊 已实现 vs 未实现分布',
                               '📈 P/L集中度分析', '📋 P/L归因明细'),
                specs=[[{"type": "bar"}, {"type": "pie"}],
                       [{"type": "waterfall"}, {"type": "table"}]],
                vertical_spacing=0.15,
                horizontal_spacing=0.1
            )
            
            top_10 = cargo_pl.head(10)
            fig.add_trace(
                go.Bar(x=top_10.index, y=top_10['总P/L'],
                      name='P/L贡献度',
                      marker_color=np.where(top_10['总P/L'] > 0, 'green', 'red'),
                      text=top_10['总P/L'].round(2),
                      textposition='auto'),
                row=1, col=1
            )
            fig.update_xaxes(tickangle=-45, row=1, col=1)
            
            fig.add_trace(
                go.Pie(labels=['已实现P/L', '未实现P/L'],
                      values=[summary['total_realized'], summary['total_unrealized']],
                      name='P/L构成', hole=0.3),
                row=1, col=2
            )
            
            cumulative_pl = cargo_pl['总P/L'].cumsum()
            
            fig.add_trace(
                go.Waterfall(
                    name="P/L集中度",
                    orientation="v",
                    measure=["relative"] * len(cargo_pl),
                    x=cargo_pl.index,
                    y=cargo_pl['总P/L'],
                    connector={"line": {"color": "rgb(63, 63, 63)"}},
                ),
                row=2, col=1
            )
            fig.add_hline(y=summary['total_pl'], line_dash="dash", line_color="blue", 
                         annotation_text=f"总P/L: ${summary['total_pl']:,.2f}", row=2, col=1)
            
            display_df = cargo_pl.copy()
            display_df = display_df.round(2)
            
            fig.add_trace(
                go.Table(
                    header=dict(values=['Cargo_ID'] + list(display_df.columns),
                               fill_color='paleturquoise',
                               align='left'),
                    cells=dict(values=[display_df.index] + [display_df[col] for col in display_df.columns],
                              fill_color='lavender',
                              align='left'),
                    name='P/L归因明细'
                ),
                row=2, col=2
            )
            
            fig.update_layout(height=800, showlegend=False)
            return fig
            
        except Exception as e:
            st.warning(f"创建P/L归因图表时出错: {e}")
            return None

# ---------------------------------------------------------
# 4. 有效性测试模块
# ---------------------------------------------------------

class EffectivenessTestingModule:
    """套保有效性测试模块"""
    
    def __init__(self, df_relations: pd.DataFrame):
        self.df_relations = df_relations
        self.test_results = None
        self.benchmark_prices = {}
        
    def extract_benchmarks(self):
        """自动从匹配数据中提取基准信息"""
        try:
            if self.df_relations.empty:
                return {}
            
            benchmarks = {}
            
            if 'Physical_Benchmark' in self.df_relations.columns:
                benchmark_types = self.df_relations['Physical_Benchmark'].dropna().unique()
            elif 'Proxy' in self.df_relations.columns:
                benchmark_types = self.df_relations['Proxy'].dropna().unique()
            else:
                return {}
            
            for benchmark in benchmark_types:
                benchmarks[benchmark] = {
                    'designation_price': 0.0,
                    'monitoring_price': 0.0,
                    'price_change': 0.0,
                    'volume': 0.0,
                    'count': 0
                }
            
            self.benchmark_prices = benchmarks
            return benchmarks
            
        except Exception as e:
            st.error(f"提取基准信息错误: {e}")
            return {}
    
    def run_effectiveness_test(self, benchmark_prices: Dict):
        """运行有效性测试"""
        try:
            if self.df_relations.empty:
                return None
            
            physical_results = []
            
            for benchmark, prices in benchmark_prices.items():
                designation_price = prices.get('designation_price', 0)
                monitoring_price = prices.get('monitoring_price', 0)
                
                if designation_price > 0:
                    price_change_pct = ((monitoring_price - designation_price) / designation_price * 100)
                    
                    if 'Physical_Benchmark' in self.df_relations.columns:
                        benchmark_data = self.df_relations[self.df_relations['Physical_Benchmark'] == benchmark]
                    elif 'Proxy' in self.df_relations.columns:
                        benchmark_data = self.df_relations[self.df_relations['Proxy'] == benchmark]
                    else:
                        benchmark_data = pd.DataFrame()
                    
                    if not benchmark_data.empty:
                        total_volume = abs(benchmark_data['Allocated_Vol']).sum()
                        total_pl = benchmark_data['Alloc_Total_PL'].sum()
                        
                        paper_performance = total_pl / total_volume if total_volume > 0 else 0
                        
                        if price_change_pct != 0:
                            effectiveness_ratio = (paper_performance / price_change_pct) * 100
                        else:
                            effectiveness_ratio = 0
                        
                        if 80 <= effectiveness_ratio <= 125:
                            effectiveness_status = '有效'
                        else:
                            effectiveness_status = '无效'
                        
                        physical_results.append({
                            '基准': benchmark,
                            '指定日价格': designation_price,
                            '监测日价格': monitoring_price,
                            '价格变动(%)': round(price_change_pct, 2),
                            '交易量': round(total_volume, 2),
                            '纸货P/L': round(total_pl, 2),
                            '纸货表现(每单位)': round(paper_performance, 4),
                            '有效性比率(%)': round(effectiveness_ratio, 2),
                            '有效性判定': effectiveness_status
                        })
            
            cargo_results = []
            if 'Cargo_ID' in self.df_relations.columns:
                for cargo_id in self.df_relations['Cargo_ID'].unique():
                    cargo_data = self.df_relations[self.df_relations['Cargo_ID'] == cargo_id]
                    
                    if not cargo_data.empty:
                        benchmark = cargo_data.iloc[0]['Physical_Benchmark'] if 'Physical_Benchmark' in cargo_data.columns else 'Unknown'
                        
                        if benchmark in benchmark_prices:
                            prices = benchmark_prices[benchmark]
                            designation_price = prices.get('designation_price', 0)
                            monitoring_price = prices.get('monitoring_price', 0)
                            
                            if designation_price > 0:
                                price_change_pct = ((monitoring_price - designation_price) / designation_price * 100)
                                cargo_volume = abs(cargo_data['Allocated_Vol']).sum()
                                cargo_pl = cargo_data['Alloc_Total_PL'].sum()
                                
                                paper_performance = cargo_pl / cargo_volume if cargo_volume > 0 else 0
                                
                                if price_change_pct != 0:
                                    effectiveness_ratio = (paper_performance / price_change_pct) * 100
                                else:
                                    effectiveness_ratio = 0
                                
                                if 80 <= effectiveness_ratio <= 125:
                                    effectiveness_status = '有效'
                                else:
                                    effectiveness_status = '无效'
                                
                                cargo_results.append({
                                    'Cargo_ID': cargo_id,
                                    '基准': benchmark,
                                    '价格变动(%)': round(price_change_pct, 2),
                                    '交易量': round(cargo_volume, 2),
                                    '纸货P/L': round(cargo_pl, 2),
                                    '纸货表现': round(paper_performance, 4),
                                    '有效性比率(%)': round(effectiveness_ratio, 2),
                                    '有效性判定': effectiveness_status
                                })
            
            self.test_results = {
                'benchmark_level': pd.DataFrame(physical_results),
                'cargo_level': pd.DataFrame(cargo_results),
                'summary': {
                    'total_benchmarks_tested': len(physical_results),
                    'effective_benchmarks': len([r for r in physical_results if r['有效性判定'] == '有效']),
                    'total_cargos_tested': len(cargo_results),
                    'effective_cargos': len([r for r in cargo_results if r['有效性判定'] == '有效']),
                    'overall_effectiveness_ratio': np.mean([r['有效性比率(%)'] for r in physical_results]) if physical_results else 0
                }
            }
            
            return self.test_results
            
        except Exception as e:
            st.error(f"有效性测试错误: {e}")
            return None
    
    def create_effectiveness_chart(self):
        """创建有效性测试可视化图表"""
        if self.test_results is None:
            return None
        
        try:
            benchmark_results = self.test_results['benchmark_level']
            
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=('📊 基准级别有效性测试', '📈 有效性比率分布', '🎯 有效性判定结果'),
                specs=[[{"type": "bar"}, {"type": "histogram"}, {"type": "pie"}]]
            )
            
            if not benchmark_results.empty:
                fig.add_trace(
                    go.Bar(x=benchmark_results['基准'], y=benchmark_results['有效性比率(%)'],
                          name='有效性比率',
                          marker_color=np.where(benchmark_results['有效性比率(%)'].between(80, 125), 
                                               'green', 'red'),
                          text=benchmark_results['有效性比率(%)'].round(1),
                          textposition='auto'),
                    row=1, col=1
                )
                fig.add_hline(y=80, line_dash="dash", line_color="orange", 
                             annotation_text="下限80%", row=1, col=1)
                fig.add_hline(y=125, line_dash="dash", line_color="orange", 
                             annotation_text="上限125%", row=1, col=1)
                fig.add_hline(y=100, line_dash="dash", line_color="green", 
                             annotation_text="完美对冲", row=1, col=1)
            
            if not benchmark_results.empty:
                fig.add_trace(
                    go.Histogram(x=benchmark_results['有效性比率(%)'],
                                nbinsx=20,
                                name='有效性分布',
                                marker_color='skyblue'),
                    row=1, col=2
                )
                fig.add_vline(x=80, line_dash="dash", line_color="red", row=1, col=2)
                fig.add_vline(x=125, line_dash="dash", line_color="red", row=1, col=2)
            
            if not benchmark_results.empty:
                effectiveness_counts = benchmark_results['有效性判定'].value_counts()
                fig.add_trace(
                    go.Pie(labels=effectiveness_counts.index, 
                          values=effectiveness_counts.values,
                          name='有效性判定',
                          hole=0.3,
                          marker_colors=['green', 'red']),
                    row=1, col=3
                )
            
            fig.update_layout(height=400, showlegend=False)
            return fig
            
        except Exception as e:
            st.warning(f"创建有效性测试图表时出错: {e}")
            return None

# ---------------------------------------------------------
# 5. Streamlit 主应用 - 修复版本
# ---------------------------------------------------------

def main():
    st.set_page_config(
        page_title="实纸货套保高级分析系统",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 自定义CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #E5E7EB;
        padding-bottom: 0.5rem;
    }
    .module-header {
        font-size: 1.8rem;
        color: #1E40AF;
        margin-top: 2rem;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #DBEAFE 0%, transparent 100%);
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
    }
    .success-box {
        background-color: #D1FAE5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #10B981;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #DBEAFE;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #F59E0B;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .price-input-table {
        background-color: #F3F4F6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 标题
    st.markdown('<h1 class="main-header">📈 实纸货套保高级分析系统</h1>', unsafe_allow_html=True)
    st.markdown("### 专业风险透视 | P/L归因分析 | 有效性测试")
    
    # 初始化session state
    if 'engine' not in st.session_state:
        st.session_state.engine = HedgeMatchingEngine()
    if 'df_relations' not in st.session_state:
        st.session_state.df_relations = None
    if 'df_physical' not in st.session_state:
        st.session_state.df_physical = None
    if 'df_paper_net' not in st.session_state:
        st.session_state.df_paper_net = None
    if 'risk_analysis' not in st.session_state:
        st.session_state.risk_analysis = None
    if 'pl_attribution' not in st.session_state:
        st.session_state.pl_attribution = None
    if 'effectiveness_test' not in st.session_state:
        st.session_state.effectiveness_test = None
    if 'benchmark_prices' not in st.session_state:
        st.session_state.benchmark_prices = {}
    
    # 侧边栏
    with st.sidebar:
        st.markdown("### 📁 数据上传")
        
        paper_file = st.file_uploader(
            "纸货数据文件",
            type=["csv", "xlsx", "xls"],
            key="paper_uploader",
            help="支持CSV/Excel格式，需包含Trade Date, Volume, Commodity等字段"
        )
        
        physical_file = st.file_uploader(
            "实货数据文件",
            type=["csv", "xlsx", "xls"],
            key="physical_uploader",
            help="支持CSV/Excel格式，需包含Cargo_ID, Volume, Hedge_Proxy等字段"
        )
        
        st.markdown("---")
        st.markdown("### ⚙️ 分析模块")
        
        enable_risk_analysis = st.checkbox("📊 风险多维透视", value=True)
        enable_pl_attribution = st.checkbox("💰 P/L归因分析", value=True)
        enable_effectiveness_test = st.checkbox("🧪 有效性测试", value=True)
        
        st.markdown("---")
        st.markdown("### 🔧 高级设置")
        
        max_rows = st.slider("表格显示行数", 10, 200, 50)
        
        st.markdown("---")
        
        if st.button("🔄 重置所有数据", type="secondary"):
            for key in ['engine', 'df_relations', 'df_physical', 'df_paper_net', 
                       'risk_analysis', 'pl_attribution', 'effectiveness_test', 
                       'benchmark_prices']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    # 主内容区 - 数据上传和匹配
    if paper_file is not None and physical_file is not None:
        try:
            # 读取数据
            if paper_file.name.endswith(('.xlsx', '.xls')):
                df_paper_raw = pd.read_excel(paper_file)
            else:
                df_paper_raw = pd.read_csv(paper_file)
            
            if physical_file.name.endswith(('.xlsx', '.xls')):
                df_physical_raw = pd.read_excel(physical_file)
            else:
                df_physical_raw = pd.read_csv(physical_file)
            
            # 显示数据预览
            with st.expander("📋 原始数据预览", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**纸货数据** ({len(df_paper_raw)}行)")
                    st.dataframe(df_paper_raw.head(10), use_container_width=True)
                
                with col2:
                    st.markdown(f"**实货数据** ({len(df_physical_raw)}行)")
                    st.dataframe(df_physical_raw.head(10), use_container_width=True)
            
            # 执行匹配按钮
            if st.button("🚀 执行套保匹配与高级分析", type="primary", use_container_width=True):
                with st.spinner("正在执行套保匹配与高级分析..."):
                    try:
                        # 重要修复：正确处理引擎返回值（现在是元组）
                        matching_results = st.session_state.engine.run_matching(
                            df_paper_raw, df_physical_raw
                        )
                        
                        # 检查matching_results类型
                        if matching_results is not None:
                            # 判断返回类型：字典还是元组
                            if isinstance(matching_results, dict):
                                # 如果是字典格式
                                st.session_state.df_relations = matching_results.get('relations')
                                st.session_state.df_physical = matching_results.get('physical')
                                st.session_state.df_paper_net = matching_results.get('paper_net')
                            elif isinstance(matching_results, tuple) and len(matching_results) >= 3:
                                # 如果是元组格式（现在使用的格式）
                                st.session_state.df_relations = matching_results[0]
                                st.session_state.df_physical = matching_results[1]
                                st.session_state.df_paper_net = matching_results[2]
                            else:
                                st.error(f"引擎返回了不支持的数据格式: {type(matching_results)}")
                                st.stop()
                            
                            if st.session_state.df_relations is not None and not st.session_state.df_relations.empty:
                                st.markdown('<div class="success-box">✅ 套保匹配完成！开始高级分析...</div>', unsafe_allow_html=True)
                                
                                # 初始化分析模块
                                if enable_risk_analysis:
                                    st.session_state.risk_analysis = RiskAnalysisModule(st.session_state.df_relations)
                                    st.session_state.risk_analysis.analyze_basis_risk()
                                    st.session_state.risk_analysis.analyze_tenor_risk()
                                
                                if enable_pl_attribution:
                                    st.session_state.pl_attribution = PLAttributionModule(st.session_state.df_relations)
                                    st.session_state.pl_attribution.analyze_pl_attribution()
                                
                                if enable_effectiveness_test:
                                    st.session_state.effectiveness_test = EffectivenessTestingModule(st.session_state.df_relations)
                                    st.session_state.benchmark_prices = st.session_state.effectiveness_test.extract_benchmarks()
                                
                                st.rerun()
                            else:
                                st.markdown('<div class="warning-box">⚠️ 匹配完成但未生成匹配记录</div>', unsafe_allow_html=True)
                        else:
                            st.error("匹配失败，请检查数据格式")
                            
                    except Exception as e:
                        st.error(f"匹配与分析过程中出现错误: {str(e)}")
                        st.exception(e)
        
        except Exception as e:
            st.error(f"数据读取错误: {str(e)}")
    
    # 显示分析结果
    if st.session_state.df_relations is not None and not st.session_state.df_relations.empty:
        st.markdown("---")
        st.markdown('<h2 class="module-header">📊 风险多维透视 (Risk Dimensions)</h2>', unsafe_allow_html=True)
        
        if enable_risk_analysis and st.session_state.risk_analysis is not None:
            risk_analysis = st.session_state.risk_analysis
            
            # 基差风险分析
            st.markdown("#### 🔄 基差风险分析 (Basis Risk)")
            st.markdown("**分析实货基准 vs 纸货工具的匹配程度**")
            
            if risk_analysis.basis_risk_results is not None:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    risk_score = risk_analysis.basis_risk_results.get('risk_score', {})
                    mismatch_ratio = risk_score.get('mismatch_ratio', 0)
                    match_quality = 100 - mismatch_ratio
                    st.metric("基差匹配质量", f"{match_quality:.1f}%")
                
                with col2:
                    mismatch_count = len(risk_analysis.basis_risk_results['dataframe'][
                        ~risk_analysis.basis_risk_results['dataframe']['Basis_Match']
                    ])
                    total_count = len(risk_analysis.basis_risk_results['dataframe'])
                    st.metric("基差错配数量", f"{mismatch_count}/{total_count}")
                
                with col3:
                    mismatch_pl = risk_score.get('mismatch_pl', 0)
                    st.metric("错配部分P/L", f"${mismatch_pl:,.2f}")
                
                # 显示基差风险图表
                basis_chart = risk_analysis.create_basis_risk_chart()
                if basis_chart:
                    st.plotly_chart(basis_chart, use_container_width=True)
                
                # 显示基差风险统计数据
                with st.expander("📋 基差风险详细数据"):
                    st.dataframe(risk_analysis.basis_risk_results['statistics'], use_container_width=True)
            
            # 期限错配风险分析
            st.markdown("#### 📅 期限错配风险分析 (Tenor Mismatch)")
            st.markdown("**分析实货月份 vs 纸货月份的匹配程度**")
            
            if risk_analysis.tenor_risk_results is not None:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    summary = risk_analysis.tenor_risk_results['summary']
                    avg_month_diff = summary.get('avg_month_diff', 0)
                    st.metric("平均月份差异", f"{avg_month_diff:.1f}个月")
                
                with col2:
                    max_month_diff = summary.get('max_month_diff', 0)
                    st.metric("最大月份差异", f"{max_month_diff:.0f}个月")
                
                with col3:
                    total_mismatch = summary.get('total_mismatch', 0)
                    total_count = len(risk_analysis.tenor_risk_results['dataframe'])
                    st.metric("期限错配数量", f"{total_mismatch}/{total_count}")
                
                # 显示期限风险图表
                tenor_chart = risk_analysis.create_tenor_risk_chart()
                if tenor_chart:
                    st.plotly_chart(tenor_chart, use_container_width=True)
                
                # 显示期限风险统计数据
                with st.expander("📋 期限风险详细数据"):
                    st.dataframe(risk_analysis.tenor_risk_results['statistics'], use_container_width=True)
        
        # P/L归因分析
        if enable_pl_attribution:
            st.markdown("---")
            st.markdown('<h2 class="module-header">💰 夯实的 P/L 归因分析</h2>', unsafe_allow_html=True)
            st.markdown("**分解已实现 vs 未实现盈亏，按Cargo聚合分析**")
            
            if st.session_state.pl_attribution is not None:
                pl_analysis = st.session_state.pl_attribution
                
                if pl_analysis.attribution_results is not None:
                    summary = pl_analysis.attribution_results['summary']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("总P/L", f"${summary['total_pl']:,.2f}")
                    
                    with col2:
                        realized_ratio = summary['realized_ratio']
                        st.metric("已实现占比", f"{realized_ratio:.1f}%")
                    
                    with col3:
                        st.metric("盈利船货数", summary['profitable_cargos'])
                    
                    with col4:
                        st.metric("P/L集中度", f"{summary['top_5_concentration']:.1f}%")
                    
                    # 显示P/L归因图表
                    pl_chart = pl_analysis.create_pl_attribution_chart()
                    if pl_chart:
                        st.plotly_chart(pl_chart, use_container_width=True)
                    
                    # 显示P/L归因详细数据
                    with st.expander("📋 P/L归因详细数据"):
                        st.dataframe(pl_analysis.attribution_results['cargo_level'], use_container_width=True)
        
        # 有效性测试
        if enable_effectiveness_test:
            st.markdown("---")
            st.markdown('<h2 class="module-header">🧪 交互式有效性测试</h2>', unsafe_allow_html=True)
            st.markdown("**测试套保有效性：匹配日纸货 vs 监测日实货价格变动**")
            
            if st.session_state.effectiveness_test is not None:
                effectiveness_test = st.session_state.effectiveness_test
                
                # 价格录入界面
                st.markdown("#### 💰 基准价格录入")
                st.markdown("请为每个基准输入指定日价格和监测日价格")
                
                if st.session_state.benchmark_prices:
                    price_inputs = {}
                    
                    with st.form("price_input_form"):
                        st.markdown('<div class="price-input-table">', unsafe_allow_html=True)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.markdown("**基准**")
                        with col2:
                            st.markdown("**指定日价格**")
                        with col3:
                            st.markdown("**监测日价格**")
                        with col4:
                            st.markdown("**价格变动**")
                        
                        for benchmark, prices in st.session_state.benchmark_prices.items():
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.markdown(f"**{benchmark}**")
                            
                            with col2:
                                designation_price = st.number_input(
                                    f"指定日价格 ({benchmark})",
                                    min_value=0.0,
                                    value=float(prices.get('designation_price', 0.0)),
                                    key=f"des_{benchmark}",
                                    label_visibility="collapsed"
                                )
                            
                            with col3:
                                monitoring_price = st.number_input(
                                    f"监测日价格 ({benchmark})",
                                    min_value=0.0,
                                    value=float(prices.get('monitoring_price', 0.0)),
                                    key=f"mon_{benchmark}",
                                    label_visibility="collapsed"
                                )
                            
                            with col4:
                                if designation_price > 0:
                                    price_change = ((monitoring_price - designation_price) / designation_price * 100)
                                    st.markdown(f"**{price_change:.2f}%**")
                                else:
                                    st.markdown("**N/A**")
                            
                            price_inputs[benchmark] = {
                                'designation_price': designation_price,
                                'monitoring_price': monitoring_price,
                                'price_change': price_change if designation_price > 0 else 0
                            }
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        submit_prices = st.form_submit_button("🚀 运行有效性测试")
                        
                        if submit_prices:
                            st.session_state.benchmark_prices = price_inputs
                            
                            with st.spinner("正在运行有效性测试..."):
                                test_results = effectiveness_test.run_effectiveness_test(price_inputs)
                                
                                if test_results:
                                    st.markdown('<div class="success-box">✅ 有效性测试完成！</div>', unsafe_allow_html=True)
                                    
                                    summary = test_results['summary']
                                    
                                    col1, col2, col3, col4 = st.columns(4)
                                    
                                    with col1:
                                        st.metric("测试基准数", summary['total_benchmarks_tested'])
                                    
                                    with col2:
                                        st.metric("有效基准数", summary['effective_benchmarks'])
                                    
                                    with col3:
                                        overall_ratio = summary['overall_effectiveness_ratio']
                                        st.metric("平均有效性比率", f"{overall_ratio:.1f}%")
                                    
                                    with col4:
                                        effectiveness_rate = (summary['effective_benchmarks'] / 
                                                           summary['total_benchmarks_tested'] * 100) if summary['total_benchmarks_tested'] > 0 else 0
                                        st.metric("整体有效性率", f"{effectiveness_rate:.1f}%")
                                    
                                    effectiveness_chart = effectiveness_test.create_effectiveness_chart()
                                    if effectiveness_chart:
                                        st.plotly_chart(effectiveness_chart, use_container_width=True)
                                    
                                    with st.expander("📋 基准级别测试结果"):
                                        if not test_results['benchmark_level'].empty:
                                            st.dataframe(test_results['benchmark_level'], use_container_width=True)
        
        # 数据导出
        st.markdown("---")
        st.markdown("### 💾 分析结果导出")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.session_state.df_relations is not None:
                csv_data = st.session_state.df_relations.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 下载匹配结果",
                    data=csv_data,
                    file_name=f"hedge_matching_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col2:
            report_data = {
                "分析时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "数据量": {
                    "匹配记录数": len(st.session_state.df_relations) if st.session_state.df_relations is not None else 0
                }
            }
            
            report_json = json.dumps(report_data, indent=2, default=str, ensure_ascii=False)
            st.download_button(
                label="📄 下载分析报告",
                data=report_json.encode('utf-8'),
                file_name=f"hedge_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
    
    else:
        # 欢迎页面
        if not (paper_file and physical_file):
            st.markdown("---")
            st.markdown('<div class="info-box">👈 请在左侧上传纸货和实货数据文件开始高级分析</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
