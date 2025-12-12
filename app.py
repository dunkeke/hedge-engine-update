import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import io
from datetime import datetime

# ==============================================================================
# 导入核心引擎
# ==============================================================================
try:
    import hedge_engine as engine
except ImportError:
    st.error("❌ 严重错误: 找不到 hedge_engine.py 模块！请确保该文件在同一目录下。")
    st.stop()

# ==============================================================================
# UI 配置
# ==============================================================================
st.set_page_config(page_title="Hedge Master Pro", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #fff; border-radius: 5px; box-shadow: 0 2px 2px rgba(0,0,0,0.1); }
    .stDataFrame { border: 1px solid #e0e0e0; border-radius: 5px; }
    h1, h2, h3 { color: #2c3e50; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 辅助分析函数
# ==============================================================================

def analyze_risk_dimensions(df_rels, df_ph):
    """分析基差风险和期限风险"""
    # 1. 基差风险: 实货基准 vs 纸货品种
    # 需要将 df_rels (包含 Paper Info) 与 df_ph (包含 Pricing Benchmark) 连接
    # df_rels 已经有 Cargo_ID, Proxy (Paper), 我们需要去 df_ph 找 Pricing_Benchmark
    
    risk_df = df_rels.merge(df_ph[['Cargo_ID', 'Pricing_Benchmark', 'Target_Contract_Month']], on='Cargo_ID', how='left')
    
    # 标记是否完全匹配
    # 简单逻辑：如果 Proxy 包含在 Benchmark 名字里 (如 Brent vs Brent)，或者 Benchmark 包含 Proxy
    # 这里做个简单的字符串包含判断，实际可能需要更复杂的映射表
    def check_basis(row):
        p = str(row['Proxy']).upper()
        b = str(row['Pricing_Benchmark']).upper()
        if p in b or b in p:
            return "Perfect Match"
        else:
            return f"Basis Risk ({b} vs {p})"
            
    risk_df['Basis_Status'] = risk_df.apply(check_basis, axis=1)
    
    # 2. 期限风险: 实货月 vs 纸货月
    def check_tenor(row):
        t_ph = str(row['Target_Contract_Month']).upper() # 实货月 (来自 df_ph merge)
        t_pa = str(row['Month']).upper() # 纸货月
        if t_ph == 'NAN' or t_pa == 'NAN': return "Unknown"
        if t_ph == t_pa: return "Matched"
        return "Mismatched"
        
    risk_df['Tenor_Status'] = risk_df.apply(check_tenor, axis=1)
    
    return risk_df

def calculate_effectiveness(df_rels, df_ph, prices_map):
    """
    有效性测试逻辑
    df_rels: 包含纸货盈亏 (Delta Paper)
    df_ph: 实货信息
    prices_map: 用户录入的价格字典 {Benchmark: {'Start': 80, 'End': 75}}
    """
    # 1. 计算实货盈亏 (Delta Physical)
    # Delta Phy = Volume * (End_Price - Start_Price) * Direction_Sign
    # 买入实货(Buy, +Vol): 价格涨(End>Start) -> 盈利(+). 
    # 但有效性测试中，通常比较的是变动值。
    # 此时需注意：实货是现货头寸。
    
    results = []
    
    for idx, row in df_ph.iterrows():
        bench = row['Pricing_Benchmark']
        vol = row['Volume'] # 注意：这里应该用"已匹配的量"还是"总量"？通常用已匹配量来测有效性
        
        # 找到该实货对应的匹配记录，计算已匹配量
        cargo_rels = df_rels[df_rels['Cargo_ID'] == row['Cargo_ID']]
        if cargo_rels.empty: continue
        
        matched_vol = cargo_rels['Allocated_Vol'].abs().sum() # 绝对值
        # 恢复方向符号: 如果实货是Buy(正)，这里用正
        matched_vol_signed = matched_vol * (1 if row['Volume'] > 0 else -1)
        
        # 获取纸货总变动 (Delta Paper)
        # MTM_PL 列已经是 (MTM - Open) * Alloc_Vol
        # 如果 Alloc_Vol 是负数(空头)，价格跌(MTM<Open)，结果为正(盈利)。逻辑正确。
        delta_paper = cargo_rels['MTM_PL'].sum() + cargo_rels['Total_PL_Alloc'].sum() # 包含已实现和未实现
        
        # 获取实货价格
        price_info = prices_map.get(bench)
        if not price_info:
            delta_phy = 0
            note = "Missing Price"
        else:
            p_start = price_info['Start']
            p_end = price_info['End']
            # 实货盈亏 = 量 * (期末 - 期初)
            delta_phy = matched_vol_signed * (p_end - p_start)
            note = "Calculated"
            
        # 计算比率: - (Delta Paper / Delta Physical)
        # 理想情况：实货亏100，纸货赚100。 比率 = - (100 / -100) = 1.0 (100%)
        if abs(delta_phy) < 1:
            ratio = 0
            status = "N/A (No Phy Delta)"
        else:
            ratio = -1 * (delta_paper / delta_phy)
            if 0.8 <= ratio <= 1.25: status = "✅ Highly Effective"
            elif 0.5 <= ratio <= 1.5: status = "⚠️ Effective"
            else: status = "❌ Ineffective"
            
        results.append({
            'Cargo_ID': row['Cargo_ID'],
            'Benchmark': bench,
            'Matched_Vol': matched_vol,
            'Start_Price': price_info['Start'] if price_info else 0,
            'End_Price': price_info['End'] if price_info else 0,
            'Delta_Phy': delta_phy,
            'Delta_Paper': delta_paper,
            'Ratio': ratio,
            'Status': status
        })
        
    return pd.DataFrame(results)

# ==============================================================================
# 主程序
# ==============================================================================

col_logo, col_header = st.columns([1, 6])
with col_header:
    st.title("Hedge Master Pro 🛡️")
    st.markdown("**智能套保风险分析与有效性测试系统** | Powered by v22 Engine")

st.markdown("---")

# --- 侧边栏 ---
with st.sidebar:
    st.header("📂 数据中心")
    ticket_file = st.file_uploader("1. 上传纸货水单 (Ticket Data)", type=['xlsx', 'csv'])
    phys_file = st.file_uploader("2. 上传实货台账 (Physical Ledger)", type=['xlsx', 'csv'])
    
    st.markdown("---")
    run_btn = st.button("🚀 启动分析引擎", type="primary", use_container_width=True)
    
    st.markdown("### ⚙️ 引擎设置")
    st.caption("Core: FIFO Netting + Time Priority")
    st.caption("P/L: Unrealized MTM + Realized")

if run_btn and ticket_file and phys_file:
    # --- 1. 核心计算 ---
    with st.spinner("正在进行多维匹配与净仓计算..."):
        try:
            # 加载
            df_p, df_ph = engine.load_data_v19(ticket_file, phys_file)
            
            # 计算
            df_p_net = engine.calculate_net_positions_corrected(df_p)
            df_rels, df_ph_final, df_p_final = engine.auto_match_hedges(df_ph, df_p_net)
            
            # 将结果存入 Session State 防止刷新丢失
            st.session_state['results'] = {
                'rels': df_rels,
                'ph': df_ph_final,
                'p': df_p_final
            }
            st.success(f"计算完成！成功生成 {len(df_rels)} 条匹配记录")
            
        except Exception as e:
            st.error(f"引擎运行错误: {str(e)}")
            st.stop()

# --- 结果展示逻辑 ---
if 'results' in st.session_state:
    res = st.session_state['results']
    df_rels = res['rels']
    df_ph = res['ph']
    df_p = res['p']
    
    # 准备风险数据
    risk_df = analyze_risk_dimensions(df_rels, df_ph)

    # --- 顶部 KPI ---
    total_exp = df_ph['Volume'].abs().sum()
    unhedged = df_ph['Unhedged_Volume'].abs().sum()
    hedged_val = total_exp - unhedged
    # 纸货总盈亏 (MTM + Realized)
    total_paper_pl = df_rels['MTM_PL'].sum() + df_rels['Total_PL_Alloc'].sum() if not df_rels.empty else 0
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("📦 实货总敞口", f"{total_exp:,.0f} BBL", help="Physical Volume Total")
    k2.metric("🛡️ 套保覆盖率", f"{(hedged_val/total_exp*100):.1f}%", help="Hedged / Total")
    k3.metric("⚠️ 风险裸露", f"{unhedged:,.0f} BBL", delta_color="inverse")
    k4.metric("💰 纸货端总盈亏", f"${total_paper_pl:,.0f}", delta=f"MTM: {df_rels['MTM_PL'].sum():,.0f}" if not df_rels.empty else 0)

    # --- 功能标签页 ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 匹配概览 (Overview)", 
        "🔍 风险透视 (Risk Analysis)", 
        "🧪 有效性测试 (Effectiveness)",
        "📋 详细账本 (Ledger)"
    ])

    # === Tab 1: 匹配概览 ===
    with tab1:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("月度敞口与覆盖")
            if 'Target_Contract_Month' in df_ph.columns:
                chart_data = df_ph.groupby('Target_Contract_Month')[['Volume', 'Unhedged_Volume']].sum().abs().reset_index()
                chart_data['Hedged'] = chart_data['Volume'] - chart_data['Unhedged_Volume']
                fig = px.bar(chart_data, x='Target_Contract_Month', y=['Hedged', 'Unhedged_Volume'],
                             color_discrete_map={'Hedged': '#2ecc71', 'Unhedged_Volume': '#e74c3c'})
                st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            st.subheader("敞口构成")
            fig_pie = px.pie(values=[hedged_val, unhedged], names=['已套保', '未套保'], 
                             color_discrete_sequence=['#2ecc71', '#e74c3c'], hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)

    # === Tab 2: 风险透视 (新增维度) ===
    with tab2:
        if not df_rels.empty:
            st.info("💡 这里的分析基于已匹配的套保关系，展示基差风险和期限错配情况。")
            rc1, rc2 = st.columns(2)
            
            with rc1:
                st.subheader("1. 基差风险 (Basis Risk)")
                # 统计 Basis_Status
                basis_stats = risk_df.groupby(['Pricing_Benchmark', 'Proxy']).size().reset_index(name='Count')
                fig_sun = px.sunburst(basis_stats, path=['Pricing_Benchmark', 'Proxy'], values='Count',
                                      color='Pricing_Benchmark', title="实货基准 -> 纸货工具 映射")
                st.plotly_chart(fig_sun, use_container_width=True)
                
            with rc2:
                st.subheader("2. 期限错配 (Tenor Risk)")
                # 统计月份对应关系
                tenor_stats = risk_df.groupby(['Target_Contract_Month', 'Month'])['Allocated_Vol'].sum().abs().reset_index()
                fig_hm = px.density_heatmap(tenor_stats, x='Target_Contract_Month', y='Month', z='Allocated_Vol',
                                            title="实货月份(X) vs 纸货月份(Y) 热力图", color_continuous_scale='Viridis')
                st.plotly_chart(fig_hm, use_container_width=True)
        else:
            st.warning("暂无匹配数据，无法分析风险。")

    # === Tab 3: 有效性测试 (新增交互逻辑) ===
    with tab3:
        st.subheader("⚖️ 回顾性有效性测试 (Retrospective Testing)")
        st.markdown("通过对比 **实货价格变动 ($\Delta P_{phy}$)** 与 **纸货实际盈亏 ($\Delta P_{paper}$)** 来评估套保效果。")
        
        if not df_rels.empty:
            # 1. 提取所有用到的基准
            benchmarks = df_ph['Pricing_Benchmark'].unique()
            benchmarks = [b for b in benchmarks if str(b) != 'NAN' and b != '']
            
            if len(benchmarks) > 0:
                with st.form("price_input_form"):
                    st.markdown("#### 第一步：录入实货基准价格")
                    st.caption("请输入指定日(期初)和监测日(期末)的市场价格。对于 JCC，请输入指数值；对于 Brent，请输入月均价。")
                    
                    # 动态生成输入表格
                    input_data = []
                    cols = st.columns(len(benchmarks))
                    
                    price_inputs = {}
                    
                    for i, bench in enumerate(benchmarks):
                        with cols[i]:
                            st.markdown(f"**{bench}**")
                            p_start = st.number_input(f"期初价格 ({bench})", value=60.0, key=f"s_{bench}")
                            p_end = st.number_input(f"期末/当前价格 ({bench})", value=60.0, key=f"e_{bench}")
                            price_inputs[bench] = {'Start': p_start, 'End': p_end}
                    
                    submitted = st.form_submit_button("🔄 运行有效性测试")
                
                if submitted:
                    # 2. 运行计算
                    eff_df = calculate_effectiveness(df_rels, df_ph, price_inputs)
                    
                    # 3. 展示结果
                    st.markdown("#### 第二步：测试结果")
                    
                    # 总体统计
                    pass_count = len(eff_df[eff_df['Status'].str.contains("Effective")])
                    total_count = len(eff_df)
                    pass_rate = pass_count / total_count if total_count > 0 else 0
                    
                    st.metric("总体通过率 (80-125%)", f"{pass_rate:.1%}")
                    
                    # 详细表格
                    st.dataframe(eff_df.style.applymap(
                        lambda v: 'color: green; font-weight: bold' if 'Effective' in str(v) else 'color: red', 
                        subset=['Status']
                    ), use_container_width=True)
                    
                    # 下载
                    csv_eff = eff_df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 下载有效性报告", csv_eff, "effectiveness_report.csv", "text/csv")
                    
            else:
                st.warning("未在实货数据中识别到有效的 Pricing_Benchmark。")
        else:
            st.warning("请先完成匹配计算。")

    # === Tab 4: 详细账本 ===
    with tab4:
        st.subheader("📝 原始数据明细")
        
        show_option = st.radio("选择查看的数据表:", ["匹配明细 (Hedge Allocation)", "实货剩余敞口", "纸货剩余头寸"], horizontal=True)
        
        if show_option == "匹配明细 (Hedge Allocation)":
            if not df_rels.empty:
                st.dataframe(df_rels, use_container_width=True)
                st.download_button("📥 下载 CSV", df_rels.to_csv(index=False).encode('utf-8'), "allocation.csv")
            else:
                st.info("无数据")
                
        elif show_option == "实货剩余敞口":
            un = df_ph[abs(df_ph['Unhedged_Volume']) > 1]
            st.dataframe(un, use_container_width=True)
            
        else: # 纸货剩余
            if 'Allocated_To_Phy' in df_p.columns:
                df_p['Remaining'] = df_p['Volume'] - df_p['Allocated_To_Phy']
                un_p = df_p[abs(df_p['Remaining']) > 1]
                st.dataframe(un_p[['Recap No', 'Std_Commodity', 'Month', 'Volume', 'Remaining', 'Price']], use_container_width=True)
            else:
                st.warning("纸货数据未包含分配信息")

else:
    # 欢迎页
    st.info("👈 请在左侧上传文件并点击运行")
