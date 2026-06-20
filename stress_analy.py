import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

st.set_page_config(page_title="应力状态分析 · 智能学习系统", layout="wide")

# ========== 侧边栏 ==========
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/stress.png", width=80)
    st.title("📊 应力状态")
    st.markdown("**智能学习系统**")
    st.markdown("---")

    module = st.radio(
        "选择模块",
        ["📖 理论体系",
         "📊 平面应力计算",
         "🎨 莫尔圆交互",
         "💪 强度理论应用",
         "🏗️ 工程应用",
         "📐 案例分析",
         "🎨 应力云图",
         "🔄 3D应力可视化",
         "🧪 虚拟实验"]
    )
    st.markdown("---")
    st.caption("💡 交互式学习 | 实时可视化 | 工程实战")

# ========== 材料数据库 ==========
MATERIALS = {
    "Q235钢": {"E": 210, "nu": 0.30, "sigma_s": 235, "sigma_b": 400},
    "45钢": {"E": 210, "nu": 0.28, "sigma_s": 355, "sigma_b": 600},
    "40Cr钢": {"E": 210, "nu": 0.30, "sigma_s": 500, "sigma_b": 750},
    "铝合金": {"E": 70, "nu": 0.33, "sigma_s": 280, "sigma_b": 310},
    "铸铁": {"E": 120, "nu": 0.25, "sigma_s": 200, "sigma_b": 250},
    "铜合金": {"E": 110, "nu": 0.34, "sigma_s": 70, "sigma_b": 220},
}


def set_chinese_font(fig):
    fig.update_layout(
        font=dict(family="SimHei, Microsoft YaHei, Arial Unicode MS, sans-serif")
    )
    return fig


def principal_stresses(sx, sy, txy):
    """计算主应力"""
    avg = (sx + sy) / 2
    R = np.sqrt(((sx - sy) / 2) ** 2 + txy ** 2)
    sigma1 = avg + R
    sigma2 = avg - R
    tau_max = R
    theta_p = 0.5 * np.arctan2(2 * txy, sx - sy)
    return sigma1, sigma2, tau_max, theta_p, avg, R


def stress_on_plane(sx, sy, txy, theta):
    """计算任意斜截面上的应力"""
    theta_rad = theta * np.pi / 180
    sigma_theta = (sx + sy) / 2 + (sx - sy) / 2 * np.cos(2 * theta_rad) + txy * np.sin(2 * theta_rad)
    tau_theta = -(sx - sy) / 2 * np.sin(2 * theta_rad) + txy * np.cos(2 * theta_rad)
    return sigma_theta, tau_theta


# ============================================================
# 1. 理论体系
# ============================================================
if module == "📖 理论体系":
    st.title("📖 一点的应力状态 · 理论体系")

    tab1, tab2, tab3, tab4 = st.tabs(["📌 基本概念", "📐 核心公式", "📊 强度理论", "🔗 知识图谱"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### 📌 什么是应力状态？

            **一点的应力状态**是指通过某一点所有截面上的应力集合。

            ### 📌 为什么要研究应力状态？
            - 确定**最危险截面**和**最危险点**
            - 判断材料**失效形式**（屈服/断裂）
            - 为**强度理论**提供依据

            ### 📌 平面应力状态
            物体内某点的应力状态可用三个应力分量表示：
            - σ_x — x方向正应力
            - σ_y — y方向正应力  
            - τ_xy — xy平面切应力

            ### 📌 应力分量的符号约定
            - **正应力**：拉为正，压为负
            - **切应力**：使微元顺时针转动为正
            """)
        with col2:
            st.markdown("""
            ### 📌 主应力与主平面

            **主平面**：切应力为零的截面

            **主应力**：主平面上的正应力

            **主应力排序**：σ₁ ≥ σ₂ ≥ σ₃

            ### 📌 应力状态的分类
            | 类型 | 特点 |
            |------|------|
            | 单向应力 | 一个主应力非零 |
            | 二向应力 | 两个主应力非零 |
            | 三向应力 | 三个主应力非零 |
            | 纯剪切 | σ_x = -σ_y, τ_xy ≠ 0 |

            ### 📌 工程意义
            复杂受力构件的**强度设计**必须基于一点应力状态的分析。
            """)

    with tab2:
        st.markdown("""
        ### 📐 核心公式

        #### 1. 主应力公式
        $$\\sigma_{1,2} = \\frac{\\sigma_x + \\sigma_y}{2} \\pm \\sqrt{\\left(\\frac{\\sigma_x - \\sigma_y}{2}\\right)^2 + \\tau_{xy}^2}$$

        #### 2. 最大切应力
        $$\\tau_{\\max} = \\sqrt{\\left(\\frac{\\sigma_x - \\sigma_y}{2}\\right)^2 + \\tau_{xy}^2}$$

        #### 3. 主平面方位
        $$\\tan 2\\theta_p = \\frac{2\\tau_{xy}}{\\sigma_x - \\sigma_y}$$

        #### 4. 任意斜截面应力
        $$\\sigma_\\theta = \\frac{\\sigma_x+\\sigma_y}{2} + \\frac{\\sigma_x-\\sigma_y}{2}\\cos 2\\theta + \\tau_{xy}\\sin 2\\theta$$
        $$\\tau_\\theta = -\\frac{\\sigma_x-\\sigma_y}{2}\\sin 2\\theta + \\tau_{xy}\\cos 2\\theta$$

        #### 5. 莫尔圆方程
        $$\\left(\\sigma - \\frac{\\sigma_x+\\sigma_y}{2}\\right)^2 + \\tau^2 = \\left(\\frac{\\sigma_x-\\sigma_y}{2}\\right)^2 + \\tau_{xy}^2$$

        #### 6. 广义胡克定律
        $$\\varepsilon_x = \\frac{1}{E}(\\sigma_x - \\nu\\sigma_y)$$
        $$\\varepsilon_y = \\frac{1}{E}(\\sigma_y - \\nu\\sigma_x)$$
        $$\\gamma_{xy} = \\frac{\\tau_{xy}}{G} = \\frac{2(1+\\nu)}{E}\\tau_{xy}$$
        """)

    with tab3:
        st.markdown("""
        ### 📊 四大强度理论

        | 理论 | 名称 | 适用材料 | 等效应力公式 |
        |------|------|----------|--------------|
        | 第一 | 最大拉应力理论 | 脆性材料 | $\\sigma_{r1} = \\sigma_1$ |
        | 第二 | 最大拉应变理论 | 脆性材料 | $\\sigma_{r2} = \\sigma_1 - \\nu\\sigma_2$ |
        | 第三 | 最大切应力理论 | 塑性材料 | $\\sigma_{r3} = \\sigma_1 - \\sigma_3$ |
        | 第四 | 畸变能理论 | 塑性材料 | $\\sigma_{r4} = \\sqrt{\\frac{(\\sigma_1-\\sigma_2)^2+(\\sigma_2-\\sigma_3)^2+(\\sigma_3-\\sigma_1)^2}{2}}$ |

        #### 平面应力状态下的简化
        - 第三强度理论：$\\sigma_{r3} = \\sqrt{(\\sigma_x-\\sigma_y)^2 + 4\\tau_{xy}^2}$
        - 第四强度理论：$\\sigma_{r4} = \\sqrt{\\sigma_x^2 + \\sigma_y^2 - \\sigma_x\\sigma_y + 3\\tau_{xy}^2}$
        """)

    with tab4:
        st.markdown("### 🧠 知识图谱")
        st.graphviz_chart('''
        digraph {
            "应力状态" -> "平面应力"
            "应力状态" -> "主应力"
            "应力状态" -> "强度理论"
            "平面应力" -> "σ_x, σ_y, τ_xy"
            "平面应力" -> "任意截面应力"
            "主应力" -> "σ₁, σ₂"
            "主应力" -> "主平面"
            "主应力" -> "最大切应力"
            "强度理论" -> "第一理论(脆性)"
            "强度理论" -> "第二理论(脆性)"
            "强度理论" -> "第三理论(塑性)"
            "强度理论" -> "第四理论(塑性)"
            "任意截面应力" -> "莫尔圆"
        }
        ''')

# ============================================================
# 2. 平面应力计算
# ============================================================
elif module == "📊 平面应力计算":
    st.title("📊 平面应力状态计算器")

    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.markdown("### ⚙️ 输入应力分量")
        sigma_x = st.slider("σ_x (MPa)", -200, 300, 100, 5)
        sigma_y = st.slider("σ_y (MPa)", -150, 200, 40, 5)
        tau_xy = st.slider("τ_xy (MPa)", -120, 120, 30, 5)

        sigma1, sigma2, tau_max, theta_p, avg, R = principal_stresses(sigma_x, sigma_y, tau_xy)

        st.markdown("---")
        st.markdown("### 📊 主应力与主平面")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("主应力 σ₁", f"{sigma1:.2f} MPa")
            st.metric("主应力 σ₂", f"{sigma2:.2f} MPa")
        with col_b:
            st.metric("最大切应力 τ_max", f"{tau_max:.2f} MPa")
            st.metric("主平面方位 θ_p", f"{theta_p * 180 / np.pi:.1f}°")

    with col2:
        # 应力单元体3D显示
        st.markdown("### 📐 应力单元体")

        # 绘制应力单元体
        fig = go.Figure()

        # 单元体框架
        x = [-1, 1, 1, -1, -1]
        y = [-1, -1, 1, 1, -1]
        z = [-1, -1, -1, -1, -1]

        # 前面
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines',
                                   line=dict(color='black', width=2), name='单元体'))

        # 应力箭头（简化为文字标注）
        annotations = [
            dict(x=1.5, y=0, z=0, text=f"σ_x={sigma_x}", showarrow=False, font=dict(color='red', size=14)),
            dict(x=0, y=1.5, z=0, text=f"σ_y={sigma_y}", showarrow=False, font=dict(color='blue', size=14)),
            dict(x=1, y=1, z=0, text=f"τ_xy={tau_xy}", showarrow=False, font=dict(color='green', size=12)),
        ]

        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[-2, 2], showgrid=False, title='x'),
                yaxis=dict(range=[-2, 2], showgrid=False, title='y'),
                zaxis=dict(range=[-2, 2], showgrid=False, title='z'),
                annotations=annotations,
                aspectmode='cube'
            ),
            height=400,
            showlegend=False
        )
        fig = set_chinese_font(fig)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 3. 莫尔圆交互
# ============================================================
elif module == "🎨 莫尔圆交互":
    st.title("🎨 莫尔圆 · 交互式分析")
    st.markdown("拖动滑块观察应力状态变化，实时更新莫尔圆")

    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.markdown("### ⚙️ 应力分量")
        sigma_x = st.slider("σ_x (MPa)", -200, 300, 100, 5, key="mohr_sx")
        sigma_y = st.slider("σ_y (MPa)", -150, 200, 40, 5, key="mohr_sy")
        tau_xy = st.slider("τ_xy (MPa)", -120, 120, 30, 5, key="mohr_txy")

        sigma1, sigma2, tau_max, theta_p, avg, R = principal_stresses(sigma_x, sigma_y, tau_xy)

        st.markdown("---")
        st.markdown("### 📊 关键参数")
        st.metric("圆心 (σ_avg, 0)", f"({avg:.1f}, 0) MPa")
        st.metric("半径 R", f"{R:.2f} MPa")
        st.metric("主应力差 σ₁-σ₂", f"{sigma1 - sigma2:.2f} MPa")

        # 斜截面分析
        st.markdown("---")
        st.markdown("### 🔄 斜截面分析")
        theta_deg = st.slider("截面角度 θ (°)", -90, 90, 30, 5)
        sigma_theta, tau_theta = stress_on_plane(sigma_x, sigma_y, tau_xy, theta_deg)
        st.metric(f"θ = {theta_deg}° 截面", f"σ = {sigma_theta:.1f} MPa, τ = {tau_theta:.1f} MPa")

    with col2:
        # 绘制莫尔圆
        theta_circle = np.linspace(0, 2 * np.pi, 200)
        circle_x = avg + R * np.cos(theta_circle)
        circle_y = R * np.sin(theta_circle)

        fig = go.Figure()

        # 莫尔圆
        fig.add_trace(go.Scatter(x=circle_x, y=circle_y, mode='lines',
                                 name='莫尔圆', line=dict(color='blue', width=3)))

        # 坐标轴
        fig.add_hline(y=0, line_dash="dash", line_color="gray", name='')
        fig.add_vline(x=0, line_dash="dash", line_color="gray", name='')

        # 应力点
        fig.add_trace(go.Scatter(x=[sigma_x, sigma_y], y=[tau_xy, -tau_xy],
                                 mode='markers', marker=dict(color='red', size=14, symbol='circle'),
                                 name='给定应力点'))

        # 主应力点
        fig.add_trace(go.Scatter(x=[sigma1, sigma2], y=[0, 0],
                                 mode='markers', marker=dict(color='green', size=16, symbol='star'),
                                 name='主应力'))

        # 最大切应力点
        fig.add_trace(go.Scatter(x=[avg, avg], y=[tau_max, -tau_max],
                                 mode='markers', marker=dict(color='orange', size=12, symbol='diamond'),
                                 name='τ_max'))

        # 斜截面应力点
        if abs(theta_deg) > 0.5:
            sigma_theta_calc, tau_theta_calc = stress_on_plane(sigma_x, sigma_y, tau_xy, theta_deg)
            fig.add_trace(go.Scatter(x=[sigma_theta_calc], y=[tau_theta_calc],
                                     mode='markers', marker=dict(color='purple', size=18, symbol='x'),
                                     name=f'θ={theta_deg}°'))

        fig.update_layout(
            title="莫尔圆 · 应力状态可视化",
            xaxis_title="正应力 σ (MPa)",
            yaxis_title="切应力 τ (MPa)",
            xaxis=dict(zeroline=True),
            yaxis=dict(zeroline=True),
            height=500,
            showlegend=True,
            legend=dict(x=1.05, y=0.5)
        )
        fig = set_chinese_font(fig)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 4. 强度理论应用
# ============================================================
elif module == "💪 强度理论应用":
    st.title("💪 强度理论应用")
    st.markdown("基于应力状态选择强度理论进行校核")

    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.markdown("### ⚙️ 应力状态")
        sigma_x = st.slider("σ_x (MPa)", -200, 300, 100, 5, key="str_sx")
        sigma_y = st.slider("σ_y (MPa)", -150, 200, 40, 5, key="str_sy")
        tau_xy = st.slider("τ_xy (MPa)", -120, 120, 30, 5, key="str_txy")

        st.markdown("### 🔧 材料参数")
        material = st.selectbox("材料", list(MATERIALS.keys()), key="str_mat")
        mat = MATERIALS[material]
        sigma_s = mat["sigma_s"]
        n = st.slider("安全系数 n", 1.5, 4.0, 2.0, 0.5)
        sigma_allow = sigma_s / n

        sigma1, sigma2, tau_max, theta_p, avg, R = principal_stresses(sigma_x, sigma_y, tau_xy)

        st.markdown("---")
        st.markdown("### 📊 计算结果")
        st.metric("σ₁", f"{sigma1:.2f} MPa")
        st.metric("σ₂", f"{sigma2:.2f} MPa")
        st.metric("τ_max", f"{tau_max:.2f} MPa")

    with col2:
        # 四大强度理论计算
        sigma_r1 = sigma1
        sigma_r2 = sigma1 - mat["nu"] * sigma2
        sigma_r3 = sigma1 - sigma2
        sigma_r4 = np.sqrt(sigma1 ** 2 + sigma2 ** 2 - sigma1 * sigma2)

        st.markdown("### 📊 强度理论等效应力")

        data = {
            "理论": ["第一强度理论 (最大拉应力)", "第二强度理论 (最大拉应变)",
                     "第三强度理论 (最大切应力)", "第四强度理论 (畸变能)"],
            "等效应力 (MPa)": [f"{sigma_r1:.2f}", f"{sigma_r2:.2f}", f"{sigma_r3:.2f}", f"{sigma_r4:.2f}"],
            "许用应力 (MPa)": [f"{sigma_allow:.2f}"] * 4,
            "状态": ["✅" if sigma_r1 <= sigma_allow else "❌",
                     "✅" if sigma_r2 <= sigma_allow else "❌",
                     "✅" if sigma_r3 <= sigma_allow else "❌",
                     "✅" if sigma_r4 <= sigma_allow else "❌"]
        }
        df = pd.DataFrame(data)
        st.table(df)

        # 推荐理论
        st.markdown("### 💡 推荐")
        if material in ["铸铁"]:
            st.info("📌 推荐使用**第一或第二强度理论**（脆性材料）")
        else:
            st.info("📌 推荐使用**第三或第四强度理论**（塑性材料）")

        # 应力分量雷达图
        fig = go.Figure()
        categories = ["σ₁", "σ₂", "τ_max", "σ_r3", "σ_r4"]
        values = [sigma1, abs(sigma2), tau_max, sigma_r3, sigma_r4]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='应力值',
            line=dict(color='blue', width=2)
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, max(values) * 1.2])),
            title="应力分量雷达图",
            height=300
        )
        fig = set_chinese_font(fig)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 5. 工程应用
# ============================================================
elif module == "🏗️ 工程应用":
    st.title("🏗️ 工程应用案例")

    case = st.selectbox("选择案例", [
        "薄壁压力容器",
        "轴类零件弯扭组合",
        "焊接接头应力分析",
        "齿轮齿根应力"
    ])

    if case == "薄壁压力容器":
        st.markdown("""
        ### 🏗️ 薄壁压力容器应力分析

        **工程背景**：
        薄壁圆筒压力容器，内径 **D = 500 mm**，壁厚 **t = 10 mm**，
        内压 **p = 4 MPa**。
        求：筒壁的应力状态。
        """)

        col1, col2 = st.columns(2)
        with col1:
            D = st.slider("内径 D (mm)", 200, 1000, 500, 50)
            t = st.slider("壁厚 t (mm)", 3, 30, 10, 1)
            p = st.slider("内压 p (MPa)", 1, 20, 4, 0.5)

            # 薄壁容器应力公式
            sigma_hoop = p * D / (2 * t)  # 环向应力
            sigma_axial = p * D / (4 * t)  # 轴向应力
            tau = 0  # 无切应力

        with col2:
            st.markdown("### 📊 计算结果")
            st.metric("环向应力 σ_θ", f"{sigma_hoop:.2f} MPa")
            st.metric("轴向应力 σ_z", f"{sigma_axial:.2f} MPa")
            st.metric("径向应力 σ_r", "0 MPa (忽略)")
            st.metric("最大切应力", f"{sigma_hoop / 2:.2f} MPa")

            sigma1, sigma2, tau_max, theta_p, avg, R = principal_stresses(sigma_hoop, sigma_axial, 0)
            st.metric("主应力 σ₁", f"{sigma1:.2f} MPa")
            st.metric("主应力 σ₂", f"{sigma2:.2f} MPa")

    elif case == "轴类零件弯扭组合":
        st.markdown("""
        ### 🏗️ 轴类零件弯扭组合应力分析

        **工程背景**：
        某传动轴同时承受弯矩 **M = 2 kN·m** 和扭矩 **T = 1.5 kN·m**，
        轴直径 **d = 50 mm**。
        求：危险点的应力状态。
        """)

        col1, col2 = st.columns(2)
        with col1:
            M = st.slider("弯矩 M (kN·m)", 0.1, 10.0, 2.0, 0.1)
            T = st.slider("扭矩 T (kN·m)", 0.1, 8.0, 1.5, 0.1)
            d = st.slider("轴直径 d (mm)", 20, 120, 50, 2)

        with col2:
            I = np.pi * d ** 4 / 64
            W = I / (d / 2)
            J = np.pi * d ** 4 / 32

            sigma_b = M * 1000 / W  # MPa
            tau_t = T * 1000 / (J / (d / 2))  # MPa

            # 危险点应力状态
            sigma_x = sigma_b
            sigma_y = 0
            tau_xy = tau_t

            sigma1, sigma2, tau_max, theta_p, avg, R = principal_stresses(sigma_x, sigma_y, tau_xy)

            st.metric("弯曲正应力", f"{sigma_b:.2f} MPa")
            st.metric("扭转切应力", f"{tau_t:.2f} MPa")
            st.metric("主应力 σ₁", f"{sigma1:.2f} MPa")
            st.metric("主应力 σ₂", f"{sigma2:.2f} MPa")

# ============================================================
# 6. 案例分析
# ============================================================
elif module == "📐 案例分析":
    st.title("📐 案例分析：复杂应力状态分析")

    st.markdown("""
    ### 📌 问题描述

    某构件危险点处应力状态为：
    - σ_x = 120 MPa
    - σ_y = 40 MPa  
    - τ_xy = 50 MPa

    材料为 **45钢**（σ_s = 355 MPa，ν = 0.28）。
    要求：
    1. 计算主应力与最大切应力
    2. 绘制莫尔圆
    3. 用第四强度理论校核强度（n=2）
    """)

    col1, col2 = st.columns([1, 1.2])

    with col1:
        sigma_x = st.slider("σ_x (MPa)", 50, 200, 120, 5, key="case_sx")
        sigma_y = st.slider("σ_y (MPa)", 10, 100, 40, 5, key="case_sy")
        tau_xy = st.slider("τ_xy (MPa)", 10, 100, 50, 5, key="case_txy")
        material = st.selectbox("材料", ["45钢", "Q235钢", "铝合金"], key="case_mat")
        mat = {"45钢": {"sigma_s": 355, "nu": 0.28},
               "Q235钢": {"sigma_s": 235, "nu": 0.30},
               "铝合金": {"sigma_s": 280, "nu": 0.33}}[material]

    with col2:
        sigma1, sigma2, tau_max, theta_p, avg, R = principal_stresses(sigma_x, sigma_y, tau_xy)
        sigma_r4 = np.sqrt(sigma1 ** 2 + sigma2 ** 2 - sigma1 * sigma2)
        sigma_allow = mat["sigma_s"] / 2

        st.markdown("### 📊 计算结果")
        st.metric("主应力 σ₁", f"{sigma1:.2f} MPa")
        st.metric("主应力 σ₂", f"{sigma2:.2f} MPa")
        st.metric("最大切应力 τ_max", f"{tau_max:.2f} MPa")
        st.metric("主平面方位", f"{theta_p * 180 / np.pi:.1f}°")
        st.metric("第四强度理论 σ_r4", f"{sigma_r4:.2f} MPa")

        if sigma_r4 <= sigma_allow:
            st.success(f"✅ 强度满足: {sigma_r4:.1f} ≤ {sigma_allow:.1f} MPa")
        else:
            st.error(f"❌ 强度不满足: {sigma_r4:.1f} > {sigma_allow:.1f} MPa")

# ============================================================
# 7. 应力云图
# ============================================================
elif module == "🎨 应力云图":
    st.title("🎨 应力云图")
    st.markdown("可视化平面应力状态分布")

    col1, col2 = st.columns([1, 1.5])

    with col1:
        view_type = st.selectbox("显示类型", ["主应力分布", "切应力分布", "正应力分布"])
        sigma_x = st.slider("σ_x (MPa)", -100, 200, 80, 5, key="cloud_sx")
        sigma_y = st.slider("σ_y (MPa)", -80, 150, 30, 5, key="cloud_sy")
        tau_xy = st.slider("τ_xy (MPa)", -80, 80, 40, 5, key="cloud_txy")

    with col2:
        # 生成应力场
        x = np.linspace(-2, 2, 30)
        y = np.linspace(-2, 2, 30)
        X, Y = np.meshgrid(x, y)

        # 模拟应力分布
        if view_type == "主应力分布":
            Z = sigma_x * np.exp(-(X ** 2 + Y ** 2) / 2) + sigma_y * np.ones_like(X)
            title = "主应力分布"
        elif view_type == "切应力分布":
            Z = tau_xy * (np.sin(X * np.pi / 2) * np.cos(Y * np.pi / 2))
            title = "切应力分布"
        else:
            Z = sigma_x * np.cos(X * np.pi / 3) + sigma_y * np.sin(Y * np.pi / 3)
            title = "正应力分布"

        fig = go.Figure(data=go.Heatmap(
            z=Z, x=x, y=y,
            colorscale='RdBu_r',
            zmid=0,
            colorbar=dict(title="应力 (MPa)")
        ))
        fig.update_layout(title=f"{title} 云图", height=450)
        fig = set_chinese_font(fig)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 8. 3D应力可视化
# ============================================================
elif module == "🔄 3D应力可视化":
    st.title("🔄 3D应力状态可视化")
    st.markdown("三维展示应力单元体和应力张量")

    col1, col2 = st.columns([1, 1.5])

    with col1:
        sigma_x = st.slider("σ_x (MPa)", -100, 200, 80, 5, key="3d_sx")
        sigma_y = st.slider("σ_y (MPa)", -80, 150, 30, 5, key="3d_sy")
        sigma_z = st.slider("σ_z (MPa)", -80, 150, 20, 5)
        tau_xy = st.slider("τ_xy (MPa)", -80, 80, 40, 5, key="3d_txy")
        show_tensor = st.checkbox("显示应力张量", value=True)

    with col2:
        if show_tensor:
            stress_tensor = np.array([
                [sigma_x, tau_xy, 0],
                [tau_xy, sigma_y, 0],
                [0, 0, sigma_z]
            ])
            fig = go.Figure(data=go.Heatmap(
                z=stress_tensor,
                x=["σx", "τxy", "τxz"],
                y=["σx", "τyx", "τzx"],
                text=[[f"{v:.0f}" for v in row] for row in stress_tensor],
                texttemplate="%{text}",
                textfont={"size": 16},
                colorscale='RdBu_r',
                zmid=0
            ))
            fig.update_layout(title="应力张量", height=400)
            fig = set_chinese_font(fig)
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 9. 虚拟实验
# ============================================================
elif module == "🧪 虚拟实验":
    st.title("🧪 虚拟应力分析实验")

    st.markdown("""
    ### 🔬 虚拟实验：应力状态与莫尔圆

    调整应力分量，观察莫尔圆的变化。
    """)

    col1, col2 = st.columns([1, 1.5])

    with col1:
        sigma_x = st.slider("σ_x (MPa)", -100, 200, 80, 5, key="exp_sx")
        sigma_y = st.slider("σ_y (MPa)", -80, 150, 30, 5, key="exp_sy")
        tau_xy = st.slider("τ_xy (MPa)", -80, 80, 40, 5, key="exp_txy")

        sigma1, sigma2, tau_max, theta_p, avg, R = principal_stresses(sigma_x, sigma_y, tau_xy)

        st.markdown("---")
        st.markdown("### 📊 实验数据")
        st.metric("σ₁", f"{sigma1:.2f} MPa")
        st.metric("σ₂", f"{sigma2:.2f} MPa")
        st.metric("τ_max", f"{tau_max:.2f} MPa")

    with col2:
        # 莫尔圆（实时更新）
        theta_circle = np.linspace(0, 2 * np.pi, 200)
        circle_x = avg + R * np.cos(theta_circle)
        circle_y = R * np.sin(theta_circle)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=circle_x, y=circle_y, mode='lines',
                                 name='莫尔圆', line=dict(color='blue', width=3)))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.add_trace(go.Scatter(x=[sigma_x, sigma_y], y=[tau_xy, -tau_xy],
                                 mode='markers', marker=dict(color='red', size=12), name='应力点'))
        fig.add_trace(go.Scatter(x=[sigma1, sigma2], y=[0, 0],
                                 mode='markers', marker=dict(color='green', size=14, symbol='star'), name='主应力'))
        fig.update_layout(title="莫尔圆", xaxis_title="σ (MPa)", yaxis_title="τ (MPa)", height=450)
        fig = set_chinese_font(fig)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("📊 应力状态分析 · 智能学习系统 | 理论 + 计算 + 可视化")