import streamlit as st
from PIL import Image
import pandas as pd
import os

st.set_page_config(page_title="Energy Demand Analytics", page_icon="⚡", layout="wide")

# Custom CSS cho giao diện thêm đẹp
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem !important;
        font-weight: 700 !important;
        color: #ff4b4b;
        text-align: center;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 30px;
    }
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .sub-title { color: #bdc3c7; }
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown('<p class="main-title">Phân Tích & Dự Báo Tiêu Thụ Năng Lượng</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Bộ dữ liệu: Household Power Consumption</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2933/2933116.png", width=150)
    st.markdown("## Thông tin Dự Án")
    st.info("""
    **Bộ dữ liệu:** Điện năng Hộ gia đình  
    **Quy mô:** Hơn 2 triệu bản ghi  
    **Mục tiêu Khoa học:**  
    - Khám phá Tương quan (EDA)  
    - Tìm luật kết hợp thiết bị  
    - Gom cụm hành vi sử dụng  
    - Dự báo chuỗi thời gian  
    - Giám sát bất thường  
    """)
    st.divider()
    st.success(" Hệ thống đã được đào tạo và xuất báo cáo tự động bằng Papermill.")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Tổng quan EDA", 
    "2. Luật kết hợp", 
    "3. Gom cụm", 
    "4. Dự báo",
    "5. Dị thường"
])

def load_and_display_image(img_path, caption):
    if os.path.exists(img_path):
        st.image(img_path, caption=caption, use_container_width=True)
    else:
        st.error(f"⚠️ Không tìm thấy ảnh tại: `{img_path}`. Vui lòng chạy các Notebook để sinh output.")

with tab1:
    st.header("Khám Phá Cấu Trúc Dữ Liệu Năng Lượng")
    st.markdown("Bước đầu tiên trong quy trình Data Mining nhằm thấu hiểu bức tranh tổng quan, chuẩn đoán phân bố và phát hiện đặc tính chu kỳ thời gian (Seasonality) của mạng lưới.")
    
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("####  Phân bố Công suất Điện")
            load_and_display_image('outputs/figures/power_distribution.png', "Phân bố cường độ")
    with col2:
        with st.container(border=True):
            st.markdown("####  Ma trận Tương Quan")
            load_and_display_image('outputs/figures/correlation_heatmap.png', "Độ tương quan Pearson")
    
    st.divider()
    st.markdown("### Đặc điểm Mùa & Chu kỳ thay đổi (Seasonality)")
    with st.container(border=True):
        load_and_display_image('outputs/figures/daily_pattern.png', "Tính chu kỳ thời gian biến đổi trong ngày")

with tab2:
    st.header("Cấu trúc Giỏ Hàng Điện Năng (Apriori Rules)")
    st.info(" **Insights:** Khai phá dữ liệu dạng giỏ hàng (Market Basket Analysis) thực hiện trên giá trị điện rời rạc (Cao/TB/Thấp). Bảng này cho thấy quy luật kết hợp: khi một phòng bật thiết bị công suất cao, thiết bị khác cũng bị tác động thế nào.")
    
    try:
        rules_df = pd.read_csv('outputs/tables/association_rules.csv')
        st.dataframe(rules_df, use_container_width=True, hide_index=True)
    except:
        st.error("Chưa có dữ liệu `association_rules.csv`.")

with tab3:
    st.header("Phân Loại Hành Vi Khách Hàng (K-Means Clustering)")
    st.info(" **Insights:** Thuật toán K-Means phân loại 24 múi giờ tiêu thụ điện trong ngày thành 3 Cụm (Cluster) đặc trưng. Điển hình như nhóm dùng nhiều điện ban đêm, nhóm dùng sưởi ban ngày.")
    
    col1, col2 = st.columns([1, 1.5])
    with col1:
        with st.container(border=True):
            st.markdown("#### Đánh giá K tự động (Elbow Method)")
            load_and_display_image('outputs/figures/kmeans_elbow_method.png', "Khuỷu tay chọn K")
    with col2:
        with st.container(border=True):
            st.markdown("#### Hình hài 3 Cụm Hành Vi")
            load_and_display_image('outputs/figures/cluster_hourly_profiles.png', "Biểu đồ 24h của 3 nhóm Hộ gia đình")

with tab4:
    st.header("Mô phỏng & Dự Báo Thời Gian Thực (Time Series)")
    
    st.markdown("#### So Sánh Sai Số Các Mô Hình Máy Học")
    try:
        metrics_df = pd.read_csv('outputs/tables/forecast_comparison.csv')
        # Format df output, highlighting lowest MAE and RMSE
        st.dataframe(metrics_df.style.highlight_min(axis=0, color='lightgreen', subset=['RMSE', 'MAE']), use_container_width=True)
    except:
        st.error("Chưa có bảng Metrics.")
        
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("#### So sánh Đường đi Dự Báo")
            load_and_display_image('outputs/figures/forecast_comparison.png', "SARIMA vs ARIMA vs Holt-Winters")
    with col2:
        with st.container(border=True):
            st.markdown("#### Phân Tích Phần Dư (Residual)")
            load_and_display_image('outputs/figures/residuals_analysis.png', "Kiểm định mô hình hợp lệ")

with tab5:
    st.header("Hệ Thống Phân Tích Các Cơn Sốc Điện (Anomaly)")
    st.error(" Nhánh **Máy học Không giám sát** sử dụng kỹ thuật Rừng Cách Ly (Isolation Forest) để bóc tách 1% những ngày bị trạm biến áp sụt áp hoặc hộ dân tiêu thụ cực đoạn ngoài dự tính.")
    
    with st.container(border=True):
        load_and_display_image('outputs/figures/anomaly_detection.png', "Chấm đỏ đại diện cho dị thường")
