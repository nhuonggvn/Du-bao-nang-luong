# Data Mining Project - Household Power Consumption ⚡

## Đề tài 10: Dự báo nhu cầu năng lượng (Energy Demand Forecasting)

Dự án này thực hiện quy trình Khai phá dữ liệu (Data Mining) trên bộ dữ liệu tiêu thụ điện năng của hộ gia đình (Household Power Consumption) nhằm mục tiêu thấu hiểu hành vi sử dụng điện, phân nhóm khách hàng và dự báo nhu cầu năng lượng trong tương lai.

---

## 📊 Nguồn dữ liệu
**Dataset**: [UCI Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

- **Mô tả**: Dữ liệu đo lường điện năng phút-phút của một hộ gia đình tại Pháp từ 2006 đến 2010.
- **Quy mô**: ~2,075,259 bản ghi.
- **Giá trị thiếu (Missing)**: Được ký hiệu bằng `?` (Cần xử lý trong giai đoạn tiền xử lý).

---

## 📂 Cấu trúc dự án
```text
Du-bao-nl/
├── README.md                    # Hướng dẫn dự án
├── app.py                       # Giao diện báo cáo Streamlit (Dashboard)
├── requirements.txt             # Danh sách thư viện cần thiết
├── configs/
│   └── params.yaml              # Cấu hình tham số mô hình & đường dẫn
├── data/
│   ├── raw/                     # Chứa file household_power_consumption.txt (Gốc)
│   └── processed/               # Dữ liệu sau khi làm sạch & feature engineering
├── src/                         # Mã nguồn module của dự án
│   ├── data/                    # Xử lý Load/Clean data
│   ├── features/                # Trích xuất đặc trưng
│   ├── mining/                  # Thuật toán Gom cụm & Luật kết hợp
│   ├── models/                  # Mô hình dự báo chuỗi thời gian
│   └── visualization/           # Vẽ biểu đồ kết quả
├── notebooks/                   # Quy trình thực hiện chi tiết theo bước
│   ├── 01_eda.ipynb             # Phân tích khám phá dữ liệu (EDA)
│   ├── 02_preprocessing.ipynb   # Tiền xử lý & làm sạch dữ liệu
│   ├── 03_mining.ipynb          # Luật kết hợp & Gom cụm (K-Means)
│   ├── 04_modeling.ipynb        # Dự báo chuỗi thời gian (ARIMA/SARIMA/HW)
│   ├── 05_anomaly_detection.ipynb # Phát hiện dị thường (Isolation Forest)
│   └── 06_evaluation_report.ipynb # Báo cáo đánh giá tổng kết
└── outputs/                     # Kết quả đầu ra của dự án
    ├── figures/                # Biểu đồ (PNG)
    ├── tables/                 # Bảng kết quả (CSV)
    └── models/                 # Lưu trữ các model đã train
```

---

## 🚀 Hướng dẫn thực hiện

### 1. Cài đặt môi trường
Đảm bảo bạn đã cài đặt Python (>= 3.8). Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu
Tải file `household_power_consumption.txt` và đặt vào thư mục `data/raw/`. (Nếu file đang ở thư mục gốc, hệ thống vẫn sẽ hoạt động nhờ cấu hình trong `params.yaml`).

### 3. Quy trình thực hiện (Notebooks)
Người dùng nên chạy các Notebook trong thư mục `notebooks/` theo thứ tự từ `01` đến `06` để tái lập toàn bộ quy trình từ xử lý dữ liệu đến huấn luyện mô hình và xuất kết quả.

### 4. Khởi chạy Dashboard (Demo)
Đây là cách nhanh nhất để xem kết quả phân tích cuối cùng thông qua giao diện trực quan:
```bash
streamlit run app.py
```

---

## 🛠️ Các thành phần kỹ thuật chính

1.  **Exploratory Data Analysis (EDA)**: Phân tích phân phối, xu hướng theo giờ/ngày/tháng và tính chu kỳ (Seasonality).
2.  **Association Rule Mining (Apriori)**: Tìm kiếm các quy luật kết hợp giữa các thiết bị đo (Sub_metering) khi tiêu thụ điện năng ở mức Cao/TB/Thấp.
3.  **Clustering (K-Means)**: Phân loại các "Daily Profiles" (Hành vi tiêu thụ 24h) thành các nhóm đặc trưng. Tự động chọn $K$ tối ưu bằng Elbow Method & Silhouette Score.
4.  **Time Series Forecasting**: So sánh các mô hình ARIMA, SARIMA và Holt-Winters để dự báo lượng điện tiêu thụ (`Global_active_power`).
5.  **Anomaly Detection**: Sử dụng **Isolation Forest** để phát hiện các thời điểm có biến động điện năng bất thường (sụt áp hoặc quá tải).

---

## 📈 Kết quả đạt được
- Hệ thống hóa toàn bộ quy trình Data Mining cho bài toán năng lượng.
- Dashboard trực quan hóa các Insights quan trọng cho phép theo dõi sai số mô hình (MAE, RMSE).
- Tự động hóa việc chạy notebook và sinh báo cáo.

---

## 👥 Tác giả
- Nguyễn Văn Hưởng

*Dự án được thực hiện cho học phần Dữ liệu lớn (Big Data).*
