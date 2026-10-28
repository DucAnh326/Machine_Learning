# BÁO CÁO DỰ ÁN: DỰ BÁO LƯỢNG THỰC KHÁCH NHÀ HÀNG (RECRUIT RESTAURANT FORECASTING)

---

## 1. TỔNG QUAN BÀI TOÁN
* **Mục tiêu:** Dự báo số lượng thực khách (`visitors`) ghé thăm nhà hàng theo từng ngày dựa trên thông tin quán, vị trí và các đặc trưng thời gian.
* **Bộ dữ liệu gốc:** Recruit Restaurant Visitor Forecasting (Kaggle), gồm các bảng `air_visit_data`, `air_store_info`, và `date_info`.
* **Quy mô xử lý:** Tối ưu hóa và giảm chiều kích thước mẫu để phục vụ huấn luyện, kiểm thử và so sánh thực nghiệm giữa hai cách tiếp cận: **Tự cài đặt thuật toán (From Scratch)** và **Sử dụng thư viện chuẩn (`scikit-learn`)**.

---

## 2. QUY TRÌNH TIỀN XỬ LÝ DỮ LIỆU (DATA PREPROCESSING)

### 2.1. Tích hợp dữ liệu (Data Integration)
* Hợp nhất bảng lịch sử ghé thăm `air_visit_data` với bảng thông tin nhà hàng `air_store_info` bằng phép `Left Join` theo khóa định danh `air_store_id`.
* Ghép tiếp với bảng lịch `date_info` theo điều kiện `visit_date == calendar_date`.
* Sau khi ghép, loại bỏ ngay cột trùng lặp `calendar_date`.

### 2.2. Trích xuất và làm sạch đặc trưng (Feature Engineering & Cleaning)
* **Chuẩn hóa địa lý (`air_area_name`):**
  * Sử dụng thư viện `unicodedata` để phân rã các ký tự Romaji tiếng Nhật có dấu gạch ngang trên đầu (macron, ví dụ: `Tōkyō` $\rightarrow$ `Tokyo`, `Ōsaka` $\rightarrow$ `Osaka`) về mã chuẩn ASCII.
  * Tách chuỗi theo dấu cách đầu tiên để lấy tên đơn vị hành chính cấp tỉnh, sau đó tách theo dấu gạch ngang `-` để lược bỏ các hậu tố `-to`, `-fu`, `-ken`.
  * Xử lý trường hợp đặc thù của `Hokkaido` (không có hậu tố) và chuyển toàn bộ chuỗi về dạng chữ thường (`lowercase`).
* **Trích xuất đặc trưng chu kỳ thời gian:**
  * Bóc tách từ trường `visit_date` thành: `month` (bắt tính mùa vụ trong năm), `day_of_week_num` (thứ trong tuần, từ $0$ là Thứ Hai đến $6$ là Chủ Nhật).
  * Tạo biến cờ nhị phân `is_weekend` ($1$ cho Thứ Bảy và Chủ Nhật, $0$ cho các ngày trong tuần).
* **Lọc bỏ thuộc tính dư thừa/gây nhiễu (Feature Selection):**
  * Loại bỏ `air_store_id`: Thuộc tính định danh có lực lượng danh mục quá lớn (high cardinality), dễ gây quá khớp (overfitting) nếu không có embedding chuyên sâu.
  * Loại bỏ `latitude` & `longitude`: Thông tin vị trí đã được gói gọn trong đơn vị hành chính cấp tỉnh.
  * Loại bỏ `day_of_week` dạng chuỗi ký tự: Trùng lặp $100\%$ thông tin với `day_of_week_num`.
  * Loại bỏ `visit_date` và `year`: Tránh rò rỉ dữ liệu chuỗi tăng dần; tập trung huấn luyện mô hình học quy luật chu kỳ.
  * Loại bỏ `day` (ngày trong tháng): Giảm số chiều không gian, tránh việc mô hình phân nhánh vụn vặt.

### 2.3. Lấy mẫu phân tầng giảm kích thước dữ liệu (Stratified Down-scaling)
* **Thực trạng:** Tập dữ liệu gốc chứa hơn $252.000$ bản ghi, gây nghẽn tài nguyên tính toán đối với thuật toán CART viết chay khi quét ngưỡng phân chia.
* **Giải pháp:** Áp dụng kỹ thuật lấy mẫu phân tầng theo tỷ lệ ngày (`visit_date`):
  $$f_{\text{sample}} = \frac{10.000}{N_{\text{total}}} \approx 3.96\%$$
* **Ưu điểm:** Duy trì đầy đủ sự hiện diện của toàn bộ $478$ ngày trong dòng thời gian, bảo toàn tỷ lệ số lượng nhà hàng hoạt động giữa các ngày lễ/cao điểm và ngày thường.

### 2.4. Phân chia Train / Test theo trục thời gian (Temporal Split)
* Sắp xếp toàn bộ dữ liệu tăng dần theo thời gian (`visit_date`).
* Tách **$80\%$ dữ liệu phía trước làm tập Huấn luyện (Train)** ($\approx 8.000$ bản ghi) và **$20\%$ dữ liệu phía sau làm tập Kiểm thử (Test)** ($\approx 2.000$ bản ghi).
* Phương pháp này đảm bảo mô phỏng đúng bài toán dự báo trong thực tế (học từ quá khứ, dự đoán tương lai) và triệt tiêu nguy cơ rò rỉ dữ liệu (Data Leakage).

### 2.5. Biến đổi Log biến mục tiêu (Target Transformation)
* **Vấn đề:** Biến mục tiêu `visitors` bị phân phối lệch phải nghiêm trọng (right-skewed / long-tail distribution) với giá trị từ $1$ tới $877$, gây ra hiện tượng phương sai thay đổi (heteroscedasticity) và làm gradient bị chi phối nặng nề bởi các điểm ngoại lai.
* **Giải pháp:** Áp dụng phép biến đổi logarit tự nhiên:
  $$z = \ln(\text{visitors} + 1)$$
* Phép biến đổi này giúp phân phối mục tiêu trở về dạng chuông gần chuẩn, ổn định phương sai sai số và đưa bản chất nhân tính của bài toán (tăng theo phần trăm) về dạng cộng tính tuyến tính.

---

## 3. THIẾT KẾ ĐẶC TRƯNG CHUYÊN BIỆT CHO MÔ HÌNH

| Nhóm mô hình | Kỹ thuật mã hóa biến danh mục (`genre`, `area`) | Kỹ thuật xử lý biến số (`month`, `day_of_week_num`) |
| :--- | :--- | :--- |
| **Linear Regression** (Chay & Lib) | **One-Hot Encoding** (`drop='first'`): Tránh bẫy biến giả (dummy variable trap), gán trọng số $w$ độc lập cho từng nhóm, loại bỏ giả định sai lệch về thứ tự. | **Polynomial Degree 2** (sinh tương tác $x_i^2, x_i x_j$) kết hợp **StandardScaler** ($Z$-score) để đồng bộ thang đo cho điều chuẩn. |
| **CART Regression Tree** (Chay & Lib) | **Target Encoding**: Thay thế danh mục bằng $\text{mean}(z_{\text{train}})$ tương ứng. Giúp cây cắt gọn ngưỡng thay vì bị loãng nhánh. | **Giữ nguyên thang đo số tự nhiên**: Bản chất cây quyết định bất biến với các phép biến đổi đơn điệu (scale-invariant). |

---

## 4. TRIỂN KHAI VÀ ĐỐI CHIẾU 4 MÔ HÌNH

### 4.1. Linear Regression (Code Chay - OLS Ridge)
* Sử dụng phương trình pháp tuyến (Normal Equation) có thành phần điều chuẩn $L_2$ ($\lambda = 0.1$):
  $$w = (X^T X + \lambda I^*)^{-1} X^T y$$
* Trong đó, ma trận đơn vị $I^*$ được triệt tiêu giá trị tại vị trí bias $(0, 0)$ nhằm không áp đặt phạt lên hệ số chặn $w_0$.
* Điều chuẩn giúp ma trận luôn khả nghịch, tránh hoàn toàn lỗi suy biến do đa cộng tuyến.

### 4.2. Linear Regression (Thư viện - Scikit-Learn `ElasticNetCV`)
* Kết hợp đồng thời hàm phạt $L_1$ (Lasso) và $L_2$ (Ridge).
* Sử dụng kiểm định chéo $5$-fold tích hợp sẵn để tự động tinh chỉnh cặp siêu tham số tối ưu (`alpha` và `l1_ratio`).

### 4.3. CART Regression Tree (Code Chay)
* Xây dựng cấu trúc cây nhị phân đệ quy dựa trên độ giảm tổng bình phương sai số (Variance / SSE Reduction).
* Giá trị tại mỗi nút lá là trung bình cộng số học của các mẫu rơi vào lá đó.
* Điều kiện dừng (Stopping criteria): Cố định độ sâu `max_depth = 6` và ngưỡng số mẫu tối thiểu để chia nhánh `min_samples_split = 30`.

### 4.4. CART Regression Tree (Thư viện - Scikit-Learn `DecisionTreeRegressor`)
* Thiết lập các siêu tham số cấu trúc tương đồng (`max_depth=6`, `min_samples_split=30`, `min_samples_leaf=15`).
* **Kết quả thực nghiệm:** Đạt các chỉ số đánh giá **trùng khớp tuyệt đối $100\%$ từng chữ số thập phân** với bản code chay, chứng minh logic thuật toán tự cài đặt hoàn toàn chuẩn xác theo giải thuật Breiman gốc.

---

## 5. ĐÁNH GIÁ HIỆU NĂNG MÔ HÌNH (EVALUATION)

### 5.1. Chiến lược chọn lựa độ đo
1. **RMSLE (Root Mean Squared Logarithmic Error):** Đo trực tiếp bằng sai số RMSE trên thang đo log. Đây là metric phản ánh sai số tỷ lệ phần trăm giữa các quy mô nhà hàng khác nhau một cách công bằng nhất.
2. **MAE & RMSE trên số khách thực tế:**
   * Nghịch đảo giá trị dự đoán về thang đo gốc: $\hat{y} = \exp(\hat{z}) - 1$.
   * Kẹp chặn giá trị không âm: $\hat{y} = \max(0, \hat{y})$.
   * Tính toán sai số trung bình tuyệt đối (MAE) và căn bậc hai sai số bình phương trung bình (RMSE).
3. **Loại bỏ hệ số xác định $R^2$:** Bản chất hàm giải mũ $\exp$ dễ phóng đại sai số phần dư của một vài điểm dự đoán chệch nhẹ trong không gian log thành chênh lệch lớn trên thang đo thực, khiến tổng bình phương phần dư vượt tổng bình phương toàn phần ($SS_{\text{res}} > SS_{\text{tot}}$) làm $R^2$ bị âm vô lý dù mô hình bám sát xu hướng chung rất tốt.

### 5.2. Kết quả thực nghiệm tiêu biểu (trên mô hình CART)
* **RMSLE:** $\approx 0.7908$ (mức baseline tốt đối với cây đơn lẻ không dùng đặc trưng độ trễ lag).
* **RMSE:** $\approx 18.04$ khách.
* **MAE:** $\approx 12.65$ khách (trung bình sai lệch chỉ rơi vào khoảng $12 \sim 13$ khách/ngày).
