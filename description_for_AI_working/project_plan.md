# **House Price Prediction Using MLP Regression**

## **Phase 0. Define Problem**

Dự án tập trung vào bài toán dự đoán giá nhà dựa trên các đặc trưng như vị trí, diện tích, số phòng, số tầng, năm xây dựng, chất lượng nhà và các thông tin liên quan khác. Đây là bài toán **Regression** vì biến mục tiêu cần dự đoán là **giá nhà**, một giá trị số liên tục.

Mục tiêu chính của dự án là xây dựng mô hình **Multilayer Perceptron Regression bằng PyTorch** để dự đoán giá nhà. Sau đó, mô hình MLP sẽ được so sánh với các mô hình hồi quy truyền thống như **Linear Regression, Decision Tree Regression và Random Forest Regression** nhằm đánh giá hiệu suất dự đoán.

## **Phase 1. Data Collection**

### **Data Source**

Dataset sử dụng từ Kaggle:

https://www.kaggle.com/datasets/vallabhadattap/kingcountyhousing

### **Lý do chọn dataset**

Dataset King County Housing phù hợp với bài toán dự đoán giá nhà vì có quy mô tương đối lớn, khoảng **21,613 mẫu dữ liệu**, đủ để huấn luyện, validation và testing mô hình Machine Learning. Bộ dữ liệu chứa nhiều đặc trưng quan trọng ảnh hưởng đến giá nhà như diện tích, số phòng ngủ, số phòng tắm, vị trí, chất lượng nhà, năm xây dựng và tình trạng nhà. Ngoài ra, dữ liệu có ít missing values hơn so với nhiều bộ dữ liệu nhà ở khác, giúp giảm khối lượng tiền xử lý.

### **Công việc cần thực hiện**

| **Nội dung**          | **Mục đích**                               |
|-----------------------|--------------------------------------------|
| Load file dữ liệu     | Đọc dataset vào môi trường Python          |
| Kiểm tra shape        | Xem số dòng, số cột                        |
| Kiểm tra tên cột      | Hiểu các feature ban đầu                   |
| Kiểm tra kiểu dữ liệu | Phân biệt numerical, categorical, datetime |
| Xác định target       | Biến mục tiêu là price                     |

## **Phase 2. Train / Validation / Test Split theo tỷ lệ 80/10/10**

Sau khi load dữ liệu, cần chia dữ liệu ngay thành ba tập:

| **Tập dữ liệu** | **Tỷ lệ** | **Vai trò**                                                                  |
|-----------------|-----------|------------------------------------------------------------------------------|
| Training set    | 80%       | Dùng để huấn luyện mô hình                                                   |
| Validation set  | 10%       | Dùng để theo dõi quá trình training, tuning hyperparameter và early stopping |
| Testing set     | 10%       | Dùng để đánh giá cuối cùng sau khi mô hình đã hoàn tất                       |

Việc chia dữ liệu sớm giúp tránh **data leakage**. Các bước như xử lý missing values, scaling, encoding hoặc feature engineering nên được fit trên training set, sau đó chỉ transform validation set và testing set.

## **Phase 3. Data Overview and EDA**

EDA nên được thực hiện chủ yếu trên **training set** để tránh nhìn trước thông tin từ validation và testing set.

### **Nội dung cần kiểm tra**

| **Nội dung EDA**    | **Mục đích**                            |
|---------------------|-----------------------------------------|
| head()              | Xem 5 dòng dữ liệu đầu tiên             |
| info()              | Kiểm tra kiểu dữ liệu và missing values |
| describe()          | Xem thống kê mô tả của các biến số      |
| Target distribution | Kiểm tra phân phối giá nhà              |
| Correlation matrix  | Xem mối quan hệ giữa các biến           |
| Scatter plot        | Xem quan hệ giữa feature và price       |
| Boxplot             | Phát hiện outlier                       |
| Histogram           | Kiểm tra phân phối của từng feature     |

### **Visualization bắt buộc nên có**

| **Biểu đồ**                        | **Ý nghĩa**                             |
|------------------------------------|-----------------------------------------|
| Histogram của price                | Xem giá nhà có bị lệch phải hay không   |
| Boxplot của price                  | Phát hiện outlier về giá                |
| Correlation heatmap                | Xác định feature liên quan mạnh đến giá |
| Scatter plot sqft_living vs price  | Kiểm tra quan hệ diện tích và giá       |
| Price by location / zipcode        | Xem ảnh hưởng của vị trí                |
| Pairplot một số feature quan trọng | Khảo sát không gian đặc trưng ban đầu   |

## **Phase 4. Data Cleaning**

Ở phase này, dữ liệu được làm sạch trước khi đưa vào mô hình.

### **Các bước xử lý**

| **Bước**                   | **Nội dung**                                                     |
|----------------------------|------------------------------------------------------------------|
| Missing values             | Kiểm tra và xử lý giá trị thiếu                                  |
| Duplicates                 | Kiểm tra và loại bỏ bản ghi trùng lặp nếu có                     |
| Outliers                   | Kiểm tra outlier ở price, sqft_living, bedrooms, bathrooms       |
| Invalid values             | Xử lý giá trị bất hợp lý, ví dụ số phòng ngủ bằng 0 hoặc quá lớn |
| Remove unnecessary columns | Loại bỏ cột ít ý nghĩa như id nếu không dùng                     |
| Date processing            | Tách date thành year, month nếu cần                              |

### **Lưu ý chuyên ngành**

Không nên xóa outlier một cách máy móc. Với giá nhà, một số outlier có thể là nhà thật sự đắt do vị trí đẹp, diện tích lớn hoặc view tốt. Vì vậy, cần kiểm tra ngữ cảnh trước khi loại bỏ.

## **Phase 5. Feature Engineering**

Feature Engineering giúp mô hình học tốt hơn bằng cách tạo ra các đặc trưng có ý nghĩa hơn từ dữ liệu gốc.

### **Các hướng xử lý**

| **Nhóm xử lý**         | **Ví dụ**                                                  |
|------------------------|------------------------------------------------------------|
| Tạo feature mới        | house_age = sale_year - yr_built                           |
| Tạo feature cải tạo    | renovated = 1 nếu yr_renovated \> 0                        |
| Feature theo diện tích | price_per_sqft, living_lot_ratio                           |
| Xử lý biến thời gian   | Tách năm, tháng từ ngày bán                                |
| Xử lý skewness         | Có thể dùng log1p(price) nếu giá nhà lệch phải mạnh        |
| Encoding               | One-Hot Encoding cho biến categorical như zipcode nếu dùng |
| Feature selection      | Chọn feature có ý nghĩa, loại feature gây nhiễu            |

### **Khảo sát không gian đặc trưng**

Nên trực quan hóa không gian đặc trưng sau khi xử lý bằng:

| **Visualization**                | **Mục đích**                                          |
|----------------------------------|-------------------------------------------------------|
| PCA 2D plot                      | Giảm chiều feature về 2D để quan sát cấu trúc dữ liệu |
| PCA colored by price             | Xem vùng nào tương ứng giá cao/thấp                   |
| Feature correlation heatmap      | Kiểm tra đa cộng tuyến                                |
| Random Forest feature importance | Tham khảo feature quan trọng từ mô hình baseline      |

## **Phase 6. Split Data into X and y**

Sau khi xác định biến mục tiêu:

| **Thành phần** | **Ý nghĩa**           |
|----------------|-----------------------|
| X              | Tập đặc trưng đầu vào |
| y              | Giá nhà cần dự đoán   |

Ví dụ:

X = các cột đặc trưng sau xử lý

y = price

Nếu price bị lệch phải mạnh, có thể dùng:

y_log = log1p(price)

Khi dự đoán xong cần chuyển ngược bằng:

price_pred = expm1(y_pred_log)

## **Phase 7. Preprocessing Pipeline bằng scikit-learn**

Phase này sử dụng **scikit-learn** để xử lý dữ liệu trước khi đưa vào PyTorch.

### **Các bước chính**

| **Bước**                 | **Thư viện** | **Mục đích**                                        |
|--------------------------|--------------|-----------------------------------------------------|
| Missing value imputation | sklearn      | Điền missing values                                 |
| One-Hot Encoding         | sklearn      | Mã hóa biến phân loại                               |
| StandardScaler           | sklearn      | Chuẩn hóa biến số                                   |
| ColumnTransformer        | sklearn      | Tách pipeline cho numerical và categorical features |
| Pipeline                 | sklearn      | Gộp các bước preprocessing                          |

### **Nguyên tắc chống data leakage**

| **Tập dữ liệu** | **Cách xử lý**  |
|-----------------|-----------------|
| Training set    | fit_transform() |
| Validation set  | transform()     |
| Testing set     | transform()     |

Không được fit scaler hoặc encoder trên toàn bộ dữ liệu.

## **Phase 8. Feature Scaling**

MLP là mô hình Neural Network nên dữ liệu đầu vào cần được chuẩn hóa để quá trình học ổn định hơn.

### **Lý do cần scaling**

| **Lý do**                            | **Giải thích**                               |
|--------------------------------------|----------------------------------------------|
| Tránh feature lớn áp đảo feature nhỏ | Ví dụ sqft_living lớn hơn bedrooms rất nhiều |
| Giúp gradient ổn định                | Neural Network học tốt hơn                   |
| Giúp hội tụ nhanh hơn                | Loss giảm đều và ít dao động hơn             |
| Giảm nguy cơ exploding gradient      | Đặc biệt khi learning rate cao               |

### **Phương pháp sử dụng**

StandardScaler

Công thức:

z = (x - mean) / standard deviation

## **Phase 9. Create PyTorch Dataset**

Sau khi dữ liệu đã được chia và chuẩn hóa, cần chuyển dữ liệu sang dạng tensor để PyTorch có thể xử lý.

### **Các Dataset cần tạo**

| **Dataset**        | **Dữ liệu**                    |
|--------------------|--------------------------------|
| Training Dataset   | X_train_tensor, y_train_tensor |
| Validation Dataset | X_val_tensor, y_val_tensor     |
| Testing Dataset    | X_test_tensor, y_test_tensor   |

### **Yêu cầu tensor**

| **Thành phần**  | **Shape mong muốn**     |
|-----------------|-------------------------|
| Features tensor | (n_samples, n_features) |
| Target tensor   | (n_samples, 1)          |

Target cần reshape về dạng (n_samples, 1) để phù hợp với output layer của MLP Regression.

## **Phase 10. Create PyTorch DataLoader**

Sau khi tạo Dataset, cần tạo DataLoader cho từng tập dữ liệu.

### **DataLoader cần có**

| **DataLoader** | **Shuffle** | **Vai trò**           |
|----------------|-------------|-----------------------|
| train_loader   | True        | Dùng để train mô hình |
| val_loader     | False       | Dùng để validation    |
| test_loader    | False       | Dùng để testing       |

### **Vai trò của DataLoader**

| **Vai trò**           | **Ý nghĩa**                                      |
|-----------------------|--------------------------------------------------|
| Batch training        | Chia dữ liệu thành từng batch nhỏ                |
| Shuffle training data | Giúp mô hình học ổn định hơn                     |
| Quản lý training loop | Giúp code PyTorch rõ ràng                        |
| Tối ưu bộ nhớ         | Không cần đưa toàn bộ dữ liệu vào model cùng lúc |

## **Phase 11. Build MLP Regression Model with PyTorch**

Ở phase này, mô hình **Multilayer Perceptron Regression** được xây dựng bằng PyTorch.

### **Cấu trúc mô hình**

Mô hình cần kế thừa từ:

torch.nn.Module

### **Kiến trúc đề xuất**

| **Thành phần**      | **Vai trò**                   |
|---------------------|-------------------------------|
| Input layer         | Nhận số lượng feature đầu vào |
| Hidden layer 1      | Học quan hệ phi tuyến ban đầu |
| Hidden layer 2      | Học đặc trưng phức tạp hơn    |
| Activation function | ReLU                          |
| Dropout             | Giảm overfitting nếu cần      |
| Output layer        | 1 neuron để dự đoán giá nhà   |

### **Ví dụ kiến trúc**

Input → Linear → ReLU → Dropout → Linear → ReLU → Linear → Output(1)

### **Lưu ý**

MLP Regression không có **centroid** và không có **K** như K-Means. Với MLP, quá trình học được theo dõi thông qua:

weights, bias, loss, gradient, epoch, validation loss, learning curve

## **Phase 12. Model Configuration**

Trước khi huấn luyện, cần xác định các hyperparameters.

| **Hyperparameter**      | **Ý nghĩa**                           |
|-------------------------|---------------------------------------|
| Number of hidden layers | Số lớp ẩn                             |
| Hidden units            | Số neuron trong từng lớp              |
| Activation function     | Hàm kích hoạt, ví dụ ReLU             |
| Learning rate           | Tốc độ cập nhật trọng số              |
| Batch size              | Số mẫu trong một batch                |
| Number of epochs        | Số vòng lặp qua toàn bộ training set  |
| Optimizer               | Adam hoặc SGD                         |
| Loss function           | MSELoss                               |
| Dropout rate            | Tỷ lệ dropout để chống overfitting    |
| Weight decay            | L2 regularization                     |
| Patience                | Số epoch chờ trước khi early stopping |

## **Phase 13. Train MLP with PyTorch**

Quá trình huấn luyện MLP được viết bằng PyTorch training loop.

### **Thành phần chính**

| **Thành phần**         | **Cấu hình đề xuất** |
|------------------------|----------------------|
| Model                  | MLP Regression       |
| Loss function          | nn.MSELoss()         |
| Optimizer              | torch.optim.Adam()   |
| Metric theo dõi        | MAE, MSE, RMSE, R²   |
| Training mode          | model.train()        |
| Evaluation mode        | model.eval()         |
| No gradient validation | torch.no_grad()      |

### **Quy trình trong mỗi epoch**

| **Bước** | **Nội dung**                                        |
|----------|-----------------------------------------------------|
| 1        | Chuyển model sang training mode bằng model.train()  |
| 2        | Lặp qua từng batch trong train_loader               |
| 3        | Dự đoán output                                      |
| 4        | Tính loss                                           |
| 5        | Reset gradient bằng optimizer.zero_grad()           |
| 6        | Backpropagation bằng loss.backward()                |
| 7        | Cập nhật trọng số bằng optimizer.step()             |
| 8        | Chuyển model sang evaluation mode bằng model.eval() |
| 9        | Tính validation loss trên val_loader                |
| 10       | Lưu log của epoch                                   |

## **Phase 14. Training Log and Monitoring**

Trong quá trình train, cần vừa run vừa log để theo dõi mô hình học có tốt không.

### **Bảng log cần lưu sau mỗi epoch**

| **Thông tin cần log** | **Ý nghĩa**                                 |
|-----------------------|---------------------------------------------|
| Epoch                 | Epoch hiện tại                              |
| Train Loss            | Sai số trên training set                    |
| Validation Loss       | Sai số trên validation set                  |
| Train MAE             | Sai số tuyệt đối trung bình trên train      |
| Validation MAE        | Sai số tuyệt đối trung bình trên validation |
| Validation RMSE       | Sai số căn bậc hai trung bình               |
| Validation R²         | Mức độ giải thích biến động giá nhà         |
| Learning Rate         | Theo dõi tốc độ học                         |
| Gradient Norm         | Kiểm tra gradient có ổn định không          |
| Epoch Time            | Thời gian chạy mỗi epoch                    |
| Best Epoch            | Epoch có validation loss tốt nhất           |

### **File nên xuất ra**

training_log.csv

File này dùng để vẽ curve và chứng minh quá trình huấn luyện minh bạch.

## **Phase 15. Visualize Training Process**

Đây là phase rất quan trọng vì giúp chứng minh mô hình học thật sự, không chỉ chạy ra kết quả cuối.

### **Visualization bắt buộc nên có**

| **Biểu đồ**                   | **Mục đích**                               |
|-------------------------------|--------------------------------------------|
| Train Loss vs Validation Loss | Theo dõi hội tụ và overfitting             |
| MAE Curve                     | Xem sai số tuyệt đối giảm qua từng epoch   |
| RMSE Curve                    | Xem sai số lớn có giảm không               |
| R² Curve                      | Xem khả năng giải thích dữ liệu tăng không |
| Learning Rate Curve           | Theo dõi learning rate nếu dùng scheduler  |
| Gradient Norm Curve           | Phát hiện vanishing/exploding gradient     |
| Prediction Snapshot           | So sánh actual vs predicted ở một số epoch |
| Residual Distribution         | Kiểm tra sai số dự đoán sau khi train      |

### **Cách nhận biết mô hình hội tụ**

| **Dấu hiệu**                        | **Diễn giải**                                 |
|-------------------------------------|-----------------------------------------------|
| Train loss giảm dần                 | Mô hình đang học từ training data             |
| Validation loss giảm dần            | Mô hình generalize tốt hơn                    |
| Train loss và val loss cùng ổn định | Mô hình có khả năng hội tụ                    |
| Validation loss không giảm thêm     | Có thể đã đến điểm tối ưu thực tế             |
| Train loss giảm nhưng val loss tăng | Có dấu hiệu overfitting                       |
| Loss dao động mạnh                  | Learning rate có thể quá lớn                  |
| Loss gần như không giảm             | Learning rate quá nhỏ hoặc model underfitting |

## **Phase 16. Early Stopping and Best Model Saving**

Early Stopping giúp dừng training khi mô hình không còn cải thiện trên validation set.

### **Nguyên tắc hoạt động**

Sau mỗi epoch, tính validation loss:

| **Trường hợp**                | **Hành động**                 |
|-------------------------------|-------------------------------|
| Validation loss giảm          | Lưu model tốt nhất            |
| Validation loss không giảm    | Tăng patience counter         |
| Patience counter đạt giới hạn | Dừng training                 |
| Loss bị NaN                   | Dừng và kiểm tra lỗi          |
| Gradient quá lớn              | Có thể dùng gradient clipping |

### **Điều kiện dừng đề xuất**

| **Điều kiện**      | **Giá trị gợi ý**          |
|--------------------|----------------------------|
| Max epochs         | 100 đến 300                |
| Patience           | 10 đến 20 epoch            |
| Min delta          | 0.0001 hoặc tùy scale loss |
| Best metric        | Validation loss thấp nhất  |
| Restore best model | Có                         |

### **Output cần lưu**

best_mlp_model.pth

training_log.csv

early_stopping_summary.json

## **Phase 17. Hyperparameter Exploration**

Phase này dùng để khảo sát mô hình và tìm cấu hình tốt hơn.

### **Các hyperparameter cần thử**

| **Nhóm khảo sát** | **Giá trị có thể thử** |
|-------------------|------------------------|
| Learning rate     | 0.01, 0.001, 0.0005    |
| Hidden units      | 64, 128, 256           |
| Hidden layers     | 1, 2, 3                |
| Batch size        | 32, 64, 128            |
| Dropout           | 0.0, 0.1, 0.2          |
| Weight decay      | 0, 1e-5, 1e-4          |

### **Output cần có**

| **Output**                       | **Ý nghĩa**                |
|----------------------------------|----------------------------|
| Hyperparameter result table      | So sánh các cấu hình       |
| Validation loss by configuration | Chọn model tốt nhất        |
| Best configuration               | Cấu hình cuối              |
| Training time                    | So sánh chi phí huấn luyện |

### **Lưu ý**

Trong bài MLP Regression, không khảo sát theo **K** như K-Means. Thay vào đó, cần khảo sát theo **số layer, số neuron, learning rate, batch size, dropout và weight decay**.

## **Phase 18. Evaluate MLP Model on Test Set**

Sau khi huấn luyện xong và chọn best model, mô hình MLP được đánh giá trên testing set.

### **Metrics sử dụng**

| **Metric** | **Ý nghĩa**                                          |
|------------|------------------------------------------------------|
| MAE        | Sai số tuyệt đối trung bình                          |
| MSE        | Sai số bình phương trung bình                        |
| RMSE       | Sai số cùng đơn vị với giá nhà                       |
| R² Score   | Tỷ lệ biến thiên của giá nhà được mô hình giải thích |

### **Diễn giải kết quả**

Mô hình tốt thường có:

MAE thấp

MSE thấp

RMSE thấp

R² Score cao

Tuy nhiên cần phân tích thêm residual để biết mô hình sai nhiều ở nhóm nhà giá thấp, trung bình hay cao.

## **Phase 19. Predictions and Visualization**

Sau khi đánh giá, mô hình được dùng để dự đoán giá nhà trên testing set.

### **Visualization cần có**

| **Biểu đồ**                             | **Mục đích**                                |
|-----------------------------------------|---------------------------------------------|
| Actual Price vs Predicted Price         | Xem dự đoán gần đường y=x không             |
| Residual Plot                           | Kiểm tra sai số có phân bố ngẫu nhiên không |
| Residual Histogram                      | Kiểm tra sai số lệch trái/phải              |
| Error by Price Range                    | Xem model sai nhiều ở nhóm giá nào          |
| Top 20 Largest Errors                   | Phân tích các case dự đoán sai nhất         |
| Training Loss and Validation Loss Curve | Chứng minh quá trình học                    |

### **Cách diễn giải**

Nếu điểm dự đoán nằm gần đường chéo trong biểu đồ Actual vs Predicted, mô hình dự đoán tương đối tốt. Nếu residual phân bố ngẫu nhiên quanh 0, mô hình ít bị bias hệ thống. Nếu residual tăng mạnh ở nhà giá cao, mô hình có thể dự đoán kém với phân khúc cao cấp.

## **Phase 20. Build and Train Comparison Models bằng scikit-learn**

Để đánh giá MLP có thật sự tốt hơn không, cần huấn luyện thêm các mô hình baseline bằng scikit-learn.

### **Mô hình so sánh**

| **Model**                | **Mục đích**                                               |
|--------------------------|------------------------------------------------------------|
| Linear Regression        | Baseline tuyến tính đơn giản                               |
| Decision Tree Regression | Mô hình phi tuyến dạng cây                                 |
| Random Forest Regression | Ensemble model mạnh, giảm overfitting so với Decision Tree |

### **Quy tắc so sánh công bằng**

Tất cả mô hình phải dùng:

Cùng training set

Cùng validation/test split

Cùng preprocessing pipeline

Cùng testing set

Cùng metrics đánh giá

## **Phase 21. Compare All Models**

Sau khi train MLP và các mô hình baseline, cần so sánh trên cùng testing set.

### **Bảng so sánh đề xuất**

| **Model**                | **MAE** | **MSE** | **RMSE** | **R²** | **Training Time** | **Inference Time** |
|--------------------------|---------|---------|----------|--------|-------------------|--------------------|
| Linear Regression        |         |         |          |        |                   |                    |
| Decision Tree Regression |         |         |          |        |                   |                    |
| Random Forest Regression |         |         |          |        |                   |                    |
| MLP Regression PyTorch   |         |         |          |        |                   |                    |

### **Visualization nên có**

| **Biểu đồ**                        | **Ý nghĩa**                   |
|------------------------------------|-------------------------------|
| Bar chart RMSE by model            | So sánh sai số                |
| Bar chart R² by model              | So sánh khả năng giải thích   |
| Training time comparison           | So sánh chi phí huấn luyện    |
| Actual vs Predicted của từng model | So sánh trực quan             |
| Residual plot của từng model       | Xem model nào sai ổn định hơn |

## **Phase 22. Analyze Results**

Sau khi có bảng kết quả, cần phân tích chuyên sâu.

### **Câu hỏi cần trả lời**

| **Câu hỏi**                           | **Ý nghĩa**                                       |
|---------------------------------------|---------------------------------------------------|
| Mô hình nào có MAE thấp nhất?         | Model nào sai số trung bình thấp nhất             |
| Mô hình nào có RMSE thấp nhất?        | Model nào ít sai số lớn hơn                       |
| Mô hình nào có R² cao nhất?           | Model nào giải thích tốt nhất biến động giá nhà   |
| MLP có tốt hơn baseline không?        | Đánh giá hiệu quả Neural Network                  |
| Có dấu hiệu overfitting không?        | Dựa vào train loss và validation loss             |
| Training time của từng model ra sao?  | Mô hình nào tốn tài nguyên hơn                    |
| Inference time của từng model ra sao? | Mô hình nào phù hợp triển khai hơn                |
| Model nào phù hợp nhất?               | Cân bằng giữa accuracy, speed và interpretability |

## **Phase 23. Model Interpretation**

MLP thường khó giải thích hơn Linear Regression hoặc Random Forest, vì vậy cần bổ sung các phân tích hỗ trợ.

### **Cách giải thích kết quả**

| **Phương pháp**                  | **Mục đích**                              |
|----------------------------------|-------------------------------------------|
| Correlation analysis             | Xem feature nào liên quan mạnh đến price  |
| Random Forest feature importance | Tham khảo feature quan trọng              |
| Permutation importance           | Đánh giá ảnh hưởng của từng feature       |
| Residual analysis                | Biết model sai ở nhóm dữ liệu nào         |
| Error by price segment           | Xem model dự đoán tốt/kém ở phân khúc nào |

### **Lưu ý**

Không nên nói MLP tự động tốt hơn mọi mô hình. MLP chỉ tốt nếu dữ liệu đủ lớn, preprocessing tốt, hyperparameter phù hợp và validation loss ổn định.

## **Phase 24. Save Artifacts**

Sau khi hoàn thành, cần lưu lại các artifact để tái sử dụng.

### **File nên lưu**

| **File**                   | **Ý nghĩa**             |
|----------------------------|-------------------------|
| best_mlp_model.pth         | Mô hình MLP tốt nhất    |
| preprocess_pipeline.joblib | Pipeline xử lý dữ liệu  |
| training_log.csv           | Log quá trình training  |
| metrics_summary.csv        | Bảng kết quả các model  |
| predictions.csv            | Giá thật và giá dự đoán |
| config.json                | Cấu hình hyperparameter |
| plots/                     | Thư mục chứa biểu đồ    |

## **Phase 25. Final Conclusion**

Dự án xây dựng pipeline dự đoán giá nhà bằng **MLP Regression với PyTorch**. Dữ liệu được chia theo tỷ lệ **80/10/10**cho training, validation và testing. Các bước xử lý dữ liệu như cleaning, feature engineering, encoding và scaling được thực hiện bằng scikit-learn, trong khi mô hình MLP được xây dựng và huấn luyện bằng PyTorch.

Trong quá trình training, mô hình được theo dõi bằng training loss, validation loss, MAE, RMSE, R², gradient norm và early stopping. Các biểu đồ như loss curve, actual vs predicted, residual plot và model comparison chart được sử dụng để trực quan hóa hiệu suất mô hình. Cuối cùng, MLP được so sánh với Linear Regression, Decision Tree Regression và Random Forest Regression để đánh giá xem Neural Network có phù hợp với bài toán dự đoán giá nhà hay không.

Mô hình tốt nhất sẽ được lựa chọn dựa trên sự cân bằng giữa sai số dự đoán, khả năng tổng quát hóa, thời gian huấn luyện, thời gian inference và mức độ phù hợp với mục tiêu triển khai.
