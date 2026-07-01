# AI Generation Log

Tài liệu này ghi lại quá trình sử dụng công cụ AI để hỗ trợ xây dựng project **House Price Prediction Using MLP Regression with PyTorch**. Thay vì chỉ liệt kê từng prompt theo kiểu máy móc, phần log dưới đây được viết lại theo dạng nhật ký triển khai để thể hiện rõ hơn cách project được hình thành, chỉnh sửa và kiểm tra qua từng giai đoạn.

## Khảo sát project ban đầu

Ở bước đầu tiên, project được kiểm tra lại toàn bộ cấu trúc thư mục, file kế hoạch và các tài liệu hướng dẫn đi kèm. Mục tiêu lúc này chưa phải viết code ngay, mà là hiểu rõ yêu cầu của bài Lab 4, xác định dữ liệu đã có hay chưa, notebook cần nằm ở đâu và workflow tổng thể có khớp với `project_plan.md` không. Sau khi rà soát, hướng triển khai được xác nhận là phù hợp: bài toán dùng dữ liệu King County House Sales, target là `price`, mô hình chính là Multilayer Perceptron Regression bằng PyTorch và cần so sánh với các baseline regression truyền thống. Ở giai đoạn này, project vẫn chủ yếu ở trạng thái chuẩn bị; code thực tế chưa được xây dựng.

## Chốt hướng triển khai và refactor project

Sau khi đọc lại project plan, rules và cấu trúc hiện có, project được định hướng lại rõ ràng theo PyTorch. Các phần nhắc tới TensorFlow/Keras hoặc những file cũ không còn phù hợp được loại khỏi hướng triển khai. Việc này giúp notebook đi theo một pipeline nhất quán hơn: chia dữ liệu trước, xử lý dữ liệu bằng Pandas/Scikit-learn, sau đó dùng PyTorch Dataset, DataLoader và MLP model để huấn luyện.

Một thay đổi lớn trong ngày này là refactor project sang cấu trúc phase-based. Thay vì viết toàn bộ logic trực tiếp trong notebook, code được tách ra thành các file trong thư mục `processing_own_phase/`, từ Phase 1 đến Phase 25. Notebook `mlp_regression.ipynb` vì vậy trở nên gọn hơn: mỗi phase chủ yếu reload module, gọi function và hiển thị kết quả. Cách làm này giúp project dễ đọc, dễ debug và dễ mở rộng hơn so với một notebook quá dài chứa toàn bộ code.

## Xây dựng các phase đầu của pipeline

Phase 1 được hoàn thiện để quản lý config, random seed, đường dẫn output và trạng thái thư viện. Phần hiển thị trong notebook cũng được làm trực quan hơn bằng HTML dashboard, giúp người đọc thấy nhanh target, split ratio, thư mục output và các thư viện cần thiết. Sau đó, Phase 2 được triển khai để load file `kc_house_data.csv` và chia dữ liệu raw thành train/validation/test theo tỷ lệ 80/10/10. Điểm quan trọng ở phase này là chưa preprocessing gì cả, nhằm tránh data leakage.

Phase 3 tập trung vào EDA trên training set. Project bổ sung các bảng xem nhanh dữ liệu, thống kê mô tả, phân phối target, correlation matrix và các biểu đồ như histogram, boxplot, heatmap, scatter plot, pairplot và giá theo zipcode. Đây là bước giúp hiểu rõ dữ liệu trước khi làm sạch: giá nhà bị lệch phải, một số biến có outlier và nhiều đặc trưng liên quan mạnh đến `price`.

Phase 4 xử lý data cleaning theo hướng bảo thủ. Các giá trị missing token được chuẩn hóa về `NaN`, duplicate được loại trong từng split, các giá trị feature vô lý được đưa về `NaN`, cột `id` được drop, cột `date` được parse và tạo thêm `sale_year`, `sale_month`. Với outlier giá nhà, project không xóa tự động vì nhà giá cao có thể là dữ liệu thật do vị trí, view, diện tích hoặc chất lượng nhà. Các outlier được flag để phân tích thay vì loại bỏ máy móc.

Phase 5 và Phase 6 bổ sung feature engineering và tách `X`, `y`. Một số feature được tạo thêm như tuổi nhà, trạng thái cải tạo, basement, các ratio diện tích và feature phục vụ EDA như `price_per_sqft`. Vì `price_per_sqft` dùng trực tiếp target `price`, Phase 6 có cơ chế tự động loại feature này khỏi tập input model để tránh data leakage.

## Chuẩn bị dữ liệu cho PyTorch

Sau khi dữ liệu đã được làm sạch và tạo feature, Phase 7 đảm nhiệm preprocessing pipeline bằng Scikit-learn. Missing values, encoding và scaling được fit trên training set, sau đó chỉ transform validation/test. Phase 8 không fit scaler mới nữa mà đóng vai trò audit, kiểm tra dữ liệu sau preprocessing có đúng shape, không còn NaN/infinite và đã được scale hợp lý hay chưa.

Phase 9 chuyển dữ liệu đã preprocessing thành PyTorch `TensorDataset`, còn Phase 10 tạo `DataLoader` cho train, validation và test. Train loader dùng `shuffle=True` để giảm bias theo thứ tự dữ liệu, trong khi validation/test giữ `shuffle=False` để đánh giá ổn định. Các bước này tạo cầu nối giữa phần xử lý dữ liệu bằng Scikit-learn và phần huấn luyện model bằng PyTorch.

## Xây dựng và huấn luyện MLP

Phase 11 xây dựng class `MLPRegression` bằng PyTorch. Mô hình nhận input dimension từ số feature sau preprocessing, dùng hidden layers theo config, activation ReLU, optional Dropout và output layer một neuron cho bài toán regression. Phase 12 gom các hyperparameter quan trọng như learning rate, batch size, epochs, hidden units, dropout, weight decay, patience và min delta vào một cấu hình dễ theo dõi.

Phase 13 viết training loop chuẩn của PyTorch: chuyển model sang train mode, forward pass, tính loss, backward, update bằng Adam, rồi đánh giá lại trên validation set bằng eval mode. Training log lưu lại train/validation loss, MAE, MSE, RMSE, R2, learning rate, gradient norm, epoch time và device. Vì target được log-transform bằng `log1p(price)`, các metric trong training log chủ yếu phản ánh target scale khi train; metric theo giá nhà thật được xử lý sau bằng inverse transform.

## Theo dõi training, early stopping và tuning

Phase 14 được dùng để lưu và kiểm tra training log, còn Phase 15 tạo các biểu đồ training process như loss curve, MAE curve, RMSE curve, R2 curve, learning rate và gradient norm. Phase 16 tóm tắt early stopping và best model saving. Cách tách này giúp notebook không chỉ có kết quả cuối, mà còn cho thấy quá trình model học qua từng epoch.

Phase 17 ban đầu được thiết kế như một hyperparameter exploration plan nhẹ, chưa train grid nặng mặc định. Sau đó, khi cần thử thật nhiều cấu hình hơn, Phase 17 được mở rộng để chạy một grid nhỏ gồm các learning rate, hidden layers, dropout, weight decay và loss function khác nhau. Project cũng thử các loss bền hơn như Huber và SmoothL1, đồng thời thêm gradient clipping để training ổn định hơn.

## Evaluation, visualization và so sánh baseline

Phase 18 đánh giá MLP trên test set. Prediction được inverse-transform về thang giá gốc bằng `expm1`, sau đó tính MAE, MSE, RMSE và R2. Kết quả được lưu vào `mlp_metrics.csv` và prediction từng dòng được lưu vào `mlp_predictions.csv`. Phase 19 dùng các prediction này để vẽ actual vs predicted, residual plot, residual histogram, error by price range, top 20 lỗi lớn nhất và phân tích riêng high-end houses.

Phase 20 train các baseline gồm Linear Regression, Decision Tree Regression và Random Forest Regression trên cùng dữ liệu đã preprocessing. Phase 21 gộp kết quả MLP với baseline để so sánh công bằng bằng MAE, RMSE, R2, training time và inference time. Phase 22 đọc bảng so sánh này để trả lời các câu hỏi chính của project: model nào có MAE thấp nhất, RMSE thấp nhất, R2 cao nhất, MLP có vượt baseline không và có dấu hiệu overfitting không.

Phase 23 bổ sung phần interpretation. Vì MLP khó giải thích trực tiếp, project dùng Random Forest feature importance như một nguồn tham khảo để hiểu feature nào có ảnh hưởng mạnh. Phase này cũng tạo residual summary và error by price segment để chỉ ra model đang yếu ở đâu. Phase 24 lưu artifact như model weights, preprocessing pipeline, config, metrics, predictions và artifact manifest. Phase 25 tạo final conclusion bằng tiếng Việt dựa trên kết quả thực tế của Phase 21 và Phase 22.

## Bổ sung EDA và xử lý outlier theo góp ý

Sau khi rà soát notebook, phần EDA được bổ sung thêm nhiều nội dung để báo cáo thuyết phục hơn. Correlation heatmap được thêm số trực tiếp trên từng ô để dễ đọc. Project cũng thêm KDE plot, boxplot cho các feature quan trọng, bảng tứ phân vị/IQR, bảng phân loại kiểu biến thành liên tục, rời rạc, phân loại hoặc datetime, và bảng skewness để xem dữ liệu có bị lệch hay không.

Với outlier, hướng xử lý được điều chỉnh cho đúng góp ý: không xóa các house-price outlier hợp lệ, đặc biệt là nhà cao cấp. Thay vào đó, project giữ lại các quan sát hợp lệ, phân tích riêng bằng bảng/plot, dùng `log1p(price)` để giảm độ lệch của target và thêm L1/L2 regularization trong quá trình huấn luyện. L1 được đưa vào training loss thông qua `l1_lambda`, còn L2 dùng `weight_decay` trong Adam.

Phase 17 cũng được chạy thật với nhiều cấu hình MLP hơn. Kết quả cho thấy một số cấu hình robust loss tạo prediction rất lớn sau inverse transform, làm RMSE tăng bất thường. Điều này giúp project có thêm cơ sở để phân tích rằng không phải loss bền hơn lúc nào cũng tốt; cần kiểm soát prediction sau khi đưa về thang giá gốc.

## Làm sạch project và cải thiện kết quả MLP

Ở lần chỉnh cuối, notebook được sửa để bỏ hard-code path tuyệt đối. Thay vì gắn trực tiếp đường dẫn máy cá nhân, notebook tự tìm project root dựa trên thư mục `processing_own_phase`. Điều này giúp notebook dễ chạy hơn trên máy khác. Project cũng được thêm `.gitignore`, xóa các file rác như `.DS_Store`, `__pycache__` và script tạm `fix_notebook.py`.

Phần data cleaning được bổ sung rule xử lý target price bất thường. Các dòng có `price <= 0` hoặc giá quá thấp như `5000` được xem là lỗi dữ liệu và bị loại bỏ bằng ngưỡng `minimum_valid_price = 10000`. Điểm này được tách rõ khỏi outlier nhà giá cao: giá quá thấp bất hợp lý thì loại, còn nhà high-end hợp lệ thì giữ lại để model học và phân tích riêng.

MLP cũng được cải thiện bằng cách dùng cấu hình tốt nhất tìm được từ Phase 17 làm default chính: `hidden_units=(64,)`, `learning_rate=0.0005`, `batch_size=64`, `dropout=0.0`, `weight_decay=0.0`, `l1_lambda=1e-6`. Sau khi chạy lại toàn bộ notebook bằng kernel `tf-mnist`, kết quả MLP cải thiện rõ rệt: RMSE giảm từ khoảng `396,912` xuống khoảng `289,546`, R2 tăng từ khoảng `0.272` lên khoảng `0.612`.

Tuy vậy, kết quả cuối vẫn cho thấy Linear Regression đang có RMSE thấp nhất, còn Random Forest có MAE tốt nhất. MLP hiện đã tiến gần baseline hơn nhiều, nhưng chưa phải model tốt nhất theo RMSE. Đây là một kết luận hợp lý cho báo cáo: pipeline MLP đã được xây dựng đầy đủ, model đã cải thiện sau cleaning và tuning, nhưng với dữ liệu tabular này các baseline truyền thống vẫn cạnh tranh rất mạnh.

Phase 17 cũng được thêm prediction guardrail để kiểm soát những trial có prediction quá lớn. Bảng hyperparameter results hiện có thêm các cột như `max_prediction`, `max_reasonable_prediction`, `negative_prediction_count`, `extreme_high_prediction_count`, `unstable_reason` và `status`. Nhờ đó, các trial có prediction vượt quá ngưỡng hợp lý được đánh dấu là `unstable_prediction` thay vì được xem như trial bình thường.

## Tổng kết

Nhìn chung, AI được dùng như một công cụ hỗ trợ triển khai, refactor, kiểm tra lỗi và cải thiện cách trình bày project. Phần quan trọng nhất của quá trình này không chỉ là sinh code, mà là dần làm project rõ ràng hơn: dữ liệu được split đúng cách, preprocessing tránh leakage, MLP được huấn luyện bằng PyTorch, baseline được so sánh công bằng, kết quả được phân tích theo metric và high-end houses được xem riêng vì đây là nhóm model dễ lỗi lớn.

Project hiện đã có notebook chạy end-to-end, các phase được tách thành file riêng, output được lưu có hệ thống và phần kết luận phản ánh đúng kết quả thực nghiệm. MLP đã được cải thiện đáng kể sau khi xử lý giá bất thường và dùng cấu hình tốt hơn, nhưng vẫn cần tuning thêm nếu mục tiêu là vượt hoàn toàn Linear Regression hoặc Random Forest.

## Cải thiện biểu đồ và thời gian Inference

Trong quá trình xem xét tiếp, một số cải thiện nhỏ về biểu đồ và thời gian inference đã được đưa vào để đảm bảo độ chính xác.
- Ở Phase 3, correlation heatmap được làm sạch thêm: cột `date`, `id` và `zipcode` được drop cẩn thận hơn bằng cách kiểm tra tên cột, và loại bỏ hoàn toàn các cột/hàng bị trống rỗng (NaN) để heatmap gọn gàng. Biểu đồ `log_price` phân bố histogram và boxplot cũng đã được ghi nhận tự động xuất hiện.
- Ở Phase 15, các chú thích được thêm vào notebook để giải thích rằng loss metric trong suốt quá trình training được đo trên thang đo log transform nên không thể đọc tương tự RMSE/MAE thực tế tính bằng USD.
- Ở Phase 19, thêm markdown cho thấy residual histogram đã được rút ra theo bản cắt (clipped) 1% - 99% để quan sát sự phân phối phần lớn dữ liệu, bỏ đi những điểm ngoại lai kéo quá xa hai đuôi.
- Ở Phase 18 và 20, quá trình tính thời gian dự đoán (inference time) được chuẩn hóa để đo lặp lại trung bình (ví dụ 30 lần) nhằm ra một con số tin cậy cho `inference_time_seconds`, từ đó bar chart thời gian trong Phase 21 đánh giá công bằng hơn.

