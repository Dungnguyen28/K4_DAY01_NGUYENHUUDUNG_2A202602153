# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy: 11/9/2026**

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** Python 3.x, PyTorch, Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
  `sample_id = traffic`, `rank = 1`, `class_id = 468`, `class_name = cab`, `taxonomy_name = ImageNet-1K`, `score = 0.510915`.
  Đây là prediction cấp ảnh, không phải prediction cho một đối tượng riêng lẻ. `rank = 1` là lớp có xác suất cao nhất theo mô hình khi xét cả bức ảnh. Mô hình không xác định vị trí từng vật thể hay mask cho từng pixel; nó chỉ gán một nhãn mô tả toàn bộ ảnh.

- Record này mô tả toàn ảnh như thế nào?
- Bản ghi trong `classification_predictions.json` thuộc task `image_classification`, nghĩa là mỗi record biểu diễn một nhãn cho toàn bộ bức ảnh. Repo đã nêu rõ trong `GUIDE.md`: file này chứa “một lớp được xếp hạng cho toàn ảnh”. Điều này khác với `detection_predictions.json` và `segmentation_predictions.json`, nơi mỗi record ứng với một vật thể hoặc một instance riêng.

- Ai định nghĩa class list mà checkpoint có thể dự đoán?
- Danh mục lớp được định nghĩa bởi chính model/checkpoint và taxonomy của nó. Trong notebook, code lấy `names = result.names`, sau đó ánh xạ `class_id -> class_name` bằng `names[int(class_id)]`. Kết quả cho thấy checkpoint `yolo11n-cls.pt` sử dụng taxonomy `ImageNet-1K`, tức là danh sách 1.000 lớp ImageNet, không phải danh sách do repo tự tạo.

- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
- `class_id` là mã máy đọc được, ổn định cho xử lý dữ liệu; `class_name` là tên dễ đọc cho người review; `taxonomy_name` cho biết lớp đó thuộc taxonomy nào. Nếu không lưu taxonomy, cùng một `class_id` có thể bị hiểu sai khi chuyển giữa các tập nhãn khác nhau. Ví dụ, một ID trong ImageNet không đồng nghĩa với cùng một ID trong COCO. Vì vậy, cần giữ cả ba để tránh hiểu nhầm và đảm bảo traceability.

- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
- Với bài toán phân loại ảnh, repo đã mô hình hóa dạng single-label classification: “một lớp cho mỗi ảnh”. Tuy nhiên, nếu ảnh bao gồm nhiều vật thể hoặc nhiều chủ đề, quy trình cần ghi rõ hướng dẫn chọn lớp chủ đạo/chiếm ưu thế, hoặc quy định chuyển sang multi-label nếu dự án yêu cầu. Một ảnh có thể chứa nhiều đối tượng nhưng output phân loại vẫn chỉ trả về một top label duy nhất.

- Vì sao model score không phải ground truth?
- `score` là độ tin cậy mà mô hình gán cho lớp đó, dùng để xếp hạng hoặc lọc prediction. Nó không phải là nhãn đã được con người xác nhận. Theo `GUIDE.md` và `README.md`, `score/confidence` là “model score dùng để xếp hạng hoặc lọc prediction; không phải điểm chất lượng của ground truth”. `rank` cho biết lớp nào đứng đầu trong xác suất của mô hình, nhưng không chứng minh rằng lớp đó là đúng tuyệt đối. Ground truth là nhãn do annotator/reviewer xác nhận theo guideline; `score` chỉ là output từ mô hình trong quá trình dự đoán.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `traffic`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
  Ví dụ record đầu tiên: `class_name = "bus"`, `score = 0.912558`, `bbox_xyxy = [93.17, 187.95, 223.01, 320.91]`, `bbox_width = 129.84`, `bbox_height = 132.96`.
  Đây là một prediction cho một vật thể riêng lẻ trong ảnh, không phải nhãn cho toàn ảnh như phần 1.

- Diễn giải vị trí box bằng lời:
  Box bắt đầu ở hoành độ x ≈ 93 px và tung độ y ≈ 188 px, kết thúc ở x ≈ 223 px và y ≈ 321 px. Chiều rộng khoảng 130 px, chiều cao khoảng 133 px. Theo dữ liệu và hình ảnh, đây là một xe buýt lớn nằm ở nửa trái của cảnh giao thông, với box bao quanh phần vật thể mà mô hình nhận diện được.

- So sánh số prediction ở hai threshold:
  Dữ liệu dùng `score_threshold = 0.35`. Nếu giảm ngưỡng xuống thấp hơn, mô hình sẽ giữ thêm nhiều box có độ tin cậy thấp hơn; nếu tăng ngưỡng lên, số prediction giảm và chỉ còn các box chắc chắn hơn. Threshold là cấu hình lọc prediction, không phải quy định ground truth. Vì vậy, threshold chỉ đổi lượng object được xem, không làm “sửa” nhãn đúng/sai cho từng vật thể.

- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
  Khi giảm threshold, số box tăng lên, độ bao phủ cảnh tăng nhưng đồng thời reviewer phải kiểm tra nhiều hơn: thêm các false positive, thêm đối tượng mờ, các box chồng lấn, và các vật thể nhỏ dễ bị nhầm. Khi tăng threshold, số lượng review giảm nhưng có nguy cơ bỏ sót vật thể thật, đặc biệt với object nhỏ hoặc bị che.

- Đề xuất một quy tắc box chặt:
  Box nên được đặt vừa khít quanh phần vật thể nhìn thấy, không để quá rộng hoặc quá lệch với đối tượng thật. Nói cách khác, x_min/y_min nên sát mép trái/trên của vật thể rõ nhất, x_max/y_max sát mép phải/dưới thấy được, và không mở rộng quá mức để “bọc” vùng nền. Quy tắc này giúp giảm sai số và dễ khớp với annotation guideline.

- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
  Nếu một vật thể chỉ hiện một phần (che, cắt mép ảnh, mờ, hoặc lẫn với đối tượng khác), guideline cần quy định cách vẽ box: giữ phần nhìn thấy, đánh dấu là occluded/partial, và nếu không chắc về class hoặc ranh giới thì escalate cho reviewer để quyết định. Nói ngắn gọn: phần nào nhìn thấy thì giữ, phần nào không rõ thì không phỏng đoán quá mức.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `traffic`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
  Ví dụ: `instance_id = "traffic-001"`, `class_name = "bus"`, `score = 0.925745`, `polygon_point_count = 120` và `polygon_xy` chứa 120 điểm pixel mô tả biên của vật thể. Đây là một mask theo instance, không phải chỉ một box khung.

- Polygon bổ sung chi tiết gì so với box?
  Box chỉ là hình chữ nhật bao quanh vật thể; polygon cho biết hình dạng thực của đối tượng, giúp mô tả ranh giới, góc, và phần cong lồi lõm của vật thể. Với xe buýt, polygon cho biết phần thân xe, mặt trước, góc và các cạnh không thẳng, trong khi box chỉ có thể bao trùm hình đó một cách khái quát.

- `instance_id` dùng để làm gì và không phải loại ID nào?
  `instance_id` dùng để phân biệt hai instance cùng lớp trong cùng một ảnh, ví dụ hai chiếc xe buýt hoặc hai chiếc ô tô. Nó không phải `class_id` (mã lớp), không phải tracking ID liên tục qua video, và không phải id của ảnh. Trong bài lab, `instance_id` chỉ đánh dấu “một instance/đối tượng trong output hiện tại”.

- Đề xuất một quy tắc biên mask:
  Biên mask nên theo đúng hình dạng của phần vật thể nhìn thấy, không bao lồi ra khỏi vật thể quá nhiều, không “đi chệch” vào nền, và không nối liền hai instance khác nhau. Nếu lớp vật thể chồng lấp, cần tách rõ instance và giữ ranh giới theo contour hợp lý.

- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
  Khi vùng bị mờ, che bởi vật thể khác, hoặc tiếp xúc với đối tượng gần kề, annotator cần theo guideline về mask “visible region only” với nhãn rõ là ambig/occluded. Nếu không chắc ranh giới, phải escalate cho reviewer hoặc annotator senior để quyết định trước khi chốt ground truth.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn chính cho toàn ảnh theo taxonomy | Nhiều đối tượng trong ảnh, lớp chủ đạo chưa rõ | Chọn nhãn chính theo guideline hoặc multi-label nếu được phép | Kiểm tra tên lớp, taxonomy và top-1 có phù hợp không |
| Phát hiện vật thể | Một box cho mỗi object, có `class_name`, `score`, `bbox_xyxy` | Box quá lớn, thiếu object, class nhầm, threshold gây nhiều box | Vẽ box khít quanh phần thể hiện rõ nhất của object | Kiểm tra class, kích thước box, coverage, độ tin cậy và threshold |
| Instance segmentation | Một mask/polygon cho mỗi instance, có `instance_id` | Biên không khít, hai instance dính nhau, vùng mờ | Tách rõ instance, giữ contour theo vùng nhìn thấy | Kiểm tra contour, instance separation, và mức độ mơ hồ do occlusion |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
  Không tải lên môi trường công khai các ảnh, khuôn mặt, biển số xe, dữ liệu khách hàng hoặc dữ liệu nội bộ không phù hợp. Chỉ làm việc với dữ liệu phù hợp phạm vi lab và quy định bảo mật đã nêu trong repo.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:
  Lab Coach / chủ sở hữu dữ liệu / reviewer của dự án, đồng thời ngừng tiếp tục annotate hoặc upload dữ liệu đó lên GitHub/Colab công khai.

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
