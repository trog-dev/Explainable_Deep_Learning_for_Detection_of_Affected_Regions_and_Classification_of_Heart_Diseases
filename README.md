# Phát hiện vùng bệnh và phân loại bệnh tim bằng học sâu có khả năng giải thích

*Explainable Deep Learning for Detection of Affected Regions and Classification of Heart Diseases*
*Graduation Thesis Score: 8.7/10*
Khóa luận tốt nghiệp ngành Khoa học máy tính tại Trường Đại học Sư phạm Kỹ thuật Vĩnh Long, 2026. Dự án sử dụng MRI tim 3D để phân đoạn cấu trúc tim và vùng nhồi máu cơ tim. Từ mặt nạ dự đoán, hệ thống trích xuất các đặc trưng về thể tích, chức năng tim hoặc tổn thương để phân loại theo từng bệnh nhân.

## Dữ liệu và bài toán

| Bộ dữ liệu | Đầu vào | Phân đoạn | Phân loại | Notebook |
| --- | --- | --- | --- | --- |
| [ACDC 2017](https://www.creatis.insa-lyon.fr/Challenge/acdc/) | MRI tim ở cuối tâm trương (ED) và cuối tâm thu (ES) | Thất trái (LV), thất phải (RV) và cơ tim (MYO) | Bình thường (NOR), cơ tim giãn (DCM), cơ tim phì đại (HCM), nhồi máu cơ tim (MINF) và bất thường thất phải (RV) | [Classification.ipynb](./Classification.ipynb) |
| [EMIDEC 2020](https://emidec.com/dataset) | MRI tim tăng cường trễ (LGE-MRI) và mặt nạ giải phẫu | Buồng tim, cơ tim và vùng nhồi máu (MI). Nhãn *no-reflow* được gộp vào lớp MI trong notebook | Bình thường (Normal) và nhồi máu cơ tim (Infarcted) | [Segmentation.ipynb](./Segmentation.ipynb) |

## Phương pháp

1. **Phân đoạn MRI 3D.** Khóa luận so sánh SegResNet, Attention U-Net, U-Net++ và DynUNet. U-Net++ được chọn để tạo mặt nạ cho bước trích xuất đặc trưng.
2. **Trích xuất đặc trưng theo bệnh nhân.** Nhánh ACDC tính các chỉ số như thể tích thất ở ED và ES và phân suất tống máu (EF). Nhánh EMIDEC tính thể tích buồng tim, cơ tim, vùng nhồi máu, tỷ lệ thể tích nhồi máu so với cơ tim và độ dày tối đa của vùng nhồi máu.
3. **Phân loại.** So sánh Random Forest, SVM, MLP và mô hình kết hợp Soft Voting sử dụng xác suất từ ba bộ phân loại.
4. **Giải thích mô hình.** Grad-CAM 3D trực quan hóa vùng ảnh ảnh hưởng đến dự đoán của mô hình phân đoạn. SHAP cho thấy mức đóng góp của từng đặc trưng vào dự đoán của bộ phân loại.

## Chạy notebook

### 1. Chuẩn bị môi trường và dữ liệu

Notebook được viết cho **Kaggle Notebooks**. Bật GPU khi huấn luyện mô hình phân đoạn 3D. Nếu chạy trên Jupyter, cài PyTorch phù hợp với môi trường CPU hoặc CUDA trước. Có thể chạy cell sau để cài các thư viện còn lại.

```python
%pip install monai medpy nibabel numpy scipy pandas scikit-learn matplotlib seaborn shap tqdm
```

Hai bộ dữ liệu cần được tải riêng dưới dạng NIfTI. Trên Kaggle, thêm bộ dữ liệu vào notebook rồi đối chiếu đường dẫn thực tế với các biến sau.

| Notebook | Biến cần kiểm tra | Thư mục dữ liệu mong đợi |
| --- | --- | --- |
| `Classification.ipynb` | `BASE_DIR` trong các khối phân đoạn | `database/` chứa `training/` và `testing/`. Mỗi ca cần `Info.cfg`, ảnh ED/ES và mặt nạ tương ứng để chạy đầy đủ các bước đánh giá. |
| `Segmentation.ipynb` | `EMIDEC_RAW_DIR` trong khối mô hình và `EMIDEC_DIR` nếu chạy cell trực quan hóa dữ liệu đầu tiên | Thư mục chứa các ca `Case_*`, mỗi ca có `Images/` và `Contours/`. Dùng bộ 100 ca có nhãn. |

Khi chạy ngoài Kaggle, đổi các đường dẫn `/kaggle/input/...` sang vị trí dữ liệu và checkpoint của bạn. Đổi các đường dẫn `/kaggle/working/...` sang thư mục kết quả có quyền ghi.

### 2. Chọn nhánh và checkpoint

| Checkpoint trong repo | Nhánh | Đầu vào U-Net++ 3D |
| --- | --- | --- |
| [best_unet++.pth](./best_unet%2B%2B.pth) | ACDC | 1 kênh MRI |
| [best_unetpp_emidec_multiclass_gt_prior (1).pth](./best_unetpp_emidec_multiclass_gt_prior%20%281%29.pth) | EMIDEC | 2 kênh gồm MRI và mặt nạ prior giải phẫu |

- **ACDC.** Mở `Classification.ipynb` và bắt đầu ở khối `unetpp`. Chạy phần cấu hình, đọc dữ liệu và chia ca để tạo `patient_data`, `train_ids`, `val_ids` và `test_ids`. Nếu dùng checkpoint có sẵn, có thể bỏ qua vòng lặp huấn luyện và trỏ các lệnh nạp trọng số đến `best_unet++.pth`. Các cell đánh giá phân đoạn mặc định đọc `SAVE_DIR / "best_model.pth"`. Các khối phân loại có đường dẫn `model_path` riêng. Chạy khối `random` để tạo các bảng đặc trưng trước khi chạy các cell phân loại sử dụng `df_train`, `df_val` và `df_test`.
- **EMIDEC.** Mở `Segmentation.ipynb` và bắt đầu ở khối `GT UNET ++`. Chạy phần tiền xử lý, chia ca và khởi tạo `val_transforms` trong khối này. Nếu dùng checkpoint có sẵn, có thể bỏ qua vòng lặp huấn luyện và cập nhật `best_model_path` cùng các lệnh `torch.load` có đường dẫn ghi trực tiếp. Sau đó chạy các cell đánh giá, Grad-CAM hoặc khối `ENSEMBLE` để trích xuất đặc trưng và phân loại. Khối này dùng danh sách ca và biến `model` của U-Net++, nên hãy nạp lại U-Net++ nếu vừa chạy một kiến trúc khác trong cùng kernel.

Chạy theo từng khối mô hình vì các notebook dùng lại những tên biến như `model` và `best_model_path`. Nếu dùng checkpoint tải sẵn, các cell vẽ biểu đồ huấn luyện vẫn cần tệp log CSV sinh ra khi huấn luyện.

> Quy trình EMIDEC hiện tại dùng mặt nạ gốc (`Contours/`) để cắt vùng tim và tạo kênh prior. Vì vậy, chạy checkpoint này theo notebook cần cả ảnh MRI lẫn mặt nạ tương ứng.

## Kết quả trong khóa luận

Các kết quả tiêu biểu dưới đây được trích từ Bảng 4.2 và 4.4 của [báo cáo khóa luận](./22022012_NguyenDucTrong.pdf).

| Bộ dữ liệu | Mô hình phân đoạn | Dice | IoU |
| --- | --- | ---: | ---: |
| ACDC 2017 | U-Net++ | **86,83%** | 77,44% |
| ACDC 2017 | SegResNet | 86,80% | **77,58%** |
| EMIDEC 2020 | U-Net++ | 66,36% | 50,37% |
| EMIDEC 2020 | DynUNet | **67,19%** | **50,92%** |

| Bộ dữ liệu | Mô hình phân loại | Accuracy | F1-score | ROC-AUC |
| --- | --- | ---: | ---: | ---: |
| ACDC 2017 | Soft Voting | 86% | 86% | 98,5% |
| EMIDEC 2020 | Soft Voting | 97% | 98% | 99,7% |

Kết quả phân loại ACDC được báo cáo trên tập kiểm thử 50 ca của dự án. Trên EMIDEC, kết quả phân loại đến từ *Stratified 5-Fold Cross-Validation* trên 100 ca. Notebook chuẩn hóa đặc trưng trên toàn bộ 100 ca trước khi chia fold và dùng chung một checkpoint phân đoạn. Các con số EMIDEC vì vậy phản ánh thiết lập thực nghiệm trong notebook, với tiền xử lý và phân đoạn dùng chung giữa các fold.

## Tài liệu

- [Báo cáo khóa luận (PDF)](./22022012_NguyenDucTrong.pdf)
- [Slide bảo vệ (PowerPoint)](./9_22022012_NguyenDucTrong_KLTN.pptx)
- Bernard và cộng sự, bài báo giới thiệu ACDC. [DOI 10.1109/TMI.2018.2837502](https://doi.org/10.1109/TMI.2018.2837502)
- Lalande và cộng sự, bài báo giới thiệu EMIDEC. [DOI 10.3390/data5040089](https://doi.org/10.3390/data5040089)

**Sinh viên thực hiện** Nguyễn Đức Trọng

**Giảng viên hướng dẫn** ThS. Nguyễn Thị Mỹ Nga
