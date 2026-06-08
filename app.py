import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import io

# ---------------------------------------------------------
# 1. CẤU HÌNH TRANG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Hệ thống Nhận diện Gãy xương X-quang",
    page_icon="🦴",
    layout="wide"
)

# ---------------------------------------------------------
# 2. SIDEBAR - CHỌN MÔ HÌNH VÀ TÙY CHỈNH THAM SỐ
# ---------------------------------------------------------
st.sidebar.title("⚙️ Cấu hình Hệ thống")

# Chọn Model (Đã thêm YOLOv8n)
st.sidebar.markdown("### 1. Chọn Mô hình")
model_choice = st.sidebar.radio(
    "Phiên bản YOLOv8:",
    (
        "YOLOv8n (Cực nhẹ, Tốc độ tối đa)", 
        "YOLOv8s (Cân bằng)", 
        "YOLOv8m (Ưu tiên Độ chính xác)"
    )
)

# Map lựa chọn với đường dẫn file model
model_paths = {
    "YOLOv8n (Cực nhẹ, Tốc độ tối đa)": "weights/yolov8n_best.pt",
    "YOLOv8s (Cân bằng)": "weights/yolov8s_best.pt",
    "YOLOv8m (Ưu tiên Độ chính xác)": "weights/yolov8m_best.pt"
}

# Load model với cache
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

try:
    current_model_path = model_paths[model_choice]
    model = load_model(current_model_path)
    model_loaded = True
    # Lấy tên ngắn gọn của model để hiển thị
    model_short_name = model_choice.split(' ')[0]
    st.sidebar.success(f"Đã tải thành công {model_short_name}!")
except Exception as e:
    model_loaded = False
    st.sidebar.error(f"Lỗi tải mô hình. Vui lòng kiểm tra file `{current_model_path}`.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 2. Tùy chỉnh Ngưỡng")
conf_threshold = st.sidebar.slider("Ngưỡng tin cậy (Confidence)", 0.0, 1.0, 0.25, 0.05)
iou_threshold = st.sidebar.slider("Ngưỡng giao thoa (IoU NMS)", 0.0, 1.0, 0.45, 0.05)

st.sidebar.markdown("---")
st.sidebar.warning("⚠️ Ứng dụng này chỉ mang tính chất nghiên cứu học thuật, không thay thế chẩn đoán y khoa.")

# ---------------------------------------------------------
# 3. GIAO DIỆN CHÍNH (TABS)
# ---------------------------------------------------------
st.title("🦴 Nhận diện Gãy xương qua X-quang")
st.markdown(f"**Đang sử dụng:** `{model_choice}`")

tab_inference, tab_metrics, tab_about = st.tabs([
    "🔍 Dự đoán", 
    "📊 So sánh 3 Mô hình", 
    "ℹ️ Tổng quan"
])

# ==========================================
# TAB 1: DỰ ĐOÁN (INFERENCE)
# ==========================================
with tab_inference:
    uploaded_file = st.file_uploader("Tải ảnh X-quang lên (JPG, PNG)", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Ảnh gốc")
            st.image(image, use_container_width=True)
            
        with col2:
            st.subheader("Kết quả Nhận diện")
            if st.button("Phân tích hình ảnh", type="primary") and model_loaded:
                with st.spinner(f"Đang xử lý bằng {model_short_name}..."):
                    results = model.predict(source=image, conf=conf_threshold, iou=iou_threshold)
                    
                    res_plotted = results[0].plot()
                    res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                    
                    st.image(res_plotted_rgb, use_container_width=True)
                    
                    # Nút tải xuống
                    img_pil = Image.fromarray(res_plotted_rgb)
                    buf = io.BytesIO()
                    img_pil.save(buf, format="PNG")
                    st.download_button(
                        label="📥 Tải ảnh kết quả",
                        data=buf.getvalue(),
                        file_name=f"ket_qua_{model_short_name}.png",
                        mime="image/png"
                    )
                    
                    num_detections = len(results[0].boxes)
                    if num_detections > 0:
                        st.success(f"Phát hiện {num_detections} vị trí nghi ngờ gãy xương.")
                    else:
                        st.info("Không phát hiện dấu hiệu với ngưỡng hiện tại.")

# ==========================================
# TAB 2: SO SÁNH MODEL (METRICS)
# ==========================================
with tab_metrics:
    st.header("So sánh Hiệu năng: Nano vs Small vs Medium")
    st.markdown("Đánh giá sự đánh đổi (trade-off) giữa Tốc độ suy luận (FPS), Dung lượng lưu trữ và Độ chính xác (mAP).")
    
    # Chia 3 cột để so sánh 3 mô hình
    col_n, col_s, col_m = st.columns(3)
    
    with col_n:
        st.info("🟢 YOLOv8n (Nano)")
        st.markdown("""
        * **Đặc điểm:** Dung lượng cực nhẹ, phù hợp Edge AI (Mobile, Raspberry Pi).
        * **mAP@50:** ~0.700 *(Nhập số thực tế của bạn)*
        * **Dung lượng:** ~6 MB
        * **Tốc độ:** Nhanh nhất
        """)
        
    with col_s:
        st.warning("🟡 YOLOv8s (Small)")
        st.markdown("""
        * **Đặc điểm:** Sự cân bằng tuyệt vời giữa tốc độ và độ chính xác.
        * **mAP@50:** ~0.750 *(Nhập số thực tế của bạn)*
        * **Dung lượng:** ~21 MB
        * **Tốc độ:** Khá nhanh
        """)
        
    with col_m:
        st.error("🔴 YOLOv8m (Medium)")
        st.markdown("""
        * **Đặc điểm:** Trích xuất đặc trưng sâu, nhận diện tốt các vết gãy phức tạp.
        * **mAP@50:** ~0.772 *(Nhập số thực tế của bạn)*
        * **Dung lượng:** ~50 MB
        * **Tốc độ:** Chậm nhất (Cần GPU)
        """)

# ==========================================
# TAB 3: THÔNG TIN TỔNG QUAN
# ==========================================
with tab_about:
    st.header("Về Dự án này")
    st.write("""
    Hệ thống hỗ trợ nhận diện vị trí gãy xương trên phim X-quang, tích hợp đồng thời 3 biến thể của YOLOv8 để 
    người dùng tự do trải nghiệm và so sánh:
    
    1. **YOLOv8n (Nano):** Nhẹ nhất, dùng cho thiết bị cấu hình yếu.
    2. **YOLOv8s (Small):** Bản tiêu chuẩn, cân bằng tốt các chỉ số.
    3. **YOLOv8m (Medium):** Nặng nhất, bù lại cho kết quả y tế đáng tin cậy hơn.
    
    Tùy thuộc vào phần cứng (có GPU hay không) và yêu cầu thực tế, hệ thống cho phép linh hoạt chuyển đổi mô hình 
    để đạt hiệu quả tốt nhất.
    """)