import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import io
import os

# ---------------------------------------------------------
# 1. CẤU HÌNH TRANG
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Chẩn đoán Gãy xương X-quang",
    page_icon="🦴",
    layout="wide"
)

# ---------------------------------------------------------
# 2. SIDEBAR - CHỌN MÔ HÌNH VÀ TÙY CHỈNH THAM SỐ
# ---------------------------------------------------------
st.sidebar.title("⚙️ Cấu hình Hệ thống")

model_choice = st.sidebar.radio(
    "1. Lựa chọn Mô hình AI:",
    (
        "YOLOv8n (Ưu tiên tốc độ)", 
        "YOLOv8s (Cân bằng)", 
        "YOLOv8m (Ưu tiên độ chính xác)"
    )
)

model_paths = {
    "YOLOv8n (Ưu tiên tốc độ)": "weights/yolov8n_best.pt",
    "YOLOv8s (Cân bằng)": "weights/yolov8s_best.pt",
    "YOLOv8m (Ưu tiên độ chính xác)": "weights/yolov8m_best.pt"
}

@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

try:
    current_model_path = model_paths[model_choice]
    model = load_model(current_model_path)
    model_loaded = True
    model_short_name = model_choice.split(' ')[0] # Lấy ra chữ YOLOv8n, YOLOv8s hoặc YOLOv8m
except Exception as e:
    model_loaded = False
    st.sidebar.error(f"Lỗi tải mô hình. Vui lòng kiểm tra file `{current_model_path}`.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 2. Tùy chỉnh Tham số AI")
conf_threshold = st.sidebar.slider("Ngưỡng tin cậy (Confidence)", 0.0, 1.0, 0.25, 0.05)
iou_threshold = st.sidebar.slider("Ngưỡng giao thoa (IoU)", 0.0, 1.0, 0.45, 0.05)

st.sidebar.markdown("---")
st.sidebar.warning("⚠️ Báo cáo từ AI chỉ mang tính chất hỗ trợ, không thay thế chẩn đoán của Bác sĩ.")

# Hàm xử lý tăng cường ảnh X-quang (CLAHE)
def enhance_xray(image):
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    enhanced_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(enhanced_rgb)

# ---------------------------------------------------------
# 3. GIAO DIỆN CHÍNH
# ---------------------------------------------------------
st.title("🦴 Hệ thống Hỗ trợ Chẩn đoán Gãy xương X-quang")
st.markdown(f"**Trạng thái:** Sẵn sàng 🟢 | **Engine:** `{model_short_name}`")

tab_inference, tab_metrics, tab_about = st.tabs([
    "🔍 Phân tích Lâm sàng", 
    "📊 Đánh giá Kỹ thuật", 
    "ℹ️ Về Ứng dụng"
])

# ==========================================
# TAB 1: PHÂN TÍCH LÂM SÀNG (ĐÃ CẬP NHẬT DỊCH NHÃN VỊ TRÍ)
# ==========================================
with tab_inference:
    input_method = st.radio("Phương thức nạp ảnh X-quang:", ("📁 Tải file từ máy tính", "📷 Dùng Camera chụp phim"), horizontal=True)
    
    image = None
    if input_method == "📁 Tải file từ máy tính":
        uploaded_file = st.file_uploader("Chọn ảnh X-quang (JPG, PNG, JPEG)", type=['jpg', 'jpeg', 'png'])
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
    else:
        camera_file = st.camera_input("Đưa phim X-quang ra trước Camera")
        if camera_file:
            image = Image.open(camera_file).convert('RGB')
    
    if image is not None:
        st.markdown("---")
        
        st.markdown("### 🎛️ Tiền xử lý hình ảnh (Tùy chọn)")
        apply_enhancement = st.checkbox("Áp dụng bộ lọc CLAHE (Làm rõ cấu trúc xương & Tăng độ tương phản tia X)")
        
        processed_image = enhance_xray(image) if apply_enhancement else image

        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Ảnh đầu vào (Đã qua xử lý)**" if apply_enhancement else "**Ảnh gốc ban đầu**")
            st.image(processed_image, use_container_width=True)
            
        with col2:
            st.markdown("**Kết quả Nhận diện AI**")
            if st.button("Chẩn đoán hình ảnh", type="primary", use_container_width=True) and model_loaded:
                with st.spinner("Hệ thống đang quét phân tích..."):
                    results = model.predict(source=processed_image, conf=conf_threshold, iou=iou_threshold)
                    
                    res_plotted = results[0].plot()
                    res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                    st.image(res_plotted_rgb, use_container_width=True)
                    
                    # --- CẬP NHẬT TRÍCH XUẤT VÀ DỊCH NHÃN CHI TIẾT ---
                    boxes = results[0].boxes
                    num_detections = len(boxes)
                    
                    st.markdown("---")
                    st.markdown("### 📝 Phiếu kết quả AI")
                    
                    if num_detections > 0:
                        st.error(f"⚠️ Phát hiện {num_detections} vị trí nghi ngờ chấn thương/gãy xương.")
                        
                        # Từ điển ánh xạ nhãn tiếng Anh sang thuật ngữ tiếng Việt
                        class_translation = {
                            'Elbow_Positive': 'Tổn thương/Gãy xương vùng Khuỷu tay',
                            'Fingers_Positive': 'Tổn thương/Gãy xương vùng Ngón tay',
                            'Forearm_Fracture': 'Gãy xương Cẳng tay',
                            'Humerus_Fracture': 'Gãy xương Cánh tay',
                            'Humerus': 'Bất thường cấu trúc xương Cánh tay',
                            'Shoulder_Fracture': 'Gãy xương bả vai/Khớp vai',
                            'Wrist_Positive': 'Tổn thương/Gãy xương vùng Cổ tay',
                            'Fracture': 'Phát hiện vết nứt/gãy xương',
                            'fracture': 'Phát hiện vết nứt/gãy xương'
                        }
                        
                        with st.expander("Xem chi tiết chẩn đoán vị trí tổn thương", expanded=True):
                            for i, box in enumerate(boxes):
                                cls_id = int(box.cls[0])
                                raw_class_name = model.names[cls_id] # Lấy tên nhãn gốc từ mô hình
                                
                                # Tra từ điển để dịch sang tiếng Việt
                                vi_class_name = class_translation.get(raw_class_name, raw_class_name)
                                
                                conf = float(box.conf) * 100
                                coords = box.xyxy[0].cpu().numpy() 
                                x1, y1, x2, y2 = map(int, coords)
                                
                                st.markdown(f"**Vị trí #{i+1}: 🔴 {vi_class_name}**")
                                st.write(f"- Nhãn gốc hệ thống: `{raw_class_name}`")
                                st.write(f"- Độ tin cậy chẩn đoán: `{conf:.2f}%`")
                                st.write(f"- Tọa độ vùng tổn thương (Pixels): `[X1:{x1}, Y1:{y1}]` đến `[X2:{x2}, Y2:{y2}]`")
                                st.markdown("---")
                    else:
                        st.success("✅ Hệ thống không phát hiện điểm bất thường trên phim chụp ở ngưỡng tin cậy hiện tại.")
                    
                    # Nút tải xuống kết quả
                    img_pil = Image.fromarray(res_plotted_rgb)
                    buf = io.BytesIO()
                    img_pil.save(buf, format="PNG")
                    st.download_button(
                        label="📥 Lưu kết quả về máy",
                        data=buf.getvalue(),
                        file_name=f"XRay_Result_{model_short_name}.png",
                        mime="image/png"
                    )

# ==========================================
# TAB 2: ĐÁNH GIÁ KỸ THUẬT (TỰ ĐỘNG THEO MODEL SIDEBAR)
# ==========================================
with tab_metrics:
    st.header(f"📊 Đánh giá Hiệu năng: {model_short_name}")
    st.markdown(f"Biểu đồ số liệu thực nghiệm thu được trong quá trình huấn luyện cấu trúc mạng **{model_short_name}**.")
    
    # Chuẩn hóa tên viết tắt để khớp với file ảnh (ví dụ: YOLOv8s -> v8s)
    prefix = model_short_name.replace("YOLOv", "v").lower()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**1. Ma trận nhầm lẫn ({model_short_name} Confusion Matrix)**")
        img_path1 = f"assets/{prefix}_confusion_matrix.png"
        if os.path.exists(img_path1):
            st.image(img_path1, use_container_width=True, caption=f"Ma trận phân loại của phiên bản {model_short_name}")
        else:
            st.error(f"Thiếu file ảnh: `{img_path1}`")
            
    with col2:
        st.markdown(f"**2. Đường cong F1-Confidence ({model_short_name} F1 Curve)**")
        img_path2 = f"assets/{prefix}_F1_curve.png"
        if os.path.exists(img_path2):
            st.image(img_path2, use_container_width=True, caption=f"Biểu đồ F1-Score tối ưu của phiên bản {model_short_name}")
        else:
            st.error(f"Thiếu file ảnh: `{img_path2}`")
            
    st.info(f"💡 Hệ thống đang tự động lọc hiển thị tài liệu kỹ thuật đồng bộ cho Engine: **{model_choice}**")

# ==========================================
# TAB 3: THÔNG TIN DỰ ÁN (ABOUT)
# ==========================================
with tab_about:
    st.header("ℹ️ Về Dự án AI Chẩn đoán X-quang")
    
    st.markdown("""
    Hệ thống **AI Chẩn đoán Gãy xương X-quang** là một giải pháp ứng dụng Học sâu vào quy trình xử lý và phân tích hình ảnh y khoa. 
    Mục tiêu của hệ thống là hoạt động như một "trợ lý ảo", hỗ trợ các bác sĩ bằng cách tự động khoanh vùng các vị trí nghi ngờ nứt/gãy xương, từ đó giảm thiểu sai sót do yếu tố chủ quan và rút ngắn thời gian chẩn đoán lâm sàng.
    
    ### 🔬 Công nghệ & Thuật toán
    - **Mô hình Nhận diện (Object Detection):** Fine-tuning các kiến trúc YOLOv8 đa kích cỡ.
    - **Tiền xử lý ảnh Y khoa:** Áp dụng thuật toán **CLAHE** (Contrast Limited Adaptive Histogram Equalization) thông qua OpenCV. Thuật toán này giúp tăng cường độ tương phản cục bộ của tia X, làm nổi bật cấu trúc xương bị mờ trước khi đưa vào mạng nơ-ron phân tích.
    - **Triển khai Giao diện (Deployment):** Xây dựng
