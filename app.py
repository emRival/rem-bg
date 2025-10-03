import streamlit as st
import pandas as pd
import io
import base64
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from rembg import remove, new_session
import numpy as np
import cv2
from typing import Tuple, Dict, Any
from streamlit_cropper import st_cropper

# --- KONFIGURASI HALAMAN & GAYA (CSS) ---
st.set_page_config(page_title="Editor Foto Canggih", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0f172a; color: #e2e8f0; }
    .stApp { background-color: #0f172a; }
    [data-testid="stSidebar"] { background-color: #1e293b; }
    .stButton > button { border-radius: 0.5rem; border: 1px solid #3b82f6; background-color: transparent; }
    .stButton > button:hover { border-color: #60a5fa; background-color: #3b82f6; color: white; }
    .stButton > button[kind="primary"] { background-color: #ef4444; border: none; }
    .stDownloadButton > button { background-color: #10b981; border: none; }
    .stFileUploader > label { border: 2px dashed #334155; background-color: #1e293b; border-radius: 0.5rem; }
    .gallery-card { background-color: #1e293b; padding: 1rem; border-radius: 0.5rem; border: 1px solid #334155; }
    [data-testid="stExpander"] { background-color: #1e293b; border-radius: 0.75rem; border: 1px solid #334155; }
    [data-testid="stExpander"] summary { font-size: 1.25rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# --- FUNGSI-FUNGSI (Tidak ada perubahan) ---
@st.cache_data
def load_csv(uploaded_file):
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if 'Nama' not in df.columns or 'NISN' not in df.columns:
                st.error("CSV harus memiliki kolom 'Nama' dan 'NISN'.")
                return pd.DataFrame()
            return df[['Nama', 'NISN']]
        except Exception as e:
            st.error(f"Error membaca CSV: {e}")
    return pd.DataFrame()

@st.cache_resource
def get_rembg_session(model_name):
    return new_session(model_name)

def refine_mask_advanced(original_image, initial_mask, alpha_matting_radius, edge_shift, smooth_kernel_size):
    if edge_shift != 0:
        kernel = np.ones((3, 3), np.uint8)
        if edge_shift < 0:
            initial_mask = cv2.erode(initial_mask, kernel, iterations=abs(edge_shift))
        else:
            initial_mask = cv2.dilate(initial_mask, kernel, iterations=edge_shift)
    if alpha_matting_radius > 0:
        guide = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2GRAY)
        mask_float = initial_mask.astype(np.float32) / 255.0
        refined_mask_float = cv2.ximgproc.guidedFilter(guide, mask_float, alpha_matting_radius, 1e-5)
        initial_mask = (refined_mask_float * 255).astype(np.uint8)
    if smooth_kernel_size > 0:
        if smooth_kernel_size % 2 == 0: smooth_kernel_size += 1
        initial_mask = cv2.GaussianBlur(initial_mask, (smooth_kernel_size, smooth_kernel_size), 0)
    return initial_mask

def remove_background(image: Image.Image, model_name: str, **kwargs) -> Image.Image:
    try:
        initial_cutout = remove(image, session=get_rembg_session(model_name))
        if initial_cutout.mode != 'RGBA': return initial_cutout
        initial_mask = np.array(initial_cutout.split()[-1])
        refined_alpha = refine_mask_advanced(
            original_image=image.convert("RGB"), initial_mask=initial_mask,
            alpha_matting_radius=kwargs.get('alpha_radius', 0),
            edge_shift=kwargs.get('edge_shift', 0),
            smooth_kernel_size=kwargs.get('smooth', 0)
        )
        final_img_arr = np.dstack((np.array(image.convert("RGB")), refined_alpha))
        return Image.fromarray(final_img_arr, 'RGBA')
    except Exception as e:
        st.error(f"Gagal hapus background: {e}")
        return image.convert('RGBA')

def cm_to_pixels(cm, dpi=300): return int(cm * dpi / 2.54)
def apply_image_adjustments(image: Image.Image, b, c, s) -> Image.Image:
    img_copy = image.copy()
    if b != 1.0: img_copy = ImageEnhance.Brightness(img_copy).enhance(b)
    if c != 1.0: img_copy = ImageEnhance.Contrast(img_copy).enhance(c)
    if s != 1.0: img_copy = ImageEnhance.Sharpness(img_copy).enhance(s)
    return img_copy
def get_output_size(size_choice, custom_w, custom_h, dpi=300):
    if size_choice == "3x4 cm": return (int(3 * dpi / 2.54), int(4 * dpi / 2.54))
    elif size_choice == "4x6 cm": return (int(4 * dpi / 2.54), int(6 * dpi / 2.54))
    elif size_choice == "Ukuran Kustom (px)": return (custom_w, custom_h)
    return (0, 0)
def create_final_canvas(processed_image, size_choice, custom_dims, bg_settings):
    bg_color_hex, offset_y, transparent_bg, sharpen_amount = bg_settings
    if size_choice == "Ukuran Asli (setelah crop)":
        final_image = processed_image
        if sharpen_amount > 0:
            final_image = final_image.filter(ImageFilter.UnsharpMask(radius=1, percent=int(sharpen_amount * 1.5), threshold=2))
        if transparent_bg or final_image.mode == 'RGBA':
            return final_image
        else:
            bg_color_rgb = tuple(int(bg_color_hex[i:i+2], 16) for i in (1,3,5))
            background = Image.new('RGB', final_image.size, bg_color_rgb)
            background.paste(final_image, (0,0), final_image if final_image.mode == 'RGBA' else None)
            return background
    canvas_width, canvas_height = get_output_size(size_choice, custom_dims[0], custom_dims[1])
    if canvas_width == 0 or canvas_height == 0: return processed_image
    resample_algorithm = Image.Resampling.BICUBIC if canvas_width > processed_image.width else Image.Resampling.LANCZOS
    width_scale, height_scale = canvas_width / processed_image.width, canvas_height / processed_image.height
    scale = min(width_scale, height_scale)
    new_width, new_height = int(processed_image.width * scale), int(processed_image.height * scale)
    scaled_img = processed_image.resize((new_width, new_height), resample_algorithm)
    if sharpen_amount > 0:
        scaled_img = scaled_img.filter(ImageFilter.UnsharpMask(radius=1, percent=int(sharpen_amount * 1.5), threshold=2))
    bg_rgba = (0,0,0,0) if transparent_bg else tuple(int(bg_color_hex[i:i+2], 16) for i in (1,3,5)) + (255,)
    canvas = Image.new('RGBA', (canvas_width, canvas_height), bg_rgba)
    paste_x = (canvas_width - scaled_img.width) // 2
    canvas.paste(scaled_img, (paste_x, offset_y), scaled_img)
    return canvas

# --- INISIALISASI SESSION STATE ---
if 'edit_mode' not in st.session_state:
    st.session_state.edit_mode = False
    st.session_state.selected_image = None
    st.session_state.images: Dict[str, Dict[str, Any]] = {}

st.title("✨ Editor Foto Ijazah Canggih")
with st.sidebar:
    st.header("📋 Data Siswa")
    uploaded_csv = st.file_uploader("Unggah file CSV", type="csv")
    df = load_csv(uploaded_csv)
    if not df.empty:
        st.success(f"{len(df)} data dimuat.")
        st.dataframe(df, use_container_width=True, height=300)

# --- FUNGSI UNTUK MENYIMPAN PENGATURAN ---
def save_current_settings(fname, current_processed_img):
    st.session_state.images[fname]['processed'] = current_processed_img
    st.session_state.images[fname]['settings'] = {
        'size_choice': st.session_state.get(f"size_radio_{fname}", "Ukuran Asli (setelah crop)"),
        'custom_w': st.session_state.get(f"custom_w_{fname}", 800),
        'custom_h': st.session_state.get(f"custom_h_{fname}", 1200),
        'bg_method': st.session_state.get(f"bg_method_{fname}", "Tidak"),
        'rembg_model_selection': st.session_state.get(f"rembg_model_selection_{fname}", "High Quality (Detail Halus)"),
        'alpha_radius': st.session_state.get(f"alpha_radius_{fname}", 10),
        'edge_shift': st.session_state.get(f"edge_shift_{fname}", 0),
        'smooth': st.session_state.get(f"smooth_{fname}", 3),
        'brightness': st.session_state.get(f"brightness_{fname}", 1.0),
        'contrast': st.session_state.get(f"contrast_{fname}", 1.0),
        'sharpen_final': st.session_state.get(f"sharpen_final_{fname}", 0),
        'transparent_bg': st.session_state.get(f"transparent_bg_{fname}", False),
        'bg_color': st.session_state.get(f"bg_color_{fname}", "#c02828"),
        'offset_y': st.session_state.get(f"offset_y_{fname}", 0),
    }

# --- MODE EDITOR ---
if st.session_state.edit_mode and st.session_state.selected_image in st.session_state.images:
    fname = st.session_state.selected_image
    img_data = st.session_state.images[fname]
    
    saved_settings = img_data.get('settings', {})

    current_size_choice = saved_settings.get('size_choice', "Ukuran Asli (setelah crop)")
    current_custom_w = saved_settings.get('custom_w', 800)
    current_custom_h = saved_settings.get('custom_h', 1200)
    current_bg_method = saved_settings.get('bg_method', "Tidak")
    current_rembg_model_selection = saved_settings.get('rembg_model_selection', "High Quality (Detail Halus)")
    current_alpha_radius = saved_settings.get('alpha_radius', 10)
    current_edge_shift = saved_settings.get('edge_shift', 0)
    current_smooth = saved_settings.get('smooth', 3)
    current_brightness = saved_settings.get('brightness', 1.0)
    current_contrast = saved_settings.get('contrast', 1.0)
    current_sharpen_final = saved_settings.get('sharpen_final', 0)
    current_transparent_bg = saved_settings.get('transparent_bg', False)
    current_bg_color = saved_settings.get('bg_color', "#c02828")
    current_offset_y = saved_settings.get('offset_y', 0)

    st.info(f"Mengedit: **{fname}**")
    
    col_preview, col_controls = st.columns([3, 2])

    with col_preview:
        st.markdown("#### Pratinjau & Hasil Akhir")
        with st.expander("Langkah 1: Ukuran & Potong (Crop) 📐", expanded=True):
            size_options = ["3x4 cm", "4x6 cm", "Ukuran Asli (setelah crop)", "Ukuran Kustom (px)"]
            size_choice = st.radio("Ukuran Output", size_options, horizontal=True, 
                                   index=size_options.index(current_size_choice), key=f"size_radio_{fname}")
            
            custom_w, custom_h = (0, 0)
            if size_choice == "Ukuran Kustom (px)":
                c1, c2 = st.columns(2)
                custom_w = c1.number_input("Lebar (px)", min_value=100, max_value=4000, value=current_custom_w, key=f"custom_w_{fname}")
                custom_h = c2.number_input("Tinggi (px)", min_value=100, max_value=4000, value=current_custom_h, key=f"custom_h_{fname}")

            if size_choice == "3x4 cm": aspect_ratio = (3, 4)
            elif size_choice == "4x6 cm": aspect_ratio = (4, 6)
            elif size_choice == "Ukuran Kustom (px)":
                aspect_ratio = (custom_w, custom_h) if custom_w > 0 and custom_h > 0 else None
            else: aspect_ratio = None
            
            cropped_img = st_cropper(img_data['original'], 
                                     realtime_update=True, 
                                     box_color="#3b82f6", 
                                     aspect_ratio=aspect_ratio, 
                                     key=f"cropper_{fname}_{size_choice}")
            
            # Ini adalah gambar dasar setelah di-crop
            processed_img = cropped_img.copy().convert("RGBA")

        with col_controls:
            st.markdown("#### Panel Kontrol")

            with st.expander("Langkah 2: Hapus Background 🪄", expanded=False):
                with st.container(border=True):
                    bg_method = st.radio("Metode", ["Tidak", "AI Otomatis"], horizontal=True, 
                                        index=["Tidak", "AI Otomatis"].index(current_bg_method), key=f"bg_method_{fname}")
                    
                    # SOLUSI: Pastikan `processed_img` diperbarui di langkah ini
                    if bg_method == "AI Otomatis":
                        MODEL_OPTIONS = {"High Quality (Detail Halus)": "isnet-general-use", "Human (Untuk Foto Orang)": "u2net_human_seg", "General (Cepat & Ringan)": "u2net"}
                        selected_model_key = st.selectbox("Model AI", options=list(MODEL_OPTIONS.keys()), 
                                                        index=list(MODEL_OPTIONS.keys()).index(current_rembg_model_selection), key=f"rembg_model_selection_{fname}")
                        actual_model_name = MODEL_OPTIONS[selected_model_key]
                        
                        st.markdown("**Penyempurnaan Tepi (Edge Refinement)**")
                        alpha_radius = st.slider("Detail", 0, 50, current_alpha_radius, key=f"alpha_radius_{fname}")
                        edge = st.slider("Geser Tepi", -10, 10, current_edge_shift, key=f"edge_shift_{fname}")
                        smooth = st.slider("Perhalus", 0, 25, current_smooth, 1, key=f"smooth_{fname}")
                        
                        with st.spinner("Menghapus background..."):
                            # `processed_img` dari langkah 1 menjadi input, lalu hasilnya diperbarui
                            processed_img = remove_background(processed_img, actual_model_name, 
                                                            alpha_radius=alpha_radius, edge_shift=edge, smooth=smooth)

            with st.expander("Langkah 3: Penyesuaian Gambar 🎨", expanded=False):
                with st.container(border=True):
                    brightness = st.slider("Kecerahan", 0.5, 1.5, current_brightness, 0.05, key=f"brightness_{fname}")
                    contrast = st.slider("Kontras", 0.5, 1.5, current_contrast, 0.05, key=f"contrast_{fname}")
                    
                    # SOLUSI: `processed_img` dari langkah 2 menjadi input, lalu hasilnya diperbarui
                    processed_img = apply_image_adjustments(processed_img, brightness, contrast, 1.0)
            
            # Variabel ini sekarang berisi hasil gabungan dari crop, bg removal, dan adjustments
            final_processed_img = processed_img

            with st.expander("Langkah 4: Finalisasi & Simpan 💾", expanded=False):
                with st.container(border=True):
                    sharpen_final = st.slider("Filter Penajam", 0, 100, current_sharpen_final, 5, key=f"sharpen_final_{fname}")
                    transparent_bg = st.checkbox("Background Transparan", current_transparent_bg, key=f"transparent_bg_{fname}")
                    bg_color = st.color_picker("Warna Background", current_bg_color, disabled=transparent_bg, key=f"bg_color_{fname}")
                    offset_y = st.slider("Geser Vertikal", -200, 200, current_offset_y, key=f"offset_y_{fname}")
                    
                    bg_settings = (bg_color, offset_y, transparent_bg, sharpen_final)
                    
                    # Gunakan `final_processed_img` sebagai dasar untuk membuat kanvas akhir
                    final_image_display = create_final_canvas(final_processed_img, size_choice, (custom_w, custom_h), bg_settings)

                    st.markdown("**Opsi Unduh**")
                    default_name = fname.split('.')[0]
                    filename = st.text_input("Nama File", default_name)
                    fmt = st.radio("Format", ["PNG", "JPEG"], horizontal=True)
                    buf = io.BytesIO()
                    
                    img_to_save = final_image_display
                    if fmt == 'JPEG' and img_to_save.mode == 'RGBA':
                        img_to_save = img_to_save.convert('RGB')
                    
                    img_to_save.save(buf, fmt, quality=95 if fmt == 'JPEG' else None)
                    st.download_button(label=f"Unduh {filename}.{fmt.lower()}", data=buf, file_name=f"{filename}.{fmt.lower()}", mime=f"image/{fmt.lower()}", use_container_width=True)
        
        st.markdown("##### Hasil Akhir")
        st.image(final_image_display, use_container_width=True)

    with col_controls:
        with st.container(border=True):
            st.markdown("##### Aksi")
            if st.button("🏠 Kembali ke Galeri"):
                # SOLUSI: Simpan `final_processed_img` yang sudah berisi semua editan
                save_current_settings(fname, final_processed_img) 
                st.session_state.edit_mode, st.session_state.selected_image = False, None
                st.rerun()
            if st.button("🗑️ Hapus Foto Ini", type="primary"):
                if fname in st.session_state.images:
                    del st.session_state.images[fname]
                st.session_state.edit_mode, st.session_state.selected_image = False, None
                st.rerun()

# --- MODE GALERI ---
else:
    st.header("📂 Unggah & Pilih Foto")
    uploaded_files = st.file_uploader("Pilih file foto", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.images:
                st.session_state.images[file.name] = {
                    'original': Image.open(file),
                    'processed': Image.open(file).convert("RGBA"), # Gambar awal yang diproses
                    'settings': {},
                }
    
    if st.session_state.images:
        st.success(f"{len(st.session_state.images)} foto dimuat. Klik 'Edit' untuk memulai.")
        cols = st.columns(4)
        GALLERY_PREVIEW_SIZE = (300, 300)
        for i, (fname, data) in enumerate(list(st.session_state.images.items())):
            with cols[i % 4]:
                with st.container(border=True):
                    # Tampilkan gambar 'processed' yang terakhir disimpan
                    display_image_in_gallery = data.get('processed', data['original'])
                    preview_img = ImageOps.fit(
                        display_image_in_gallery.convert("RGB"),
                        GALLERY_PREVIEW_SIZE, 
                        Image.Resampling.LANCZOS
                    )
                    st.image(preview_img, use_container_width=True, caption=f"**{fname}**")
                    
                    btn_col1, btn_col2 = st.columns(2) 
                    with btn_col1:
                        if st.button("✏️ Edit", key=f"edit_{fname}", use_container_width=True):
                            st.session_state.edit_mode = True
                            st.session_state.selected_image = fname
                            st.rerun()
                    with btn_col2:
                        if st.button("🗑️ Hapus", key=f"delete_{fname}", use_container_width=True, type="primary"):
                            del st.session_state.images[fname]
                            st.rerun()
    else:
        st.info("💡 Silakan unggah satu atau lebih foto untuk ditampilkan di galeri.")