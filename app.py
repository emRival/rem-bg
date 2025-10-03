import streamlit as st
import pandas as pd
import io
import base64
from PIL import Image, ImageEnhance, ImageFilter
from rembg import remove, new_session
import numpy as np
import cv2
from typing import Tuple
from streamlit_cropper import st_cropper

# --- KONFIGURASI HALAMAN & GAYA (CSS) ---
st.set_page_config(page_title="Editor Foto Canggih", layout="wide")

st.markdown("""
<style>
    /* (CSS dari sebelumnya tetap sama, tidak perlu diubah) */
    .main { background-color: #0f172a; color: #e2e8f0; }
    .stApp { background-color: #0f172a; }
    [data-testid="stSidebar"] { background-color: #1e293b; }
    .stButton > button { border-radius: 0.5rem; border: 1px solid #3b82f6; background-color: transparent; }
    .stButton > button:hover { border-color: #60a5fa; background-color: #3b82f6; color: white; }
    .stButton > button[kind="primary"] { background-color: #ef4444; border: none; }
    .stDownloadButton > button { background-color: #10b981; border: none; }
    .stFileUploader > label { border: 2px dashed #334155; background-color: #1e293b; border-radius: 0.5rem; }
    .control-card { background-color: #1e293b; padding: 1.2rem 1.5rem; border-radius: 0.75rem; margin-bottom: 1rem; border: 1px solid #334155;}
    .gallery-card { background-color: #1e293b; padding: 1rem; border-radius: 0.5rem; border: 1px solid #334155; }
    [data-testid="stExpander"] { background-color: #1e293b; border-radius: 0.75rem; border: 1px solid #334155; }
    [data-testid="stExpander"] summary { font-size: 1.25rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# --- FUNGSI CACHING & UTAMA ---

@st.cache_data
def load_csv(uploaded_file):
    # (Fungsi ini tidak berubah)
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
    # (Fungsi ini tidak berubah)
    return new_session(model_name)

def refine_mask_advanced(original_image, initial_mask, alpha_matting_radius, edge_shift, smooth_kernel_size):
    """Fungsi penyempurnaan mask dengan urutan logika yang diperbaiki."""
    
    # Langkah 1: Lakukan pergeseran struktural (Erode/Dilate) terlebih dahulu.
    # Ini menentukan bentuk dasar potongan sebelum diperhalus.
    if edge_shift != 0:
        kernel = np.ones((3, 3), np.uint8)
        if edge_shift < 0: # Erode / Mengikis
            initial_mask = cv2.erode(initial_mask, kernel, iterations=abs(edge_shift))
        else: # Dilate / Memperluas
            initial_mask = cv2.dilate(initial_mask, kernel, iterations=edge_shift)

    # Langkah 2: Gunakan Alpha Matting untuk menyempurnakan detail pada tepi yang sudah digeser.
    if alpha_matting_radius > 0:
        guide = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2GRAY)
        mask_float = initial_mask.astype(np.float32) / 255.0
        refined_mask_float = cv2.ximgproc.guidedFilter(guide, mask_float, alpha_matting_radius, 1e-5)
        initial_mask = (refined_mask_float * 255).astype(np.uint8)

    # Langkah 3: Berikan sentuhan akhir dengan memperhalus tepi.
    if smooth_kernel_size > 0:
        if smooth_kernel_size % 2 == 0: smooth_kernel_size += 1
        initial_mask = cv2.GaussianBlur(initial_mask, (smooth_kernel_size, smooth_kernel_size), 0)
        
    return initial_mask
def remove_background(image: Image.Image, model_name: str, **kwargs) -> Image.Image:
    """Proses hapus background dengan opsi penyempurnaan canggih."""
    try:
        # Dapatkan hasil potongan awal dari rembg
        initial_cutout = remove(image, session=get_rembg_session(model_name))
        if initial_cutout.mode != 'RGBA': return initial_cutout
        
        # Dapatkan alpha mask awal
        initial_mask = np.array(initial_cutout.split()[-1])
        
        # Kirim ke fungsi penyempurnaan canggih
        refined_alpha = refine_mask_advanced(
            original_image=image.convert("RGB"),
            initial_mask=initial_mask,
            alpha_matting_radius=kwargs.get('alpha_radius', 0),
            edge_shift=kwargs.get('edge_shift', 0),
            smooth_kernel_size=kwargs.get('smooth', 0)
        )
        
        # Gabungkan gambar asli dengan mask yang sudah super halus
        final_img_arr = np.dstack((np.array(image.convert("RGB")), refined_alpha))
        return Image.fromarray(final_img_arr, 'RGBA')

    except Exception as e:
        st.error(f"Gagal hapus background: {e}")
        return image.convert('RGBA')

# ... (Sisa fungsi-fungsi lain seperti cm_to_pixels, dll, tidak ada perubahan) ...
def cm_to_pixels(cm, dpi=300): return int(cm * dpi / 2.54)
def apply_image_adjustments(image: Image.Image, b, c, s) -> Image.Image:
    if b != 1.0: image = ImageEnhance.Brightness(image).enhance(b)
    if c != 1.0: image = ImageEnhance.Contrast(image).enhance(c)
    if s != 1.0: image = ImageEnhance.Sharpness(image).enhance(s)
    return image
# HAPUS FUNGSI create_final_image LAMA ANDA, GANTI DENGAN DUA FUNGSI INI

def get_output_size(size_choice, custom_w, custom_h, dpi=300):
    """Mendapatkan dimensi output dalam piksel berdasarkan pilihan."""
    if size_choice == "3x4 cm":
        return (int(3 * dpi / 2.54), int(4 * dpi / 2.54))
    elif size_choice == "4x6 cm":
        return (int(4 * dpi / 2.54), int(6 * dpi / 2.54))
    elif size_choice == "Ukuran Kustom (px)":
        return (custom_w, custom_h)
    return (0, 0) # Fallback untuk ukuran asli

def create_final_canvas(processed_image, size_choice, custom_dims, bg_settings):
    """
    Membuat canvas final dengan algoritma resize cerdas dan filter penajam.
    """
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

    # Untuk ukuran preset dan kustom
    canvas_width, canvas_height = get_output_size(size_choice, custom_dims[0], custom_dims[1])
    
    # --- LOGIKA RESIZE CERDAS ---
    if canvas_width > processed_image.width:
        resample_algorithm = Image.Resampling.BICUBIC # Terbaik untuk memperbesar (upscaling)
    else:
        resample_algorithm = Image.Resampling.LANCZOS # Terbaik untuk memperkecil (downscaling)
    
    scale = canvas_width / processed_image.width
    scaled_img = processed_image.resize((canvas_width, int(processed_image.height * scale)), resample_algorithm)

    # --- TERAPKAN FILTER PENAJAM SETELAH RESIZE ---
    if sharpen_amount > 0:
        scaled_img = scaled_img.filter(ImageFilter.UnsharpMask(radius=1, percent=int(sharpen_amount * 1.5), threshold=2))

    # Proses pembuatan canvas
    bg_rgba = (0,0,0,0) if transparent_bg else tuple(int(bg_color_hex[i:i+2], 16) for i in (1,3,5)) + (255,)
    canvas = Image.new('RGBA', (canvas_width, canvas_height), bg_rgba)
    canvas.paste(scaled_img, (0, offset_y), scaled_img)
    
    return canvas
# --- INISIALISASI & TAMPILAN UTAMA ---
# (Tidak ada perubahan di bagian ini)
if 'edit_mode' not in st.session_state:
    st.session_state.edit_mode = False
    st.session_state.selected_image = None
    st.session_state.images = {}
st.title("✨ Editor Foto Ijazah Canggih")
with st.sidebar:
    st.header("📋 Data Siswa")
    uploaded_csv = st.file_uploader("Unggah file CSV", type="csv")
    df = load_csv(uploaded_csv)
    if not df.empty:
        st.success(f"{len(df)} data dimuat.")
        st.dataframe(df, use_container_width=True, height=300)

# --- MODE EDITOR (dengan UI yang diperbarui) ---
if st.session_state.edit_mode and st.session_state.selected_image in st.session_state.images:
    fname = st.session_state.selected_image
    img_data = st.session_state.images[fname]
    
    st.info(f"Mengedit: **{fname}**")
    
    col_preview, col_controls = st.columns([3, 2])

    with col_preview:
        st.markdown("#### Pratinjau & Hasil Akhir")
        
        with st.expander("Langkah 1: Ukuran & Potong (Crop) 📐", expanded=True):
            size_options = ["3x4 cm", "4x6 cm", "Ukuran Asli (setelah crop)", "Ukuran Kustom (px)"]
            size_choice = st.radio("Ukuran Output", size_options, horizontal=True, key="size_radio")
            
            custom_w, custom_h = (0, 0)
            # Tampilkan input angka jika Ukuran Kustom dipilih
            if size_choice == "Ukuran Kustom (px)":
                c1, c2 = st.columns(2)
                custom_w = c1.number_input("Lebar (px)", min_value=100, max_value=4000, value=800)
                custom_h = c2.number_input("Tinggi (px)", min_value=100, max_value=4000, value=1200)

            # Atur aspect ratio cropper secara dinamis
            if size_choice == "3x4 cm":
                aspect_ratio = (3, 4)
            elif size_choice == "4x6 cm":
                aspect_ratio = (4, 6)
            elif size_choice == "Ukuran Kustom (px)":
                aspect_ratio = (custom_w, custom_h)
            else: # Untuk "Ukuran Asli"
                aspect_ratio = None # Bebas

            cropped_img = st_cropper(img_data['original'], realtime_update=True, box_color="#3b82f6", aspect_ratio=aspect_ratio, key="cropper")
            processed_img = cropped_img.copy().convert("RGBA")

        with col_controls:
            st.markdown("#### Panel Kontrol")

            with st.expander("Langkah 2: Hapus Background 🪄", expanded=False):
                with st.container(border=True):
                    bg_method = st.radio("Metode", ["Tidak", "AI Otomatis"], key="bg_method", horizontal=True)
                    
                    if bg_method == "AI Otomatis":
                        MODEL_OPTIONS = {
                            "High Quality (Detail Halus)": "isnet-general-use",
                            "Human (Untuk Foto Orang)": "u2net_human_seg",
                            "General (Cepat & Ringan)": "u2net"
                        }
                        selected_model_key = st.selectbox("Model AI", options=list(MODEL_OPTIONS.keys()), key="rembg_model_selection")
                        actual_model_name = MODEL_OPTIONS[selected_model_key]
                        
                        st.markdown("**Penyempurnaan Tepi (Edge Refinement)**")
                        # --- FITUR BARU DITAMBAHKAN DI SINI ---
                        alpha_radius = st.slider("Penyempurnaan Detail (Alpha Matting)", 0, 50, 10, help="Memperbaiki detail halus seperti rambut. Semakin tinggi nilainya, semakin kuat efeknya. Set ke 0 untuk menonaktifkan.")
                        edge = st.slider("Geser Tepi (Edge Shift)", -10, 10, 0, help="Nilai negatif akan mengikis tepi, positif akan memperluas.")
                        smooth = st.slider("Perhalus Tepi (Smooth)", 0, 25, 3, 1, help="Memperhalus tepi potongan agar lebih natural.")
                        
                        with st.spinner("Menghapus & menyempurnakan background..."):
                            processed_img = remove_background(processed_img, actual_model_name, 
                                                            alpha_radius=alpha_radius, edge_shift=edge, smooth=smooth)
            
            # ... Sisa expander lain (Langkah 3 & 4) tidak ada perubahan ...
            with st.expander("Langkah 3: Penyesuaian Gambar 🎨", expanded=False):
                with st.container(border=True):
                    brightness = st.slider("Kecerahan", 0.5, 1.5, 1.0, 0.05, key="brightness")
                    contrast = st.slider("Kontras", 0.5, 1.5, 1.0, 0.05, key="contrast")
                    # SLIDER SHARPNESS LAMA DIHAPUS, DIGANTI DENGAN YANG DI BAWAH
                    
            with st.expander("Langkah 4: Finalisasi & Simpan 💾", expanded=False):
                 with st.container(border=True):
                    # --- SLIDER BARU DITAMBAHKAN DI SINI ---
                    sharpen_final = st.slider("Filter Penajam (Final)", 0, 100, 0, 5, 
                                            help="Terapkan setelah resize untuk melawan blur. Coba nilai 20-50 untuk hasil yang lebih tajam.")
                    
                    transparent_bg = st.checkbox("Background Transparan", key="transparent_bg")
                    bg_color = st.color_picker("Warna Background", "#c02828", disabled=transparent_bg, key="bg_color")
                    offset_y = st.slider("Geser Vertikal (px)", -200, 200, 0, key="offset_y")
                    
                    # Gabungkan semua setting untuk dikirim ke fungsi
                    bg_settings = (bg_color, offset_y, transparent_bg, sharpen_final)
                    final_image = create_final_canvas(processed_img, size_choice, (custom_w, custom_h), bg_settings)

                    st.markdown("**Opsi Unduh**")
                    default_name = fname.split('.')[0]
                    # Anda bisa menambahkan kembali logika penamaan file dengan data siswa di sini jika perlu
                    filename = st.text_input("Nama File", default_name)
                    
                    fmt = st.radio("Format", ["PNG", "JPEG"], horizontal=True)
                    buf = io.BytesIO()
                    
                    # Siapkan gambar untuk disimpan
                    img_to_save = final_image
                    if fmt == 'JPEG' and img_to_save.mode == 'RGBA':
                        img_to_save = img_to_save.convert('RGB')
                    
                    img_to_save.save(buf, fmt, quality=95 if fmt == 'JPEG' else None)
                    
                    st.download_button(
                        label=f"Unduh {filename}.{fmt.lower()}",
                        data=buf,
                        file_name=f"{filename}.{fmt.lower()}",
                        mime=f"image/{fmt.lower()}",
                        use_container_width=True
                    )

        st.markdown("##### Hasil Akhir")
        st.image(locals().get("final_image", processed_img), use_column_width=True)

    with col_controls:
        with st.container(border=True):
            st.markdown("##### Aksi")
            if st.button("🏠 Kembali ke Galeri"):
                st.session_state.edit_mode = False
                st.session_state.selected_image = None
                st.rerun()
            if st.button("🗑️ Hapus Foto Ini", type="primary"):
                del st.session_state.images[fname]
                st.session_state.edit_mode = False
                st.session_state.selected_image = None
                st.rerun()

# --- MODE GALERI ---
# (Tidak ada perubahan di bagian ini)
else:
    st.header("📂 Unggah & Pilih Foto")
    uploaded_files = st.file_uploader("Pilih file foto", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.images:
                st.session_state.images[file.name] = {'original': Image.open(file)}
    
    if st.session_state.images:
        st.success(f"{len(st.session_state.images)} foto dimuat. Klik 'Edit Foto' untuk memulai.")
        cols = st.columns(4)
        for i, (fname, data) in enumerate(st.session_state.images.items()):
            with cols[i % 4]:
                with st.container(border=True):
                    st.image(data['original'], use_column_width=True)
                    st.markdown(f"**{fname}**")
                    if st.button("✏️ Edit Foto", key=f"edit_{fname}", use_container_width=True):
                        st.session_state.edit_mode = True
                        st.session_state.selected_image = fname
                        st.rerun()
    else:
        st.info("💡 Silakan unggah satu atau lebih foto untuk ditampilkan di galeri.")