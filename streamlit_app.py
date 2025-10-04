import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import tempfile
import sqlite3
from datetime import datetime
from PIL import Image
import pytesseract
import easyocr
import cv2
from pdf2image import convert_from_bytes
import re
import matplotlib.pyplot as plt

# ----------------------
# Configuration / Parameters
# ----------------------
# OCR settings
OCR_LANGS = ['id', 'en']  # bahasa indonesia + english
EASYOCR_MODEL = None  # will be created lazily
TESSERACT_CMD = None  # if user installed tesseract in custom path, set here

# Database
DB_PATH = 'mutasi_barang.db'

# Confidence thresholds and parsing heuristics
MIN_CONFIDENCE = 0.30  # EasyOCR confidence threshold to accept a text block
UNIT_CANDIDATES = ['pcs', 'buah', 'kg', 'ltr', 'liter', 'box', 'roll', 'pak', 'paket']

# ----------------------
# Helper functions
# ----------------------

def init_db(path=DB_PATH):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS mutasi (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            no TEXT,
            nama_barang TEXT,
            jumlah TEXT,
            satuan TEXT,
            keterangan TEXT,
            tanggal_keluar TEXT,
            source_file TEXT,
            inserted_at TEXT
        )
    ''')
    conn.commit()
    return conn


def save_rows(rows, source_file=None):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            'INSERT INTO mutasi (no,nama_barang,jumlah,satuan,keterangan,tanggal_keluar,source_file,inserted_at) VALUES (?,?,?,?,?,?,?,?)',
            (r.get('NO', ''), r.get('NAMA BARANG', ''), r.get('JUMLAH', ''), r.get('SATUAN', ''), r.get('KETERANGAN',''), r.get('TANGGAL KELUAR',''), source_file or '', datetime.utcnow().isoformat())
        )
    conn.commit()
    conn.close()


def query_all():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('SELECT * FROM mutasi ORDER BY id DESC', conn)
    conn.close()
    return df


# Try to discover tesseract binary automatically (user can override TESSERACT_CMD env var)
if 'TESSERACT_CMD' in os.environ:
    pytesseract.pytesseract.tesseract_cmd = os.environ['TESSERACT_CMD']


@st.cache_resource
def get_easyocr_reader(langs=OCR_LANGS):
    return easyocr.Reader(langs, gpu=False)


def image_to_text_easyocr(img):
    global EASYOCR_MODEL
    if EASYOCR_MODEL is None:
        EASYOCR_MODEL = get_easyocr_reader()
    results = EASYOCR_MODEL.readtext(img)
    # results -> list of [bbox, text, confidence]
    lines = []
    for (bbox, text, conf) in results:
        if conf >= MIN_CONFIDENCE:
            lines.append({'text': text.strip(), 'conf': float(conf)})
    return lines


def image_to_lines_pytesseract(img):
    # img is PIL or numpy
    if isinstance(img, np.ndarray):
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    elif isinstance(img, Image.Image):
        pil = img
    else:
        pil = Image.open(io.BytesIO(img))
    raw = pytesseract.image_to_string(pil, lang='eng+ind')
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    return lines


# Basic table parsing heuristic: if the uploaded is excel/csv, read directly. If image/pdf,
# try extracting by detecting header row keywords and splitting lines by separators or multiple spaces.

def parse_table_from_text_lines(lines):
    """
    lines: list of strings (or dicts with 'text').
    Attempt to locate header and parse columns.
    """
    text_lines = []
    if not lines:
        return []
    # unify
    for l in lines:
        if isinstance(l, dict):
            text_lines.append(l.get('text',''))
        else:
            text_lines.append(str(l))
    # find header line that contains 'NAMA' and 'JUMLAH' or 'NO'
    header_idx = -1
    for i, l in enumerate(text_lines[:10]):
        low = l.lower()
        if 'nama' in low and 'jumlah' in low:
            header_idx = i
            break
        if 'no' in low and 'nama' in low:
            header_idx = i
            break
    if header_idx == -1:
        # fallback: assume first line is header
        header_idx = 0

    header = re.split(r'\s{2,}|\t|;', text_lines[header_idx])
    header = [h.strip().upper() for h in header if h.strip()]

    # normalize known variants
    mapping = {}
    for i, h in enumerate(header):
        hh = h.replace('.', '').strip()
        if 'NAMA' in hh:
            mapping[i] = 'NAMA BARANG'
        elif 'JUMLAH' in hh or 'QTY' in hh or 'QTY' in hh:
            mapping[i] = 'JUMLAH'
        elif 'KETERANGAN' in hh or 'KET' in hh:
            mapping[i] = 'KETERANGAN'
        elif 'TANGGAL' in hh or 'TGL' in hh:
            mapping[i] = 'TANGGAL KELUAR'
        elif 'NO' in hh:
            mapping[i] = 'NO'
        else:
            mapping[i] = hh

    rows = []
    for l in text_lines[header_idx+1:]:
        parts = re.split(r'\s{2,}|\t|;', l)
        parts = [p.strip() for p in parts if p.strip()]
        if not parts:
            continue
        # if number of parts matches header, map directly
        if len(parts) == len(header):
            r = {}
            for i, p in enumerate(parts):
                key = mapping.get(i, f'COL{i}')
                r[key] = p
            # try to fill SATUAN from KETERANGAN
            r['SATUAN'] = guess_unit(r.get('KETERANGAN',''))
            rows.append(r)
        else:
            # try to split by commas
            joined = ' | '.join(parts)
            # heuristic: split by ' - ' or ' , '
            candidate = re.split(r'\s{2,}|\s-\s|,', l)
            candidate = [c.strip() for c in candidate if c.strip()]
            if len(candidate) >= 3:
                # place: NO, NAMA, JUMLAH, KETERANGAN, TANGGAL if possible
                r = {}
                # crude mapping
                try:
                    r['NO'] = candidate[0]
                    r['NAMA BARANG'] = candidate[1]
                    r['JUMLAH'] = candidate[2]
                    if len(candidate) > 3:
                        r['KETERANGAN'] = candidate[3]
                    if len(candidate) > 4:
                        r['TANGGAL KELUAR'] = candidate[4]
                except Exception:
                    pass
                r['SATUAN'] = guess_unit(r.get('KETERANGAN',''))
                rows.append(r)
            else:
                # skip
                continue
    return rows


def guess_unit(ket):
    if not ket:
        return ''
    k = ket.lower()
    for u in UNIT_CANDIDATES:
        if u in k:
            return u
    # try to capture patterns like '10 kg' inside keterangan
    m = re.search(r"(kg|pcs|buah|box|paket|ltr|liter)\b", k)
    if m:
        return m.group(1)
    return ''


def process_image_file_bytes(file_bytes):
    # Convert to OpenCV image
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    # try EasyOCR -> get lines
    lines = image_to_text_easyocr(img)
    parsed = parse_table_from_text_lines(lines)
    if parsed:
        return parsed
    # fallback to pytesseract line parsing
    lines2 = image_to_lines_pytesseract(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    parsed2 = parse_table_from_text_lines(lines2)
    return parsed2


def process_pdf_bytes(file_bytes):
    pages = convert_from_bytes(file_bytes)
    all_rows = []
    for p in pages:
        # convert PIL to bytes
        buf = io.BytesIO()
        p.save(buf, format='PNG')
        rows = process_image_file_bytes(buf.getvalue())
        if rows:
            all_rows.extend(rows)
    return all_rows


# ----------------------
# Streamlit App UI
# ----------------------

st.set_page_config(page_title='Chatbot Mutasi Barang', layout='wide')
st.title('Chatbot Mutasi Barang — Streamlit')
st.write('Sistem ini mengekstraksi data tabel (handwriting & printed), menyimpan ke database, menampilkan visualisasi, dan mengekspor data. Semua tools menggunakan komponen gratis/open-source.')

conn = init_db()

with st.sidebar:
    st.header('Pengaturan')
    st.markdown('Atur parameter OCR dan jalur Tesseract jika perlu.')
    tess_path = st.text_input('Tesseract binary path (opsional)', value=os.environ.get('TESSERACT_CMD',''))
    if tess_path:
        pytesseract.pytesseract.tesseract_cmd = tess_path
    min_conf = st.slider('Minimal confidence (EasyOCR)', min_value=0.0, max_value=1.0, value=MIN_CONFIDENCE, step=0.05)
    st.write('Catatan: jika kualitas tulisan rendah, turunkan confidence.')

# file uploader
uploaded = st.file_uploader('Unggah file (image/pdf/xlsx/csv)', type=['png','jpg','jpeg','pdf','xls','xlsx','csv'])

if uploaded is not None:
    st.info(f'Nama file: {uploaded.name} — {uploaded.type}')
    data_rows = []
    if uploaded.type == 'application/pdf' or uploaded.name.lower().endswith('.pdf'):
        st.write('Memproses PDF...')
        file_bytes = uploaded.read()
        data_rows = process_pdf_bytes(file_bytes)

    elif uploaded.type.startswith('image') or any(uploaded.name.lower().endswith(ext) for ext in ['.png','.jpg','.jpeg']):
        st.write('Memproses gambar...')
        file_bytes = uploaded.read()
        data_rows = process_image_file_bytes(file_bytes)

    elif uploaded.name.lower().endswith(('.xls','.xlsx')):
        st.write('Memproses spreadsheet...')
        try:
            df = pd.read_excel(uploaded)
        except Exception:
            df = pd.read_csv(uploaded)
        # normalize expected columns
        expected = ['NO', 'NAMA BARANG', 'JUMLAH', 'KETERANGAN', 'TANGGAL KELUAR']
        df_cols = [c.upper().strip() for c in df.columns]
        mapping = {}
        for c in df.columns:
            cu = c.upper().strip()
            if 'NAMA' in cu:
                mapping[c] = 'NAMA BARANG'
            elif 'JUMLAH' in cu or 'QTY' in cu:
                mapping[c] = 'JUMLAH'
            elif 'KETERANGAN' in cu or 'KET' in cu:
                mapping[c] = 'KETERANGAN'
            elif 'TANGGAL' in cu or 'TGL' in cu:
                mapping[c] = 'TANGGAL KELUAR'
            elif 'NO' in cu:
                mapping[c] = 'NO'
            else:
                mapping[c] = cu
        df_renamed = df.rename(columns=mapping)
        rows = []
        for _, r in df_renamed.iterrows():
            row = {
                'NO': str(r.get('NO','')),
                'NAMA BARANG': str(r.get('NAMA BARANG','')),
                'JUMLAH': str(r.get('JUMLAH','')),
                'KETERANGAN': str(r.get('KETERANGAN','')),
                'TANGGAL KELUAR': str(r.get('TANGGAL KELUAR','')),
                'SATUAN': guess_unit(str(r.get('KETERANGAN','')))
            }
            rows.append(row)
        data_rows = rows

    elif uploaded.name.lower().endswith('.csv'):
        df = pd.read_csv(uploaded)
        # same handling as excel
        # ... (reuse code)
        expected = ['NO', 'NAMA BARANG', 'JUMLAH', 'KETERANGAN', 'TANGGAL KELUAR']
        df_cols = [c.upper().strip() for c in df.columns]
        mapping = {}
        for c in df.columns:
            cu = c.upper().strip()
            if 'NAMA' in cu:
                mapping[c] = 'NAMA BARANG'
            elif 'JUMLAH' in cu or 'QTY' in cu:
                mapping[c] = 'JUMLAH'
            elif 'KETERANGAN' in cu or 'KET' in cu:
                mapping[c] = 'KETERANGAN'
            elif 'TANGGAL' in cu or 'TGL' in cu:
                mapping[c] = 'TANGGAL KELUAR'
            elif 'NO' in cu:
                mapping[c] = 'NO'
            else:
                mapping[c] = cu
        df_renamed = df.rename(columns=mapping)
        rows = []
        for _, r in df_renamed.iterrows():
            row = {
                'NO': str(r.get('NO','')),
                'NAMA BARANG': str(r.get('NAMA BARANG','')),
                'JUMLAH': str(r.get('JUMLAH','')),
                'KETERANGAN': str(r.get('KETERANGAN','')),
                'TANGGAL KELUAR': str(r.get('TANGGAL KELUAR','')),
                'SATUAN': guess_unit(str(r.get('KETERANGAN','')))
            }
            rows.append(row)
        data_rows = rows

    st.success(f'Ditemukan {len(data_rows)} baris.')

    if data_rows:
        df_preview = pd.DataFrame(data_rows)
        # normalize columns to expected
        if 'SATUAN' not in df_preview.columns:
            df_preview['SATUAN'] = df_preview.get('KETERANGAN','').apply(lambda x: guess_unit(str(x)))
        st.subheader('Pratinjau data hasil ekstraksi')
        st.dataframe(df_preview)

        if st.button('Simpan ke database'):
            save_rows(data_rows, source_file=uploaded.name)
            st.success('Data berhasil disimpan ke database lokal.')

        # export mapping as required
        export_df = pd.DataFrame({
            'URAIAN BARANG': df_preview.get('NAMA BARANG', ''),
            'ITEM KELUAR': df_preview.get('JUMLAH',''),
            'SATUAN': df_preview.get('SATUAN',''),
            'TANGGAL KELUAR': df_preview.get('TANGGAL KELUAR',''),
            'KETERANGAN': df_preview.get('KETERANGAN','')
        })

        st.download_button('Export CSV (format yang diminta)', data=export_df.to_csv(index=False).encode('utf-8'), file_name='export_mutasi.csv', mime='text/csv')
        st.download_button('Export Excel', data=to_excel(export_df), file_name='export_mutasi.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

# show DB contents, visualisasi sederhana
st.sidebar.markdown('---')
if st.sidebar.button('Tampilkan isi database'):
    df_all = query_all()
    st.subheader('Isi database (terbaru di atas)')
    st.dataframe(df_all)
    if not df_all.empty:
        # simple chart: jumlah per nama barang (try to convert jumlah numeric)
        tmp = df_all.copy()
        tmp['ITEM'] = pd.to_numeric(tmp['jumlah'].astype(str).str.extract(r'(\d+)')[0].fillna(0))
        grouped = tmp.groupby('nama_barang', dropna=False)['ITEM'].sum().reset_index().sort_values('ITEM', ascending=False).head(20)
        st.subheader('Rekap item keluar (top 20)')
        fig, ax = plt.subplots()
        ax.bar(grouped['nama_barang'], grouped['ITEM'])
        ax.set_xticklabels(grouped['nama_barang'], rotation=45, ha='right')
        ax.set_ylabel('Jumlah')
        st.pyplot(fig)

# Chatbot style QA about mutasi barang (simple rule-based + retrieval)
st.markdown('---')
st.header('Chatbot — Tanya Jawab (domain: mutasi barang)')
query = st.text_input('Tanyakan sesuatu tentang data atau mutasi barang (contoh: berapa total barang X keluar bulan lalu)')
if query:
    # minimal NLP: try to answer simple retrieval and aggregation queries
    df = query_all()
    if df.empty:
        st.info('Database kosong. Silakan unggah file terlebih dahulu.')
    else:
        # naive intent detection
        q = query.lower()
        # check for keyword 'total' and product name
        if 'total' in q or 'jumlah' in q or 'berapa' in q:
            # find product name in query
            names = df['nama_barang'].astype(str).unique()
            matched = [n for n in names if n and n.lower() in q]
            if matched:
                sel = df[df['nama_barang'].str.lower() == matched[0].lower()]
                nums = pd.to_numeric(sel['jumlah'].astype(str).str.extract(r'(\d+)')[0].fillna(0))
                total = int(nums.sum())
                st.success(f"Total keluar untuk '{matched[0]}' adalah {total} (berdasarkan data yang tersedia).")
            else:
                st.info('Tidak menemukan nama barang spesifik dalam kueri. Silakan sebutkan nama barang secara lengkap.')

# ----------------------
# Utility helper: export excel
# ----------------------

def to_excel(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='mutasi')
        writer.save()
    processed_data = output.getvalue()
    return processed_data

from supabase_example import upload_to_supabase

if st.button("Upload ke Supabase"):
    if not df.empty:
        with st.spinner("Mengunggah data ke Supabase..."):
            try:
                response = upload_to_supabase(df)
                st.success("✅ Data berhasil dikirim ke Supabase!")
            except Exception as e:
                st.error(f"❌ Gagal mengunggah: {e}")
    else:
        st.warning("Tidak ada data untuk diunggah.")

from ai_suggestion import generate_ai_recommendations

st.markdown("### 🤖 Rekomendasi Otomatis (AI Suggestion)")
if st.button("Lihat Rekomendasi AI"):
    if not df.empty:
        with st.spinner("Menganalisis data mutasi barang..."):
            recs = generate_ai_recommendations(df)
            for rec in recs:
                st.info(rec)
    else:
        st.warning("Tidak ada data untuk dianalisis.")

from ai_visualization import plot_monthly_trend, plot_item_distribution, plot_trend_by_item

st.markdown("## 📊 Visualisasi Tren Barang")

if not df.empty:
    with st.expander("🔹 Tren Total Barang Keluar per Bulan"):
        plot_monthly_trend(df)

    with st.expander("🔹 Distribusi Barang Keluar Berdasarkan Jenis Barang"):
        plot_item_distribution(df)

    with st.expander("🔹 Tren Barang Spesifik"):
        item_names = df["uraian_barang"].unique().tolist()
        selected_item = st.selectbox("Pilih barang untuk melihat trennya:", item_names)
        plot_trend_by_item(df, selected_item)
else:
    st.info("Silakan unggah data terlebih dahulu untuk menampilkan grafik.")
