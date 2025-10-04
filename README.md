# Chatbot Mutasi Barang (Streamlit)


A Streamlit-deployable AI-powered chatbot for processing handwriting and typed documents that contain stock/mutation tables. It extracts table rows with columns `NO., NAMA BARANG, JUMLAH, KETERANGAN, TANGGAL KELUAR`, stores them in a local SQLite database (with optional Supabase integration), and allows export in the requested format:
`URAIAN BARANG, ITEM KELUAR, SATUAN, TANGGAL KELUAR, KETERANGAN`.


This project uses free/open-source tools only (EasyOCR, pytesseract, OpenCV, pandas, sqlite3, matplotlib). For handwriting recognition we use EasyOCR with pytesseract fallback.