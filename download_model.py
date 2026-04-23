"""
download_model.py
=================
Download model.pkl dari Google Drive dengan benar.

Masalah umum:
  - Google Drive mengembalikan HTML "virus scan warning page" untuk file besar
  - File yang tersimpan adalah HTML bukan binary pickle
  - pickle.load() / joblib.load() gagal dengan error 60 (ENOEXEC)

Fix:
  - Gunakan requests.Session()
  - Ekstrak confirm token dari cookies ATAU dari HTML body
  - Verifikasi file dengan magic bytes sebelum dianggap valid
"""

import os
import re
import logging
import requests

logger = logging.getLogger(__name__)

# ── Konfigurasi ──
# Ambil dari URL Google Drive:
# https://drive.google.com/file/d/FILE_ID_DI_SINI/view
# Ganti dengan FILE_ID model.pkl Anda
GDRIVE_FILE_ID = os.environ.get("MODEL_FILE_ID", "1jb59fnfXX1Jtcv2-5xv70UDGIHCM4J-6")
MODEL_PATH = os.environ.get("MODEL_PATH", "model.pkl")

GDRIVE_URL = "https://drive.google.com/uc"
CHUNK_SIZE = 32768  # 32 KB


def _get_confirm_token(response: requests.Response) -> str | None:
    """
    Ekstrak confirm token dari response Google Drive.
    Google Drive menyimpan token di cookies ATAU di dalam HTML body.
    """
    # Cara 1: dari cookies (file kecil-menengah)
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    # Cara 2: dari HTML body (file besar, Google Drive baru)
    try:
        content = response.content.decode("utf-8", errors="ignore")
        # Cari pola confirm=TOKEN di dalam href atau action form
        match = re.search(r'confirm=([0-9A-Za-z_\-]+)', content)
        if match:
            return match.group(1)

        # Fallback: cari di dalam input hidden
        match = re.search(r'name="confirm"\s+value="([^"]+)"', content)
        if match:
            return match.group(1)

        # Google Drive terbaru: uuid di dalam URL download
        match = re.search(r'"(/uc\?export=download[^"]+confirm=[^"]+)"', content)
        if match:
            # Return URL langsung sebagai signal khusus
            return "__URL__:" + match.group(1).replace("&amp;", "&")

    except Exception:
        pass

    return None


def _is_valid_model(file_path: str) -> bool:
    """
    Verifikasi bahwa file adalah model yang valid (pickle atau joblib),
    BUKAN HTML page dari Google Drive.

    Magic bytes:
      - Pickle : dimulai dengan 0x80 (opcode PROTO)
      - Joblib  : format pickle juga, sama
      - HTML    : dimulai dengan '<' (0x3C) -> TIDAK VALID
    """
    try:
        with open(file_path, "rb") as f:
            magic = f.read(4)

        if not magic:
            logger.error("File kosong.")
            return False

        # Cek apakah isi file adalah HTML (tanda gagal download)
        if magic[:1] in (b'<', b'{') or magic[:4] in (b'<!DO', b'<htm', b'<HTM'):
            logger.error("File berisi HTML — Google Drive mengembalikan confirmation page, bukan model.")
            return False

        # Pickle/joblib dimulai dengan byte 0x80
        if magic[0] == 0x80:
            return True

        # Beberapa format joblib lama menggunakan header berbeda
        # Cek apakah bisa dibuka (soft check)
        logger.warning(f"Magic bytes tidak dikenali: {magic.hex()} — mencoba lanjut...")
        return True  # Biarkan joblib.load() yang memvalidasi

    except Exception as e:
        logger.error(f"Gagal membaca file: {e}")
        return False


def _save_stream(response: requests.Response, dest: str) -> None:
    """Simpan response stream ke file dengan chunked writing."""
    os.makedirs(os.path.dirname(dest) if os.path.dirname(dest) else ".", exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter keep-alive empty chunks
                f.write(chunk)


def download_model(
    file_id: str = GDRIVE_FILE_ID,
    dest_path: str = MODEL_PATH,
    force: bool = False,
) -> bool:
    """
    Download model.pkl dari Google Drive.

    Args:
        file_id  : ID file Google Drive (dari URL /file/d/FILE_ID/view)
        dest_path: Path tujuan penyimpanan model
        force    : Paksa download ulang meski file sudah ada

    Returns:
        True jika berhasil, False jika gagal
    """
    # Skip jika sudah ada dan valid
    if not force and os.path.exists(dest_path):
        if _is_valid_model(dest_path):
            logger.info(f"Model sudah ada dan valid: {dest_path}")
            return True
        else:
            logger.warning(f"Model ada tapi tidak valid, download ulang...")
            os.remove(dest_path)

    if file_id == "GANTI_DENGAN_FILE_ID_ANDA":
        logger.error(
            "FILE_ID belum diset! Set environment variable MODEL_FILE_ID "
            "atau edit GDRIVE_FILE_ID di download_model.py"
        )
        return False

    logger.info(f"Downloading model dari Google Drive (file_id={file_id})...")
    session = requests.Session()

    try:
        # ── Request pertama ──
        resp = session.get(
            GDRIVE_URL,
            params={"id": file_id, "export": "download"},
            stream=True,
            timeout=30,
        )
        resp.raise_for_status()

        # Cek apakah ada confirmation token (file besar)
        token = _get_confirm_token(resp)

        if token:
            if token.startswith("__URL__:"):
                # Google Drive baru: gunakan URL langsung dari HTML
                redirect_url = "https://drive.google.com" + token[8:]
                logger.info("Menggunakan redirect URL dari HTML...")
                resp = session.get(redirect_url, stream=True, timeout=60)
            else:
                logger.info(f"Confirm token ditemukan, mengirim ulang request...")
                resp = session.get(
                    GDRIVE_URL,
                    params={"id": file_id, "export": "download", "confirm": token},
                    stream=True,
                    timeout=60,
                )
            resp.raise_for_status()

        # ── Simpan file ──
        _save_stream(resp, dest_path)
        logger.info(f"Download selesai → {dest_path}")

        # ── Verifikasi ──
        if not _is_valid_model(dest_path):
            logger.error(
                "File yang didownload tidak valid.\n"
                "Kemungkinan penyebab:\n"
                "  1. FILE_ID salah\n"
                "  2. File Google Drive tidak bisa diakses publik\n"
                "  3. Google Drive mengubah format confirmation page\n"
                "Solusi: Pastikan sharing = 'Anyone with the link' dan coba lagi."
            )
            os.remove(dest_path)
            return False

        size_mb = os.path.getsize(dest_path) / 1024 / 1024
        logger.info(f"Model valid. Ukuran: {size_mb:.2f} MB")
        return True

    except requests.exceptions.Timeout:
        logger.error("Timeout saat download model. Coba lagi atau periksa koneksi.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error saat download: {e}")
    except Exception as e:
        logger.error(f"Error tidak terduga: {e}")

    # Bersihkan file parsial jika ada
    if os.path.exists(dest_path):
        os.remove(dest_path)
    return False


# ── Entrypoint langsung ──
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    success = download_model(force=True)
    print("✓ Berhasil" if success else "✗ Gagal")
