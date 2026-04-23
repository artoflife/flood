"""
download_model.py
=================
Download model.pkl dari Google Drive dengan retry logic.

Perubahan dari versi sebelumnya:
  - Tambah retry dengan exponential backoff untuk handle 503
  - Gunakan requests.adapters.HTTPAdapter dengan Retry
  - Tambah fallback URL (export=download langsung)
  - Timeout lebih panjang untuk file 454MB
"""

import os
import re
import time
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# ── Konfigurasi ──
GDRIVE_FILE_ID = os.environ.get("MODEL_FILE_ID", "1jb59fnfXX1Jtcv2-5xv70UDGIHCM4J-6")
MODEL_PATH = os.environ.get("MODEL_PATH", "model.pkl")

GDRIVE_URL = "https://drive.google.com/uc"
GDRIVE_DOWNLOAD_URL = "https://drive.usercontent.google.com/download"  # URL baru Google
CHUNK_SIZE = 32768  # 32 KB
MAX_RETRIES = 5
BACKOFF_FACTOR = 2  # 1s, 2s, 4s, 8s, 16s


def _make_session() -> requests.Session:
    """
    Buat session dengan retry otomatis untuk 503, 502, 429.
    Exponential backoff: 1s → 2s → 4s → 8s → 16s
    """
    session = requests.Session()
    retry = Retry(
        total=MAX_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _get_confirm_token(response: requests.Response) -> str | None:
    """
    Ekstrak confirm token dari response Google Drive.
    Coba dari cookies dulu, lalu parse HTML body.
    """
    # Cara 1: dari cookies
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    # Cara 2: parse HTML body
    try:
        content = response.content.decode("utf-8", errors="ignore")

        # Pola confirm=TOKEN
        match = re.search(r'confirm=([0-9A-Za-z_\-]+)', content)
        if match:
            return match.group(1)

        # Input hidden
        match = re.search(r'name="confirm"\s+value="([^"]+)"', content)
        if match:
            return match.group(1)

        # URL redirect langsung dari HTML
        match = re.search(r'"(/uc\?export=download[^"]+confirm=[^"]+)"', content)
        if match:
            return "__URL__:" + match.group(1).replace("&amp;", "&")

        # Format Google Drive terbaru (usercontent.google.com)
        match = re.search(
            r'href="(https://drive\.usercontent\.google\.com/download[^"]+)"',
            content
        )
        if match:
            return "__FULLURL__:" + match.group(1).replace("&amp;", "&")

    except Exception:
        pass

    return None


def _is_valid_model(file_path: str) -> bool:
    """
    Verifikasi file adalah pickle/joblib, bukan HTML page.
    Magic bytes pickle: 0x80
    HTML: dimulai '<'
    """
    try:
        with open(file_path, "rb") as f:
            magic = f.read(4)

        if not magic:
            logger.error("File kosong.")
            return False

        if magic[:1] in (b'<', b'{') or magic[:4] in (b'<!DO', b'<htm', b'<HTM'):
            logger.error("File berisi HTML — bukan model yang valid.")
            return False

        if magic[0] == 0x80:
            return True

        logger.warning(f"Magic bytes tidak dikenali: {magic.hex()} — mencoba lanjut...")
        return True

    except Exception as e:
        logger.error(f"Gagal membaca file: {e}")
        return False


def _save_stream(response: requests.Response, dest: str) -> None:
    """Simpan response stream ke file secara chunked."""
    os.makedirs(os.path.dirname(dest) if os.path.dirname(dest) else ".", exist_ok=True)
    total = 0
    with open(dest, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                total += len(chunk)
    logger.info(f"Ditulis {total / 1024 / 1024:.1f} MB ke {dest}")


def _try_download_with_session(session: requests.Session, file_id: str, dest_path: str) -> bool:
    """
    Satu percobaan download lengkap: request pertama → token → request final → simpan.
    Return True jika berhasil.
    """
    # ── Coba URL lama (drive.google.com/uc) ──
    for base_url, params in [
        (GDRIVE_URL, {"id": file_id, "export": "download"}),
        (GDRIVE_DOWNLOAD_URL, {"id": file_id, "export": "download", "confirm": "t"}),
    ]:
        try:
            logger.info(f"Mencoba: {base_url}")
            resp = session.get(base_url, params=params, stream=True, timeout=30)

            if resp.status_code == 503:
                logger.warning(f"503 dari {base_url}, lanjut ke URL berikutnya...")
                continue

            resp.raise_for_status()

            # Cek apakah perlu confirm token
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" in content_type:
                token = _get_confirm_token(resp)

                if not token:
                    logger.warning("HTML response tapi tidak ada confirm token.")
                    continue

                if token.startswith("__FULLURL__:"):
                    final_url = token[12:]
                    logger.info("Menggunakan full URL dari HTML...")
                    resp = session.get(final_url, stream=True, timeout=120)
                elif token.startswith("__URL__:"):
                    redirect_url = "https://drive.google.com" + token[8:]
                    logger.info("Menggunakan redirect URL dari HTML...")
                    resp = session.get(redirect_url, stream=True, timeout=120)
                else:
                    logger.info(f"Confirm token: {token[:8]}..., mengirim ulang request...")
                    resp = session.get(
                        GDRIVE_URL,
                        params={"id": file_id, "export": "download", "confirm": token},
                        stream=True,
                        timeout=120,
                    )
                resp.raise_for_status()

            _save_stream(resp, dest_path)
            return True

        except requests.exceptions.Timeout:
            logger.warning(f"Timeout dari {base_url}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request error dari {base_url}: {e}")

    return False


def download_model(
    file_id: str = GDRIVE_FILE_ID,
    dest_path: str = MODEL_PATH,
    force: bool = False,
) -> bool:
    """
    Download model.pkl dari Google Drive dengan retry + exponential backoff.

    Args:
        file_id  : ID file Google Drive
        dest_path: Path tujuan
        force    : Download ulang meski sudah ada

    Returns:
        True jika berhasil
    """
    if not force and os.path.exists(dest_path):
        if _is_valid_model(dest_path):
            logger.info(f"Model sudah ada dan valid: {dest_path}")
            return True
        else:
            logger.warning("Model ada tapi tidak valid, download ulang...")
            os.remove(dest_path)

    logger.info(f"Downloading model (file_id={file_id})...")
    session = _make_session()

    # Retry manual di atas retry HTTP adapter, untuk handle
    # kasus 503 yang tidak tertangkap adapter (e.g., stream timeout)
    for attempt in range(1, MAX_RETRIES + 1):
        logger.info(f"Percobaan {attempt}/{MAX_RETRIES}...")

        success = _try_download_with_session(session, file_id, dest_path)

        if success and os.path.exists(dest_path):
            if _is_valid_model(dest_path):
                size_mb = os.path.getsize(dest_path) / 1024 / 1024
                logger.info(f"✓ Model valid. Ukuran: {size_mb:.2f} MB")
                return True
            else:
                logger.error("File didownload tapi tidak valid (mungkin HTML).")
                os.remove(dest_path)

        if attempt < MAX_RETRIES:
            wait = BACKOFF_FACTOR ** attempt
            logger.warning(f"Gagal, tunggu {wait}s sebelum retry...")
            time.sleep(wait)

    logger.error(
        "Semua percobaan gagal.\n"
        "Kemungkinan penyebab:\n"
        "  1. Google Drive sedang down (503) — coba lagi nanti\n"
        "  2. File tidak dibagikan publik — set ke 'Anyone with the link'\n"
        "  3. FILE_ID salah\n"
        "Solusi alternatif: Gunakan Hugging Face Hub atau object storage (S3/R2)"
    )

    if os.path.exists(dest_path):
        os.remove(dest_path)
    return False


# ── Entrypoint langsung ──
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    success = download_model(force=True)
    print("✓ Berhasil" if success else "✗ Gagal")
