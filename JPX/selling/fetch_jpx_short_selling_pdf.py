# fetch_jpx_short_selling_pdf.py
"""
File: fetch_jpx_short_selling_pdf.py

ソースコードの役割:
JPX（日本取引所グループ）のウェブサイトから「空売り集計」のPDFファイルを自動的に収集・ダウンロードするスクリプトです。
日次集計PDFを対象とし、インデックスページおよびバックナンバーページをクロールして、
指定された年月（MIN_TARGET_YEARMONTH）以降のデータをローカルの階層ディレクトリに保存します。
既存ファイルのスキップ機能、リクエスト間のウェイト、およびWindows環境に配慮したファイル名サニタイズ機能を備えています。
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urldefrag, urlparse
from urllib.request import Request, urlopen


BASE_DIR_URL = "https://www.jpx.co.jp/markets/statistics-equities/short-selling/"
DAILY_INDEX_URL = urljoin(BASE_DIR_URL, "index.html")

# ダウンロード対象の開始年月
MIN_TARGET_YEARMONTH = (2025, 3)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/145.0.0.0 Safari/537.36"
)


class LinkExtractor(HTMLParser):
    """HTMLから<a>タグのhref属性を抽出するパーサー。"""

    def __init__(self) -> None:
        """パーサーの初期化。"""
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """開始タグを処理し、リンクを抽出する。

        Args:
            tag (str): タグ名。
            attrs (list[tuple[str, str | None]]): タグの属性リスト。
        """
        if tag.lower() != "a":
            return
        attr_dict = dict(attrs)
        href = attr_dict.get("href")
        if href:
            self.links.append(href)


class BacknumberOptionExtractor(HTMLParser):
    """バックナンバー選択用の<select>からURLを抽出するパーサー。"""

    def __init__(self) -> None:
        """パーサーの初期化。"""
        super().__init__()
        self.in_target_select = False
        self.options: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """タグを解析し、特定のクラスを持つselect内のoption値を抽出する。

        Args:
            tag (str): タグ名。
            attrs (list[tuple[str, str | None]]): タグの属性リスト。
        """
        attr_dict = dict(attrs)

        if tag.lower() == "select":
            class_name = attr_dict.get("class") or ""
            if "backnumber" in class_name.split():
                self.in_target_select = True
            return

        if tag.lower() == "option" and self.in_target_select:
            value = attr_dict.get("value")
            if value:
                self.options.append(value)

    def handle_endtag(self, tag: str) -> None:
        """終了タグを処理。

        Args:
            tag (str): タグ名。
        """
        if tag.lower() == "select" and self.in_target_select:
            self.in_target_select = False


def fetch_bytes(url: str, timeout: int = 30) -> bytes:
    """URLからバイナリデータを取得する。

    Args:
        url (str): 取得先URL。
        timeout (int): タイムアウト秒数。

    Returns:
        bytes: 取得したデータ。
    """
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_text(url: str, timeout: int = 30) -> str:
    """URLからテキストデータを取得する。

    Args:
        url (str): 取得先URL。
        timeout (int): タイムアウト秒数。

    Returns:
        str: デコードされたテキストデータ。
    """
    raw = fetch_bytes(url, timeout=timeout)
    return raw.decode("utf-8", errors="replace")


def normalize_url(url: str) -> str:
    """URLからフラグメントを除去し正規化する。

    Args:
        url (str): 元のURL。

    Returns:
        str: 正規化されたURL。
    """
    url, _frag = urldefrag(url)
    return url


def is_same_short_selling_dir(url: str) -> bool:
    """URLが空売り集計のディレクトリ配下であるか判定する。

    Args:
        url (str): 判定対象URL。

    Returns:
        bool: 同一ディレクトリ配下であればTrue。
    """
    parsed = urlparse(url)
    return (
        parsed.scheme in ("http", "https")
        and parsed.netloc == "www.jpx.co.jp"
        and parsed.path.startswith("/markets/statistics-equities/short-selling/")
    )


def is_pdf_url(url: str) -> bool:
    """URLがPDFファイルを指しているか判定する。

    Args:
        url (str): 判定対象URL。

    Returns:
        bool: PDFであればTrue。
    """
    return urlparse(url).path.lower().endswith(".pdf")


def is_daily_summary_pdf_url(url: str) -> bool:
    """URLが「日次集計」のPDF（形式: YYMMDD-m.pdf）か判定する。

    Args:
        url (str): 判定対象URL。

    Returns:
        bool: 日次集計PDFであればTrue。
    """
    path = urlparse(url).path.lower()
    name = os.path.basename(path)
    return bool(re.fullmatch(r"\d{6}-m\.pdf", name))


def extract_links(html: str, base_url: str) -> list[str]:
    """HTMLから絶対パスのリンク一覧を抽出する。

    Args:
        html (str): 解析対象HTML。
        base_url (str): 相対パス解決のためのベースURL。

    Returns:
        list[str]: 絶対URLのリスト。
    """
    parser = LinkExtractor()
    parser.feed(html)
    result: list[str] = []
    for href in parser.links:
        abs_url = normalize_url(urljoin(base_url, href))
        result.append(abs_url)
    return result


def extract_backnumber_pages(html: str, base_url: str) -> list[str]:
    """HTML内のバックナンバーアーカイブ一覧ページURLを抽出する。

    Args:
        html (str): 解析対象HTML。
        base_url (str): 相対パス解決のためのベースURL。

    Returns:
        list[str]: アーカイブページのURLリスト。
    """
    parser = BacknumberOptionExtractor()
    parser.feed(html)

    pages: list[str] = []
    seen: set[str] = set()

    for value in parser.options:
        abs_url = normalize_url(urljoin(base_url, value))
        if not is_same_short_selling_dir(abs_url):
            continue
        if abs_url not in seen:
            seen.add(abs_url)
            pages.append(abs_url)

    return pages


def sanitize_filename(name: str) -> str:
    """ファイル名として使用できない文字を置換する。

    Args:
        name (str): 置換前ファイル名。

    Returns:
        str: サニタイズ済みファイル名。
    """
    return re.sub(r'[<>:"/\\\\|?*]+', "_", name)


def filename_from_url(url: str) -> str:
    """URLからサニタイズ済みのファイル名を抽出する。

    Args:
        url (str): 対象URL。

    Returns:
        str: サニタイズ済みファイル名。
    """
    path = urlparse(url).path
    name = os.path.basename(path) or "download.pdf"
    return sanitize_filename(name)


def ensure_dir(path: Path) -> None:
    """ディレクトリが存在しない場合に作成する（親階層含む）。

    Args:
        path (Path): 作成対象ディレクトリ。
    """
    path.mkdir(parents=True, exist_ok=True)


def infer_pdf_date_from_filename(url: str) -> tuple[int, int, int] | None:
    """ファイル名から日付（年、月、日）を推測する。

    Args:
        url (str): PDFのURL。

    Returns:
        tuple[int, int, int] | None: (year, month, day) のタプル。推測不可時はNone。
    """
    path = urlparse(url).path
    name = os.path.basename(path).lower()
    m = re.match(r"^(\d{2})(\d{2})(\d{2})(?:-[a-z0-9_]+)?\.pdf$", name)
    if not m:
        return None

    yy = int(m.group(1))
    mm = int(m.group(2))
    dd = int(m.group(3))
    # 80年以降は1900年代、それ以外は2000年代と仮定
    year = 2000 + yy if yy <= 79 else 1900 + yy

    if not (1 <= mm <= 12 and 1 <= dd <= 31):
        return None

    return (year, mm, dd)


def infer_archive_yearmonth_from_url(url: str) -> tuple[int, int] | None:
    """アーカイブページのURLから対象年月を推測する。

    Args:
        url (str): アーカイブページのURL。

    Returns:
        tuple[int, int] | None: (year, month) のタプル。推測不可時はNone。
    """
    path = urlparse(url).path.lower()

    if path.endswith("/index.html"):
        return None

    m = re.search(r"/00-archives-(\d{2})\.html$", path)
    if not m:
        return None

    archive_no = int(m.group(1))
    # 00-archives-01.html -> 2026-03 (現在の運用からの推測)
    # 00-archives-02.html -> 2026-02
    year = 2026
    month = 3 - (archive_no - 1)

    while month <= 0:
        year -= 1
        month += 12

    return (year, month)


def is_in_target_range(url: str) -> bool:
    """PDFの日付がダウンロード対象範囲内か判定する。

    Args:
        url (str): PDFのURL。

    Returns:
        bool: 対象範囲内であればTrue。
    """
    dt = infer_pdf_date_from_filename(url)
    if dt is None:
        return False

    year, month, _day = dt
    min_year, min_month = MIN_TARGET_YEARMONTH

    return (year, month) >= (min_year, min_month)


def is_target_archive_page(url: str) -> bool:
    """アーカイブページが対象範囲のデータを含む可能性があるか判定する。

    Args:
        url (str): アーカイブページのURL。

    Returns:
        bool: 対象範囲のデータを含む可能性があればTrue。
    """
    ym = infer_archive_yearmonth_from_url(url)
    if ym is None:
        return normalize_url(url) == normalize_url(DAILY_INDEX_URL)

    min_year, min_month = MIN_TARGET_YEARMONTH
    return ym >= (min_year, min_month)


def collect_target_pages() -> list[str]:
    """クロール対象となるHTMLページ一覧（インデックス＋バックナンバー）を収集する。

    Returns:
        list[str]: 収集されたURLのリスト。
    """
    print(f"[PAGE] {DAILY_INDEX_URL}")

    html = fetch_text(DAILY_INDEX_URL)
    pages = [normalize_url(DAILY_INDEX_URL)]

    for page_url in extract_backnumber_pages(html, DAILY_INDEX_URL):
        if is_target_archive_page(page_url):
            pages.append(page_url)

    return pages


def collect_pdf_urls_from_pages(page_urls: list[str]) -> list[str]:
    """ページ群から対象となるPDFのURL一覧を重複なく抽出する。

    Args:
        page_urls (list[str]): 解析対象のHTMLページURLリスト。

    Returns:
        list[str]: 重複を除去しソートされたPDF URLリスト。
    """
    pdf_urls: set[str] = set()

    for page_url in page_urls:
        if page_url != normalize_url(DAILY_INDEX_URL):
            print(f"[PAGE] {page_url}")

        try:
            html = fetch_text(page_url)
        except Exception as exc:
            print(f"  [WARN] failed to fetch page: {exc}", file=sys.stderr)
            continue

        for link in extract_links(html, page_url):
            if not is_same_short_selling_dir(link):
                continue
            if not is_daily_summary_pdf_url(link):
                continue
            if not is_in_target_range(link):
                continue
            pdf_urls.add(link)

        time.sleep(0.2)

    return sorted(pdf_urls)


def download_pdfs(
    pdf_urls: list[str], output_dir: Path, overwrite: bool = False
) -> None:
    """PDFを指定ディレクトリにダウンロードする。

    Args:
        pdf_urls (list[str]): ダウンロード対象のURLリスト。
        output_dir (Path): 保存先のルートディレクトリ。
        overwrite (bool): 既存ファイルを上書きするかどうか。
    """
    total = len(pdf_urls)
    print(f"[INFO] {total} PDFs found.")

    for i, url in enumerate(pdf_urls, start=1):
        filename = filename_from_url(url)
        year = "20" + filename[:2]
        dst = output_dir / "ShortSelling" / year / filename
        ensure_dir(dst.parent)

        if dst.exists() and not overwrite:
            print(f"[SKIP] ({i}/{total}) {filename}")
            continue

        print(f"[GET ] ({i}/{total}) {filename}")
        try:
            data = fetch_bytes(url)
            dst.write_bytes(data)
        except Exception as exc:
            print(f"  [WARN] failed to download {url}: {exc}", file=sys.stderr)

        time.sleep(0.2)


def main() -> int:
    """メインエントリポイント。引数解析と実行管理を行う。

    Returns:
        int: 終了ステータスコード。
    """
    parser = argparse.ArgumentParser(description="Fetch all JPX short-selling PDFs.")
    parser.add_argument(
        "--output-dir",
        default="C:\\transformer_futures_data\\pdf",
        help="保存先ディレクトリ",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="既存ファイルを上書き",
    )
    args = parser.parse_args()

    pages = collect_target_pages()
    print(f"[INFO] crawled pages: {len(pages)}")
    print(
        f"[INFO] target range: {MIN_TARGET_YEARMONTH[0]}-{MIN_TARGET_YEARMONTH[1]:02d} and newer"
    )

    pdf_urls = collect_pdf_urls_from_pages(pages)

    output_dir = Path(args.output_dir)
    download_pdfs(pdf_urls, output_dir=output_dir, overwrite=args.overwrite)

    print("[DONE]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
