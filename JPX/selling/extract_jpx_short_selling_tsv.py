# extract_jpx_short_selling_tsv.py
"""
File: extract_jpx_short_selling_tsv.py

ソースコードの役割:
JPX（日本取引所グループ）が公開する「空売り集計」の日次PDFファイルから、市場全体の集計表を抽出してTSV形式に変換するスクリプトです。
fetch_jpx_short_selling_pdf.py によって整理された階層ディレクトリ（ShortSelling/年/...）を再帰的にスキャンし、
「年月日」「売買代金(a)〜(c)」「比率」「合計(d)」等のデータを構造化して出力します。
抽出した日付は SQLite 等での利用を想定し 'YYYY-MM-DD' 形式に変換されます。
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import pdfplumber
from pydantic import BaseModel, Field, ValidationError


class ExtractedRow(BaseModel):
    """抽出された1日分の空売り集計データモデル。

    Pydanticを使用して型安全とバリデーションを担保します。
    """

    年月日: str = Field(
        ..., description="SQLite向けにフォーマットされた日付 (YYYY-MM-DD)"
    )
    売買代金_a: str = Field(..., description="実注文の売買代金(a)")
    比率_a_d: str = Field(..., description="実注文の比率 (a)/(d)")
    売買代金_b: str = Field(..., description="空売り(価格規制あり)の売買代金(b)")
    比率_b_d: str = Field(..., description="空売り(価格規制あり)の比率 (b)/(d)")
    売買代金_c: str = Field(..., description="空売り(価格規制なし)の売買代金(c)")
    比率_c_d: str = Field(..., description="空売り(価格規制なし)の比率 (c)/(d)")
    合計_d: str = Field(..., description="売買代金合計(d)")
    source_file: str = Field(..., description="抽出元のPDFファイル名")


def to_sqlite_date(s: str) -> str:
    """日付文字列を ISO 8601 形式に変換する。

    Args:
        s (str): 変換前の日付文字列（例: 2025年3月10日）。

    Returns:
        str: 'YYYY-MM-DD' 形式の文字列。

    Raises:
        ValueError: 日付形式が想定と異なる場合。
    """
    m = re.fullmatch(r"(\d{4})年(\d{1,2})月(\d{1,2})日", s)
    if not m:
        raise ValueError(f"SQLite向け日付に変換できません: {s}")

    year = int(m.group(1))
    month = int(m.group(2))
    day = int(m.group(3))
    return f"{year:04d}-{month:02d}-{day:02d}"


def parse_data_line(line: str, source_file: str) -> ExtractedRow | None:
    """PDFから抽出した1行のテキストから、正規表現を用いて集計データを抽出する。

    列の結合や構造崩れに対応するため、特定のキーワードと正規表現パターンを用いて
    テキストストリームから直接データ要素を拾い上げます。

    Args:
        line (str): 解析対象のテキスト行。
        source_file (str): エラー追跡用のソースファイル名。

    Returns:
        ExtractedRow | None: 抽出に成功した場合はデータモデル、対象行でない場合はNone。
    """
    date_match = re.search(r"(\d{4}年\d{1,2}月\d{1,2}日)", line)
    if not date_match:
        return None

    date_str = date_match.group(1)

    # 年月日をテキストから除去して、数値・比率のみを純粋に抽出できるようにする
    line_no_date = line.replace(date_str, " ")

    # 比率 (e.g., 63.5%) を全て抽出
    percents = re.findall(r"\d+\.\d+%", line_no_date)

    # 金額の抽出精度を上げるため、パーセンテージも除去
    line_no_percent = re.sub(r"\d+\.\d+%", " ", line_no_date)

    # 金額 (カンマ区切りの数値) を全て抽出
    amounts = re.findall(r"\b\d+(?:,\d{3})*\b", line_no_percent)

    if len(amounts) >= 4 and len(percents) >= 3:
        try:
            return ExtractedRow(
                年月日=to_sqlite_date(date_str),
                売買代金_a=amounts[0],
                比率_a_d=percents[0],
                売買代金_b=amounts[1],
                比率_b_d=percents[1],
                売買代金_c=amounts[2],
                比率_c_d=percents[2],
                合計_d=amounts[3],
                source_file=source_file,
            )
        except ValidationError as e:
            print(
                f"[WARN] Pydantic validation error in {source_file}: {e}",
                file=sys.stderr,
            )
            return None

    return None


def extract_from_pdf(pdf_path: Path) -> ExtractedRow:
    """PDFファイルを開き、集計データを抽出する。

    Args:
        pdf_path (Path): 解析対象PDFのパス。

    Returns:
        ExtractedRow: 抽出されたデータ。

    Raises:
        ValueError: ページが存在しない、または対象データが見つからない場合。
    """
    with pdfplumber.open(str(pdf_path)) as pdf:
        if not pdf.pages:
            raise ValueError(f"ページがありません: {pdf_path.name}")

        page = pdf.pages[0]

        # 表の構造崩れを回避するため、表内のテキストをスペース区切りで結合して解析
        tables = page.extract_tables()
        for table in tables:
            for row in table:
                row_text = " ".join([str(c) if c is not None else "" for c in row])
                # 改行による分断を防ぐ
                row_text = row_text.replace("\n", " ").replace("\r", " ")
                row_text = re.sub(r"[ \t]+", " ", row_text)

                res = parse_data_line(row_text, pdf_path.name)
                if res:
                    return res

        # テーブルとして認識されなかった場合のフォールバック（直接テキスト抽出）
        text = page.extract_text() or ""
        text = text.replace("\n", " ").replace("\r", " ")
        text = re.sub(r"[ \t]+", " ", text)

        res = parse_data_line(text, pdf_path.name)
        if res:
            return res

        raise ValueError(f"対象のデータ行が見つかりません: {pdf_path.name}")


def find_pdfs(input_dir: Path) -> list[Path]:
    """ディレクトリ内を再帰的にスキャンし、対象となるPDFファイル一覧を取得する。

    Args:
        input_dir (Path): スキャン対象のルートディレクトリ。

    Returns:
        list[Path]: 見つかったPDFパスのリスト。
    """
    pdfs: list[Path] = []
    for p in input_dir.rglob("*.pdf"):
        if not p.is_file():
            continue
        if re.fullmatch(r"\d{6}-m\.pdf", p.name.lower()):
            pdfs.append(p)
    return sorted(pdfs)


def write_tsv(rows: list[ExtractedRow], output_tsv: Path) -> None:
    """抽出されたデータをTSVファイルに書き出す。

    Args:
        rows (list[ExtractedRow]): 抽出済みデータモデルのリスト。
        output_tsv (Path): 出力先TSVファイルのパス。
    """
    output_tsv.parent.mkdir(parents=True, exist_ok=True)

    with output_tsv.open("w", encoding="utf-8-sig", newline="") as f:
        # Pydantic モデルからフィールド名を動的に取得
        field_names = list(ExtractedRow.model_fields.keys())
        writer = csv.DictWriter(f, fieldnames=field_names, delimiter="\t")
        writer.writeheader()

        for row in rows:
            writer.writerow(row.model_dump())


def main() -> int:
    """メインエントリポイント。"""
    parser = argparse.ArgumentParser(
        description="JPX空売りPDF（日次）を再帰的にスキャンしてTSV化する"
    )
    parser.add_argument(
        "--input-dir",
        default="C:\\transformer_futures_data\\pdf\\ShortSelling",
        help="PDF格納ルートディレクトリ (fetch_jpx_short_selling_pdf.py の保存先)",
    )
    parser.add_argument(
        "--output-tsv",
        default="C:\\transformer_futures_data\\tsv\\ShortSelling\\jpx_short_selling_daily.tsv",
        help="出力TSVパス",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_tsv = Path(args.output_tsv)

    if not input_dir.exists():
        print(f"[ERROR] Directory not found: {input_dir}", file=sys.stderr)
        return 1

    pdf_paths = find_pdfs(input_dir)
    print(f"[INFO] Found {len(pdf_paths)} PDF files.")

    results: list[ExtractedRow] = []
    for path in pdf_paths:
        try:
            row = extract_from_pdf(path)
            results.append(row)
            print(f"[OK  ] {path.name} -> {row.年月日}")
        except Exception as exc:
            print(f"[FAIL] {path.name}: {exc}", file=sys.stderr)

    if results:
        write_tsv(results, output_tsv)
        print(f"[DONE] Saved {len(results)} rows to {output_tsv}")
    else:
        print("[WARN] No data extracted.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
