#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import glob
import re

def summarize_tsvs(directory_path):
    # 処理対象のTSVファイルをパターンで取得
    search_pattern = os.path.join(directory_path, "*.tsv")
    tsv_files = glob.glob(search_pattern)
    
    if not tsv_files:
        print(f"Error: No TSV files found in '{directory_path}'.", file=sys.stderr)
        return

    # 日付ごとにPnLリストを集計する辞書
    daily_pnls = {}
    # ファイル名から日付（YYYY-MM-DD）を抽出する正規表現
    date_pattern = re.compile(r"test_(\d{4}-\d{2}-\d{2})\.tsv")
    
    for filepath in tsv_files:
        match = date_pattern.search(filepath)
        if match:
            trade_date = match.group(1)
        else:
            continue  # パターンに合わないファイルはスキップ

        if trade_date not in daily_pnls:
            daily_pnls[trade_date] = []

        # TSVファイルを読み込んで pnl を抽出
        with open(filepath, 'r', encoding='utf-8') as f:
            header = f.readline()
            if not header: continue
            
            # ヘッダー列から pnl のインデックスを取得
            cols = header.strip().split('\t')
            if 'pnl' not in cols:
                continue
            pnl_idx = cols.index('pnl')

            for line in f:
                if not line.strip(): continue
                parts = line.strip().split('\t')
                if len(parts) > pnl_idx:
                    try:
                        # PnLの値を辞書に追加
                        daily_pnls[trade_date].append(float(parts[pnl_idx]))
                    except ValueError:
                        pass

    if not daily_pnls:
        print("Error: No valid trade data found.", file=sys.stderr)
        return

    # ヘッダー出力
    print("trade_date\ttrades\twin%\tpf\tpnl\tcum_pnl")
    cum_pnl = 0.0

    # 日付順にソートして各指標を計算
    for trade_date in sorted(daily_pnls.keys()):
        pnls = daily_pnls[trade_date]
        n_trades = len(pnls)
        
        if n_trades == 0:
            continue
            
        # 勝率 (プラス決済の割合)
        wins = sum(1 for x in pnls if x > 0)
        win_rate = (wins / n_trades) * 100
        
        # プロフィットファクター (総利益 / 総損失)
        gross_profit = sum(x for x in pnls if x > 0)
        gross_loss = abs(sum(x for x in pnls if x < 0))
        
        if gross_loss > 0:
            pf = gross_profit / gross_loss
        else:
            pf = float('inf') if gross_profit > 0 else 0.0
            
        # 日次PnLと累積PnL
        day_pnl = sum(pnls)
        cum_pnl += day_pnl
        
        # TSV形式で出力
        print(f"{trade_date}\t{n_trades}\t{win_rate:.2f}\t{pf:.3f}\t{day_pnl:.0f}\t{cum_pnl:.0f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python summarize_tsvs.py <directory_path>", file=sys.stderr)
        sys.exit(1)
    
    # 引数で渡されたディレクトリパスを処理
    summarize_tsvs(sys.argv[1])
