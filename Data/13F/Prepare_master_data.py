# prepare_master_data.py

import pandas as pd
import os

# 缓存文件名
info_cache_path = "all_info_combined_newest.pkl"
cover_cache_path = "all_cover_combined_newest.pkl"

# 如果已经缓存，跳过
if os.path.exists(info_cache_path) and os.path.exists(cover_cache_path):
    print("✅ 缓存已存在，跳过合并。")
else:
    print("🔄 正在读取季度文件夹并合并 INFOTABLE & COVERPAGE...")

    # 文件夹位置
    base_dir = "."
    form13f_dirs = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and ("form13f" in d.lower() or "2024dec2025feb" in d.lower())
    ]

    info_all = []
    cover_all = []

    for folder in form13f_dirs:
        info_path = os.path.join(base_dir, folder, "INFOTABLE.tsv")
        cover_path = os.path.join(base_dir, folder, "COVERPAGE.tsv")

        if os.path.exists(info_path) and os.path.exists(cover_path):
            try:
                info_df = pd.read_csv(info_path, sep="\t")
                cover_df = pd.read_csv(cover_path, sep="\t")

                info_all.append(info_df)
                cover_all.append(cover_df)
                print(f"✅ Loaded: {folder}")
            except Exception as e:
                print(f"⚠️ Failed to load {folder}: {e}")

    # 合并
    info = pd.concat(info_all, ignore_index=True)
    cover = pd.concat(cover_all, ignore_index=True)

    # 保存缓存
    info.to_pickle(info_cache_path)
    cover.to_pickle(cover_cache_path)
    print("💾 已保存为 all_info_combined.pkl 和 all_cover_combined.pkl")