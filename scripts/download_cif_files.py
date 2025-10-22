#!/usr/bin/env python3
"""
Download mmCIF files from PDB for HARIBOSS complexes
"""

import argparse
import pandas as pd
from pathlib import Path
import urllib.request
import time
from tqdm import tqdm

import subprocess
import os

def download_cif(pdb_id: str, output_dir: Path, proxy_url: str = None) -> bool:
    pdb_id = pdb_id.lower()
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    output_file = output_dir / f"{pdb_id}.cif"

    if output_file.exists():
        return True

    try:
        # 构建 curl 命令
        cmd = ["curl", "-o", str(output_file), "--max-time", "1000", url]
        
        # 如果有代理
        if proxy_url:
            cmd.extend(["-x", proxy_url])
        
        # 执行 curl
        result = subprocess.run(cmd, capture_output=True, timeout=99999)
        
        if result.returncode == 0:
            return True
        else:
            print(f"  ✗ 下载失败 {pdb_id}: {result.stderr.decode()}")
            return False
    except Exception as e:
        print(f"  ✗ 下载失败 {pdb_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download mmCIF files from PDB")
    parser.add_argument("--hariboss_csv", type=str, required=True,
                       help="Path to HARIBOSS Complexes.csv")
    parser.add_argument("--output_dir", type=str, default="data/raw/mmCIF",
                       help="Directory to save CIF files")
    parser.add_argument("--max_downloads", type=int, default=None,
                       help="Maximum number of files to download (for testing)")
    parser.add_argument("--delay", type=float, default=0.5,
                       help="Delay between downloads in seconds (to be nice to PDB servers)")

    args = parser.parse_args()

    # Read HARIBOSS CSV
    print(f"Reading HARIBOSS CSV from {args.hariboss_csv}...")
    df = pd.read_csv(args.hariboss_csv)

    # Get unique PDB IDs
    pdb_ids = df['id'].unique()
    print(f"Found {len(pdb_ids)} unique PDB IDs")

    if args.max_downloads:
        pdb_ids = pdb_ids[:args.max_downloads]
        print(f"Limiting to first {args.max_downloads} complexes")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {output_dir}\n")

    # Download files
    successful = 0
    failed = []

    for pdb_id in tqdm(pdb_ids, desc="Downloading CIF files"):
        if download_cif(pdb_id, output_dir):
            successful += 1
        else:
            failed.append(pdb_id)

        # Be nice to PDB servers
        time.sleep(args.delay)

    # Summary
    print(f"\n{'='*70}")
    print(f"Download Summary")
    print(f"{'='*70}")
    print(f"Total PDB IDs: {len(pdb_ids)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(failed)}")

    if failed:
        print(f"\nFailed downloads:")
        for pdb_id in failed:
            print(f"  - {pdb_id}")


if __name__ == "__main__":
    main()
