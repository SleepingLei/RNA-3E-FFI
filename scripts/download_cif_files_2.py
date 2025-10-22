#!/usr/bin/env python3
"""
Download mmCIF files from PDB for failed HARIBOSS complexes
"""

import argparse
import pandas as pd
from pathlib import Path
import subprocess
import time
from tqdm import tqdm

def read_failed_ids(failed_file: Path) -> list:
    """从failed.txt读取PDB ID列表"""
    pdb_ids = []
    with open(failed_file, 'r') as f:
        for line in f:
            line = line.strip()
            # 跳过空行和注释
            if line and not line.startswith('#'):
                pdb_ids.append(line)
    return pdb_ids

def download_cif(pdb_id: str, output_dir: Path, proxy_url: str = None, retry: int = 3) -> bool:
    """下载单个mmCIF文件"""
    pdb_id = pdb_id.lower()
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    output_file = output_dir / f"{pdb_id}.cif"


    # 重试机制
    for attempt in range(retry):
        try:
            # 构建 curl 命令
            cmd = ["curl", "-o", str(output_file), "--max-time", "300000", "-L", url]
            
            # 如果有代理
            if proxy_url:
                cmd.extend(["-x", proxy_url])
            
            # 执行 curl
            result = subprocess.run(cmd, capture_output=True, timeout=350)
            
            if result.returncode == 0:
                # 检查文件大小，确保下载成功
                if output_file.stat().st_size > 0:
                    return True
                else:
                    output_file.unlink()  # 删除空文件
            else:
                stderr = result.stderr.decode()
                if attempt < retry - 1:
                    print(f"  ⟳ 重试 {pdb_id} (尝试 {attempt + 1}/{retry}): {stderr[:100]}")
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    print(f"  ✗ 下载失败 {pdb_id}: {stderr[:200]}")
                    
        except subprocess.TimeoutExpired:
            if attempt < retry - 1:
                print(f"  ⟳ 超时重试 {pdb_id} (尝试 {attempt + 1}/{retry})")
                time.sleep(2 ** attempt)
            else:
                print(f"  ✗ 下载超时 {pdb_id}")
        except Exception as e:
            print(f"  ✗ 下载异常 {pdb_id}: {e}")
            return False
    
    return False

def main():
    parser = argparse.ArgumentParser(description="下载失败的PDB mmCIF文件")
    parser.add_argument(
        "--failed-file",
        type=Path,
        default=Path("failed.txt"),
        help="包含失败PDB ID的文件 (默认: failed.txt)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/mmCIF"),
        help="输出目录 (默认: data/raw/mmCIF)"
    )
    parser.add_argument(
        "--proxy",
        type=str,
        default=None,
        help="代理URL (可选)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="请求间隔(秒) (默认: 0.5)"
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取失败的PDB ID
    if not args.failed_file.exists():
        print(f"✗ 找不到文件: {args.failed_file}")
        return
    
    pdb_ids = read_failed_ids(args.failed_file)
    print(f"📋 读取到 {len(pdb_ids)} 个PDB ID")
    
    # 下载文件
    successful = 0
    failed = 0
    
    for pdb_id in tqdm(pdb_ids, desc="下载进度"):
        if download_cif(pdb_id, args.output_dir, args.proxy):
            successful += 1
        else:
            failed += 1
        
        # 请求间隔，避免过度请求
        time.sleep(args.delay)
    
    # 统计结果
    print(f"\n✓ 成功: {successful}")
    print(f"✗ 失败: {failed}")
    print(f"📁 输出目录: {args.output_dir}")

if __name__ == "__main__":
    main()
