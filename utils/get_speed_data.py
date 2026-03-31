import argparse
import os
import os.path as osp
import random

import numpy as np
from tqdm import tqdm


def filter_and_select_samples(data_path, output_path, min_length, max_length, num_samples):
    print(f"加载数据文件: {data_path}")
    data_dict = np.load(data_path, allow_pickle=True).item()

    print(f"过滤运动长度在 {min_length}-{max_length} 帧之间的样本...")
    filtered_keys = []
    for key, value in tqdm(data_dict.items(), desc="过滤样本"):
        motion_length = value["motion"].shape[0]
        if min_length <= motion_length <= max_length:
            filtered_keys.append(key)

    print(f"找到 {len(filtered_keys)} 个符合条件的样本")
    if len(filtered_keys) < num_samples:
        print(f"警告: 只有 {len(filtered_keys)} 个样本，少于请求的 {num_samples} 个")
        num_samples = len(filtered_keys)

    selected_keys = random.sample(filtered_keys, num_samples)
    np.save(output_path, np.array(selected_keys))
    print(f"已保存选择的样本到: {output_path}")
    return selected_keys


def main():
    parser = argparse.ArgumentParser(description="从 data_test.npy 中随机选择测速样本")
    parser.add_argument("--data_dir", type=str, default="./datasets/HumanML3D", help="数据集目录路径")
    parser.add_argument("--output_dir", type=str, default="./datasets/HumanML3D", help="输出目录路径")
    parser.add_argument("--min_length", type=int, default=40, help="最小运动长度（帧数）")
    parser.add_argument("--max_length", type=int, default=200, help="最大运动长度（帧数）")
    parser.add_argument("--num_samples", type=int, default=100, help="要选择的样本数量")
    parser.add_argument("--output_name", type=str, default="random_selected_data.npy", help="输出文件名")
    args = parser.parse_args()

    data_path = osp.join(args.data_dir, "data_test.npy")
    output_path = osp.join(args.output_dir, args.output_name)
    os.makedirs(args.output_dir, exist_ok=True)

    selected_samples = filter_and_select_samples(
        data_path=data_path,
        output_path=output_path,
        min_length=args.min_length,
        max_length=args.max_length,
        num_samples=args.num_samples,
    )

    print("\n前10个选择的样本:")
    for i, sample in enumerate(selected_samples[:10]):
        print(f"{i + 1}. {sample}")


if __name__ == "__main__":
    main()
