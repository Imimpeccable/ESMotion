import argparse
import os
import random
from os.path import join as pjoin

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.AE import AE_models
from models.ESMotion import ESMotion_models
from utils.datasets import Text2MotionDataset, collate_fn
from utils.evaluators import Evaluators
from utils.eval_utils import evaluation_esmotion


MODEL_KEY_ALIASES = {
    "ESMotion-SiT-XL": "ESMotion-Score-XL",
    "ESMotion": "ESMotion-Score-XL",
}


def resolve_model_key(model_key: str) -> str:
    if model_key in ESMotion_models:
        return model_key
    alias_key = MODEL_KEY_ALIASES.get(model_key)
    if alias_key in ESMotion_models:
        print(f"[Info] model key '{model_key}' -> '{alias_key}'")
        return alias_key
    available = ", ".join(sorted(ESMotion_models.keys()))
    raise KeyError(f"Unsupported model '{model_key}'. Available: {available}")


def main(args):
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    data_root = f"{args.dataset_dir}/HumanML3D/"
    dim_pose = 67
    motion_dir = pjoin(data_root, "new_joint_vecs")
    text_dir = pjoin(data_root, "texts")

    mean = np.load(pjoin(data_root, "Mean_mar.npy"))
    std = np.load(pjoin(data_root, "Std_mar.npy"))
    eval_mean = np.load(f"utils/eval_mean_std/{args.dataset_name}/eval_mean.npy")
    eval_std = np.load(f"utils/eval_mean_std/{args.dataset_name}/eval_std.npy")

    eval_dataset = Text2MotionDataset(
        eval_mean,
        eval_std,
        pjoin(data_root, "test.txt"),
        args.dataset_name,
        motion_dir,
        text_dir,
        4,
        196,
        20,
        evaluation=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=collate_fn,
        shuffle=True,
    )

    model_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name, "model")

    ae = AE_models[args.ae_model](input_width=dim_pose)
    ae_ckpt = torch.load(
        pjoin(args.checkpoints_dir, args.dataset_name, args.ae_name, "model", "latest.tar"),
        map_location="cpu",
    )
    ae.load_state_dict(ae_ckpt["ae"])

    model_key = resolve_model_key(args.model)
    ema_esmotion = ESMotion_models[model_key](ae_dim=ae.output_emb_width, cond_mode="text")
    checkpoint = torch.load(pjoin(model_dir, "latest.tar"), map_location="cpu")
    ema_esmotion.load_state_dict(checkpoint["ema_ESMotion"], strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_wrapper = Evaluators(args.dataset_name, device=device)

    out_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name, "eval")
    os.makedirs(out_dir, exist_ok=True)

    ae.eval().to(device)
    ema_esmotion.eval().to(device)

    fid, div, top1, top2, top3, matching, mm, clip_scores = [], [], [], [], [], [], [], []
    repeat_time = 20

    with open(pjoin(out_dir, "eval.log"), "w", encoding="utf-8") as f:
        for i in range(repeat_time):
            print(f"Evaluation iteration {i + 1}/{repeat_time}")
            print(f"Evaluation iteration {i + 1}/{repeat_time}", file=f, flush=True)

            with torch.no_grad():
                (
                    best_fid,
                    best_div,
                    best_top1,
                    best_top2,
                    best_top3,
                    best_matching,
                    best_mm,
                    clip_score,
                    _,
                    _,
                ) = evaluation_esmotion(
                    model_dir,
                    eval_loader,
                    ema_esmotion,
                    ae,
                    None,
                    i,
                    best_fid=1000,
                    clip_score_old=-1,
                    best_div=0,
                    best_top1=0,
                    best_top2=0,
                    best_top3=0,
                    best_matching=100,
                    eval_wrapper=eval_wrapper,
                    device=device,
                    train_mean=mean,
                    train_std=std,
                    time_steps=args.time_steps,
                    cond_scale=args.cfg,
                    temperature=args.temperature,
                    cal_mm=args.cal_mm,
                    draw=False,
                    hard_pseudo_reorder=args.hard_pseudo_reorder,
                )

            msg_iter = (
                f"Iteration {i + 1}:\n"
                f"\tFID: {best_fid:.3f}\n"
                f"\tDiversity: {best_div:.3f}\n"
                f"\tTOP1: {best_top1:.3f}, TOP2: {best_top2:.3f}, TOP3: {best_top3:.3f}\n"
                f"\tMatching: {best_matching:.3f}\n"
                f"\tMultimodality: {best_mm:.3f}\n"
                f"\tCLIP-Score: {clip_score:.3f}\n"
            )
            print(msg_iter)
            print(msg_iter, file=f, flush=True)

            fid.append(best_fid)
            div.append(best_div)
            top1.append(best_top1)
            top2.append(best_top2)
            top3.append(best_top3)
            matching.append(best_matching)
            mm.append(best_mm)
            clip_scores.append(clip_score)

        msg_final = (
            f"FID: {np.mean(fid):.3f}, conf. {np.std(fid) * 1.96 / np.sqrt(repeat_time):.3f}\n"
            f"Diversity: {np.mean(div):.3f}, conf. {np.std(div) * 1.96 / np.sqrt(repeat_time):.3f}\n"
            f"TOP1: {np.mean(top1):.3f}, conf. {np.std(top1) * 1.96 / np.sqrt(repeat_time):.3f}, "
            f"TOP2: {np.mean(top2):.3f}, conf. {np.std(top2) * 1.96 / np.sqrt(repeat_time):.3f}, "
            f"TOP3: {np.mean(top3):.3f}, conf. {np.std(top3) * 1.96 / np.sqrt(repeat_time):.3f}\n"
            f"Matching: {np.mean(matching):.3f}, conf. {np.std(matching) * 1.96 / np.sqrt(repeat_time):.3f}\n"
            f"Multimodality: {np.mean(mm):.3f}, conf. {np.std(mm) * 1.96 / np.sqrt(repeat_time):.3f}\n"
            f"CLIP-Score: {np.mean(clip_scores):.3f}, conf. {np.std(clip_scores) * 1.96 / np.sqrt(repeat_time):.3f}\n"
        )
        print(msg_final)
        print(msg_final, file=f, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="ESMotion")
    parser.add_argument("--ae_name", type=str, default="AE")
    parser.add_argument("--ae_model", type=str, default="AE_Model")
    parser.add_argument("--model", type=str, default="ESMotion-Score-XL")
    parser.add_argument("--dataset_name", type=str, default="t2m", choices=["t2m"])
    parser.add_argument("--dataset_dir", type=str, default="./datasets")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints")
    parser.add_argument("--time_steps", default=18, type=int)
    parser.add_argument("--cfg", default=4.5, type=float)
    parser.add_argument("--temperature", default=1, type=float)
    parser.add_argument("--cal_mm", action="store_false")
    parser.add_argument("--hard_pseudo_reorder", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    main(parser.parse_args())
