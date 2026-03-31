import argparse
import copy
import os
import random
import time
from collections import OrderedDict, defaultdict
from os.path import join as pjoin

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.AE import AE_models
from models.ESMotion import ESMotion_models
from utils.datasets import Text2MotionDataset, collate_fn
from utils.evaluators import Evaluators
from utils.eval_utils import evaluation_esmotion
from utils.train_utils import def_value, print_current_loss, save, update_ema, update_lr_warm_up


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

    train_dataset = Text2MotionDataset(
        mean,
        std,
        pjoin(data_root, "train.txt"),
        args.dataset_name,
        motion_dir,
        text_dir,
        args.unit_length,
        args.max_motion_length,
        20,
        evaluation=False,
    )
    val_dataset = Text2MotionDataset(
        mean,
        std,
        pjoin(data_root, "val.txt"),
        args.dataset_name,
        motion_dir,
        text_dir,
        args.unit_length,
        args.max_motion_length,
        20,
        evaluation=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.num_workers,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.num_workers,
        shuffle=True,
    )

    eval_loader = None
    eval_wrapper = None
    if args.need_evaluation:
        eval_mean = np.load(f"utils/eval_mean_std/{args.dataset_name}/eval_mean.npy")
        eval_std = np.load(f"utils/eval_mean_std/{args.dataset_name}/eval_std.npy")
        eval_dataset = Text2MotionDataset(
            eval_mean,
            eval_std,
            pjoin(data_root, "val.txt"),
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
            batch_size=32,
            num_workers=args.num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            shuffle=True,
        )

    model_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name, "model")
    os.makedirs(model_dir, exist_ok=True)

    ae = AE_models[args.ae_model](input_width=dim_pose)
    ae_ckpt = torch.load(
        pjoin(args.checkpoints_dir, args.dataset_name, args.ae_name, "model", "latest.tar"),
        map_location="cpu",
    )
    ae.load_state_dict(ae_ckpt["ae"])

    model_key = resolve_model_key(args.model)
    esmotion = ESMotion_models[model_key](ae_dim=ae.output_emb_width, cond_mode="text")

    ema_esmotion = copy.deepcopy(esmotion)
    ema_esmotion.eval()
    for param in ema_esmotion.parameters():
        param.requires_grad_(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.need_evaluation:
        eval_wrapper = Evaluators(args.dataset_name, device=device)

    tb_logger = SummaryWriter(model_dir)
    ae.eval().to(device)
    esmotion.to(device)
    ema_esmotion.to(device)

    optimizer = optim.AdamW(esmotion.parameters(), betas=(0.9, 0.99), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_decay)

    epoch = 0
    it = 0
    if args.is_continue:
        ckpt_path = pjoin(model_dir, "latest.tar")
        checkpoint = torch.load(ckpt_path, map_location=device)
        esmotion_key = "esmotion" if "esmotion" in checkpoint else "ESMotion"
        opt_key = "opt_esmotion" if "opt_esmotion" in checkpoint else "opt_ESMotion"
        esmotion.load_state_dict(checkpoint[esmotion_key], strict=False)
        ema_esmotion.load_state_dict(checkpoint["ema_ESMotion"], strict=False)
        optimizer.load_state_dict(checkpoint[opt_key])
        scheduler.load_state_dict(checkpoint["scheduler"])
        epoch, it = checkpoint["ep"], checkpoint["total_it"]
        print(f"Resume from epoch={epoch}, iter={it}")

    start_time = time.time()
    total_iters = args.epoch * len(train_loader)
    print(f"Total Epochs: {args.epoch}, Total Iters: {total_iters}")

    logs = defaultdict(def_value, OrderedDict())
    best_fid, best_div, best_top1, best_top2, best_top3, best_matching, clip_score = 1000, 0, 0, 0, 0, 100, -1

    while epoch < args.epoch:
        ae.eval()
        esmotion.train()

        for i, batch_data in enumerate(train_loader):
            it += 1
            if it < args.warm_up_iter:
                update_lr_warm_up(it, args.warm_up_iter, optimizer, args.lr)

            conds, motion, m_lens = batch_data
            motion = motion.detach().float().to(device)
            m_lens = m_lens.detach().long().to(device) // 4

            latent = ae.encode(motion)
            conds = conds.to(device).float() if torch.is_tensor(conds) else conds
            loss = esmotion.forward_loss(latent, conds, m_lens)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            update_ema(esmotion, ema_esmotion, 0.9999)

            logs["loss"] += loss.item()
            logs["lr"] += optimizer.param_groups[0]["lr"]

            if it % args.log_every == 0:
                mean_loss = OrderedDict()
                for tag, value in logs.items():
                    tb_logger.add_scalar(f"Train/{tag}", value / args.log_every, it)
                    mean_loss[tag] = value / args.log_every
                logs = defaultdict(def_value, OrderedDict())
                print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

        save(pjoin(model_dir, "latest.tar"), epoch, esmotion, optimizer, scheduler, it, "esmotion", ema_ESMotion=ema_esmotion)
        epoch += 1

        esmotion.eval()
        val_loss = []
        with torch.no_grad():
            for conds, motion, m_lens in val_loader:
                motion = motion.detach().float().to(device)
                m_lens = m_lens.detach().long().to(device) // 4
                latent = ae.encode(motion)
                conds = conds.to(device).float() if torch.is_tensor(conds) else conds
                val_loss.append(esmotion.forward_loss(latent, conds, m_lens).item())

        val_loss_mean = float(np.mean(val_loss))
        tb_logger.add_scalar("Val/loss", val_loss_mean, epoch)
        print(f"Validation loss: {val_loss_mean:.4f}")

        if args.need_evaluation and eval_loader is not None:
            (
                best_fid,
                best_div,
                best_top1,
                best_top2,
                best_top3,
                best_matching,
                _,
                clip_score,
                _,
                save_now,
            ) = evaluation_esmotion(
                model_dir,
                eval_loader,
                ema_esmotion,
                ae,
                tb_logger,
                epoch - 1,
                best_fid=best_fid,
                clip_score_old=clip_score,
                best_div=best_div,
                best_top1=best_top1,
                best_top2=best_top2,
                best_top3=best_top3,
                best_matching=best_matching,
                eval_wrapper=eval_wrapper,
                device=device,
                train_mean=mean,
                train_std=std,
            )
            if save_now:
                save(
                    pjoin(model_dir, "net_best_fid.tar"),
                    epoch - 1,
                    esmotion,
                    optimizer,
                    scheduler,
                    it,
                    "esmotion",
                    ema_ESMotion=ema_esmotion,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="ESMotion")
    parser.add_argument("--ae_name", type=str, default="AE")
    parser.add_argument("--ae_model", type=str, default="AE_Model")
    parser.add_argument("--model", type=str, default="ESMotion-Score-XL")
    parser.add_argument("--dataset_name", type=str, default="t2m", choices=["t2m"])
    parser.add_argument("--dataset_dir", type=str, default="./datasets")
    parser.add_argument("--max_motion_length", type=int, default=196)
    parser.add_argument("--unit_length", type=int, default=4)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epoch", default=500, type=int)
    parser.add_argument("--warm_up_iter", default=2000, type=int)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--milestones", default=[50_000], nargs="+", type=int)
    parser.add_argument("--lr_decay", default=0.1, type=float)
    parser.add_argument("--need_evaluation", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--is_continue", action="store_true")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_every", default=50, type=int)
    parser.add_argument("--hard_pseudo_reorder", action="store_true")
    main(parser.parse_args())
