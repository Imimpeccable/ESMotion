import os
from os.path import join as pjoin
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
import random
from models.AE import AE_models
from models.ESMotion import ESMotion_models
from models.LengthEstimator import LengthEstimator
from utils.motion_process import recover_from_ric, plot_3d_motion, t2m_kinematic_chain
import argparse

MODEL_KEY_ALIASES = {
    "ESMotion-SiT-XL": "ESMotion-Score-XL",
    "ESMotion": "ESMotion-Score-XL",
}


def resolve_model_key(model_key):
    if model_key in ESMotion_models:
        return model_key
    alias_key = MODEL_KEY_ALIASES.get(model_key)
    if alias_key in ESMotion_models:
        print(f"[Info] model key '{model_key}' -> '{alias_key}'")
        return alias_key
    available = ", ".join(sorted(ESMotion_models.keys()))
    raise KeyError(f"Unsupported model '{model_key}'. Available: {available}")


def parse_prompt_and_length(raw_line):
    infos = raw_line.strip().split("#")
    caption = infos[0].strip()
    if caption == "":
        raise ValueError("Empty caption is not allowed.")
    if len(infos) <= 1:
        return caption, None

    raw_len = infos[-1].strip()
    if raw_len == "" or raw_len.upper() == "NA":
        return caption, None
    return caption, int(raw_len)


def main(args):
    #################################################################################
    #                                      Seed                                     #
    #################################################################################
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # setting this to true significantly increase training and sampling speed
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    #################################################################################
    #                                       Data                                    #
    #################################################################################
    dim_pose = 67
    nb_joints = 22
    data_root = f'{args.dataset_dir}/HumanML3D/'
    mean = np.load(pjoin(data_root, 'Mean_mar.npy'))
    std = np.load(pjoin(data_root, 'Std_mar.npy'))
    #################################################################################
    #                                      Models                                   #
    #################################################################################
    model_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'model')
    result_dir = pjoin('./generation1', args.name)
    os.makedirs(result_dir, exist_ok=True)

    ae = AE_models[args.ae_model](input_width=dim_pose)
    ckpt = torch.load(
        pjoin(args.checkpoints_dir, args.dataset_name, args.ae_name, 'model', 'latest.tar'),
        map_location='cpu',
    )
    model_key = 'ae'
    ae.load_state_dict(ckpt[model_key])

    model_key = resolve_model_key(args.model)
    ema_ESMotion = ESMotion_models[model_key](ae_dim=ae.output_emb_width, cond_mode='text')
    model_dir = pjoin(model_dir, 'latest.tar')
    checkpoint = torch.load(model_dir, map_location='cpu')
    ema_state_key = 'ema_ESMotion' if 'ema_ESMotion' in checkpoint else 'ema_esmotion'
    missing_keys2, unexpected_keys2 = ema_ESMotion.load_state_dict(checkpoint[ema_state_key], strict=False)
    assert len(unexpected_keys2) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys2])

    length_estimator = LengthEstimator(512, 50)
    ckpt = torch.load(pjoin(args.checkpoints_dir, args.dataset_name, 'length_estimator', 'model', 'finest.tar'),
                      map_location='cpu')
    length_estimator.load_state_dict(ckpt['estimator'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #################################################################################
    #                                     Sampling                                  #
    #################################################################################
    prompt_list = []
    length_list = []
    if args.text_prompt != "":
        caption, motion_len = parse_prompt_and_length(args.text_prompt)
        prompt_list.append(caption)
        length_list.append(motion_len)
    elif args.text_path != "":
        with open(args.text_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line == "":
                    continue
                caption, motion_len = parse_prompt_and_length(line)
                prompt_list.append(caption)
                length_list.append(motion_len)
    else:
        raise "A text prompt, or a file a text prompts are required!!!"

    ae.to(device)
    ema_ESMotion.to(device)
    length_estimator.to(device)

    ae.eval()
    ema_ESMotion.eval()
    length_estimator.eval()

    needs_estimation = any(x is None for x in length_list)
    if needs_estimation:
        print("Using length estimator for prompts without explicit motion length.")
        text_embedding = ema_ESMotion.encode_text(prompt_list)
        pred_dis = length_estimator(text_embedding)
        probs = F.softmax(pred_dis, dim=-1)
        token_lens = Categorical(probs).sample()
        for i, motion_len in enumerate(length_list):
            if motion_len is not None:
                token_lens[i] = motion_len // 4
    else:
        token_lens = torch.LongTensor(length_list).to(device).long() // 4
    token_lens = token_lens.clamp(min=1)
    m_length = (token_lens * 4).detach().cpu().tolist()
    captions = prompt_list

    kinematic_chain = t2m_kinematic_chain

    for r in range(args.repeat_times):
        print("-->Repeat %d" % r)
        with torch.no_grad():
            pred_latents = ema_ESMotion.generate(captions, token_lens, args.time_steps, args.cfg,
                                              temperature=args.temperature, hard_pseudo_reorder=args.hard_pseudo_reorder)
            pred_motions = ae.decode(pred_latents)
            pred_motions = pred_motions.detach().cpu().numpy()
            data = pred_motions * std[:pred_motions.shape[2]] + mean[:pred_motions.shape[2]]

        for k, (caption, joint_data) in enumerate(zip(captions, data)):
            print("---->Sample %d: %s %d" % (k, caption, m_length[k]))
            s_path = pjoin(result_dir, str(k))
            os.makedirs(s_path, exist_ok=True)
            joint_data = joint_data[:m_length[k]]
            joint = recover_from_ric(torch.from_numpy(joint_data).float(), nb_joints).numpy()
            # save_path = pjoin(s_path, "caption:%s_sample%d_repeat%d_len%d.mp4" % (caption, k, r, m_length[k]))
            save_path = pjoin(s_path, "caption:%s_sample%d_repeat%d_len%d.gif" % (caption, k, r, m_length[k]))
            plot_3d_motion(save_path, kinematic_chain, joint, title=caption, fps=20)
            np.save(pjoin(s_path, "caption:%s_sample%d_repeat%d_len%d.npy" % (caption, k, r, m_length[k])), joint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='ESMotion-Score-XL')
    parser.add_argument('--ae_name', type=str, default="AE")
    parser.add_argument('--ae_model', type=str, default='AE_Model')
    parser.add_argument('--model', type=str, default='ESMotion-Score-XL')
    parser.add_argument('--dataset_name', type=str, default='t2m', choices=['t2m'])
    parser.add_argument('--dataset_dir', type=str, default='./datasets')

    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
    parser.add_argument("--time_steps", default=18, type=int)
    parser.add_argument("--cfg", default=4.5, type=float)
    parser.add_argument("--temperature", default=1, type=float)

    parser.add_argument('--text_prompt', default='', type=str)
    parser.add_argument('--text_path', type=str, default="")
    parser.add_argument("--motion_length", default=0, type=int)
    parser.add_argument("--repeat_times", default=1, type=int)
    parser.add_argument('--hard_pseudo_reorder', action="store_true")
    arg = parser.parse_args()
    main(arg)

