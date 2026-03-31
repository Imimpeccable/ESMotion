# ESMotion


## Setup
Environment setup is the same as the official MARDM repo:
https://github.com/neu-vi/MARDM

## Data Preparation
Please follow the dataset preparation instructions in:
https://github.com/neu-vi/MARDM

Then place HumanML3D under `./datasets/HumanML3D`.

## Inference
Single prompt:
```bash
python sample.py --name ESMotion_SiT_XL --model ESMotion-Score-XL --dataset_name t2m --text_prompt "A person is running on a treadmill."
```

Prompt file:
```bash
python sample.py --name ESMotion_SiT_XL --model ESMotion-Score-XL --dataset_name t2m --text_path ./text_prompt.txt
```


## Training
Train AE:
```bash
python train_AE.py --name AE --dataset_name t2m --batch_size 256 --epoch 50 --lr_decay 0.05
```

Train ESMotion:
```bash
python train_ESMotion.py --name ESMotion_SiT_XL --model ESMotion-Score-XL --dataset_name t2m --batch_size 64 --ae_name AE
```

## Evaluation
Evaluate AE:
```bash
python evaluation_AE.py --name AE --dataset_name t2m
```

Evaluate ESMotion:
```bash
python evaluation_ESMotion.py --name ESMotion_SiT_XL --model ESMotion-Score-XL --dataset_name t2m --cfg 4.5
```
