import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.resnet_simclr import ResNetSimCLR
from simclr import SimCLR
from sklearn.manifold import TSNE
from torchvision import datasets
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms as transforms

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', '--data', metavar='DIR', required=True,
                    help='path to dataset')
parser.add_argument('-dataset-name', '--dataset-name', default='MSTAR',
                    help='dataset name', choices=['stl10', 'cifar10','MSTAR'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-output', '--output', metavar='DIR', default='runs/simclr',
                    help='path to output')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('--visualize-only', action='store_true', help='Run t-SNE visualization only, without training.')
parser.add_argument('--vis-data', metavar='../MSTAR/val', default=None, help='Path to the labeled dataset for t-SNE visualization (must be in ImageFolder format).')

def main():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    if args.visualize_only:
        if not args.vis_data:
            raise ValueError("错误: 使用 --visualize-only 时必须提供 --vis-data 路径。")
        print("--- 仅运行 t-SNE 可视化 ---")
        visualize_tsne(args)
    else:
        print("--- 开始 SimCLR 训练 ---")
        dataset = ContrastiveLearningDataset(args.data)
        train_dataset = dataset.get_dataset(args.dataset_name, args.n_views)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=True)

        model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)

        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        if args.device.type == 'cuda':
            with torch.cuda.device(args.gpu_index):
                simclr.train(train_loader)
        else:
            simclr.train(train_loader)

        if args.vis_data:
            print("\n--- 训练完成，开始进行 t-SNE 可视化 ---")
            visualize_tsne(args)
        else:
            print("\n--- 训练完成 ---")


def visualize_tsne(args):
    """
    依次生成 Epoch 0, 50, 100 的图像，
    使用热启动 (Hot Start) 保证图像形态的连续演变。
    """

    # 1. 定义目标 Epoch
    target_epochs = [0, 50, args.epochs]
    print(f"--- 准备生成序列图象 ---")
    print(f"目标 Epoch: {target_epochs}")

    # 数据准备
    vis_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    vis_dataset = datasets.ImageFolder(args.vis_data, transform=vis_transform)
    vis_loader = torch.utils.data.DataLoader(
        vis_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # 必须 False，保证点一一对应
        num_workers=args.workers,
        pin_memory=True
    )

    prev_embedding = None # 用于存储上一轮坐标，实现热启动

    # --- 循环开始 ---
    for epoch_idx in target_epochs:
        print(f"\n========================================")
        print(f"正在处理 Epoch {epoch_idx} ...")
        print(f"========================================")

        # 1. 加载模型
        model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)
        model = model.to(args.device)

        if epoch_idx == 0:
            print("  -> 加载随机初始化权重 (Random Init)")
        else:
            checkpoint_path = os.path.join(args.output, f'checkpoint_{epoch_idx:04d}.pth.tar')
            if not os.path.exists(checkpoint_path):
                print(f"  警告: 找不到 {checkpoint_path}，跳过此阶段")
                continue
            print(f"  -> 加载权重: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=args.device)
            state_dict = checkpoint.get('state_dict', checkpoint)
            model.load_state_dict(state_dict, strict=False)

        model.eval()

        # 2. 提取特征
        features_list, labels_list = [], []
        with torch.no_grad():
            for images, labels in tqdm(vis_loader, desc=f"  提取特征"):
                images = images.to(args.device)
                feats = model.backbone(images)
                if len(feats.shape) > 2: feats = feats.view(feats.size(0), -1)

                # --- 【核心修复】归一化 ---


                features_list.append(feats.cpu().numpy())
                labels_list.append(labels.cpu().numpy())

        features = np.concatenate(features_list, axis=0)
        all_labels = np.concatenate(labels_list, axis=0)

        # Epoch 0 的微小噪声处理 (防止SVD崩溃)
        std_val = np.std(features)
        if epoch_idx == 0 or std_val < 1e-4:
            noise = np.random.normal(0, 1e-5, features.shape)
            features = features + noise

        # 3. 计算 t-SNE (关键的热启动逻辑)
        if prev_embedding is None:
            # 第一张图 (Epoch 0)：只能随机初始化
            init_mode = 'random'
            print("  -> t-SNE 初始化: Random (冷启动)")
        else:
            # 后续图：使用上一轮的坐标作为起点
            init_mode = prev_embedding
            print("  -> t-SNE 初始化: Hot Start (继承上一轮坐标)")

        tsne = TSNE(n_components=2, init=init_mode, perplexity=45, n_iter=1000, random_state=args.seed)

        try:
            # 计算当前 Epoch 的坐标
            current_embedding = tsne.fit_transform(features)
            # 更新 prev_embedding 供下一轮使用
            prev_embedding = current_embedding
        except Exception as e:
            print(f"  [Error] t-SNE 计算出错: {e}")
            continue

        # 4. 立即绘图并保存 (在循环内部)
        print(f"  -> 正在绘制 Epoch {epoch_idx} 的图像...")
        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(
            current_embedding[:, 0],
            current_embedding[:, 1],
            c=all_labels,
            s=8,
            alpha=0.8,
            cmap=plt.cm.get_cmap("jet", len(vis_dataset.classes))
        )

        plt.axis('off')

        # 动态文件名
        save_name = f'tsne_hot_start_epoch_{epoch_idx}.png'
        save_path = os.path.join(args.output, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close() # 这一步很重要，防止内存溢出

        print(f"  -> [成功] 图片已保存至: {save_path}")

    print("\n--- 全部处理完成 ---")

if __name__ == "__main__":
    main()
