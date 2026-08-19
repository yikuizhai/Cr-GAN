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
            raise ValueError("--vis-data is required when --visualize-only is set.")
        print("--- Running t-SNE visualization only ---")
        visualize_tsne(args)
    else:
        print("--- Starting SimCLR training ---")
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
            print("\n--- Training complete; starting t-SNE visualization ---")
            visualize_tsne(args)
        else:
            print("\n--- Training complete ---")


def visualize_tsne(args):
    """
    Generate feature-space plots at epochs 0, 50, and the final epoch.
    Warm-start t-SNE with the coordinates from the previous epoch.
    """

    # Define the target epochs.
    target_epochs = [0, 50, args.epochs]
    print("--- Preparing the feature-space visualization sequence ---")
    print(f"Target epochs: {target_epochs}")

    # Prepare the visualization dataset.
    vis_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    vis_dataset = datasets.ImageFolder(args.vis_data, transform=vis_transform)
    vis_loader = torch.utils.data.DataLoader(
        vis_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Keep sample order aligned across epochs.
        num_workers=args.workers,
        pin_memory=True
    )

    prev_embedding = None  # Coordinates from the previous epoch.

    # Iterate over the target epochs.
    for epoch_idx in target_epochs:
        print("\n========================================")
        print(f"Processing epoch {epoch_idx} ...")
        print("========================================")

        # Load the encoder.
        model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)
        model = model.to(args.device)

        if epoch_idx == 0:
            print("  -> Using random initialization")
        else:
            checkpoint_path = os.path.join(args.output, f'checkpoint_{epoch_idx:04d}.pth.tar')
            if not os.path.exists(checkpoint_path):
                print(f"  Warning: {checkpoint_path} was not found; skipping this epoch")
                continue
            print(f"  -> Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=args.device)
            state_dict = checkpoint.get('state_dict', checkpoint)
            model.load_state_dict(state_dict, strict=False)

        model.eval()

        # Extract features.
        features_list, labels_list = [], []
        with torch.no_grad():
            for images, labels in tqdm(vis_loader, desc="  Extracting features"):
                images = images.to(args.device)
                feats = model.backbone(images)
                if len(feats.shape) > 2: feats = feats.view(feats.size(0), -1)

                features_list.append(feats.cpu().numpy())
                labels_list.append(labels.cpu().numpy())

        features = np.concatenate(features_list, axis=0)
        all_labels = np.concatenate(labels_list, axis=0)

        # Add a small perturbation when the features are degenerate.
        std_val = np.std(features)
        if epoch_idx == 0 or std_val < 1e-4:
            noise = np.random.normal(0, 1e-5, features.shape)
            features = features + noise

        # Compute t-SNE coordinates.
        if prev_embedding is None:
            # Initialize the first plot randomly.
            init_mode = 'random'
            print("  -> t-SNE initialization: random")
        else:
            # Warm-start subsequent plots from the previous coordinates.
            init_mode = prev_embedding
            print("  -> t-SNE initialization: warm start")

        tsne = TSNE(n_components=2, init=init_mode, perplexity=45, n_iter=1000, random_state=args.seed)

        try:
            # Compute the coordinates for the current epoch.
            current_embedding = tsne.fit_transform(features)
            # Reuse these coordinates for the next epoch.
            prev_embedding = current_embedding
        except Exception as e:
            print(f"  [Error] t-SNE failed: {e}")
            continue

        # Draw and save the plot immediately.
        print(f"  -> Drawing the plot for epoch {epoch_idx} ...")
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

        # Use an epoch-specific filename.
        save_name = f'tsne_hot_start_epoch_{epoch_idx}.png'
        save_path = os.path.join(args.output, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()  # Close the figure to prevent memory growth.

        print(f"  -> Saved visualization to: {save_path}")

    print("\n--- All visualizations complete ---")

if __name__ == "__main__":
    main()
