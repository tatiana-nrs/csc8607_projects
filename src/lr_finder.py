import argparse
import copy
import os
import time
import yaml
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import src.model as model
from src import augmentation, data_loading


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # ======================
    # Load config
    # ======================
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config["train"]["seed"])

    # ======================
    # Dataset (subset)
    # ======================
    dataset = augmentation.AugmentationDataset(
        data_path=config["dataset"]["split"]["train"]["chemin"],
        transform=None,
    )

    train_loader = data_loading.create_stratified_subset_loader_manual(
        dataset=dataset,
        subset_size=10000,          # comme ton camarade
        batch_size=config["train"]["batch_size"],
    )

    # ======================
    # Model
    # ======================
    net = model.build_model(config["basic_model"]).to(device)
    criterion = nn.CrossEntropyLoss()
    initial_state = copy.deepcopy(net.state_dict())

    # ======================
    # LR finder params (FIXÉS)
    # ======================
    lr_list = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    wd_list = [0.0, 1e-5, 1e-4]
    iters = 100

    run_name = f"lr_finder_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join(config["paths"]["runs_dir"], run_name)
    writer = SummaryWriter(log_dir)

    step = 0

    # ======================
    # LR finder (PROPRE)
    # ======================
    for wd in wd_list:
        net.load_state_dict(initial_state)

        for lr in lr_list:
            optimizer = model.get_optimizer(
                net,
                config,
                lr=lr,
                weight_decay=wd,
            )

            net.train()
            losses = []

            for i, (x, y) in enumerate(train_loader):
                if i >= iters:
                    break

                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                out = net(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

            avg_loss = sum(losses) / len(losses)

            writer.add_scalar("lr_finder/loss", avg_loss, step)
            writer.add_scalar("lr_finder/lr", lr, step)
            writer.add_scalar("lr_finder/wd", wd, step)

            print(f"[step {step:03d}] lr={lr:.1e} wd={wd:.1e} loss={avg_loss:.4f}")
            step += 1

    writer.close()
    print(f"[DONE] logs -> {log_dir}")
    print("TensorBoard tags: lr_finder/lr , lr_finder/loss")


if __name__ == "__main__":
    main()
