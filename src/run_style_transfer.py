import torch.optim as optim
from utils import load_image, save_image, get_device, load_yaml
from model import VGGFeatures
from loss import content_loss, style_loss, gram_matrix
import sys

def run_style_transfer(exp_config_path):
    cfg = load_yaml(exp_config_path)
    device = get_device()

    content = load_image(
        cfg["data"]["content_path"],
        cfg["image"]["size"],
        device
    )
    style = load_image(
        cfg["data"]["style_path"],
        cfg["image"]["size"],
        device
    )

    model = VGGFeatures().to(device)

    generated = content.clone().requires_grad_(True)
    optimizer = optim.Adam(
        [generated],
        lr=cfg["optimization"]["learning_rate"]
    )

    content_feats = model(content)
    style_feats = model(style)

    style_weights = cfg["loss"]["style_layers"]

    style_grams = {
        k: gram_matrix(style_feats[k])
        for k in style_weights
    }

    for step in range(cfg["optimization"]["steps"]):
        gen_feats = model(generated)

        c_loss = content_loss(
            gen_feats["conv5"],
            content_feats["conv5"]
        )
        s_loss = style_loss(
            gen_feats,
            style_grams,
            style_weights
        )

        loss = (
            cfg["loss"]["content_weight"] * c_loss +
            cfg["loss"]["style_weight"] * s_loss
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(
                f"[{cfg['experiment']['name']}] "
                f"Step {step} | Loss {loss.item():.4f}"
            )

    save_image(generated, cfg["output"]["path"])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/train.py <exp_config.yaml>")
        exit(1)

    run_style_transfer(sys.argv[1])
