import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from ignite.engine import Engine
from ignite.metrics import FID, InceptionScore
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG_SIZE = 128
LATENT_DIM = 128
CHANNELS = [32, 64, 128, 256, 512]
EPOCHS = 100
BATCH_SIZE = 64
LR = 1e-3
BETA = 1.0
KL_WARMUP_EPOCHS = 10
N_EVAL = 2500
EVAL_BATCH = 50
SEED = 0

_sqrtm = scipy.linalg.sqrtm
scipy.linalg.sqrtm = lambda A, disp=True: _sqrtm(A) if disp else (_sqrtm(A), 0.0)


def load_split(split_dir):
    paths = sorted(p for p in Path(split_dir).rglob("*")
                   if p.suffix.lower() in {".jpeg", ".jpg", ".png"} and not p.name.startswith("."))
    imgs = []
    for p in paths:
        with Image.open(p) as im:
            im.draft("L", (2 * IMG_SIZE, 2 * IMG_SIZE))
            im = im.convert("L").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
            imgs.append(np.asarray(im, dtype=np.uint8))
    return torch.from_numpy(np.stack(imgs))[:, None]


class ConvVAE(nn.Module):
    def __init__(self):
        super().__init__()
        enc, c_in = [], 1
        for c in CHANNELS:
            enc += [nn.Conv2d(c_in, c, 4, 2, 1, bias=False), nn.BatchNorm2d(c), nn.LeakyReLU(0.2, inplace=True)]
            c_in = c
        self.encoder = nn.Sequential(*enc)
        feat = CHANNELS[-1] * 4 * 4
        self.fc_mu = nn.Linear(feat, LATENT_DIM)
        self.fc_logvar = nn.Linear(feat, LATENT_DIM)

        self.fc_dec = nn.Sequential(nn.Linear(LATENT_DIM, feat), nn.ReLU(inplace=True))
        rev, dec = CHANNELS[::-1], []
        for a, b in zip(rev[:-1], rev[1:]):
            dec += [nn.ConvTranspose2d(a, b, 4, 2, 1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True)]
        dec += [nn.ConvTranspose2d(rev[-1], 1, 4, 2, 1)]
        self.decoder = nn.Sequential(*dec)

    def encode(self, x):
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h).clamp(-30.0, 20.0)

    def decode(self, z):
        return self.decoder(self.fc_dec(z).view(-1, CHANNELS[-1], 4, 4))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(z), mu, logvar


def vae_loss(logits, x, mu, logvar):
    rec = F.binary_cross_entropy_with_logits(logits, x, reduction="none").flatten(1).sum(1)
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)
    return rec.mean(), kl.mean()


@torch.no_grad()
def eval_loss(model, x_u8, bs=256):
    model.eval()
    s_rec = s_kl = 0.0
    for i in range(0, len(x_u8), bs):
        x = x_u8[i:i + bs].float().div_(255)
        logits, mu, logvar = model(x)
        rec, kl = vae_loss(logits, x, mu, logvar)
        s_rec += rec.item() * len(x)
        s_kl += kl.item() * len(x)
    return s_rec / len(x_u8), s_kl / len(x_u8)


def train(model, xtr, xte, device):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR * 0.01)
    steps_per_epoch = len(xtr) // BATCH_SIZE
    warm_steps = KL_WARMUP_EPOCHS * steps_per_epoch
    history, step = [], 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(xtr), device=device)
        s_rec = s_kl = 0.0
        for i in range(steps_per_epoch):
            x = xtr[perm[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]].float().div_(255)
            logits, mu, logvar = model(x)
            rec, kl = vae_loss(logits, x, mu, logvar)
            beta = BETA * min(1.0, (step + 1) / warm_steps)
            opt.zero_grad(set_to_none=True)
            (rec + beta * kl).backward()
            opt.step()
            step += 1
            s_rec += rec.item()
            s_kl += kl.item()
        sched.step()

        tr_rec, tr_kl = s_rec / steps_per_epoch, s_kl / steps_per_epoch
        te_rec, te_kl = eval_loss(model, xte)
        history.append(dict(epoch=epoch, train_loss=tr_rec + tr_kl, train_rec=tr_rec, train_kl=tr_kl,
                            test_loss=te_rec + te_kl, test_rec=te_rec, test_kl=te_kl))
        print(f"ep {epoch:3d} | train {tr_rec + tr_kl:8.1f} (rec {tr_rec:8.1f}, kl {tr_kl:6.1f}) | "
              f"held-out {te_rec + te_kl:8.1f} (rec {te_rec:8.1f}, kl {te_kl:6.1f}) | "
              f"beta {beta:.2f} | {time.time() - t0:.1f}s", flush=True)
    return history


def plot_losses(history, out):
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.0))
    panels = [("loss", "Negative ELBO (rec + KL)"), ("rec", "Reconstruction (BCE)"), ("kl", "KL divergence")]
    for ax, (key, title) in zip(axes, panels):
        ax.plot(ep, [h[f"train_{key}"] for h in history], label="train", lw=1.6)
        ax.plot(ep, [h[f"test_{key}"] for h in history], label="held-out", lw=1.6, ls="--")
        ax.axvline(KL_WARMUP_EPOCHS, color="gray", lw=0.8, ls=":")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("nats / image")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "loss_curves.png", dpi=200)
    fig.savefig(out / "loss_curves.pdf")
    plt.close(fig)


@torch.no_grad()
def sample(model, n, device, seed, bs=250):
    model.eval()
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(0, n, bs):
        z = torch.randn(min(bs, n - i), LATENT_DIM, generator=g).to(device)
        out.append(torch.sigmoid(model.decode(z)).cpu())
    return torch.cat(out)


@torch.no_grad()
def reconstruct(model, x_u8, device, bs=250):
    model.eval()
    out = []
    for i in range(0, len(x_u8), bs):
        mu, _ = model.encode(x_u8[i:i + bs].to(device).float().div_(255))
        out.append(torch.sigmoid(model.decode(mu)).cpu())
    return torch.cat(out)


@torch.no_grad()
def make_figures(model, test_u8, device, out):
    vutils.save_image(sample(model, 64, device, seed=SEED + 100), out / "samples_final.png",
                      nrow=8, padding=2, pad_value=1.0)

    idx = torch.randperm(len(test_u8), generator=torch.Generator().manual_seed(SEED))
    real = test_u8[idx[:10]]
    vutils.save_image(torch.cat([real.float() / 255, reconstruct(model, real, device)]),
                      out / "reconstructions.png", nrow=10, padding=2, pad_value=1.0)


def to_inception(batch):
    return F.interpolate(batch.clamp(0, 1).repeat(1, 3, 1, 1), size=(299, 299), mode="bilinear", align_corners=False)


def evaluate_fid_is(model, train_u8, device):
    n = min(N_EVAL, len(train_u8) // 2)
    perm = torch.randperm(len(train_u8), generator=torch.Generator().manual_seed(SEED))
    real_a = train_u8[perm[:n]].float() / 255
    real_b_u8 = train_u8[perm[n:2 * n]]
    sets = {
        "vae_samples": sample(model, n, device, seed=SEED + 1),
        "vae_reconstructions": reconstruct(model, real_b_u8, device),
        "real_images": real_b_u8.float() / 255,
    }

    def evaluation_step(engine, batch):
        with torch.no_grad():
            fake, real = batch
            return to_inception(fake.to(device)), to_inception(real.to(device))

    results = {"n_images_per_set": n}
    for name, imgs in sets.items():
        evaluator = Engine(evaluation_step)
        FID(device=device).attach(evaluator, "fid")
        InceptionScore(device=device, output_transform=lambda x: x[0]).attach(evaluator, "is")
        evaluator.run(DataLoader(TensorDataset(imgs, real_a), batch_size=EVAL_BATCH), max_epochs=1)
        m = evaluator.state.metrics
        results[name] = {"fid": float(m["fid"]), "is": float(m["is"])}
        print(f"  {name:20s} FID {m['fid']:8.4f} | IS {m['is']:.3f}", flush=True)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="chest_xray", help="folder containing train/ val/ test/")
    p.add_argument("--out_dir", default="runs/vae_cxr")
    args = p.parse_args()

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    root = Path(args.data_root)
    train_u8 = load_split(root / "train")
    test_u8 = torch.cat([load_split(root / "val"), load_split(root / "test")])
    print(f"train {len(train_u8)} | held-out {len(test_u8)} | device {device}")

    model = ConvVAE().to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    history = train(model, train_u8.to(device), test_u8.to(device), device)
    torch.save(model.state_dict(), out / "vae.pt")
    with open(out / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)

    model.eval()
    plot_losses(history, out)
    make_figures(model, test_u8, device, out)
    print("computing FID / IS ...")
    metrics = evaluate_fid_is(model, train_u8, device)
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()