# -*- coding: utf-8 -*-
"""RunPod-Handler für TTrain.

Aufgaben:
  {"task": "embed", "images": [{"id": "...", "b64": "..."}], "resolution": 336, "model": "dinov2_vits14"}
      → {"embeddings": [{"id": "...", "vector": [384 floats]}]}

  {"task": "lora_train", "images": [{"id": "...", "y": 0|1, "b64": "..."}], "blocks": [8,9,10,11],
   "resolution": 336, "epochs": 8, "lr": 1e-4}
      → {"metrics": {"precision":…, "recall":…, "precision_at_100":…}, "adapter_b64": "..."}

Ohne GPU/Modell lässt sich alles mit TTRAIN_FAKE=1 testen: dann werden
inhaltliche, aber billige Ersatzvektoren berechnet.
"""
import base64
import copy
import io
import os

import numpy as np

FAKE = os.environ.get("TTRAIN_FAKE", "0") == "1"
DIM = 384
_MODEL = {}


def _pil():
    from PIL import Image
    return Image


def _decode(b64):
    from PIL import Image
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return img


def fake_vector(img, resolution=336):
    """Billiger, aber inhaltlich brauchbarer Vektor: 12x8 Graustufen + Farbe."""
    small = img.convert("L").resize((8, 12))
    gray = np.asarray(small, dtype=np.float32).ravel() / 255.0
    rgb = np.asarray(img.resize((4, 4)), dtype=np.float32).ravel() / 255.0
    vec = np.zeros(DIM, dtype=np.float32)
    vec[: gray.size] = gray
    vec[gray.size: gray.size + rgb.size] = rgb
    extra = np.asarray(img.resize((2, 2)), dtype=np.float32).ravel() / 255.0
    vec[gray.size + rgb.size: gray.size + rgb.size + extra.size] = extra
    n = np.linalg.norm(vec) + 1e-8
    return (vec / n).tolist()


def load_model(name="dinov2_vits14"):
    if name in _MODEL:
        return _MODEL[name]
    import torch
    model = torch.hub.load("facebookresearch/dinov2", name, trust_repo=True)
    model.eval()
    _MODEL[name] = model
    return model


def embed_real(images, resolution=336, model_name="dinov2_vits14"):
    import torch
    from torchvision import transforms
    model = load_model(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    tf = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    out = []
    with torch.no_grad():
        for item in images:
            img = _decode(item["b64"])
            x = tf(img).unsqueeze(0).to(device)
            v = model(x)[0].float().cpu().numpy()
            v = v / (np.linalg.norm(v) + 1e-8)
            out.append({"id": str(item["id"]), "vector": v.tolist()})
    return out


def handle_embed(inp):
    images = inp.get("images") or []
    res = int(inp.get("resolution", 336))
    if FAKE:
        return {"embeddings": [{"id": str(i["id"]), "vector": fake_vector(_decode(i["b64"]), res)} for i in images]}
    return {"embeddings": embed_real(images, res, inp.get("model", "dinov2_vits14"))}


def handle_lora_predict(inp):
    images = inp["images"]
    buf = io.BytesIO(base64.b64decode(inp["adapter_b64"], validate=True))
    if FAKE:
        with np.load(buf, allow_pickle=False) as adapter:
            w, b = adapter["w"], float(adapter["b"][0])
        logits = np.asarray([fake_vector(_decode(i["b64"])) for i in images]) @ w + b
        scores = (1 / (1 + np.exp(-np.clip(logits, -40, 40)))).tolist()
    else:
        import torch
        from torchvision import transforms

        adapter = torch.load(buf, map_location="cpu", weights_only=True)
        model = copy.deepcopy(load_model())
        layers = [getattr(block.attn, name)
                  for index, block in enumerate(model.blocks) if index in adapter["blocks"]
                  for name in ("qkv", "proj")]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        res = adapter["resolution"]
        tf = transforms.Compose([
            transforms.Resize(res, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(res),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        with torch.inference_mode():
            for layer, weights in zip(layers, adapter["lora"], strict=True):
                delta = weights["B"] @ weights["A"]
                layer.weight.add_((delta * 2).to(layer.weight))
            model.to(device).eval()
            head = {key: value.to(device) for key, value in adapter["head"].items()}
            x = torch.stack([tf(_decode(i["b64"])) for i in images]).to(device)
            logits = torch.nn.functional.linear(model(x), head["weight"], head["bias"])
            scores = logits.sigmoid().flatten().cpu().tolist()
    return {"scores": {str(item["id"]): score for item, score in zip(images, scores)}}


def handle_lora(inp):
    """Trainiert Low-Rank-Adapter auf den letzten Blöcken und liefert sie zurück."""
    images = inp.get("images") or []
    blocks = inp.get("blocks") or [8, 9, 10, 11]
    res = int(inp.get("resolution", 336))
    epochs = int(inp.get("epochs", 8))

    if FAKE:
        # Ersatz-Training: lineares Modell auf den Ersatzvektoren
        X = np.stack([fake_vector(_decode(i["b64"]), res) for i in images])
        y = np.array([float(i.get("y", 0)) for i in images])
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        w = np.zeros(X.shape[1], dtype=np.float32)
        b = 0.0
        for _ in range(300):
            p = 1 / (1 + np.exp(-np.clip(X @ w + b, -40, 40)))
            g = (p - y) / len(y)
            w -= 0.5 * (X.T @ g)
            b -= 0.5 * float(g.sum())
        p = 1 / (1 + np.exp(-np.clip(X @ w + b, -40, 40)))
        pred = p >= 0.5
        tp = float(((pred) & (y == 1)).sum()); fp = float(((pred) & (y == 0)).sum())
        fn = float(((~pred) & (y == 1)).sum())
        order = np.argsort(-p)[: min(100, len(p))]
        metrics = {"precision": round(tp / max(1.0, tp + fp), 3),
                   "recall": round(tp / max(1.0, tp + fn), 3),
                   "precision_at_100": round(float(y[order].mean()), 3)}
        buf = io.BytesIO()
        np.savez_compressed(buf, w=w, b=np.array([b]))
        adapter = base64.b64encode(buf.getvalue()).decode()
        return {"metrics": metrics, "adapter_b64": adapter, "blocks": blocks, "epochs": epochs}

    return train_lora_real(images, blocks, res, epochs, float(inp.get("lr", 1e-4)))


def train_lora_real(images, blocks, res, epochs, lr):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    class DS(Dataset):
        def __init__(self, items):
            self.items = items
            self.tf = transforms.Compose([
                transforms.Resize(res, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(res),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            it = self.items[i]
            return self.tf(_decode(it["b64"])), torch.tensor(float(it.get("y", 0)))

    class LoRA(nn.Module):
        def __init__(self, base: nn.Linear, r=8, alpha=16):
            super().__init__()
            self.base = base
            self.scale = alpha / r
            self.A = nn.Parameter(torch.zeros(r, base.in_features))
            self.B = nn.Parameter(torch.zeros(base.out_features, r))
            nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)

        def forward(self, x):
            return self.base(x) + self.scale * (x @ self.A.T @ self.B.T)

    model = copy.deepcopy(load_model())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    for p_ in model.parameters():
        p_.requires_grad_(False)

    wrapped = []
    for idx, blk in enumerate(model.blocks):
        if idx not in blocks:
            continue
        for name in ("qkv", "proj"):
            layer = getattr(blk.attn, name)
            lora = LoRA(layer).to(device)
            setattr(blk.attn, name, lora)
            wrapped.append(lora)

    head = nn.Linear(model.embed_dim, 1).to(device)
    params = [head.weight, head.bias] + [p_ for m in wrapped for p_ in (m.A, m.B)]
    opt = torch.optim.AdamW(params, lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    ds = DS(images)
    dl = DataLoader(ds, batch_size=8, shuffle=True)
    model.train()
    for _ in range(epochs):
        for x, y in dl:
            x, y = x.to(device), y.float().to(device)
            opt.zero_grad()
            feat = model(x)
            loss = loss_fn(head(feat).squeeze(1), y)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        scores, labels = [], []
        for x, y in DataLoader(ds, batch_size=8):
            x = x.to(device)
            scores.append(torch.sigmoid(head(model(x)).squeeze(1)).cpu().numpy())
            labels.append(y.numpy())
    p = np.concatenate(scores); y = np.concatenate(labels)
    pred = p >= 0.5
    tp = float((pred & (y == 1)).sum()); fp = float((pred & (y == 0)).sum()); fn = float(((~pred) & (y == 1)).sum())
    order = np.argsort(-p)[: min(100, len(p))]
    metrics = {"precision": round(tp / max(1.0, tp + fp), 3), "recall": round(tp / max(1.0, tp + fn), 3),
               "precision_at_100": round(float(y[order].mean()), 3)}

    buf = io.BytesIO()
    torch.save({"lora": [{k: v.detach().cpu() for k, v in (("A", m.A), ("B", m.B))} for m in wrapped],
                "head": head.state_dict(), "blocks": list(blocks), "resolution": res}, buf)
    return {"metrics": metrics, "adapter_b64": base64.b64encode(buf.getvalue()).decode(),
            "blocks": list(blocks), "epochs": epochs}


def handler(job):
    inp = job.get("input") or {}
    task = inp.get("task", "embed")
    if task == "embed":
        return handle_embed(inp)
    if task == "lora_predict":
        return handle_lora_predict(inp)
    if task == "lora_train":
        return handle_lora(inp)
    return {"error": f"unbekannte Aufgabe: {task}"}


if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
