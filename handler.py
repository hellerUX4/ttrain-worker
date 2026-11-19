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
import time

import numpy as np

FAKE = os.environ.get("TTRAIN_FAKE", "0") == "1"
DIM = 384
_MODEL = {}
SWIN_MODELS = {
    "swinv2_base": "swinv2_base_window12to16_192to256.ms_in22k_ft_in1k",
    "swinv2_large": "swinv2_large_window12to16_192to256.ms_in22k_ft_in1k",
}
TIMM_MODELS = {
    **SWIN_MODELS,
    "dinov3_vits16": "vit_small_patch16_dinov3.lvd1689m",
    "siglip2_vitb16": "vit_base_patch16_siglip_224.v2_webli",
    "convnextv2_tiny": "convnextv2_tiny.fcmae_ft_in22k_in1k",
}


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
    if name in TIMM_MODELS:
        import timm
        model = timm.create_model(f"hf-hub:timm/{TIMM_MODELS[name]}", pretrained=True, num_classes=0)
    else:
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


def adapter_layers(model, model_name, blocks):
    if model_name == "convnextv2_tiny" or model_name in SWIN_MODELS:
        stages = model.layers if model_name in SWIN_MODELS else model.stages
        return [(block.mlp, name)
                for index, stage in enumerate(stages) if index in blocks
                for block in stage.blocks for name in ("fc1", "fc2")]
    if model_name in TIMM_MODELS:
        return [(block.mlp, name)
                for index, block in enumerate(model.blocks) if index in blocks
                for name in ("fc1", "fc2")]
    return [(block.attn, name)
            for index, block in enumerate(model.blocks) if index in blocks
            for name in ("qkv", "proj")]


def image_transform(model, model_name, resolution):
    from torchvision import transforms

    if model_name in TIMM_MODELS:
        from timm.data import create_transform, resolve_model_data_config
        config = resolve_model_data_config(model)
        config["input_size"] = (3, resolution, resolution)
        return create_transform(**config, is_training=False)
    return transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def load_adapter_model(adapter, device=None):
    import torch

    name = adapter.get("model", "dinov2_vits14")
    model = copy.deepcopy(load_model(name))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    layers = [getattr(parent, key) for parent, key in adapter_layers(model, name, adapter["blocks"])]
    with torch.no_grad():
        for layer, weights in zip(layers, adapter["lora"], strict=True):
            layer.weight.add_((2 * (weights["B"] @ weights["A"])).to(layer.weight))
    model.to(device).eval().requires_grad_(False)
    head = {key: value.to(device) for key, value in adapter["head"].items()}
    return model, name, device, head


def visualize_yes(inp, progress=None):
    import torch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        return optimize_yes_pattern(inp, device, progress)
    except RuntimeError as error:
        if device.type != "cuda" or "Expected all tensors to be on the same device" not in str(error):
            raise
    if progress:
        progress({"fallback": "cpu"})
    device = torch.device("cpu")
    return optimize_yes_pattern(inp, device, progress)


def optimize_yes_pattern(inp, device, progress):
    import torch
    import torch.nn.functional as F
    from PIL import Image

    if FAKE:
        raise ValueError("Das Ja-Muster braucht einen echten Worker.")
    local_head = "linear_head" in inp
    if local_head:
        weights = inp["linear_head"]
        name = "dinov2_vits14"
        model = copy.deepcopy(load_model(name)).to(device).eval().requires_grad_(False)
        resolution = 336
        head = {"weight": torch.tensor([weights["w"]], device=device),
                "bias": torch.tensor([weights["b"]], device=device)}
    else:
        adapter = torch.load(io.BytesIO(base64.b64decode(inp["adapter_b64"], validate=True)),
                             map_location="cpu", weights_only=True)
        model, name, device, head = load_adapter_model(adapter, device=device)
        resolution = adapter["resolution"]
    normalization = image_transform(model, name, resolution).transforms[-1]
    noise = (0.2 * torch.randn(1, 3, 20, 20, device=device)).requires_grad_()
    optimizer = torch.optim.Adam([noise], lr=0.08)
    mean = noise.new_tensor(normalization.mean).view(1, 3, 1, 1)
    std = noise.new_tensor(normalization.std).view(1, 3, 1, 1)
    best_logit, best = float("-inf"), None
    steps = 32
    for step in range(steps + 1):
        pixels = F.interpolate(noise.sigmoid(), size=(resolution, resolution),
                               mode="bilinear", align_corners=False)
        with device:
            features = model((pixels - mean) / std)
        if local_head:
            features = F.normalize(features)
        logit = F.linear(features, head["weight"], head["bias"]).squeeze()
        value = logit.detach().item()
        if value > best_logit:
            best_logit, best = value, pixels.detach()
        if progress and (step % 2 == 0 or step == steps):
            progress({"step": step, "steps": steps})
        if step < steps:
            optimizer.zero_grad()
            (-logit + 0.01 * noise.square().mean()).backward()
            optimizer.step()
    if best is None:
        raise RuntimeError("Das Modell hat keinen gültigen Ja-Score geliefert.")
    image = Image.fromarray((best[0].permute(1, 2, 0) * 255).round().byte().cpu().numpy())
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return {"png": base64.b64encode(buf.getvalue()).decode("ascii"),
            "score": torch.sigmoid(torch.tensor(best_logit)).item()}


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

        adapter = torch.load(buf, map_location="cpu", weights_only=True)
        model, model_name, device, head = load_adapter_model(adapter)
        tf = image_transform(model, model_name, adapter["resolution"])
        with torch.inference_mode():
            x = torch.stack([tf(_decode(i["b64"])) for i in images]).to(device)
            logits = torch.nn.functional.linear(model(x), head["weight"], head["bias"])
            scores = logits.sigmoid().flatten().cpu().tolist()
    return {"scores": {str(item["id"]): score for item, score in zip(images, scores)}}


def training_images(inp):
    urls = inp.get("dataset_urls")
    if urls is None:
        url = inp.get("dataset_url")
        if not url:
            return inp.get("images") or []
        urls = [url]
    if not isinstance(urls, list) or not urls or any(not isinstance(url, str) for url in urls):
        raise ValueError("Ungültige Liste von Trainingspaketen.")
    images = []
    for url in urls:
        images.extend(download_training_part(url))
    return images


def download_training_part(url):
    import json
    from urllib.parse import urlsplit
    from urllib.request import HTTPRedirectHandler, build_opener

    parsed = urlsplit(url)
    if (parsed.scheme != "https" or not parsed.hostname
            or not parsed.hostname.endswith(".r2.cloudflarestorage.com")
            or parsed.netloc != parsed.hostname):
        raise ValueError("Ungültige Adresse für den privaten Trainingsspeicher.")

    class NoRedirect(HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    limit = 256 * 1024 * 1024
    try:
        with build_opener(NoRedirect()).open(url, timeout=120) as response:
            data = response.read(limit + 1)
    except OSError:
        raise RuntimeError("RunPod konnte die Trainingsbilder nicht aus R2 laden.") from None
    if len(data) > limit:
        raise ValueError("Trainingspaket überschreitet 256 MiB.")
    images = json.loads(data)
    if not isinstance(images, list) or not images:
        raise ValueError("Trainingspaket enthält keine Bilder.")
    return images


def handle_lora(inp, progress=None):
    """Trainiert Low-Rank-Adapter auf den letzten Blöcken und liefert sie zurück."""
    images = training_images(inp)
    blocks = inp.get("blocks", [8, 9, 10, 11])
    res = int(inp.get("resolution", 336))
    epochs = int(inp.get("epochs", 8))

    early = inp.get("task") == "lora_train_early"
    validation = [item for item in images if item.get("validation")] if early else []
    if early:
        images = [item for item in images if not item.get("validation")]
        if not images or {item["y"] for item in validation} != {0, 1}:
            raise ValueError("Early Stopping braucht separate Ja- und Nein-Prüfbilder.")
        if FAKE:
            raise ValueError("Early Stopping benötigt einen echten Trainingsworker.")

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

    model = inp.get("model", "dinov2_vits14")
    if model not in (*TIMM_MODELS, "dinov2_vits14", "dinov2_vits14_reg", "dinov2_vitb14", "dinov2_vitl14_reg"):
        raise ValueError("Unbekanntes LoRA-Basismodell.")
    expected_resolution = 256 if model in SWIN_MODELS else 224
    if model in TIMM_MODELS and res != expected_resolution:
        raise ValueError(f"Dieses Basismodell erwartet {expected_resolution} Pixel.")
    lr = float(inp.get("lr", 1e-4))
    head_lr = float(inp.get("head_lr", lr))
    if not all(0 < rate < float("inf") for rate in (lr, head_lr)):
        raise ValueError("Lernraten müssen endlich und größer als 0 sein.")
    return train_lora_real(images, blocks, res, epochs, lr, progress, model, validation, head_lr=head_lr)


def train_lora_real(images, blocks, res, epochs, lr, progress=None, model_name="dinov2_vits14", validation=(), head_lr=None):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    model = copy.deepcopy(load_model(model_name))
    tf = image_transform(model, model_name, res)

    class DS(Dataset):
        def __init__(self, items):
            self.items = items
            self.tf = tf

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    for p_ in model.parameters():
        p_.requires_grad_(False)

    wrapped = []
    for parent, name in adapter_layers(model, model_name, blocks):
        layer = getattr(parent, name)
        lora = LoRA(layer).to(device)
        setattr(parent, name, lora)
        wrapped.append(lora)

    features = model.num_features if model_name in TIMM_MODELS else model.embed_dim
    head = nn.Linear(features, 1).to(device)
    params = [head.weight, head.bias] + [p_ for m in wrapped for p_ in (m.A, m.B)]
    groups = [{"params": params[:2], "lr": lr if head_lr is None else head_lr}]
    if wrapped:
        groups.append({"params": params[2:], "lr": lr})
    opt = torch.optim.AdamW(groups, lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    ds = DS(images)
    dl = DataLoader(ds, batch_size=8, shuffle=True)
    validation_loader = DataLoader(DS(validation), batch_size=8) if validation else None
    best_loss, best_epoch, best_params = float("inf"), 0, None
    last_update = float("-inf")
    for epoch in range(1, epochs + 1):
        model.train()
        head.train()
        processed = 0
        for x, y in dl:
            x, y = x.to(device), y.float().to(device)
            opt.zero_grad()
            feat = model(x)
            loss = loss_fn(head(feat).squeeze(1), y)
            loss.backward()
            opt.step()
            processed += len(y)
            now = time.monotonic()
            if progress and (now - last_update >= 2 or processed == len(ds)):
                progress({"epoch": epoch, "epochs": epochs, "processed": processed, "images": len(ds)})
                last_update = now

        if validation_loader is not None:
            model.eval()
            head.eval()
            total_loss = 0.0
            with torch.no_grad():
                for x, y in validation_loader:
                    logits = head(model(x.to(device))).squeeze(1)
                    total_loss += loss_fn(logits, y.to(device)).item() * len(y)
            validation_loss = total_loss / len(validation)
            if not np.isfinite(validation_loss):
                raise RuntimeError("Der Validierungsverlust ist ungültig.")
            if validation_loss < best_loss:
                best_loss, best_epoch = validation_loss, epoch
                best_params = [param.detach().cpu().clone() for param in params]
            if epoch - best_epoch >= 3:
                break

    if best_params is not None:
        with torch.no_grad():
            for param, best in zip(params, best_params):
                param.copy_(best)
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
                "head": head.state_dict(), "blocks": list(blocks), "resolution": res, "model": model_name}, buf)
    return {"metrics": metrics, "adapter_b64": base64.b64encode(buf.getvalue()).decode(),
            "blocks": list(blocks), "epochs": epoch, "best_epoch": best_epoch or epoch,
            "early_stopping": validation_loader is not None, "model": model_name}


def handler(job):
    inp = job.get("input") or {}
    task = inp.get("task", "embed")
    if task == "embed":
        return handle_embed(inp)
    if task == "visualize_yes":
        return visualize_yes(inp)
    if task == "lora_predict":
        return handle_lora_predict(inp)
    if task in ("lora_train", "lora_train_early"):
        return handle_lora(inp)
    return {"error": f"unbekannte Aufgabe: {task}"}


def streaming_handler(job):
    inp = job.get("input") or {}
    if inp.get("task") not in ("lora_train", "lora_train_early", "visualize_yes") or FAKE:
        yield handler(job)
        return

    from queue import Queue
    from threading import Thread

    updates = Queue()

    def train():
        try:
            function = visualize_yes if inp.get("task") == "visualize_yes" else handle_lora
            result = function(inp, lambda progress: updates.put(("progress", progress)))
            updates.put(("result", result))
        except Exception as error:
            updates.put(("error", error))

    Thread(target=train, daemon=True).start()
    while True:
        kind, value = updates.get()
        if kind == "error":
            raise value
        if kind == "result":
            yield value
            return
        yield {"training_progress": value}


if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": streaming_handler, "return_aggregate_stream": True})
