import secrets
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


def _decode(b64, processing_version=1):
    from PIL import Image
    raw = base64.b64decode(b64)
    from PIL import ImageOps
    with Image.open(io.BytesIO(raw)) as source:
        oriented = ImageOps.exif_transpose(source) if processing_version >= 2 else source
        img = oriented.convert("RGB")
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


def embed_real(images, resolution=336, model_name="dinov2_vits14", processing_version=1):
    import torch
    from torchvision import transforms
    model = load_model(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    tf = configured_transform(model, model_name, resolution,
                              {"crop": "pad", "processing_version": 2} if processing_version >= 2 else {})
    out = []
    with torch.no_grad():
        for item in images:
            img = _decode(item["b64"], processing_version)
            x = tf(img).unsqueeze(0).to(device)
            v = model(x)[0].float().cpu().numpy()
            v = v / (np.linalg.norm(v) + 1e-8)
            out.append({"id": str(item["id"]), "vector": v.tolist(), "signature": item.get("signature")})
    return out


def handle_embed(inp):
    images = training_images(inp)
    res = int(inp.get("resolution", 336))
    if FAKE:
        return {"embeddings": [{"id": str(i["id"]), "vector": fake_vector(_decode(i["b64"]), res)} for i in images]}
    return {"embeddings": embed_real(images, res, inp.get("model", "dinov2_vits14"), inp.get("processing_version", 1)),
            "processing_version": inp.get("processing_version", 1), "resolution": res,
            "model": inp.get("model", "dinov2_vits14")}


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
    steps, grid_size = inp.get("steps", 32), inp.get("grid", 20)
    if type(steps) is not int or steps not in (32, 64, 128, 256, 512):
        raise ValueError("Ungültige Schrittzahl für das Ja-Muster.")
    if type(grid_size) is not int or grid_size not in (20, 40, 64):
        raise ValueError("Ungültige Rastergröße für das Ja-Muster.")
    score, resolution = yes_pattern_scorer(inp, device)
    noise = (0.2 * torch.randn(1, 3, grid_size, grid_size, device=device)).requires_grad_()
    optimizer = torch.optim.Adam([noise], lr=0.08)
    best_logit, best = float("-inf"), None
    for step in range(steps + 1):
        pixels = F.interpolate(noise.sigmoid(), size=(resolution, resolution),
                               mode="bilinear", align_corners=False)
        logit = score(pixels)
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
            "score": torch.sigmoid(torch.tensor(best_logit)).item(), "steps": steps, "grid": grid_size}


def yes_pattern_scorer(inp, device):
    import torch
    import torch.nn.functional as F

    pattern_options = {}
    local_head = "linear_head" in inp
    if local_head:
        weights = inp["linear_head"]
        name = "dinov2_vits14"
        model = copy.deepcopy(load_model(name)).to(device).eval().requires_grad_(False)
        resolution = 336
        head = {"weight": torch.tensor([weights["w"]], dtype=torch.float32, device=device),
                "bias": torch.tensor([weights["b"]], dtype=torch.float32, device=device)}
    else:
        adapter = torch.load(io.BytesIO(base64.b64decode(inp["adapter_b64"], validate=True)),
                             map_location="cpu", weights_only=True)
        model, name, device, head = load_adapter_model(adapter, device=device)
        resolution = adapter["resolution"]
        pattern_options = adapter.get("training_options") or {}
    normalization = image_transform(model, name, resolution).transforms[-1]
    mean = head["weight"].new_tensor(normalization.mean).view(1, 3, 1, 1)
    std = head["weight"].new_tensor(normalization.std).view(1, 3, 1, 1)

    def score(pixels):
        pixels = F.interpolate(pixels, size=(resolution, resolution), mode="bilinear",
                               align_corners=False, antialias=True) if pixels.shape[-1] != resolution else pixels
        with device:
            normalized = (pixels - mean) / std
        if pattern_options.get("detail_grid"):
            tensors = [normalized]
            grid = pattern_options["detail_grid"]
            overlap = pattern_options.get("detail_overlap", .2)
            size = 1 / (grid - (grid - 1) * overlap)
            for row in range(grid):
                for col in range(grid):
                    left = round(col * size * (1-overlap) * resolution)
                    top = round(row * size * (1-overlap) * resolution)
                    tile = normalized[:, :, top:min(resolution, round(top+size*resolution)),
                                      left:min(resolution, round(left+size*resolution))]
                    tensors.append(F.interpolate(tile, size=(resolution,resolution),
                                                 mode="bicubic", align_corners=False))
            normalized = torch.stack(tensors, dim=1)
        features = feature_tensor(model, normalized)
        if local_head:
            features = F.normalize(features)
        return F.linear(features, head["weight"], head["bias"]).squeeze()

    return score, resolution


def visualize_yes_diffusion(inp, progress=None):
    import torch
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    from PIL import Image

    if FAKE or not torch.cuda.is_available():
        raise RuntimeError("Das fotorealistische Ja-Muster braucht einen GPU-Worker.")
    device = torch.device("cuda:0")
    score, _ = yes_pattern_scorer(inp, device)
    pipe = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        variant="fp16", torch_dtype=torch.float16, use_safetensors=True, token=False,
        safety_checker=None, requires_safety_checker=False,
    ).to(device)
    pipe.vae.eval().requires_grad_(False)
    pipe.unet.eval().requires_grad_(False)
    pipe.enable_vae_slicing()
    scheduler = DDIMScheduler.from_config(pipe.scheduler.config, clip_sample=False)
    count, steps = 3, 24
    scheduler.set_timesteps(steps, device=device)
    with torch.no_grad():
        embedding, _ = pipe.encode_prompt("", device, 1, False)
    images = []
    for sample in range(count):
        seed = secrets.randbits(32)
        rng = torch.Generator(device=device).manual_seed(seed)
        latent = torch.randn((1, pipe.unet.config.in_channels, 64, 64),
                             generator=rng, device=device, dtype=torch.float32)
        for index, timestep in enumerate(scheduler.timesteps):
            with torch.no_grad():
                noise = pipe.unet(latent.to(pipe.unet.dtype), timestep,
                                  encoder_hidden_states=embedding).sample.float()
            # Guide intermediate predictions of the clean image without retaining the U-Net graph.
            if 3 <= index < steps - 2:
                with torch.enable_grad():
                    clean = scheduler.step(noise, timestep, latent).pred_original_sample
                    clean = clean.detach().requires_grad_()
                    decoded = pipe.vae.decode((clean / pipe.vae.config.scaling_factor).to(pipe.vae.dtype)).sample
                    logit = score((decoded.float() / 2 + .5).clamp(0, 1))
                    gradient, = torch.autograd.grad(logit, clean)
                rms = gradient.square().mean().sqrt()
                if torch.isfinite(rms) and rms > 0:
                    noise = noise - 0.12 * (gradient / rms).clamp(-4, 4)
            latent = scheduler.step(noise, timestep, latent).prev_sample
            if progress:
                progress({"step": sample * steps + index + 1, "steps": count * steps})
        with torch.no_grad():
            decoded = pipe.vae.decode((latent / pipe.vae.config.scaling_factor).to(pipe.vae.dtype)).sample
            pixels = (decoded.float() / 2 + .5).clamp(0, 1)
            array = (pixels[0].permute(1, 2, 0) * 255).round().byte().cpu().numpy()
            image = Image.fromarray(array)
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=90)
        with torch.no_grad():
            import numpy as np
            shown = torch.from_numpy(np.array(Image.open(buf))).to(device).permute(2, 0, 1)[None].float() / 255
            score_value = score(shown).sigmoid().item()
        images.append({"jpg": base64.b64encode(buf.getvalue()).decode("ascii"),
                       "score": score_value, "seed": seed})
    return {"images": images, "steps": steps}


def configured_transform(model, model_name, resolution, options=None, training=False):
    from torchvision import transforms
    from PIL import ImageOps
    options = options or {}
    base = image_transform(model, model_name, resolution)
    steps = []
    if training:
        if options.get("flip", False):
            steps.append(transforms.RandomHorizontalFlip())
        brightness = options.get("brightness", 0)
        saturation = options.get("saturation", 0)
        if brightness or saturation:
            steps.append(transforms.ColorJitter(brightness=brightness, saturation=saturation))
        scale = options.get("crop_scale", 1)
        if scale < 1:
            steps.append(transforms.RandomResizedCrop(
                resolution, scale=(scale, 1), ratio=(0.95, 1.05),
                interpolation=transforms.InterpolationMode.BICUBIC))
    if options.get("crop", "center") == "pad":
        steps.append(transforms.Lambda(lambda image: ImageOps.pad(
            image, (resolution, resolution), method=3, color=(124, 116, 104))))
    spatial = (transforms.Resize, transforms.CenterCrop)
    steps.extend(step for step in base.transforms
                 if options.get("crop") != "pad" or not isinstance(step, spatial))
    return transforms.Compose(steps)


def detail_images(image, options):
    grid = options.get("detail_grid", 0)
    if not grid:
        return [image]
    overlap = options.get("detail_overlap", 0.2)
    size = 1 / (grid - (grid-1)*overlap)
    stride = size * (1-overlap)
    tiles = [image]
    for row in range(grid):
        for col in range(grid):
            left, top = col*stride, row*stride
            tiles.append(image.crop((round(left*image.width), round(top*image.height),
                                     min(image.width, round((left+size)*image.width)),
                                     min(image.height, round((top+size)*image.height)))))
    return tiles


def feature_tensor(model, x):
    import torch
    if x.ndim == 4:
        return model(x)
    batch, views = x.shape[:2]
    chunks = []
    for chunk in x.flatten(0, 1).split(1 if model.training else 2):
        if model.training:
            from torch.utils.checkpoint import checkpoint
            chunks.append(checkpoint(model, chunk, use_reentrant=False))
        else:
            chunks.append(model(chunk))
    features = torch.cat(chunks).reshape(batch, views, -1)
    if views == 1:
        # Keep the classifier input layout while omitting both detail features.
        full = features[:, 0]
        return torch.cat((full, full.new_zeros((batch, 2 * full.shape[1]))), dim=1)
    return torch.cat((features[:, 0], features[:, 1:].mean(1),
                                   features[:, 1:].max(1).values), dim=1)


def load_sam3(device):
    from huggingface_hub import hf_hub_download
    from transformers import Sam3Model, Sam3Processor

    model_id = "facebook/sam3"
    hf_hub_download(model_id, "config.json")
    processor = Sam3Processor.from_pretrained(model_id)
    model = Sam3Model.from_pretrained(model_id).to(device).eval()
    return model, processor, device


def segment_sam3_image(image, runtime, prompt="person"):
    model, processor, device = runtime
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    size = image.copy()
    size.thumbnail((512, 512))
    return processor.post_process_instance_segmentation(
        model(**inputs), threshold=.5, mask_threshold=.5,
        target_sizes=[(size.height, size.width)],
    )[0]


def encode_png(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def encode_masks(masks):
    from PIL import Image, ImageChops

    PALETTE = ((27, 206, 235), (255, 164, 66), (137, 225, 107), (211, 135, 248))

    overlay = None
    foreground = None
    count = 0
    for mask in masks:
        alpha = Image.fromarray(mask.cpu().numpy().astype("uint8") * 255)
        if alpha.getbbox() is None:
            continue
        color = PALETTE[count % len(PALETTE)]
        count += 1
        foreground = alpha if foreground is None else ImageChops.lighter(foreground, alpha)
        layer = Image.new("RGBA", alpha.size, color)
        layer.putalpha(alpha.point([0] * 255 + [120]))
        overlay = layer if overlay is None else Image.alpha_composite(overlay, layer)
    if foreground is None:
        return {"person_count": 0, "mask_png": None, "foreground_png": None}
    matte = Image.new("RGBA", foreground.size)
    matte.putalpha(foreground)
    return {"person_count": count, "mask_png": encode_png(overlay), "foreground_png": encode_png(matte)}


def sam3_group_box(image, masks):
    if not len(masks):
        return None
    foreground = masks.bool().any(dim=0).cpu().numpy()
    ys, xs = np.nonzero(foreground)
    if not len(xs):
        return None
    mask_height, mask_width = foreground.shape
    width, height = image.size
    box = (int(xs.min()) * width // mask_width, int(ys.min()) * height // mask_height,
           ((int(xs.max()) + 1) * width + mask_width - 1) // mask_width,
           ((int(ys.max()) + 1) * height + mask_height - 1) // mask_height)
    if box[2] - box[0] < 16 or box[3] - box[1] < 16:
        return None
    return box


def yolo_person_masks(image, model, device, *, confidence=.25, iou=None, resolution=640):
    resized = image.copy()
    max_edge = max(768, resolution)
    resized.thumbnail((max_edge, max_edge))
    overlap = {} if iou is None else {"iou": iou}
    result = model.predict(resized, classes=[0], conf=confidence, imgsz=resolution, retina_masks=True,
                           device=device, verbose=False, **overlap)[0]
    if result.masks is None:
        return [], []
    return result.masks.data.cpu(), result.boxes.conf.cpu().tolist()


def yolo_mask_quality(masks, scores):
    if not len(masks):
        return "Keine Personenmaske gefunden."
    if len(scores) != len(masks) or any(not np.isfinite(score) or score < .60 for score in scores):
        return "Mindestens eine Person wurde zu unsicher erkannt."
    if (masks.float().mean(dim=(1, 2)) < .003).any():
        return "Mindestens eine Personenmaske ist sehr klein."
    area = masks.bool().any(dim=0).float().mean().item()
    if not .05 <= area <= .90:
        return "Die gemeinsame Maske bedeckt weniger als 5 % oder mehr als 90 % des Bildes."
    return None


def yolo_training_mask(image, model, device):
    from PIL import Image

    masks, scores = yolo_person_masks(image, model, device)
    if yolo_mask_quality(masks, scores):
        return None
    alpha = Image.fromarray(masks.bool().any(dim=0).numpy().astype("uint8") * 255)
    buffer = io.BytesIO()
    alpha.save(buffer, format="PNG")
    return buffer.getvalue()


def apply_yolo_training_mask(image, mask):
    from PIL import Image

    if mask is None:
        return image
    with Image.open(io.BytesIO(mask)) as alpha:
        background = Image.new("RGB", image.size, (124, 116, 104))
        return Image.composite(image, background, alpha.resize(image.size, Image.Resampling.NEAREST))


def load_sam2_predictor(device):
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    return SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-tiny", device=device)


def sam2_crop_candidate(image, predictor):
    width, height = image.size
    predictor.set_image(np.array(image))
    masks, scores, _ = predictor.predict(
        point_coords=np.array([[width // 2, height // 2]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.int32),
        multimask_output=True,
    )
    fallback_mask = None
    for index in np.argsort(np.where(np.isfinite(scores), scores, -np.inf))[::-1]:
        mask = masks[index]
        if fallback_mask is None and np.isfinite(scores[index]) and mask.any():
            fallback_mask = mask
        if not np.isfinite(scores[index]) or scores[index] < .75:
            continue
        if not mask[height // 2, width // 2] or not .03 <= mask.mean() <= .8:
            continue
        ys, xs = np.nonzero(mask)
        left, right = int(xs.min()), int(xs.max()) + 1
        top, bottom = int(ys.min()), int(ys.max()) + 1
        margin_x = round((right - left) * .1)
        margin_y = round((bottom - top) * .1)
        box = (max(0, left - margin_x), max(0, top - margin_y),
               min(width, right + margin_x), min(height, bottom + margin_y))
        if (box[2] - box[0]) >= 16 and (box[3] - box[1]) >= 16 and box != (0, 0, width, height):
            return box, mask
    return None, fallback_mask


def transformed_views(image, transform, options, *, include_details=True):
    import torch
    if not options.get("detail_grid"):
        return transform(image)
    views = detail_images(image, options) if include_details else [image]
    return torch.stack([transform(view) for view in views])


def view_scores(model, head, images, transform, full_transform, options, device, training_options=None):
    import torch
    from PIL import ImageOps
    views = [images]
    for name in options.get("views", []):
        if name == "mirror":
            views.append([ImageOps.mirror(image) for image in images])
        elif name == "detail":
            views.append([image.crop((int(image.width*.075), int(image.height*.075),
                                     int(image.width*.925), int(image.height*.925))) for image in images])
        elif name == "full":
            views.append(images)
    scores = []
    for index, view in enumerate(views):
        tf = full_transform if index and options["views"][index-1] == "full" else transform
        x = torch.stack([transformed_views(image, tf, training_options or {}) for image in view]).to(device)
        scores.append(torch.nn.functional.linear(feature_tensor(model, x), head["weight"], head["bias"]).sigmoid().flatten())
    stacked = torch.stack(scores)
    method = options.get("aggregation", "mean")
    return (stacked.min(0).values if method == "min" else stacked.mean(0)).cpu().tolist()


def handle_lora_predict(inp):
    images = training_images(inp)
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
        tf = configured_transform(model, model_name, adapter["resolution"], adapter.get("training_options"))
        full = configured_transform(model, model_name, adapter["resolution"], {"crop": "pad"})
        with torch.inference_mode():
            scores = []
            for item in images:
                image = _decode(item["b64"], adapter.get("training_options", {}).get("processing_version", 1))
                scores.extend(view_scores(model, head, [image], tf, full,
                              inp.get("prediction_options") or {}, device, adapter.get("training_options") or {}))
    return {"scores": {str(item["id"]): score for item, score in zip(images, scores)},
            "prediction_options": inp.get("prediction_options") or {}}


def training_images(inp, progress=None):
    urls = inp.get("dataset_urls")
    if urls is None:
        url = inp.get("dataset_url")
        if not url:
            return inp.get("images") or []
        urls = [url]
    if not isinstance(urls, list) or not urls or any(not isinstance(url, str) for url in urls):
        raise ValueError("Ungültige Liste von Trainingspaketen.")
    images = []
    for index, url in enumerate(urls):
        if progress:
            progress(index, len(urls))
        images.extend(download_training_part(url))
        if progress:
            progress(index + 1, len(urls))
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
    def preparing(stage, count=0, total=1):
        if progress:
            progress({"stage": stage, "epoch": 1, "epochs": int(inp.get("epochs", 8)),
                      "processed": count, "images": total})
    preparing("download")
    images = training_images(inp, lambda count, total: preparing("download", count, total))
    blocks = inp.get("blocks", [8, 9, 10, 11])
    res = int(inp.get("resolution", 336))
    epochs = int(inp.get("epochs", 8))

    early = inp.get("task") in ("lora_train_early", "lora_train_early_sampled", "lora_train_early_sam2", "lora_train_early_yolo", "lora_train_early_sam3")
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
    return train_lora_real(images, blocks, res, epochs, lr, progress, model, validation, head_lr=head_lr, training_options=inp.get("training_options"),
                           sam3_result_url=inp.get("sam3_result_url"))


def train_lora_real(images, blocks, res, epochs, lr, progress=None, model_name="dinov2_vits14", validation=(), head_lr=None, training_options=None, sam3_result_url=None):
    import torch
    import torch.nn as nn
    from random import random
    from torch.utils.data import DataLoader, Dataset
    if progress:
        progress({"stage": "model_load", "epoch": 1, "epochs": epochs, "processed": 0, "images": 1})
    model = copy.deepcopy(load_model(model_name))
    if progress:
        progress({"stage": "setup", "epoch": 1, "epochs": epochs, "processed": 0, "images": 1})
    options = training_options or {}
    tf = configured_transform(model, model_name, res, options)
    train_tf = configured_transform(model, model_name, res, options, training=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam3_probability = options.get("sam3_probability", 0)
    sam3_boxes = {}
    sam3_results = {}
    sam3 = None
    if sam3_probability:
        sam3_tf = configured_transform(model, model_name, res, {**options, "crop": "pad", "crop_scale": 1}, training=True)
    yolo_probability = options.get("yolo_cutout_probability", 0)
    yolo_masks = {}
    yolo = None
    if yolo_probability:
        from ultralytics import YOLO

        yolo = YOLO("yolo11s-seg.pt")
        yolo_tf = configured_transform(model, model_name, res, {**options, "crop": "pad", "crop_scale": 1}, training=True)
    sam2_probability = options.get("sam2_probability", 0)
    sam2_boxes = {}
    predictor = None
    if sam2_probability:
        predictor = load_sam2_predictor(device)

    class DS(Dataset):
        def __init__(self, items, *, training=False):
            self.items = items
            self.training = training
            self.tf = train_tf if training else tf

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            nonlocal sam3
            it = self.items[i]
            probability = options.get("detail_probability", 100) if self.training else 100
            include_details = probability == 100 or random() < probability / 100
            image = _decode(it["b64"], options.get("processing_version", 1))
            transform = self.tf
            if self.training and sam3_probability and random() < sam3_probability / 100:
                if "sam3_crop_box" in it:
                    box = it["sam3_crop_box"]
                else:
                    if i not in sam3_boxes:
                        with torch.inference_mode():
                            if sam3 is None:
                                sam3 = load_sam3(device)
                            masks = segment_sam3_image(image, sam3)["masks"]
                            sam3_boxes[i] = sam3_group_box(image, masks)
                            if sam3_result_url:
                                sam3_results[str(it["id"])] = {**encode_masks(masks), "crop_box": sam3_boxes[i]}
                    box = sam3_boxes[i]
                if box is not None:
                    left, top, right, bottom = box
                    if 0 <= left < right <= image.width and 0 <= top < bottom <= image.height:
                        image = image.crop(box)
                        transform = sam3_tf
            if self.training and yolo is not None and random() < yolo_probability / 100:
                if i not in yolo_masks:
                    with torch.inference_mode():
                        yolo_masks[i] = yolo_training_mask(image, yolo, device)
                if yolo_masks[i] is not None:
                    image = apply_yolo_training_mask(image, yolo_masks[i])
                    transform = yolo_tf
            if self.training and predictor is not None and random() < sam2_probability / 100:
                if i not in sam2_boxes:
                    with torch.inference_mode():
                        sam2_boxes[i], _ = sam2_crop_candidate(image, predictor)
                    predictor.reset_predictor()
                box = sam2_boxes[i]
                if box is not None:
                    image = image.crop(box)
            views = transformed_views(image, transform, options, include_details=include_details)
            return views, torch.tensor(float(it.get("y", 0)))

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
    head = nn.Linear(features * (3 if options.get("detail_grid") else 1), 1).to(device)
    params = [head.weight, head.bias] + [p_ for m in wrapped for p_ in (m.A, m.B)]
    groups = [{"params": params[:2], "lr": lr if head_lr is None else head_lr}]
    if wrapped:
        groups.append({"params": params[2:], "lr": lr})
    opt = torch.optim.AdamW(groups, lr=lr)
    def loss_fn(logits, labels):
        losses = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        weights = torch.where(labels == 0, options.get("negative_weight", 1), 1)
        return (losses * weights).mean()

    ds = DS(images, training=True)
    eval_ds = DS(images)
    hard_factor = options.get("hard_factor", 1)
    sampler = None
    if hard_factor > 1:
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(torch.ones(len(ds), dtype=torch.double), len(ds), replacement=True)
    batch_size = 1 if options.get("detail_grid") else 8
    dl = DataLoader(ds, batch_size=batch_size, shuffle=sampler is None, sampler=sampler)
    validation_loader = DataLoader(DS(validation), batch_size=batch_size) if validation else None
    best_loss, best_epoch, best_params = float("inf"), 0, None
    last_update = float("-inf")
    def report_stage(stage, count, total, force=False):
        nonlocal last_update
        now = time.monotonic()
        if progress and (force or count == total or now - last_update >= 2):
            progress({"stage": stage, "epoch": epoch, "epochs": epochs,
                      "processed": count, "images": total})
            last_update = now

    for epoch in range(1, epochs + 1):
        model.train()
        head.train()
        processed = 0
        report_stage("train", 0, len(ds), force=True)
        for x, y in dl:
            x, y = x.to(device), y.float().to(device)
            opt.zero_grad()
            feat = feature_tensor(model, x)
            loss = loss_fn(head(feat).squeeze(1), y)
            loss.backward()
            opt.step()
            processed += len(y)
            now = time.monotonic()
            if progress and (now - last_update >= 2 or processed == len(ds)):
                progress({"epoch": epoch, "epochs": epochs, "processed": processed, "images": len(ds)})
                last_update = now

        if sampler is not None and epoch < epochs:
            model.eval()
            head.eval()
            weights = []
            report_stage("hard_negative", 0, len(eval_ds), force=True)
            with torch.no_grad():
                for x, y in DataLoader(eval_ds, batch_size=batch_size):
                    scores = head(feature_tensor(model, x.to(device))).sigmoid().squeeze(1).cpu()
                    difficult = (y == 0) & (scores >= options.get("hard_threshold", 0.5))
                    weights.extend(torch.where(difficult, hard_factor, 1).tolist())
                    report_stage("hard_negative", len(weights), len(eval_ds))
            sampler.weights = torch.as_tensor(weights, dtype=torch.double)

        if validation_loader is not None:
            model.eval()
            head.eval()
            total_loss = 0.0
            checked = 0
            report_stage("validation", 0, len(validation), force=True)
            with torch.no_grad():
                for x, y in validation_loader:
                    logits = head(feature_tensor(model, x.to(device))).squeeze(1)
                    total_loss += loss_fn(logits, y.to(device)).item() * len(y)
                    checked += len(y)
                    report_stage("validation", checked, len(validation))
            validation_loss = total_loss / len(validation)
            if not np.isfinite(validation_loss):
                raise RuntimeError("Der Validierungsverlust ist ungültig.")
            if validation_loss < best_loss:
                best_loss, best_epoch = validation_loss, epoch
                best_params = [param.detach().cpu().clone() for param in params]
            if epoch - best_epoch >= 3:
                break

    report_stage("checkpoint", 0, 1, force=True)
    if best_params is not None:
        with torch.no_grad():
            for param, best in zip(params, best_params):
                param.copy_(best)
    model.eval()
    report_stage("training_evaluation", 0, len(eval_ds), force=True)
    with torch.no_grad():
        scores, labels = [], []
        checked = 0
        for x, y in DataLoader(eval_ds, batch_size=batch_size):
            x = x.to(device)
            scores.append(torch.sigmoid(head(feature_tensor(model, x)).squeeze(1)).cpu().numpy())
            labels.append(y.numpy())
            checked += len(y)
            report_stage("training_evaluation", checked, len(eval_ds))
    p = np.concatenate(scores); y = np.concatenate(labels)
    pred = p >= 0.5
    tp = float((pred & (y == 1)).sum()); fp = float((pred & (y == 0)).sum()); fn = float(((~pred) & (y == 1)).sum())
    order = np.argsort(-p)[: min(100, len(p))]
    metrics = {"precision": round(tp / max(1.0, tp + fp), 3), "recall": round(tp / max(1.0, tp + fn), 3),
               "precision_at_100": round(float(y[order].mean()), 3)}

    buf = io.BytesIO()
    report_stage("saving", 0, 1, force=True)
    torch.save({"lora": [{k: v.detach().cpu() for k, v in (("A", m.A), ("B", m.B))} for m in wrapped],
                "head": head.state_dict(), "blocks": list(blocks), "resolution": res, "model": model_name, "training_options": options}, buf)
    encoded = base64.b64encode(buf.getvalue()).decode()
    report_stage("saving", 1, 1, force=True)
    if sam3_result_url and sam3_results:
        import gzip
        import json
        from urllib.request import Request, urlopen

        body = gzip.compress(json.dumps(sam3_results, separators=(",", ":")).encode("utf-8"), compresslevel=4)
        try:
            with urlopen(Request(sam3_result_url, body, method="PUT"), timeout=180) as response:
                if response.status != 200:
                    raise RuntimeError("SAM-3-Masken konnten nicht gespeichert werden.")
        except OSError:
            raise RuntimeError("SAM-3-Masken konnten nicht in R2 gespeichert werden.") from None
    return {"metrics": metrics, "adapter_b64": encoded,
            "blocks": list(blocks), "epochs": epoch, "best_epoch": best_epoch or epoch,
            "early_stopping": validation_loader is not None, "model": model_name, "training_options": options,
            **({"sam3_masks_uploaded": True} if sam3_result_url and sam3_results else {}),
            **({"sam3_crops": {str(images[i]["id"]): box for i, box in sam3_boxes.items()}} if sam3_boxes else {})}


TRAIN_TASKS = ("lora_train", "lora_train_early", "lora_train_sampled", "lora_train_early_sampled",
               "lora_train_sam2", "lora_train_early_sam2", "lora_train_yolo", "lora_train_early_yolo",
               "lora_train_sam3", "lora_train_early_sam3")


def handler(job):
    inp = job.get("input") or {}
    task = inp.get("task", "embed")
    if task == "capabilities":
        result = {"training_tasks": list(TRAIN_TASKS), "processing_version": 2, "pattern_options": True, "pattern_diffusion": True,
                  "pattern_diffusion_safety_checker": False}
        if inp.get("check_sam3_access"):
            from huggingface_hub import hf_hub_download

            try:
                hf_hub_download("facebook/sam3", "config.json")
                result["sam3_access"] = True
            except Exception:
                result["sam3_access"] = False
        return result
    if task == "embed":
        return handle_embed(inp)
    if task == "visualize_yes_diffusion":
        return visualize_yes_diffusion(inp)
    if task in ("visualize_yes", "visualize_yes_configured"):
        return visualize_yes(inp)
    if task == "lora_predict":
        return handle_lora_predict(inp)
    if task in TRAIN_TASKS:
        return handle_lora(inp)
    return {"error": f"unbekannte Aufgabe: {task}"}


def streaming_handler(job):
    inp = job.get("input") or {}
    if inp.get("task") not in (*TRAIN_TASKS, "visualize_yes", "visualize_yes_configured", "visualize_yes_diffusion") or FAKE:
        yield handler(job)
        return

    from queue import Queue, Empty
    from threading import Thread

    updates = Queue()

    def train():
        try:
            if inp.get("task") == "visualize_yes_diffusion":
                function = visualize_yes_diffusion
            elif inp.get("task") in ("visualize_yes", "visualize_yes_configured"):
                function = visualize_yes
            else:
                function = handle_lora
            result = function(inp, lambda progress: updates.put(("progress", progress)))
            updates.put(("result", result))
        except Exception as error:
            updates.put(("error", error))

    Thread(target=train, daemon=True).start()
    latest = None
    while True:
        try:
            kind, value = updates.get(timeout=10)
        except Empty:
            if latest is not None:
                yield {"training_progress": {**latest, "heartbeat": True}}
            continue
        if kind == "error":
            raise value
        if kind == "result":
            yield value
            return
        latest = value
        yield {"training_progress": value}


if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": streaming_handler, "return_aggregate_stream": True})
