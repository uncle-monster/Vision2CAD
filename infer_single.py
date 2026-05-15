#!/usr/bin/env python3
"""
End-to-end single-image inference for Img2CAD.

Takes a single image (e.g., phone photo of a chair/table/storage furniture),
runs both stages of the Img2CAD pipeline, and outputs an editable CAD model (.obj).

Usage:
    python infer_single.py --image path/to/photo.jpg --category chair

Stages:
    1. LlamaFT (VLM): image -> discrete CAD structure (h5)
    2. TrAssembler (GMFlow): image + h5 -> continuous parameters -> OBJ
"""

import os
import sys
import argparse
import time
import re
import ast
import json
from copy import deepcopy
from pathlib import Path
from math import sqrt, sin, cos, tan, radians

import numpy as np
import h5py
import cv2
import torch
from PIL import Image
from tqdm import tqdm
from enum import Enum

from torchvision.transforms import ToTensor, Normalize, Compose

# Add project root to path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)
sys.path.insert(0, os.path.join(project_dir, 'GMFlow'))

from qwen_vl_utils import process_vision_info
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from huggingface_hub import login

from utils.cmd_to_vec import cmd2vec
from cadlib.macro import ALL_COMMANDS, CMD_ARGS_MASK, N_ARGS, EOS_IDX, EOS_VEC, SOL_VEC, PAD_VAL
from cadlib.visualize import vec2CADsolid
from utils.io import cad_to_obj, cad_to_step
from utils.misc import render_nvdiffrast_rgb

import clip
from omegaconf import OmegaConf


# ============================================================================
# Text-to-h5 conversion utilities (from LlamaFT/infer.py)
# ============================================================================

EVAL_FTN = {
    'sin': lambda x: sin(radians(x)),
    'cos': lambda x: cos(radians(x)),
    'tan': lambda x: tan(radians(x)),
    'sqrt': sqrt
}

CMD_VARS = ['NewBody', 'Join', 'Intersect', 'Cut',
            'OneSided', 'TwoSided', 'Symmetric']

COMMENT_STARTS = ['//', '#', ';', '--']
SOL_STARTS = ('<SOL>', '$<SOL>$')
CUT_STARTS = ('<CUT>', '$<CUT>$')
CMD_STARTS = ('L:', 'A:', 'R:', 'E:')


class CMD_TYPE(Enum):
    SOL = 0
    CUT = 1
    EXTRUDE = 2
    LINE = 3
    ARC = 4
    CIRCLE = 5


cmd_mapping = {
    "<SOL>": CMD_TYPE.SOL,
    "<CUT>": CMD_TYPE.CUT,
    "E": CMD_TYPE.EXTRUDE,
    "L": CMD_TYPE.LINE,
    "A": CMD_TYPE.ARC,
    "R": CMD_TYPE.CIRCLE,
    "C": CMD_TYPE.CIRCLE,
}


def divide_lines_into_blocks(lines):
    blocks = []
    current_block = []
    for line in lines:
        if line.strip() == "":
            if current_block:
                blocks.append("\n".join(current_block))
                current_block = []
        else:
            current_block.append(line)
    if current_block:
        blocks.append("\n".join(current_block))
    return blocks


def filter_lines(lines):
    res = []
    for line in lines:
        if line.strip().startswith("```"):
            res.append('')
        else:
            line = line.replace('\\', '')
            if '<SOL>' in line and line.split('<SOL>')[1].strip() != '':
                res.append('<SOL>')
                res.append(line.split('<SOL>')[1].strip())
            else:
                res.append(line)
    return res


def filter_blocks(blocks):
    res = []
    for i, block in enumerate(blocks):
        if '<SOL>' in block:
            if i > 0 and block.startswith('<SOL>') and '<SOL>' not in blocks[i-1]:
                res.append(blocks[i-1] + '\n' + block)
            else:
                res.append(block)
    return res


def parse_cmd_line(line):
    line = re.sub(r'[<>$]', '', line)
    if line.startswith("SOL"):
        return CMD_TYPE.SOL, None
    elif line.startswith("CUT"):
        return CMD_TYPE.CUT, None
    pattern = re.compile(r'(\w)\s*')
    match = pattern.search(line)
    if match:
        cmd_str = match.groups()[0]
        cmd_type = cmd_mapping.get(cmd_str, None)
        if cmd_type is not None:
            return cmd_type, None
    return None, None


def extract_base_name(text):
    match = re.search(r'(.*)(\s+)\D*(\d+)', text)
    if match:
        name_part = match.group(1).strip()
    else:
        name_part = text.strip()
    name_part = name_part.lower()
    return name_part


def remove_leading_ending_symbols(text):
    text = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', text)
    return text.strip()


def replace_digits_with_zero(input_string):
    return re.sub(r'\d', '0', input_string)


def remove_parentheses(line):
    return re.sub(r'\([^()]*\)', '', line)


def remove_keywords(line):
    line = line.lower()
    line = line.replace("sketch and extrusion", "").strip()
    line = line.replace("sketch of", "").strip()
    line = line.replace("sketch for", "").strip()
    line = line.replace("extrusion of", "").strip()
    line = line.replace("start of", "").strip()
    words = line.split()
    filtered_words = [word for word in words if word.lower() not in {"sketching", "sketch", "extrude", "profile", "begin", "-"}]
    return ' '.join(filtered_words)


def extract_comments(block, blacklists=('define', 'model', 'program')):
    comments = []
    res = []
    cand_seps = ['#', '//']
    for line in block:
        sep_locations = []
        seps = []
        for sep in cand_seps:
            if sep in line:
                sep_locations.append(line.index(sep))
                seps.append(sep)
        sep = seps[np.argmin(sep_locations)] if sep_locations else None
        if sep is not None:
            comment = line.split(sep, 1)[1].strip().lower()
            if not any(bl in comment for bl in blacklists):
                comments.append(remove_parentheses(remove_keywords(comment)))
            line = line.split(sep, 1)[0].strip('#/')
        if line.strip() != '':
            res.append(line.strip())
    return comments, res


def get_vars(args):
    vars = set([])
    if not args:
        return vars
    for arg in args:
        try:
            for node in ast.walk(ast.parse(arg)):
                if isinstance(node, ast.Name) and node.id not in EVAL_FTN.keys() | CMD_VARS:
                    vars.add(node.id)
        except SyntaxError:
            pass
    return vars


def eval_vars(args, var_vals):
    if not args:
        return args
    for i in range(len(args)):
        if args[i] not in CMD_VARS:
            args[i] = eval(args[i], EVAL_FTN, var_vals)
    return args


def get_part_to_cad(lines):
    lines = filter_lines(lines)
    blocks = divide_lines_into_blocks(lines)
    blocks = filter_blocks(blocks)
    blacklists = ('define', 'model', 'program')
    cmd_blocks = {}

    for block in blocks:
        comments, block_lines = extract_comments(block.splitlines(), blacklists)
        if len(comments) > 0:
            comment = comments[0]
            cmds = []
            for j, line in enumerate(block_lines):
                cmd_type, _ = parse_cmd_line(line)
                if cmd_type == CMD_TYPE.SOL:
                    cmds.append([CMD_TYPE.SOL, None])
                    continue
                if cmd_type == CMD_TYPE.EXTRUDE:
                    cmds.append([CMD_TYPE.EXTRUDE, ['0', '1.', '0', '1.', '0', '0', '0', '0', '0', '1.', '0', 'NewBody', 'OneSided']])
                    continue
                if cmd_type == CMD_TYPE.LINE:
                    nargs = 2
                elif cmd_type == CMD_TYPE.ARC:
                    nargs = 4
                elif cmd_type == CMD_TYPE.CIRCLE:
                    nargs = 3
                cmds.append([cmd_type, ['0'] * nargs])
            cmd_blocks[comment] = cmds

    cmd_convert_map = {
        CMD_TYPE.SOL: '<SOL>',
        CMD_TYPE.EXTRUDE: 'E',
        CMD_TYPE.LINE: 'L',
        CMD_TYPE.ARC: 'A',
        CMD_TYPE.CIRCLE: 'R',
    }
    cmd_blocked_converted = {}
    for label, cmd_block in cmd_blocks.items():
        cmd_blocked_converted[label] = []
        arc_loc = -1
        for cmd in cmd_block:
            cmd_type, args = cmd
            if cmd_type == CMD_TYPE.ARC:
                arc_loc = len(cmd_blocked_converted[label])
            if cmd_type in cmd_convert_map:
                cmd_blocked_converted[label].append([cmd_convert_map[cmd_type], [str(arg) for arg in args] if args is not None else None])
        if arc_loc != -1 and arc_loc < len(cmd_blocked_converted[label]) - 2:
            cmd_block_new = deepcopy(cmd_blocked_converted[label])
            cmd_blocked_converted[label] = cmd_block_new[0:1] + cmd_block_new[arc_loc+1:-1] \
                + cmd_block_new[1:arc_loc+1] + cmd_block_new[-1:]
    return cmd_blocked_converted


def get_part_to_vec(part_to_cad, use_normal=True):
    part_to_vec = {}
    for part in part_to_cad.keys():
        cad = part_to_cad[part]
        if cad:
            full_vec = np.array([cmd2vec(*cmd, use_normal=use_normal) for cmd in cad])
            part_to_vec[part] = full_vec
    return part_to_vec


def process_text_to_h5(text_response, output_path):
    try:
        part_to_cad = get_part_to_cad(text_response.splitlines())
        part_to_vec = get_part_to_vec(part_to_cad)
        if part_to_vec:
            full_vec = np.concatenate([part_to_vec[part] for part in part_to_vec.keys()])
            with h5py.File(output_path, 'w') as fp:
                fp.create_dataset("vec", data=full_vec, dtype=float)
                for part in part_to_vec.keys():
                    fp.create_dataset(part, data=part_to_vec[part], dtype=float)
            return True
        else:
            return False
    except Exception as e:
        print(f"Error processing text to h5: {e}")
        return False


# ============================================================================
# Image preprocessing for phone photos
# ============================================================================

def preprocess_phone_photo(image_path, output_size=512, model_name='isnet-general-use', morph_kernel=5):
    """
    Preprocess a phone photo to match the training distribution:
    1. Remove background (using rembg with isnet-general-use)
    2. Apply morphological repair to recover thin structures (e.g., chair legs)
    3. Center the object on white background
    4. Resize to a standard size

    Returns the preprocessed PIL Image.
    """
    try:
        from rembg import remove, new_session
    except ImportError:
        print("Warning: rembg not installed. Skipping background removal.")
        print("Install with: pip install rembg")
        img = Image.open(image_path).convert("RGB")
        img = img.resize((output_size, output_size))
        return img

    img = Image.open(image_path).convert("RGB")

    # Remove background with specified model
    print(f"  Using rembg model: {model_name}")
    session = new_session(model_name)
    img_nobg = remove(img, session=session)

    # Convert to numpy for processing
    img_np = np.array(img_nobg)

    # Extract alpha mask
    if img_np.shape[-1] == 4:  # RGBA
        alpha = img_np[..., 3]
    else:
        gray = cv2.cvtColor(img_np[..., :3], cv2.COLOR_RGB2GRAY)
        alpha = np.where(gray < 250, 255, 0).astype(np.uint8)

    # Morphological repair: recover thin structures (chair legs, etc.)
    if morph_kernel > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        # Binarize
        _, alpha_bin = cv2.threshold(alpha, 30, 255, cv2.THRESH_BINARY)
        # Closing: connect broken thin regions
        alpha_bin = cv2.morphologyEx(alpha_bin, cv2.MORPH_CLOSE, kernel, iterations=2)
        # Dilation: expand to recover edges
        alpha_bin = cv2.dilate(alpha_bin, kernel, iterations=1)
        # Keep largest connected component to remove noise
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(alpha_bin, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            alpha_bin = np.where(labels == largest_label, 255, 0).astype(np.uint8)
        alpha = alpha_bin

    mask = alpha > 30
    coords = np.argwhere(mask)
    if len(coords) == 0:
        print("Warning: Could not find object in image. Using original image.")
        img = img.resize((output_size, output_size))
        return img

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Crop to bounding box with some padding
    pad = int(0.05 * max(y_max - y_min, x_max - x_min))
    y_min = max(0, y_min - pad)
    x_min = max(0, x_min - pad)
    y_max = min(img_np.shape[0], y_max + pad)
    x_max = min(img_np.shape[1], x_max + pad)

    obj = img_np[y_min:y_max, x_min:x_max]

    # Create white background and center the object
    bg = np.ones((output_size, output_size, 3), dtype=np.uint8) * 255
    obj_h, obj_w = obj.shape[:2]

    # Scale to fit while maintaining aspect ratio
    scale = min((output_size - 40) / obj_h, (output_size - 40) / obj_w)
    new_h, new_w = int(obj_h * scale), int(obj_w * scale)
    obj_resized = cv2.resize(obj, (new_w, new_h))

    # Center on white background
    y_offset = (output_size - new_h) // 2
    x_offset = (output_size - new_w) // 2

    if obj_resized.shape[-1] == 4:
        alpha_resized = obj_resized[..., 3:4] / 255.0
        rgb_resized = obj_resized[..., :3]
        bg_region = bg[y_offset:y_offset+new_h, x_offset:x_offset+new_w]
        blended = (rgb_resized * alpha_resized + bg_region * (1 - alpha_resized)).astype(np.uint8)
        bg[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = blended
    else:
        bg[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = obj_resized[..., :3]

    return Image.fromarray(bg)


# ============================================================================
# Stage 1: LlamaFT inference
# ============================================================================

def load_llama_model(model_id, adapter_path):
    """Load the Llama 3.2 Vision model with LoRA adapter."""
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)

    print(f"Loading base model: {model_id}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        local_files_only=True
    )

    print(f"Loading adapter: {adapter_path}")
    model.load_adapter(adapter_path)
    model.eval()
    return model, processor


def generate_description(prompt, sample_image, model, processor, max_new_tokens=1024):
    """Generate CAD description from image using fine-tuned Llama model."""
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image", "image": sample_image},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        add_special_tokens=False,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        top_p=1.0,
        temperature=1.0,
        do_sample=False
    )
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output_text[0]


def run_stage1(image, prompt, model, processor, output_path, max_new_tokens=1024, progress_callback=None):
    """Run Stage 1: image -> h5 (discrete CAD structure)."""
    if not progress_callback:
        progress_callback = lambda stage, pct, msg: print(f"  [{stage}] {msg}")

    print("\n[Stage 1] Generating discrete CAD structure...")
    start_time = time.time()

    progress_callback('stage1', 15.0, 'VLM 正在生成 CAD 结构...')
    text_response = generate_description(prompt, image, model, processor, max_new_tokens=max_new_tokens)

    print(f"  Generated text ({len(text_response)} chars):")
    print(f"  {text_response[:300]}...")

    progress_callback('stage1', 38.0, '正在解析 CAD 结构...')
    success = process_text_to_h5(text_response, output_path)
    elapsed = time.time() - start_time

    if success:
        print(f"  Stage 1 complete in {elapsed:.1f}s. h5 saved to {output_path}")
        progress_callback('stage1', 42.0, 'Stage 1 完成')
    else:
        print(f"  Stage 1 failed: could not parse CAD structure from text")
        progress_callback('stage1', 42.0, 'Stage 1 失败：无法解析 CAD 结构')

    return success, text_response


# ============================================================================
# Stage 2: TrAssembler inference
# ============================================================================

def get_clip_features(clip_model, texts, device):
    """Get CLIP text features for given texts."""
    text_tokens = clip.tokenize(texts).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features


def create_sample(h5, part_names, image, do_retrieval, name_embeddings_db, clip_model):
    """Create a sample dict for TrAssembler inference (from eval.py)."""
    part_cad_vec = []
    for name in part_names:
        part_vec = h5[name]
        while isinstance(part_vec, h5py._hl.group.Group):
            for k, v in part_vec.items():
                part_vec = v
                break
        part_cad_vec.append(part_vec[:])

    base_part_names = [extract_base_name(remove_leading_ending_symbols(part.replace('#', ' '))) for part in part_names]
    part_name_embeddings = get_clip_features(clip_model, base_part_names, 'cuda').cpu().numpy()

    if do_retrieval:
        for i in range(len(part_name_embeddings)):
            dist = np.linalg.norm(part_name_embeddings[i] - name_embeddings_db, axis=-1)
            nearest_idx = dist.argmin()
            part_name_embeddings[i] = name_embeddings_db[nearest_idx]

    part_names_no_digit = np.array([replace_digits_with_zero(n) for n in part_names])
    part_index = np.zeros(len(part_names))
    for name in np.unique(part_names_no_digit):
        group_idx = np.where(part_names_no_digit == name)[0]
        part_index[group_idx] = np.arange(len(group_idx))

    sample = {
        "part_cad": part_cad_vec,
        "part_name_emb": part_name_embeddings,
        "part_index": part_index,
        "img_tex": image,
        "img_non_tex": image,
        "data_id": 0,
    }
    return sample


def collate_fn_single(sample):
    """Collate function for a single sample (adapted from dataset.py collate_fn)."""
    data = [sample]
    max_N_part = max(len(d['part_cad']) for d in data)
    max_N_lines = max(max(len(part) for part in d['part_cad']) for d in data)

    batch_data = {
        'padded_cad': [], 'command_mask': [], 'part_name_emb': [],
        'part_index': [], 'img_tex': [], 'img_non_tex': [], 'data_id': [], 'args_mask': []
    }

    for item in data:
        part_cads = item['part_cad']
        part_name_embs = item['part_name_emb']

        padded_cads = [np.concatenate([cad, EOS_VEC[np.newaxis].repeat(max_N_lines - len(cad), axis=0)]) for cad in part_cads]
        masks = [np.concatenate([np.ones(len(cad), dtype=int), np.zeros(max_N_lines - len(cad), dtype=int)]) for cad in part_cads]

        if len(padded_cads) < max_N_part:
            extra_padding = max_N_part - len(padded_cads)
            padded_cads += [EOS_VEC[np.newaxis].repeat(max_N_lines, axis=0)] * extra_padding
            masks += [np.zeros(max_N_lines, dtype=int)] * extra_padding
            part_name_embs = np.concatenate([part_name_embs, np.zeros((extra_padding, part_name_embs.shape[-1]))])
            item['part_index'] = np.concatenate([item['part_index'], np.zeros(extra_padding)])

        padded_cads = np.stack(padded_cads)
        args_mask = (np.abs(padded_cads[..., 1:] + 1) > 1e-4).astype(int)
        ext_mask = (padded_cads[..., 0] + 0.5).astype(int) == 5
        args_mask[ext_mask, -3:] = 0
        args_mask[ext_mask, -5] = 0
        padded_cads[ext_mask, -5] = 1.0

        padded_cads[(padded_cads[..., 0] + 0.5).astype(int) == 1, 3] *= np.pi / 180.
        args_mask[(padded_cads[..., 0] + 0.5).astype(int) == 1, 3] = 0

        batch_data['args_mask'].append(args_mask)
        batch_data['padded_cad'].append(padded_cads)
        batch_data['command_mask'].append(np.stack(masks))
        batch_data['part_name_emb'].append(part_name_embs)
        batch_data['part_index'].append(item['part_index'])
        batch_data['img_tex'].append(item['img_tex'])
        batch_data['img_non_tex'].append(item['img_non_tex'])
        batch_data['data_id'].append(item['data_id'])

    for key in ['padded_cad', 'command_mask', 'part_name_emb', 'part_index', 'args_mask']:
        batch_data[key] = torch.tensor(np.stack(batch_data[key]), dtype=torch.float32)

    batch_data['img_tex'] = torch.tensor(np.stack(batch_data['img_tex']), dtype=torch.float32)
    batch_data['img_non_tex'] = torch.tensor(np.stack(batch_data['img_non_tex']), dtype=torch.float32)
    batch_data['data_id'] = np.stack(batch_data['data_id'])

    args = batch_data['padded_cad'][..., 1:]
    args_mask = batch_data['args_mask']

    return {
        'command': batch_data['padded_cad'][..., 0].round().long(),
        'command_mask': batch_data['command_mask'],
        'args': args,
        'args_mask': args_mask,
        'part_name_embedding': batch_data['part_name_emb'],
        'part_index': batch_data['part_index'],
        'img_tex': batch_data['img_tex'],
        'img_non_tex': batch_data['img_non_tex'],
        'data_id': batch_data['data_id']
    }


def run_stage2(h5_path, image_pil, model, clip_model, name_embeddings_db, do_retrieval, out_dir, progress_callback=None):
    """Run Stage 2: h5 + image -> CAD parameters -> OBJ."""
    if not progress_callback:
        progress_callback = lambda stage, pct, msg: print(f"  [{stage}] {msg}")

    print("\n[Stage 2] Predicting continuous CAD parameters...")

    image_transform = Compose([ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    # Convert PIL to numpy for OpenCV-style processing
    image_np = np.array(image_pil)
    image_tensor = image_transform(image_np)

    progress_callback('stage2', 48.0, '正在准备输入数据...')
    h5 = h5py.File(h5_path, 'r')
    part_names = list(sorted([n for n in h5 if n != 'vec' and '_bbox' not in n]))

    # Handle h5 hierarchy: part names can be groups
    flat_part_names = []
    for name in part_names:
        item = h5[name]
        if isinstance(item, h5py._hl.group.Group):
            for sub_name in item.keys():
                flat_part_names.append(f"{name}/{sub_name}")
        else:
            flat_part_names.append(name)

    if not flat_part_names:
        # Fall back to using 'vec' dataset if no parts found
        print("  Warning: No part datasets found in h5. CAD generation may fail.")
        flat_part_names = part_names if part_names else ['unknown']

    sample = create_sample(h5, flat_part_names, image_tensor, do_retrieval, name_embeddings_db, clip_model)
    h5.close()

    batch = collate_fn_single(sample)
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].cuda()

    print(f"  Parts: {len(flat_part_names)}, Commands: {batch['command'].shape[2]}")

    # Run diffusion sampling
    progress_callback('stage2', 52.0, '正在进行扩散采样...')
    try:
        pred_args = model.sample(batch)
    except Exception as e:
        print(f"  Error during diffusion sampling: {e}")
        progress_callback('stage2', 52.0, f'采样出错: {e}')
        return False

    progress_callback('stage2', 80.0, '正在生成 CAD 实体...')
    # Denormalize
    model.denormalize(pred_args, batch['args_mask'])

    # Build CAD vector
    cad_vec = []
    cmd = batch['command'][0]
    cmd_mask = batch['command_mask'][0]
    arg_mask = batch['args_mask'][0]
    batch_args = batch['args'][0]
    batch_args[arg_mask > 0] = pred_args[0][arg_mask > 0]

    for p in range(cmd_mask.shape[0]):
        if cmd_mask[p][0] > 0:
            for l in range(cmd_mask.shape[1]):
                if cmd_mask[p][l] > 0:
                    arg = batch_args[p][l].cpu().numpy()
                    cmd_type = int(cmd[p][l].item())
                    if cmd_type == 1:  # Arc
                        arg[2] /= np.pi / 180.
                        arg[3] = 1.0
                    cad_vec.append(np.concatenate([np.array([cmd_type]), arg]))

    if len(cad_vec) == 0:
        print("  Error: No CAD vectors generated")
        progress_callback('stage2', 80.0, '未生成 CAD 向量')
        return False

    cad_vec = np.array(cad_vec)
    print(f"  Generated {len(cad_vec)} CAD commands")

    # Generate CAD solid
    try:
        full_shape = vec2CADsolid(cad_vec, is_numerical=False)
        obj_path = os.path.join(out_dir, 'final.obj')
        step_path = os.path.join(out_dir, 'final.step')
        cad_to_obj(full_shape, obj_path)
        cad_to_step(full_shape, step_path)
        print(f"  OBJ saved to {obj_path}")
        print(f"  STEP saved to {step_path}")

        progress_callback('stage2', 90.0, '正在渲染预览图...')
        # Render preview
        try:
            rgb_pred = render_nvdiffrast_rgb(obj_path)
            cv2.imwrite(os.path.join(out_dir, 'final.png'), rgb_pred[..., ::-1])
            print(f"  Preview saved to {os.path.join(out_dir, 'final.png')}")
        except Exception as e:
            print(f"  Warning: Could not render preview: {e}")

        # Save h5
        with h5py.File(os.path.join(out_dir, 'final.h5'), 'w') as h5_out:
            h5_out.create_dataset('vec', data=cad_vec)

    except Exception as e:
        print(f"  Error generating CAD solid: {e}")
        progress_callback('stage2', 90.0, f'CAD 实体生成出错: {e}')
        return False

    progress_callback('stage2', 95.0, 'CAD 模型生成完成')
    return True


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="End-to-end single-image Img2CAD inference")
    parser.add_argument('--image', type=str, required=True, help="Path to input image (phone photo, PNG/JPG)")
    parser.add_argument('--category', type=str, default='chair', help="Object category (chair/table/storagefurniture)")
    parser.add_argument('--output_dir', type=str, default=None, help="Output directory (default: data/output/single_inference/)")
    parser.add_argument('--adapter_path', type=str, default=None, help="Path to LlamaFT LoRA adapter")
    parser.add_argument('--model_dir', type=str, default=None, help="Path to TrAssembler checkpoint directory")
    parser.add_argument('--num_tokens', type=int, default=1024, help="Max tokens for VLM generation")
    parser.add_argument('--hf_token', type=str, default=None, help="Hugging Face token")
    parser.add_argument('--no_background_removal', action='store_true', help="Skip background removal")
    parser.add_argument('--rembg_model', type=str, default='isnet-general-use', help="rembg model (u2net, u2netp, isnet-general-use, sam)")
    parser.add_argument('--morph_kernel', type=int, default=5, help="Morphological kernel size for mask repair (0 to disable)")
    parser.add_argument('--save_text', action='store_true', help="Save intermediate text response")
    parser.add_argument('--text_emb_retrieval', action='store_true', help="Use text embedding retrieval for part names")

    args = parser.parse_args()

    # Setup paths
    if args.output_dir is None:
        args.output_dir = os.path.join(project_dir, 'data', 'output', 'single_inference')
    os.makedirs(args.output_dir, exist_ok=True)

    image_name = Path(args.image).stem
    obj_out_dir = os.path.join(args.output_dir, image_name)
    os.makedirs(obj_out_dir, exist_ok=True)

    # HF auth (optional, only needed if models need to be downloaded)
    hf_token = args.hf_token or os.getenv('HF_TOKEN')
    if hf_token:
        try:
            login(hf_token)
        except Exception as e:
            print(f"Warning: HuggingFace login failed ({e}). Proceeding in offline mode.")

    # Auto-detect paths
    if args.adapter_path is None:
        adapter_search_dir = os.path.join(project_dir, 'data', 'ckpts', 'llamaft', args.category)
        adapter_checkpoints = sorted(Path(adapter_search_dir).glob('checkpoint-*') if os.path.exists(adapter_search_dir) else [])
        if not adapter_checkpoints:
            adapter_checkpoints = sorted(Path(os.path.join(project_dir, 'data', 'ckpts', 'hf', 'llamaft', args.category)).glob('checkpoint-*'))
        if adapter_checkpoints:
            args.adapter_path = str(adapter_checkpoints[-1])
            print(f"Auto-detected adapter: {args.adapter_path}")
        else:
            raise FileNotFoundError(f"No adapter found for category '{args.category}'")

    if args.model_dir is None:
        model_dir = os.path.join(project_dir, 'data', 'ckpts', 'trassembler', args.category)
        if not os.path.exists(model_dir):
            model_dir = os.path.join(project_dir, 'data', 'ckpts', 'hf', 'trassembler', args.category)
        args.model_dir = model_dir
        print(f"Auto-detected model dir: {args.model_dir}")

    # ================================================================
    # Step 0: Preprocess image
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Img2CAD Single Image Inference")
    print(f"Image: {args.image}")
    print(f"Category: {args.category}")
    print(f"{'='*60}")

    if args.no_background_removal:
        image_pil = Image.open(args.image).convert("RGB")
        print("Skipping background removal")
    else:
        print("\n[Preprocessing] Removing background and centering object...")
        image_pil = preprocess_phone_photo(args.image, model_name=args.rembg_model, morph_kernel=args.morph_kernel)
        print(f"  Preprocessed image size: {image_pil.size}")

    # Save preprocessed image
    preprocessed_path = os.path.join(obj_out_dir, 'preprocessed.png')
    image_pil.save(preprocessed_path)

    # ================================================================
    # Stage 1: LlamaFT (image -> h5)
    # ================================================================
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    llama_model, processor = load_llama_model(model_id, args.adapter_path)

    prompt_path = os.path.join(project_dir, 'LlamaFT', 'prompt.txt')
    prompt = open(prompt_path).read()

    h5_path = os.path.join(obj_out_dir, 'stage1_output.h5')
    success, text_response = run_stage1(image_pil, prompt, llama_model, processor, h5_path, max_new_tokens=args.num_tokens)

    if args.save_text:
        text_path = os.path.join(obj_out_dir, 'stage1_text.txt')
        with open(text_path, 'w') as f:
            f.write(text_response)

    if not success:
        print("\nERROR: Stage 1 failed. Cannot continue.")
        sys.exit(1)

    # Free Llama model memory
    del llama_model, processor
    torch.cuda.empty_cache()

    # ================================================================
    # Stage 2: TrAssembler (h5 + image -> OBJ)
    # ================================================================
    ckpt_path = os.path.join(args.model_dir, 'checkpoints', 'last.ckpt')

    # Try to load config from Hydra output
    config_path = os.path.join(args.model_dir, '.hydra', 'config.yaml')
    if not os.path.exists(config_path):
        # Fall back to default config
        config_path = os.path.join(project_dir, 'TrAssembler', 'config.yaml')
    config = OmegaConf.load(config_path)

    # Ensure category is set correctly
    if config.category != args.category:
        print(f"  Note: Config category '{config.category}' differs from requested '{args.category}', using config value")

    # Add TrAssembler to path for its internal imports
    sys.path.insert(0, os.path.join(project_dir, 'TrAssembler'))
    from TrAssembler.model import GMFlowModel

    model = GMFlowModel.load_from_checkpoint(
        ckpt_path,
        args=config,
        embed_dim=config.network.embed_dim,
        num_heads=config.network.num_heads,
        dropout=config.network.dropout,
        bias=True,
        scaling_factor=1.,
        args_range=np.array([-1., 1.])
    ).cuda().eval()

    # Load CLIP
    torch.set_grad_enabled(False)
    clip_model, _ = clip.load("ViT-B/32", device='cuda')
    clip_model.eval()

    # Build name embedding database
    partnet_common_path = os.path.join(project_dir, 'data', f'partnet2common_{args.category}.json')
    if os.path.exists(partnet_common_path):
        name_db = list(set(json.load(open(partnet_common_path)).values()))
        name_db = [name[:name.find('/')] if '/' in name else name for name in name_db]
        name_embeddings_db = clip_model.encode_text(clip.tokenize(name_db).to('cuda')).cpu().numpy()
        name_embeddings_db /= np.linalg.norm(name_embeddings_db, axis=1, keepdims=True)
        print(f"  Name database: {len(name_db)} canonical part names")
    else:
        name_embeddings_db = np.zeros((0,))
        print("  Warning: No part name database found")

    success = run_stage2(h5_path, image_pil, model, clip_model, name_embeddings_db, args.text_emb_retrieval, obj_out_dir)

    if success:
        print(f"\n{'='*60}")
        print(f"DONE! Output files saved to: {obj_out_dir}")
        print(f"  - final.step: Editable CAD solid (import in Fusion 360/SolidWorks)")
        print(f"  - final.obj: 3D mesh (import in Blender/MeshLab)")
        print(f"  - final.png: Preview render")
        print(f"  - final.h5: CAD vector data")
        print(f"  - stage1_output.h5: Stage 1 intermediate output")
        print(f"  - preprocessed.png: Preprocessed input image")
        print(f"{'='*60}")
    else:
        print("\nERROR: Stage 2 failed.")
        sys.exit(1)


if __name__ == '__main__':
    main()
