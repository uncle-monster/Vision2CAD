"""
Task state machine and inference runner for Img2CAD web server.

Uses a background thread to run the inference pipeline from infer_single.py,
with progress callbacks that feed both the WebSocket and the job state dict.
"""

import os
import sys
import time
import json
import uuid
import shutil
import threading
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# Ensure project root is importable
_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_dir)
sys.path.insert(0, os.path.join(_project_dir, 'GMFlow'))
sys.path.insert(0, os.path.join(_project_dir, 'TrAssembler'))

# ---------------------------------------------------------------------------
# Job state
# ---------------------------------------------------------------------------

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Job cleanup
# ---------------------------------------------------------------------------

JOB_MAX_AGE_SECONDS = 3600  # 1 hour
CLEANUP_INTERVAL_SECONDS = 600  # 10 minutes
_cleanup_thread_started = False


def _start_cleanup_thread():
    """Start a background daemon thread that periodically removes old jobs."""
    global _cleanup_thread_started
    if _cleanup_thread_started:
        return
    _cleanup_thread_started = True

    def _cleanup_loop():
        while True:
            time.sleep(CLEANUP_INTERVAL_SECONDS)
            _cleanup_old_jobs()

    t = threading.Thread(target=_cleanup_loop, daemon=True)
    t.start()


def _cleanup_old_jobs():
    now = time.time()
    with JOBS_LOCK:
        stale_ids = [
            jid for jid, job in JOBS.items()
            if now - job.get('last_accessed', job['created_at']) > JOB_MAX_AGE_SECONDS
        ]
    for jid in stale_ids:
        with JOBS_LOCK:
            if jid in JOBS:
                del JOBS[jid]
        out_dir = os.path.join(_project_dir, 'data', 'output', 'web_inference', jid)
        if os.path.isdir(out_dir):
            try:
                shutil.rmtree(out_dir)
            except Exception:
                pass

# ---------------------------------------------------------------------------
# Model cache (cached per category so we don't reload for every request)
# ---------------------------------------------------------------------------

_model_cache: dict[str, dict] = {}
_model_cache_lock = threading.Lock()


def _get_cached_models(category: str):
    with _model_cache_lock:
        return _model_cache.get(category, {})


def _set_cached_models(category: str, models: dict):
    with _model_cache_lock:
        _model_cache[category] = models


# ---------------------------------------------------------------------------
# Progress callback contract
# ---------------------------------------------------------------------------

def make_progress_callback(job_id: str):
    """Return a callback(stage: str, progress: float, message: str)."""

    def cb(stage: str, progress: float, message: str):
        with JOBS_LOCK:
            if job_id in JOBS:
                job = JOBS[job_id]
                prev_stage = job.get('stage', '')
                job.update({
                    'stage': stage,
                    'progress': min(max(progress, 0.0), 100.0),
                    'message': message,
                    'last_accessed': time.time(),
                })
                # Record stage entry time when transitioning to a new stage
                if stage != prev_stage:
                    stage_times = job.setdefault('stage_times', {})
                    if stage not in stage_times:
                        stage_times[stage] = time.time() - job['created_at']

    return cb


# ---------------------------------------------------------------------------
# Stage 1 helpers
# ---------------------------------------------------------------------------

STAGE1_MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"


def _load_llama_model(adapter_path: str, *, progress_cb=None):
    from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

    if progress_cb:
        progress_cb('preprocessing', 5.0, '加载 Llama 模型...')

    processor = AutoProcessor.from_pretrained(STAGE1_MODEL_ID, local_files_only=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        STAGE1_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        local_files_only=True,
    )
    model.load_adapter(adapter_path)
    model.eval()
    return model, processor


def _run_stage1_vlm(image_pil, prompt, model, processor, max_new_tokens=1024, *, progress_cb=None):
    from qwen_vl_utils import process_vision_info

    if progress_cb:
        progress_cb('stage1', 10.0, 'VLM 正在推理...')

    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image", "image": image_pil},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs,
        padding=True, add_special_tokens=False, return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    generated_ids = model.generate(
        **inputs, max_new_tokens=max_new_tokens,
        top_p=1.0, temperature=1.0, do_sample=False,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )
    return output_text[0]


# ---------------------------------------------------------------------------
# Stage 2 helpers
# ---------------------------------------------------------------------------

def _build_part_name_db(project_dir: str, category: str, clip_model):
    path = os.path.join(project_dir, 'data', f'partnet2common_{category}.json')
    if not os.path.exists(path):
        return np.zeros((0,)), []
    name_db = list(set(json.load(open(path)).values()))
    name_db = [n[:n.find('/')] if '/' in n else n for n in name_db]
    with torch.no_grad():
        import clip as clip_module
        emb = clip_model.encode_text(clip_module.tokenize(name_db).to('cuda')).cpu().numpy()
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return emb, name_db


def _load_stage2_model(model_dir: str, category: str, project_dir: str):
    from omegaconf import OmegaConf
    from TrAssembler.model import GMFlowModel

    ckpt_path = os.path.join(model_dir, 'checkpoints', 'last.ckpt')
    config_path = os.path.join(model_dir, '.hydra', 'config.yaml')
    if not os.path.exists(config_path):
        config_path = os.path.join(project_dir, 'TrAssembler', 'config.yaml')
    config = OmegaConf.load(config_path)

    model = GMFlowModel.load_from_checkpoint(
        ckpt_path, args=config,
        embed_dim=config.network.embed_dim,
        num_heads=config.network.num_heads,
        dropout=config.network.dropout,
        bias=True, scaling_factor=1.,
        args_range=np.array([-1., 1.]),
    ).cuda().eval()
    return model, config


# ---------------------------------------------------------------------------
# Main inference runner (runs in background thread)
# ---------------------------------------------------------------------------

def run_inference(job_id: str, image_data: bytes, category: str):
    """Run the full Img2CAD pipeline in a background thread.  Updates JOBS dict."""

    with JOBS_LOCK:
        JOBS[job_id]['status'] = 'processing'
        JOBS[job_id]['started_at'] = time.time()

    cb = make_progress_callback(job_id)

    # ---- determine output dir ----
    out_base = os.path.join(_project_dir, 'data', 'output', 'web_inference', job_id)
    os.makedirs(out_base, exist_ok=True)

    # ---- save uploaded image ----
    image_path = os.path.join(out_base, 'input.jpg')
    with open(image_path, 'wb') as f:
        f.write(image_data)

    try:
        # ====================================================================
        # Step 0: Preprocess
        # ====================================================================
        cb('preprocessing', 2.0, '正在移除背景...')
        from infer_single import preprocess_phone_photo
        image_pil = preprocess_phone_photo(image_path)
        preprocessed_path = os.path.join(out_base, 'preprocessed.png')
        image_pil.save(preprocessed_path)
        cb('preprocessing', 8.0, '预处理完成')

        # ====================================================================
        # Step 1: LlamaFT
        # ====================================================================
        # Detect adapter path
        adapter_base = os.path.join(_project_dir, 'data', 'ckpts', 'llamaft', category)
        adapters = sorted(Path(adapter_base).glob('checkpoint-*')) if os.path.exists(adapter_base) else []
        if not adapters:
            adapter_base = os.path.join(_project_dir, 'data', 'ckpts', 'hf', 'llamaft', category)
            adapters = sorted(Path(adapter_base).glob('checkpoint-*')) if os.path.exists(adapter_base) else []
        if not adapters:
            raise FileNotFoundError(f"未找到 {category} 类别的 LoRA adapter")
        adapter_path = str(adapters[-1])

        cached = _get_cached_models(category)
        if 'llama_model' in cached:
            llama_model, processor = cached['llama_model'], cached['processor']
            cb('stage1', 10.0, '使用已缓存的 Llama 模型')
        else:
            llama_model, processor = _load_llama_model(adapter_path, progress_cb=cb)
            _set_cached_models(category, {**cached, 'llama_model': llama_model, 'processor': processor})

        prompt_path = os.path.join(_project_dir, 'LlamaFT', 'prompt.txt')
        prompt = open(prompt_path).read()

        cb('stage1', 15.0, 'VLM 正在生成 CAD 结构...')
        text_response = _run_stage1_vlm(image_pil, prompt, llama_model, processor, progress_cb=cb)

        # Parse text → h5
        from infer_single import process_text_to_h5
        h5_path = os.path.join(out_base, 'stage1_output.h5')
        success = process_text_to_h5(text_response, h5_path)
        if not success:
            raise RuntimeError("Stage 1 失败：无法从 VLM 输出解析 CAD 结构")

        # Save text for debugging
        text_path = os.path.join(out_base, 'stage1_text.txt')
        with open(text_path, 'w') as f:
            f.write(text_response)

        cb('stage1', 42.0, 'Stage 1 完成 — 已解析 CAD 结构')

        # ====================================================================
        # Step 2: TrAssembler
        # ====================================================================
        model_dir = os.path.join(_project_dir, 'data', 'ckpts', 'trassembler', category)
        if not os.path.exists(model_dir):
            model_dir = os.path.join(_project_dir, 'data', 'ckpts', 'hf', 'trassembler', category)

        cb('stage2', 45.0, '加载 TrAssembler 模型...')

        cached = _get_cached_models(category)
        if 'stage2_model' in cached:
            model, clip_model, name_embeddings_db = (
                cached['stage2_model'], cached['clip_model'], cached['name_embeddings_db'])
        else:
            model, _config = _load_stage2_model(model_dir, category, _project_dir)
            import clip as clip_module
            torch.set_grad_enabled(False)
            clip_model, _ = clip_module.load("ViT-B/32", device='cuda')
            clip_model.eval()
            name_embeddings_db, _ = _build_part_name_db(_project_dir, category, clip_model)
            _set_cached_models(category, {
                **cached,
                'stage2_model': model, 'clip_model': clip_model,
                'name_embeddings_db': name_embeddings_db,
            })

        from infer_single import run_stage2
        success = run_stage2(h5_path, image_pil, model, clip_model, name_embeddings_db, True, out_base, progress_callback=cb)
        if not success:
            raise RuntimeError("Stage 2 失败：CAD 实体生成出错")

        cb('stage2', 95.0, '生成完成，正在整理文件...')

        # Verify outputs exist
        obj_path = os.path.join(out_base, 'final.obj')
        step_path = os.path.join(out_base, 'final.step')
        if not os.path.exists(obj_path):
            raise RuntimeError("OBJ 文件未生成")

        # Record file sizes
        obj_size = os.path.getsize(obj_path)
        step_size = os.path.getsize(step_path) if os.path.exists(step_path) else 0

        with JOBS_LOCK:
            JOBS[job_id].update({
                'status': 'done',
                'stage': 'done',
                'progress': 100.0,
                'message': 'CAD 模型生成完成！',
                'obj_size': obj_size,
                'step_size': step_size,
            })

    except Exception as exc:
        with JOBS_LOCK:
            JOBS[job_id].update({
                'status': 'error',
                'message': str(exc),
                'traceback': traceback.format_exc(),
            })
        # Write error traceback to file for debugging
        try:
            with open(os.path.join(out_base, 'error.txt'), 'w') as f:
                f.write(traceback.format_exc())
        except Exception:
            pass


def create_job(category: str) -> str:
    """Register a new job and return its id."""
    _start_cleanup_thread()
    job_id = uuid.uuid4().hex[:12]
    now = time.time()
    with JOBS_LOCK:
        JOBS[job_id] = {
            'job_id': job_id,
            'status': 'queued',
            'stage': 'queued',
            'progress': 0.0,
            'message': '任务已创建',
            'category': category,
            'created_at': now,
            'last_accessed': now,
            'started_at': None,
            'obj_size': 0,
            'step_size': 0,
            'stage_times': {},
        }
    return job_id


def get_job(job_id: str) -> dict | None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job:
            job['last_accessed'] = time.time()
        return job


def get_job_output_dir(job_id: str) -> str:
    return os.path.join(_project_dir, 'data', 'output', 'web_inference', job_id)
