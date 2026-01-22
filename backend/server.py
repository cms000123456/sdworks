"""
Stable Diffusion Backend Server
A standalone FastAPI server using the diffusers library
Compatible with AUTOMATIC1111 API format
"""

import io
import os
import base64
import logging
import gc
import shutil
import httpx
import asyncio
from typing import Optional, List
from contextlib import asynccontextmanager

from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline as transformers_pipeline

import torch
from PIL import Image
from fastapi import FastAPI, HTTPException, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from diffusers import (
    StableDiffusionPipeline, 
    StableDiffusionXLPipeline,
    StableDiffusionXLPipeline,
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DDIMScheduler
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
pipeline = None
current_model_id = "runwayml/stable-diffusion-v1-5"
MODELS_DIR = "/app/models"
LEGACY_MODELS_DIR = "/app/models/legacy"
LORAS_DIR = "/app/models/loras"

# Prompt Refinement Models
interrogator_processor = None
interrogator_model = None
magic_prompt_pipe = None

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LEGACY_MODELS_DIR, exist_ok=True)
os.makedirs(LORAS_DIR, exist_ok=True)

# Available models configuration
AVAILABLE_MODELS = [
    {
        "title": "Stable Diffusion v1.5 (RunwayML)",
        "model_name": "runwayml/stable-diffusion-v1-5",
        "hash": "v1-5",
        "sha256": "v1-5",
        "config": "standard",
        "architecture": "sd15"
    },
    {
        "title": "Stable Diffusion v2.1 (StabilityAI)",
        "model_name": "stabilityai/stable-diffusion-2-1",
        "hash": "v2-1",
        "sha256": "v2-1",
        "config": "v2",
        "architecture": "sd21"
    },
    {
        "title": "Stable Diffusion v1.4 (CompVis)",
        "model_name": "CompVis/stable-diffusion-v1-4",
        "hash": "v1-4",
        "sha256": "v1-4",
        "config": "standard",
        "architecture": "sd15"
    },
    {
        "title": "OpenJourney v4 (Midjourney Style)",
        "model_name": "prompthero/openjourney-v4",
        "hash": "openjourney-v4",
        "sha256": "openjourney-v4",
        "config": "standard",
        "architecture": "sd15"
    }
]


class LoraSelection(BaseModel):
    name: str = Field(..., description="Filename of the LoRA")
    weight: float = Field(1.0, description="Strength of the LoRA")


class GenerationRequest(BaseModel):
    """Request model matching AUTOMATIC1111 format"""
    prompt: str = Field(..., description="Text prompt for image generation")
    negative_prompt: str = Field("", description="Negative prompt to avoid certain elements")
    steps: int = Field(20, ge=1, le=150, description="Number of sampling steps")
    cfg_scale: float = Field(7.0, ge=1.0, le=30.0, description="Classifier-free guidance scale")
    width: int = Field(512, description="Image width")
    height: int = Field(512, description="Image height")
    seed: int = Field(-1, description="Random seed (-1 for random)")
    batch_size: int = Field(1, ge=1, le=8, description="Number of images to generate")
    sampler_name: str = Field("DPM++ 2M Karras", description="Sampler name")
    n_iter: int = Field(1, description="Number of iterations")
    loras: Optional[List[LoraSelection]] = Field(None, description="List of LoRAs to apply")
    init_images: Optional[List[str]] = Field(None, description="List of base64 encoded initial images for Img2Img")
    denoising_strength: float = Field(0.75, ge=0.0, le=1.0, description="Denoising strength for Img2Img")


class GenerationResponse(BaseModel):
    """Response model matching AUTOMATIC1111 format"""
    images: list[str] = Field(..., description="Base64 encoded images")
    parameters: dict = Field(..., description="Generation parameters used")
    info: str = Field("", description="Generation info")


class ModelItem(BaseModel):
    title: str
    model_name: str
    hash: Optional[str] = None
    sha256: Optional[str] = None
    filename: str = ""
    config: Optional[str] = None
    
    model_config = {
        "protected_namespaces": ()
    }


class Options(BaseModel):
    sd_model_checkpoint: Optional[str] = None


def load_model(model_id: str, is_retry=False):
    """Load Stable Diffusion model with optimizations"""
    global pipeline, current_model_id
    
    prev_model_id = current_model_id
    logger.info(f"Loading Stable Diffusion model: {model_id}...")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    try:
        # Clean up previous model to free memory
        # Use None instead of del to prevent NameError
        if pipeline is not None:
            pipeline = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Load pipeline
        if model_id.endswith(".safetensors") or model_id.endswith(".ckpt"):
            # Search for model in local directories
            model_path = None
            for scan_dir in [MODELS_DIR, LEGACY_MODELS_DIR]:
                potential_path = os.path.join(scan_dir, model_id)
                if os.path.exists(potential_path):
                    model_path = potential_path
                    break
            
            if not model_path:
                raise FileNotFoundError(f"Model file not found: {model_id}")
            
            logger.info(f"Loading local single-file model from: {model_path}")
            
            # Detect architecture for single file
            lower_file = model_id.lower()
            if "xl" in lower_file:
                pipeline_class = StableDiffusionXLPipeline
                logger.info("Detected SDXL architecture for local model")
            else:
                pipeline_class = StableDiffusionPipeline
                logger.info("Detected SD 1.5 architecture for local model")
                
            new_pipeline = pipeline_class.from_single_file(
                model_path,
                torch_dtype=dtype,
                load_safety_checker=False,
                use_safetensors=model_id.endswith(".safetensors")
            )
        else:
            # Load from Hugging Face - AutoPipeline works fine here
            new_pipeline = AutoPipelineForText2Image.from_pretrained(
                model_id,
                torch_dtype=dtype,
                safety_checker=None, 
                requires_safety_checker=False
            )
        
        # Determine architecture and optimize scheduler
        pipeline_type = type(new_pipeline).__name__
        logger.info(f"Loaded pipeline type: {pipeline_type}")
        
        # Optimize scheduler based on architecture
        if "XL" in pipeline_type:
            from diffusers import EulerDiscreteScheduler
            new_pipeline.scheduler = EulerDiscreteScheduler.from_config(
                new_pipeline.scheduler.config
            )
            logger.info("SDXL architecture detected, using EulerDiscreteScheduler")
        else:
            new_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                new_pipeline.scheduler.config
            )
            logger.info("SD 1.5 architecture detected, using DPMSolverMultistepScheduler")
        
        # Move to device
        new_pipeline = new_pipeline.to(device)
        
        # Enable memory efficient attention if available
        if device == "cuda":
            try:
                new_pipeline.enable_xformers_memory_efficient_attention()
                logger.info("Enabled xformers memory efficient attention")
            except Exception as e:
                logger.warning(f"xformers not available: {e}")
                new_pipeline.enable_attention_slicing()
                logger.info("Enabled attention slicing")
        
        pipeline = new_pipeline
        current_model_id = model_id
        logger.info(f"Model {model_id} loaded successfully on {device}")
        
    except Exception as e:
        logger.error(f"Failed to load model {model_id}: {e}")
        
        # If this wasn't already a retry and we had a working model before, try to go back
        if not is_retry and prev_model_id and prev_model_id != model_id:
            logger.info(f"Attempting to reload previous working model: {prev_model_id}")
            try:
                load_model(prev_model_id, is_retry=True)
                # If retry succeeds, we don't need to raise the exception as we've recovered
                return
            except Exception as retry_err:
                logger.error(f"Critical: Failed to reload previous model: {retry_err}")
        
        # If we reach here, we failed to load the new model and (if applicable) failed to reload the old one
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to load model {model_id}: {str(e)}. " + 
                   ("Previous model restored." if pipeline is not None else "No model currently loaded.")
        )


def load_interrogator():
    """Load BLIP model for image interrogation"""
    global interrogator_processor, interrogator_model
    if interrogator_model is None:
        logger.info("Loading BLIP interrogator model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        interrogator_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        interrogator_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(device)
        logger.info("BLIP model loaded successfully")


def load_magic_prompt():
    """Load GPT-2 model for prompt expansion"""
    global magic_prompt_pipe
    if magic_prompt_pipe is None:
        logger.info("Loading Magic Prompt model...")
        device = 0 if torch.cuda.is_available() else -1
        magic_prompt_pipe = transformers_pipeline(
            "text-generation", 
            model="succinctly/text2image-prompt-generator",
            device=device
        )
        logger.info("Magic Prompt model loaded successfully")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("Starting Stable Diffusion server...")
    
    # Log all registered routes for debugging
    logger.info("Registered routes:")
    for route in app.routes:
        logger.info(f"  {route.methods} {route.path}")
        
    load_model(current_model_id)
    logger.info("Server ready!")
    yield
    # Shutdown
    logger.info("Shutting down server...")


# Create FastAPI app
app = FastAPI(
    title="Stable Diffusion API",
    description="Standalone Stable Diffusion backend compatible with AUTOMATIC1111 format",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "Stable Diffusion API is running",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "current_model": current_model_id
    }


@app.get("/sdapi/v1/sd-models")
async def get_models():
    """Return available models (AUTOMATIC1111 compatible)"""
    models = []
    for model in AVAILABLE_MODELS:
        models.append(dict(model))
    
    # Add local models from both directories
    scan_dirs = [
        {"path": MODELS_DIR, "prefix": "Local"},
        {"path": LEGACY_MODELS_DIR, "prefix": "Legacy"}
    ]
    
    for scan in scan_dirs:
        dir_path = scan["path"]
        prefix = scan["prefix"]
        
        if os.path.exists(dir_path):
            for file in os.listdir(dir_path):
                if file.endswith((".safetensors", ".ckpt")):
                    # Basic architecture detection based on name
                    arch = "sd15"
                    lower_file = file.lower()
                    if "xl" in lower_file:
                        arch = "sdxl"
                    elif "sd2" in lower_file or "v2-" in lower_file:
                        arch = "sd21"
                    
                    models.append({
                        "title": f"{prefix}: {file} ({arch.upper()})",
                        "model_name": file,
                        "hash": "local",
                        "sha256": "local",
                        "config": "local",
                        "architecture": arch
                    })
    
    return models


@app.get("/sdapi/v1/loras")
async def get_loras():
    """Return available LoRAs"""
    loras = []
    if os.path.exists(LORAS_DIR):
        for file in os.listdir(LORAS_DIR):
            if file.endswith((".safetensors", ".ckpt")):
                loras.append({
                    "name": file,
                    "title": file.replace(".safetensors", "").replace(".ckpt", ""),
                    "path": os.path.join(LORAS_DIR, file)
                })
    return loras


@app.post("/sdapi/v1/upload-model")
async def upload_model(file: UploadFile = File(...), filename: str = Form(...)):
    """Upload a model file to the local models directory"""
    if not (filename.endswith(".safetensors") or filename.endswith(".ckpt")):
        raise HTTPException(status_code=400, detail="Only .safetensors or .ckpt files are allowed")
    
    target_path = os.path.join(MODELS_DIR, filename)
    try:
        # Stream the file to disk instead of loading it all in memory
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        logger.info(f"Model uploaded successfully: {filename}")
        return {"message": f"Model {filename} uploaded successfully"}
    except Exception as e:
        logger.error(f"Failed to upload model: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        file.file.close()


@app.get("/sdapi/v1/civitai/models")
async def search_civitai_models(query: str = "", limit: int = 20):
    """Search models on Civitai via proxy"""
    async with httpx.AsyncClient() as client:
        try:
            results = []
            seen_ids = set()
            
            # Strategy 1: Search by query (names)
            # Strategy 2: Search by tag (keywords)
            search_urls = [
                f"https://civitai.com/api/v1/models?query={query}&limit={limit}",
                f"https://civitai.com/api/v1/models?tag={query}&limit={limit}"
            ]
            
            for url in search_urls:
                try:
                    res = await client.get(url, timeout=10.0, headers={"User-Agent": "Mozilla/5.0"})
                    if res.status_code != 200: continue
                    data = res.json()
                    
                    for model in data.get("items", []):
                        m_id = model.get("id")
                        if m_id in seen_ids: continue
                        
                        m_type = model.get("type")
                        versions = model.get("modelVersions", [])
                        if not versions: continue
                        
                        latest_version = versions[0]
                        files = latest_version.get("files", [])
                        download_url = None
                        filename = None
                        
                        # Find the best file (safetensors preferred)
                        for f in files:
                            fname = f.get("name", "")
                            ftype = f.get("type")
                            if ftype in ["Model", "Pruned Model"] and fname.endswith(".safetensors"):
                                download_url = f.get("downloadUrl")
                                filename = fname
                                break
                        
                        # Fallback
                        if not download_url:
                            for f in files:
                                if f.get("type") in ["Model", "Pruned Model"]:
                                    download_url = f.get("downloadUrl")
                                    filename = f.get("name")
                                    break
                        
                        if download_url:
                            seen_ids.add(m_id)
                            # Safe image access
                            img_url = ""
                            images = latest_version.get("images", [])
                            if images and isinstance(images, list) and len(images) > 0:
                                img_url = images[0].get("url", "")
                            
                            results.append({
                                "id": m_id,
                                "name": model.get("name"),
                                "description": model.get("description"),
                                "type": m_type,
                                "image": img_url,
                                "downloadUrl": download_url,
                                "filename": filename
                            })
                            if len(results) >= limit: break
                except Exception as e:
                    logger.warning(f"Search branch failed for {url}: {e}")
                
                if len(results) >= limit: break
            
            logger.info(f"Returning {len(results)} filtered results to frontend for query '{query}'")
            return {"items": results}
        except Exception as e:
            logger.error(f"Civitai search failed: {e}")
            raise HTTPException(status_code=500, detail=f"Civitai search failed: {str(e)}")


@app.post("/sdapi/v1/civitai/download")
async def download_civitai_model(
    downloadUrl: str = Form(...), 
    filename: str = Form(...),
    isLora: bool = Form(False)
):
    """Download a model from Civitai in the background"""
    if not (filename.endswith(".safetensors") or filename.endswith(".ckpt")):
        filename += ".safetensors"
        
    save_dir = LORAS_DIR if isLora else MODELS_DIR
    target_path = os.path.join(save_dir, filename)
    
    if os.path.exists(target_path):
        return {"message": "Model already exists", "filename": filename}

    async def do_download():
        target_file_path = target_path
        try:
            logger.info(f"Starting background download: {filename}")
            async with httpx.AsyncClient(follow_redirects=True) as client:
                async with client.stream("GET", downloadUrl, timeout=None) as response:
                    response.raise_for_status()
                    
                    # Try to get filename from Content-Disposition if it looks like a generic one
                    if "downloaded_model" in filename or "civitai_" in filename:
                        cd = response.headers.get("Content-Disposition")
                        if cd and "filename=" in cd:
                            import re
                            # Extract filename from header (e.g. filename="model.safetensors")
                            fname_match = re.search('filename="?([^";]+)"?', cd)
                            if fname_match:
                                new_filename = fname_match.group(1)
                                target_file_path = os.path.join(save_dir, new_filename)
                                logger.info(f"Renaming download to header filename: {new_filename}")

                    with open(target_file_path, "wb") as f:
                        async for chunk in response.aiter_bytes():
                            f.write(chunk)
            logger.info(f"Download complete: {os.path.basename(target_file_path)}")
        except Exception as e:
            logger.error(f"Background download failed for {filename}: {e}")
            if os.path.exists(target_file_path):
                try: os.remove(target_file_path)
                except: pass

    # Run download in background
    asyncio.create_task(do_download())
    return {"message": "Download started in background", "filename": filename}


@app.get("/sdapi/v1/options")
async def get_options():
    """Get current options (including active model)"""
    return {
        "sd_model_checkpoint": current_model_id,
        "sd_model_checkpoint_hash": "fluff", # Dummy value for compatibility
    }


@app.post("/sdapi/v1/options")
async def set_options(options: Options):
    """Set options (handle model switching)"""
    global pipeline, current_model_id
    
    if options.sd_model_checkpoint:
        # Find model by title or name
        target_model = None
        
        # Check hardcoded models
        for model in AVAILABLE_MODELS:
            if options.sd_model_checkpoint == model["title"] or \
               options.sd_model_checkpoint == model["model_name"]:
                target_model = model["model_name"]
                break
        
        # Check local models if not found in hardcoded list
        if not target_model:
            scan_dirs = [
                {"path": MODELS_DIR, "prefix": "Local"},
                {"path": LEGACY_MODELS_DIR, "prefix": "Legacy"}
            ]
            
            for scan in scan_dirs:
                dir_path = scan["path"]
                prefix = scan["prefix"]
                
                if os.path.exists(dir_path):
                    for file in os.listdir(dir_path):
                        if file.endswith((".safetensors", ".ckpt")):
                            # Match against full title with architecture suffix as generated in get_models
                            # Or just the filename
                            if options.sd_model_checkpoint == file or \
                               options.sd_model_checkpoint.startswith(f"{prefix}: {file}"):
                                target_model = file
                                break
                if target_model: break
        
        if target_model and target_model != current_model_id:
            logger.info(f"Switching model to: {target_model}")
            try:
                load_model(target_model)
                logger.info("Model switched successfully")
            except Exception as e:
                logger.error(f"Failed to switch model: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        elif not target_model:
            logger.warning(f"Model not found: {options.sd_model_checkpoint}")
                
    return {"message": "Options set"}


@app.post("/sdapi/v1/txt2img", response_model=GenerationResponse)
async def text_to_image(request: GenerationRequest):
    global pipeline
    """
    Generate images from text prompt
    Compatible with AUTOMATIC1111 API format
    """
    try:
        logger.info(f"Generating {request.batch_size} image(s) with prompt: {request.prompt[:50]}...")
        
        # Validate pipeline is loaded
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Handle seed
        actual_seed = request.seed
        if actual_seed == -1:
            actual_seed = torch.randint(0, 2147483647, (1,)).item()
            logger.info(f"Generated random seed: {actual_seed}")
        
        generator = torch.Generator(device=pipeline.device).manual_seed(actual_seed)
        
        # Handle LoRAs
        active_loras = []
        if request.loras:
            for lora in request.loras:
                lora_path = os.path.join(LORAS_DIR, lora.name)
                if os.path.exists(lora_path):
                    try:
                        adapter_name = lora.name.replace(".safetensors", "").replace(".ckpt", "")
                        logger.info(f"Loading LoRA: {lora.name} with weight {lora.weight}...")
                        pipeline.load_lora_weights(lora_path, adapter_name=adapter_name)
                        active_loras.append(adapter_name)
                    except Exception as le:
                        logger.error(f"Failed to load LoRA {lora.name}: {le}")
                else:
                    logger.warning(f"LoRA file not found: {lora_path}")
            
            if active_loras:
                # Set weights for all active LoRAs
                weights = [next(l.weight for l in request.loras if l.name.startswith(name)) for name in active_loras]
                pipeline.set_adapters(active_loras, adapter_weights=weights)

        # Handle Img2Img vs Text2Img pipeline switching
        input_image = None
        if request.init_images and len(request.init_images) > 0:
            # Img2Img Mode
            try:
                # Decode first image
                image_data = base64.b64decode(request.init_images[0])
                input_image = Image.open(io.BytesIO(image_data)).convert("RGB")
                input_image = input_image.resize((request.width, request.height))
                
                # Switch to Img2Img pipeline if needed
                # AutoPipelineForImage2Image is a factory, so we check the pipeline instance name
                current_pipe_class = pipeline.__class__.__name__
                if "Img2Img" not in current_pipe_class:
                    logger.info(f"Current pipeline is {current_pipe_class}, switching to Img2Img via AutoPipeline...")
                    pipeline = AutoPipelineForImage2Image.from_pipe(pipeline)
                    logger.info(f"Switched to {pipeline.__class__.__name__}")
                else:
                    logger.info("Pipeline is already Img2Img compatible")

            except Exception as e:
                logger.error(f"Failed to process init image: {e}")
                raise HTTPException(status_code=400, detail=f"Invalid init image: {e}")
        else:
            # Text2Img Mode
            current_pipe_class = pipeline.__class__.__name__
            # If current pipeline is Img2Img (and not XL/1.5 base which can do txt2img potentially, but often better to switch back for safety)
            # Actually, standard StableDiffusionPipeline can handle txt2img, StableDiffusionImg2ImgPipeline cannot (it expects image)
            if "Img2Img" in current_pipe_class:
                 logger.info(f"Current pipeline is {current_pipe_class}, switching back to Text2Img via AutoPipeline...")
                 pipeline = AutoPipelineForText2Image.from_pipe(pipeline)
                 logger.info(f"Switched back to {pipeline.__class__.__name__}")

        # Apply Scheduler
        sampler_map = {
            "Euler a": EulerAncestralDiscreteScheduler,
            "Euler": EulerDiscreteScheduler,
            "DPM++ 2M Karras": DPMSolverMultistepScheduler,
            "DDIM": DDIMScheduler
        }
        
        SchedulerClass = sampler_map.get(request.sampler_name, DPMSolverMultistepScheduler)
        
        # Configure scheduler kwargs based on type
        scheduler_kwargs = {}
        if SchedulerClass == DPMSolverMultistepScheduler:
            scheduler_kwargs["use_karras_sigmas"] = True
        
        # Only switch if different (optimization)
        if type(pipeline.scheduler) != SchedulerClass:
            logger.info(f"Switching scheduler to {request.sampler_name}")
            pipeline.scheduler = SchedulerClass.from_config(
                pipeline.scheduler.config, 
                **scheduler_kwargs
            )
        elif request.sampler_name == "DPM++ 2M Karras" and not getattr(pipeline.scheduler.config, "use_karras_sigmas", False):
             # Ensure karras is on if requested
             pipeline.scheduler = SchedulerClass.from_config(
                pipeline.scheduler.config, 
                **scheduler_kwargs
            )

        # Generate images
        with torch.inference_mode():
            if input_image:
                # Clamp strength to avoid 0.0 or 1.0 weirdness if necessary, though diffusers usually handles it
                # High strength = more destruction. 1.0 = ignore image.
                final_strength = max(0.05, min(0.95, request.denoising_strength))
                logger.info(f"Running Img2Img with strength {final_strength} (requested {request.denoising_strength})")
                
                result = pipeline(
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt if request.negative_prompt else None,
                    image=input_image,
                    strength=final_strength,
                    num_inference_steps=request.steps,
                    guidance_scale=request.cfg_scale,
                    num_images_per_prompt=request.batch_size,
                    generator=generator
                )
            else:
                result = pipeline(
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt if request.negative_prompt else None,
                    num_inference_steps=request.steps,
                    guidance_scale=request.cfg_scale,
                    width=request.width,
                    height=request.height,
                    num_images_per_prompt=request.batch_size,
                    generator=generator
                )
        
        # Unload LoRAs to clean up the pipeline for the next request
        if active_loras:
            logger.info("Unloading LoRAs...")
            try:
                pipeline.unload_lora_weights()
            except Exception as ue:
                logger.warning(f"Failed to unload LoRA weights: {ue}")
        
        # Convert images to base64
        images_base64 = [image_to_base64(img) for img in result.images]
        
        # Build response
        response = GenerationResponse(
            images=images_base64,
            parameters={
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "steps": request.steps,
                "cfg_scale": request.cfg_scale,
                "width": request.width,
                "height": request.height,
                "height": request.height,
                "seed": actual_seed,
                "batch_size": request.batch_size,
                "sampler_name": request.sampler_name
            },
            info=f"Generated {len(images_base64)} image(s)"
        )
        
        logger.info(f"Successfully generated {len(images_base64)} image(s)")
        return response
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


class InterrogateRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image")


@app.post("/sdapi/v1/interrogate")
async def interrogate(request: InterrogateRequest):
    """Generate a prompt from an image using BLIP"""
    try:
        load_interrogator()
        
        # Decode image
        image_data = base64.b64decode(request.image)
        raw_image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Process image
        inputs = interrogator_processor(raw_image, return_tensors="pt").to(interrogator_model.device)
        out = interrogator_model.generate(**inputs)
        caption = interrogator_processor.decode(out[0], skip_special_tokens=True)
        
        logger.info(f"Interrogated image: {caption}")
        return {"prompt": caption}
    except Exception as e:
        logger.error(f"Interrogation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Interrogation failed: {str(e)}")


class MagicPromptRequest(BaseModel):
    prompt: str = Field(..., description="Short prompt to expand")


@app.post("/sdapi/v1/magic-prompt")
async def magic_prompt(request: MagicPromptRequest):
    """Expand a short prompt using GPT-2 with high variety"""
    try:
        load_magic_prompt()
        import random
        
        input_prompt = request.prompt.strip()
        
        # Enhanced parameters for variety
        # temperature=1.2 is adventurous but good for variety
        # repetition_penalty helps avoid GPT-2 loops
        results = magic_prompt_pipe(
            input_prompt, 
            max_new_tokens=70, # Use new tokens relative to input
            num_return_sequences=3, # Generate multiple candidates
            do_sample=True,
            top_p=0.9,
            top_k=50,
            temperature=1.1,
            repetition_penalty=1.2
        )
        
        # Filter out results that didn't actually add anything (if any)
        candidates = [r['generated_text'].strip() for r in results]
        valid_candidates = [c for c in candidates if len(c) > len(input_prompt) + 5]
        
        if not valid_candidates:
            # Fallback to the longest one if none are significantly longer
            result = max(candidates, key=len)
        else:
            # Pick a random one from the valid candidates for variety
            result = random.choice(valid_candidates)
        
        logger.info(f"Expanded prompt (variety pick): {result}")
        return {"prompt": result}
    except Exception as e:
        logger.error(f"Magic prompt expansion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Magic prompt expansion failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    logger.info("=" * 60)
    logger.info("Stable Diffusion Backend Server")
    logger.info("=" * 60)
    logger.info(f"Default model: {current_model_id}")
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860,
        log_level="info"
    )
