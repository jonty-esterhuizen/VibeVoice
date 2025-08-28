#!/usr/bin/env python3
"""
VibeVoice Server API - REST API equivalent to gradio_demo.py
Provides JSON-based podcast generation with streaming support
"""

import asyncio
import base64
import io
import json
import logging
import os
import sys
import tempfile
import time
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator
import traceback
import re

import numpy as np
import soundfile as sf
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, field_validator
import librosa

# Load VibeVoice modules
from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.streamer import AudioStreamer
from transformers.utils import logging as transformers_logging
from transformers import set_seed

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Configure transformers logging
transformers_logging.set_verbosity_info()

# Security
security = HTTPBearer()

class VibeVoiceServer:
    """Main server class that handles model loading and generation"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.available_voices = {}
        self.is_initialized = False
        self.active_generations = {}  # Track active generations
        
    async def initialize(self):
        """Initialize the VibeVoice model and processor"""
        if self.is_initialized:
            return
            
        logger.info("🚀 Initializing VibeVoice Server...")
        
        model_path = os.getenv("MODEL_PATH", "microsoft/VibeVoice-1.5B")
        device = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        inference_steps = int(os.getenv("INFERENCE_STEPS", "10"))
        
        logger.info(f"📁 Loading model from: {model_path}")
        logger.info(f"🔧 Device: {device}")
        logger.info(f"⚙️ Inference steps: {inference_steps}")
        
        try:
            # Auto-download model if needed
            logger.info("🔽 Checking model availability...")
            
            # Load processor (this will auto-download if needed)
            logger.info("Loading processor...")
            self.processor = VibeVoiceProcessor.from_pretrained(model_path)
            
            # Load model (this will auto-download if needed)
            logger.info("Loading model...")
            self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map='cuda',
                # attn_implementation="flash_attention_2",  # Disabled - not available
            )
            self.model.eval()
            
            # Configure noise scheduler
            self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
                self.model.model.noise_scheduler.config, 
                algorithm_type='sde-dpmsolver++',
                beta_schedule='squaredcos_cap_v2'
            )
            self.model.set_ddpm_inference_steps(num_steps=inference_steps)
            
            if hasattr(self.model.model, 'language_model'):
                logger.info(f"Language model attention: {self.model.model.language_model.config._attn_implementation}")
            
            # Setup voice presets
            await self.setup_voice_presets()
            
            self.is_initialized = True
            logger.info("✅ VibeVoice Server initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize VibeVoice Server: {e}")
            raise
    
    async def setup_voice_presets(self):
        """Setup voice presets by scanning the voices directory"""
        voices_dir = os.getenv("VOICES_DIR", "voices")
        
        if not os.path.exists(voices_dir):
            logger.warning(f"⚠️ Voices directory not found at {voices_dir}")
            self.available_voices = {}
            return
        
        logger.info(f"🔍 Scanning voices directory: {voices_dir}")
        
        # Scan for audio files
        audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac']
        audio_files = []
        
        for file in os.listdir(voices_dir):
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                if os.path.isfile(os.path.join(voices_dir, file)):
                    audio_files.append(file)
        
        # Create voice presets dictionary
        self.available_voices = {}
        for audio_file in audio_files:
            name = os.path.splitext(audio_file)[0]
            full_path = os.path.join(voices_dir, audio_file)
            self.available_voices[name] = full_path
        
        # Sort alphabetically
        self.available_voices = dict(sorted(self.available_voices.items()))
        
        logger.info(f"🎭 Found {len(self.available_voices)} voice presets: {list(self.available_voices.keys())}")
        
        if not self.available_voices:
            logger.error(f"❌ No voice presets found in {voices_dir}")
    
    def read_audio(self, audio_path: str, target_sr: int = 24000) -> np.ndarray:
        """Read and preprocess audio file"""
        try:
            wav, sr = sf.read(audio_path)
            if len(wav.shape) > 1:
                wav = np.mean(wav, axis=1)
            if sr != target_sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            return wav
        except Exception as e:
            logger.error(f"❌ Error reading audio {audio_path}: {e}")
            return np.array([])

# Global server instance
vibe_server = VibeVoiceServer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown"""
    # Startup
    logger.info("🚀 Starting VibeVoice API Server...")
    await vibe_server.initialize()
    yield
    # Shutdown
    logger.info("🛑 Shutting down VibeVoice API Server...")

# Create FastAPI app
app = FastAPI(
    title="VibeVoice API",
    description="High-Quality Dialogue Generation API with Streaming Support",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class GenerationRequest(BaseModel):
    """Request model for podcast generation"""
    script: str = Field(..., description="The podcast script")
    num_speakers: int = Field(default=2, ge=1, le=4, description="Number of speakers (1-4)")
    speakers: List[str] = Field(default=[], description="List of speaker voice names")
    cfg_scale: float = Field(default=1.3, ge=1.0, le=2.0, description="CFG scale for guidance")
    
    @field_validator('speakers')
    @classmethod
    def validate_speakers(cls, v, info):
        num_speakers = info.data.get('num_speakers', 2) if info.data else 2
        if len(v) > 0 and len(v) != num_speakers:
            raise ValueError(f"Number of speakers ({len(v)}) must match num_speakers ({num_speakers})")
        return v

class GenerationResponse(BaseModel):
    """Response model for podcast generation"""
    status: str
    message: str
    generation_id: Optional[str] = None
    audio_base64: Optional[str] = None
    duration: Optional[float] = None
    sample_rate: Optional[int] = None

class StatusResponse(BaseModel):
    """Server status response"""
    status: str
    message: str
    available_voices: List[str]
    model_loaded: bool

# Authentication
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key from Authorization header"""
    expected_key = os.getenv("API_KEY", "1234")
    
    if credentials.credentials != expected_key:
        logger.warning(f"🚫 Unauthorized access attempt with key: {credentials.credentials[:4]}...")
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return credentials.credentials

# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    start_time = time.time()
    
    # Get client IP
    client_ip = request.client.host
    
    # Log request
    logger.info(f"📥 {request.method} {request.url.path} from {client_ip}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(f"📤 {response.status_code} - {process_time:.3f}s")
    
    return response

def convert_to_16_bit_wav(data):
    """Convert audio data to 16-bit WAV format"""
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    
    data = np.array(data)
    
    # Normalize to range [-1, 1] if it's not already
    if np.max(np.abs(data)) > 1.0:
        data = data / np.max(np.abs(data))
    
    # Scale to 16-bit integer range
    data = (data * 32767).astype(np.int16)
    return data

def audio_to_base64(audio_data: np.ndarray, sample_rate: int = 24000) -> str:
    """Convert numpy audio data to base64 encoded WAV"""
    # Convert to 16-bit
    audio_16bit = convert_to_16_bit_wav(audio_data)
    
    # Create WAV bytes
    buffer = io.BytesIO()
    sf.write(buffer, audio_16bit, sample_rate, format='WAV')
    wav_bytes = buffer.getvalue()
    
    # Encode to base64
    return base64.b64encode(wav_bytes).decode('utf-8')

def save_audio_file(audio_data: np.ndarray, sample_rate: int, script: str) -> Optional[str]:
    """Save generated audio to file if enabled in settings"""
    try:
        # Check if saving is enabled
        save_enabled = os.getenv("SAVE_GENERATED_AUDIO", "false").lower() == "true"
        if not save_enabled:
            return None
        
        # Get output directory
        output_dir = os.getenv("OUTPUT_DIRECTORY", "outputs")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename: first10chars_yymmddhhmmss.wav
        # Get first 10 chars from script, remove non-alphanumeric
        first_chars = re.sub(r'[^a-zA-Z0-9]', '', script)[:10]
        if not first_chars:
            first_chars = "audio"
        
        # Get timestamp in yymmddhhmmss format
        timestamp = datetime.now().strftime("%y%m%d%H%M%S")
        
        filename = f"{first_chars}_{timestamp}.wav"
        filepath = os.path.join(output_dir, filename)
        
        # Convert to 16-bit and save
        audio_16bit = convert_to_16_bit_wav(audio_data)
        sf.write(filepath, audio_16bit, sample_rate)
        
        logger.info(f"💾 Audio saved to: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"❌ Failed to save audio file: {e}")
        return None

# API Endpoints

@app.get("/", response_model=StatusResponse)
async def root():
    """Root endpoint - server status"""
    logger.info("📊 Status check requested")
    
    return StatusResponse(
        status="online" if vibe_server.is_initialized else "initializing",
        message="VibeVoice API Server is running" if vibe_server.is_initialized else "Server is initializing...",
        available_voices=list(vibe_server.available_voices.keys()),
        model_loaded=vibe_server.is_initialized
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/voices")
async def get_voices(api_key: str = Depends(verify_api_key)):
    """Get available voices"""
    logger.info("🎭 Voice list requested")
    
    return {
        "voices": list(vibe_server.available_voices.keys()),
        "count": len(vibe_server.available_voices)
    }

@app.post("/generate", response_model=GenerationResponse)
async def generate_podcast(
    request: GenerationRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """Generate podcast from script"""
    
    generation_id = f"gen_{int(time.time() * 1000)}"
    logger.info(f"🎙️ Generation request {generation_id}: {request.num_speakers} speakers, {len(request.script)} chars")
    
    if not vibe_server.is_initialized:
        raise HTTPException(status_code=503, detail="Server is still initializing")
    
    try:
        # Validate script
        if not request.script.strip():
            raise HTTPException(status_code=400, detail="Script cannot be empty")
        
        # Clean script
        script = request.script.replace("'", "'")
        
        # Validate speakers
        selected_speakers = request.speakers
        if not selected_speakers:
            # Use default speakers if none specified
            available_speakers = list(vibe_server.available_voices.keys())
            default_speakers = ['en-Alice_woman', 'en-Carter_man', 'en-Frank_man', 'en-Maya_woman']
            selected_speakers = []
            for speaker in default_speakers[:request.num_speakers]:
                if speaker in available_speakers:
                    selected_speakers.append(speaker)
            
            # Fill remaining with available speakers
            remaining = request.num_speakers - len(selected_speakers)
            for speaker in available_speakers:
                if remaining <= 0:
                    break
                if speaker not in selected_speakers:
                    selected_speakers.append(speaker)
                    remaining -= 1
        
        # Validate all speakers exist
        for speaker in selected_speakers:
            if speaker not in vibe_server.available_voices:
                raise HTTPException(status_code=400, detail=f"Speaker '{speaker}' not found")
        
        logger.info(f"🎭 Using speakers: {selected_speakers}")
        
        # Load voice samples
        voice_samples = []
        for speaker_name in selected_speakers:
            audio_path = vibe_server.available_voices[speaker_name]
            audio_data = vibe_server.read_audio(audio_path)
            if len(audio_data) == 0:
                raise HTTPException(status_code=500, detail=f"Failed to load audio for {speaker_name}")
            voice_samples.append(audio_data)
        
        logger.info(f"✅ Loaded {len(voice_samples)} voice samples")
        
        # Format script with speaker assignments
        lines = script.strip().split('\n')
        formatted_script_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line already has speaker format
            if line.startswith('Speaker ') and ':' in line:
                formatted_script_lines.append(line)
            else:
                # Auto-assign to speakers in rotation (1-indexed)
                speaker_id = (len(formatted_script_lines) % request.num_speakers) + 1
                formatted_script_lines.append(f"Speaker {speaker_id}: {line}")
        
        formatted_script = '\n'.join(formatted_script_lines)
        logger.info(f"📝 Formatted script with {len(formatted_script_lines)} turns")
        
        start_time = time.time()
        
        # Prepare inputs
        inputs = vibe_server.processor(
            text=[formatted_script],
            voice_samples=[voice_samples],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        
        logger.info("🔄 Starting generation...")
        
        # Generate audio using streamer approach (like gradio demo)
        audio_streamer = AudioStreamer(
            batch_size=1,
            stop_signal=None,
            timeout=None
        )
        
        # Start generation in a separate thread (following gradio pattern)
        generation_thread = threading.Thread(
            target=lambda: vibe_server.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=request.cfg_scale,
                tokenizer=vibe_server.processor.tokenizer,
                generation_config={
                    'do_sample': False,
                },
                audio_streamer=audio_streamer,
                verbose=False,  # Disable verbose in streaming mode
                refresh_negative=True,
            )
        )
        generation_thread.start()
        
        # Wait for generation to start producing audio
        time.sleep(1)
        
        # Collect audio chunks
        sample_rate = int(os.getenv("SAMPLE_RATE", "24000"))
        all_audio_chunks = []
        
        # Get the stream for the first (and only) sample
        audio_stream = audio_streamer.get_stream(0)
        
        for audio_chunk in audio_stream:
            # Convert tensor to numpy (following gradio pattern)
            if torch.is_tensor(audio_chunk):
                # Convert bfloat16 to float32 first, then to numpy
                if audio_chunk.dtype == torch.bfloat16:
                    audio_chunk = audio_chunk.float()
                audio_np = audio_chunk.cpu().numpy().astype(np.float32)
            else:
                audio_np = np.array(audio_chunk, dtype=np.float32)
            
            # Ensure audio is 1D and properly normalized
            if len(audio_np.shape) > 1:
                audio_np = audio_np.squeeze()
            
            # Convert to 16-bit for consistency
            audio_16bit = convert_to_16_bit_wav(audio_np)
            all_audio_chunks.append(audio_16bit)
        
        # Wait for generation to complete
        generation_thread.join(timeout=10.0)
        
        generation_time = time.time() - start_time
        logger.info(f"⏱️ Generation completed in {generation_time:.2f} seconds")
        
        # Combine all audio chunks
        if all_audio_chunks:
            audio_data = np.concatenate(all_audio_chunks).astype(np.float32)
        else:
            raise HTTPException(status_code=500, detail="No audio chunks were generated")
        sample_rate = int(os.getenv("SAMPLE_RATE", "24000"))
        duration = len(audio_data) / sample_rate
        
        logger.info(f"🎵 Generated {duration:.2f} seconds of audio")
        
        # Save audio file if enabled
        saved_filepath = save_audio_file(audio_data, sample_rate, script)
        
        # Convert to base64
        audio_base64 = audio_to_base64(audio_data, sample_rate)
        
        logger.info(f"✅ Generation {generation_id} completed successfully")
        
        return GenerationResponse(
            status="success",
            message=f"Podcast generated successfully in {generation_time:.2f}s",
            generation_id=generation_id,
            audio_base64=audio_base64,
            duration=duration,
            sample_rate=sample_rate
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Generation {generation_id} failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/generate/wav")
async def generate_podcast_wav(
    request: GenerationRequest,
    api_key: str = Depends(verify_api_key)
):
    """Generate podcast and return WAV file directly"""
    generation_id = f"gen_{int(time.time() * 1000000 % 1000000000000)}"
    script = request.script.strip()
    
    if not script:
        raise HTTPException(status_code=400, detail="Script cannot be empty")
    
    logger.info(f"📥 POST /generate/wav")
    logger.info(f"🎙️ Generation request {generation_id}: {request.num_speakers} speakers, {len(script)} chars")
    
    try:
        # Speaker selection logic (same as main generate endpoint)
        if request.speakers:
            selected_speakers = request.speakers[:request.num_speakers]
        else:
            available_speaker_list = list(vibe_server.available_voices.keys())
            if len(available_speaker_list) < request.num_speakers:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Requested {request.num_speakers} speakers but only {len(available_speaker_list)} available"
                )
            selected_speakers = available_speaker_list[:request.num_speakers]
        
        # Validate speakers and load voice samples (same logic as main endpoint)
        for speaker in selected_speakers:
            if speaker not in vibe_server.available_voices:
                raise HTTPException(status_code=400, detail=f"Speaker '{speaker}' not found")
        
        logger.info(f"🎭 Using speakers: {selected_speakers}")
        
        voice_samples = []
        for speaker_name in selected_speakers:
            audio_path = vibe_server.available_voices[speaker_name]
            audio_data = vibe_server.read_audio(audio_path)
            if len(audio_data) == 0:
                raise HTTPException(status_code=500, detail=f"Failed to load audio for {speaker_name}")
            voice_samples.append(audio_data)
        
        logger.info(f"✅ Loaded {len(voice_samples)} voice samples")
        
        # Format script with speaker assignments (same logic)
        lines = script.strip().split('\n')
        formatted_script_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('Speaker ') and ':' in line:
                formatted_script_lines.append(line)
            else:
                speaker_id = (len(formatted_script_lines) % request.num_speakers) + 1
                formatted_script_lines.append(f"Speaker {speaker_id}: {line}")
        
        formatted_script = '\n'.join(formatted_script_lines)
        logger.info(f"📝 Formatted script with {len(formatted_script_lines)} turns")
        
        start_time = time.time()
        
        # Generate audio using streamer approach (like gradio demo)
        inputs = vibe_server.processor(
            text=[formatted_script],
            voice_samples=[voice_samples],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        
        logger.info("🔄 Starting generation...")
        
        # Create audio streamer
        audio_streamer = AudioStreamer(
            batch_size=1,
            stop_signal=None,
            timeout=None
        )
        
        # Start generation in a separate thread (following gradio pattern)
        generation_thread = threading.Thread(
            target=lambda: vibe_server.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=request.cfg_scale,
                tokenizer=vibe_server.processor.tokenizer,
                generation_config={
                    'do_sample': False,
                },
                audio_streamer=audio_streamer,
                verbose=False,  # Disable verbose in streaming mode
                refresh_negative=True,
            )
        )
        generation_thread.start()
        
        # Wait for generation to start producing audio
        time.sleep(1)
        
        # Collect audio chunks
        sample_rate = int(os.getenv("SAMPLE_RATE", "24000"))
        all_audio_chunks = []
        
        # Get the stream for the first (and only) sample
        audio_stream = audio_streamer.get_stream(0)
        
        for audio_chunk in audio_stream:
            # Convert tensor to numpy (following gradio pattern)
            if torch.is_tensor(audio_chunk):
                # Convert bfloat16 to float32 first, then to numpy
                if audio_chunk.dtype == torch.bfloat16:
                    audio_chunk = audio_chunk.float()
                audio_np = audio_chunk.cpu().numpy().astype(np.float32)
            else:
                audio_np = np.array(audio_chunk, dtype=np.float32)
            
            # Ensure audio is 1D and properly normalized
            if len(audio_np.shape) > 1:
                audio_np = audio_np.squeeze()
            
            # Convert to 16-bit for consistency
            audio_16bit = convert_to_16_bit_wav(audio_np)
            all_audio_chunks.append(audio_16bit)
        
        # Wait for generation to complete
        generation_thread.join(timeout=10.0)
        
        generation_time = time.time() - start_time
        logger.info(f"⏱️ Generation completed in {generation_time:.2f} seconds")
        
        # Combine all audio chunks
        if all_audio_chunks:
            audio_data = np.concatenate(all_audio_chunks).astype(np.float32)
        else:
            raise HTTPException(status_code=500, detail="No audio chunks were generated")
        sample_rate = int(os.getenv("SAMPLE_RATE", "24000"))
        duration = len(audio_data) / sample_rate
        
        logger.info(f"🎵 Generated {duration:.2f} seconds of audio")
        
        # Create temporary file for response
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_filepath = temp_file.name
        temp_file.close()
        
        # Convert to 16-bit and save to temp file
        audio_16bit = convert_to_16_bit_wav(audio_data)
        sf.write(temp_filepath, audio_16bit, sample_rate)
        
        # Also save to outputs directory if enabled
        save_audio_file(audio_data, sample_rate, script)
        
        logger.info(f"✅ Generation {generation_id} completed successfully")
        
        # Return WAV file as response
        return FileResponse(
            path=temp_filepath,
            media_type='audio/wav',
            filename=f"generated_audio_{generation_id}.wav",
            headers={
                "X-Generation-ID": generation_id,
                "X-Duration": str(duration),
                "X-Generation-Time": f"{generation_time:.2f}s"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Generation {generation_id} failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/generate/stream")
async def generate_podcast_stream(
    request: GenerationRequest,
    api_key: str = Depends(verify_api_key)
):
    """Generate podcast with streaming response"""
    
    generation_id = f"stream_{int(time.time() * 1000)}"
    logger.info(f"🎙️ Streaming generation request {generation_id}")
    
    if not vibe_server.is_initialized:
        raise HTTPException(status_code=503, detail="Server is still initializing")
    
    async def generate_stream():
        """Async generator for streaming response"""
        try:
            # Similar setup as regular generation
            # (Implementation would involve creating an async version of the streaming logic)
            # For now, return a simple stream
            yield json.dumps({
                "type": "status",
                "message": "Generation starting...",
                "generation_id": generation_id
            }) + "\n"
            
            # Placeholder for actual streaming implementation
            await asyncio.sleep(1)
            
            yield json.dumps({
                "type": "complete",
                "message": "Streaming not yet fully implemented",
                "generation_id": generation_id
            }) + "\n"
            
        except Exception as e:
            yield json.dumps({
                "type": "error",
                "message": str(e),
                "generation_id": generation_id
            }) + "\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

# Main function
def main():
    """Main function to run the server"""
    import uvicorn
    
    # Set seed for reproducibility
    set_seed(42)
    
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("SERVER_PORT", "8000"))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    logger.info(f"🚀 Starting VibeVoice API Server on {host}:{port}")
    logger.info(f"🔑 API Key: {os.getenv('API_KEY', '1234')}")
    logger.info(f"📁 Model path: {os.getenv('MODEL_PATH', '/tmp/vibevoice-model')}")
    
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=reload,
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )

if __name__ == "__main__":
    main()