# VibeVoice API Usage Guide

## Overview

The VibeVoice API provides a REST interface for generating high-quality AI podcasts, equivalent to the functionality of `gradio_demo.py` but accessible via HTTP endpoints.

## Quick Start

### 1. Installation

Install the required dependencies:
```bash
pip install -r requirements_server.txt
```

### 2. Configuration

The server reads configuration from `.env` file:
```bash
# Copy and modify .env as needed
API_KEY=1234
MODEL_PATH=/tmp/vibevoice-model
SERVER_PORT=8000
```

### 3. Start the Server

```bash
python server.py
```

The server will start on `http://localhost:8000`

### 4. Test the API

```bash
python test_api.py
```

## API Endpoints

### Authentication

All endpoints (except `/health`) require API key authentication via Bearer token:
```http
Authorization: Bearer 1234
```

### GET `/` - Server Status

Returns server status and available voices.

**Response:**
```json
{
    "status": "online",
    "message": "VibeVoice API Server is running",
    "available_voices": ["en-Alice_woman", "en-Carter_man", ...],
    "model_loaded": true
}
```

### GET `/health` - Health Check

Simple health check endpoint (no authentication required).

### GET `/voices` - Get Available Voices

Returns list of available speaker voices.

**Response:**
```json
{
    "voices": ["en-Alice_woman", "en-Carter_man", "en-Frank_man", "en-Maya_woman"],
    "count": 4
}
```

### POST `/generate` - Generate Podcast

Generate a complete podcast from a script.

**Request:**
```json
{
    "script": "Speaker 0: Welcome to our podcast!\nSpeaker 1: Thanks for having me!",
    "num_speakers": 2,
    "speakers": ["en-Alice_woman", "en-Carter_man"],
    "cfg_scale": 1.3
}
```

**Parameters:**
- `script` (required): The podcast script text
- `num_speakers` (default: 2): Number of speakers (1-4)
- `speakers` (optional): List of speaker voice names. If empty, uses defaults
- `cfg_scale` (default: 1.3): Guidance scale for generation (1.0-2.0)

**Response:**
```json
{
    "status": "success",
    "message": "Podcast generated successfully in 45.23s",
    "generation_id": "gen_1703123456789",
    "audio_base64": "UklGRiQAAABXQVZFZm10...",
    "duration": 12.5,
    "sample_rate": 24000
}
```

The `audio_base64` field contains the generated audio as a base64-encoded WAV file.

## Usage Examples

### Python Example

```python
import requests
import base64

# Configuration
API_BASE_URL = "http://localhost:8000"
HEADERS = {"Authorization": "Bearer 1234"}

# Generate podcast
request_data = {
    "script": "Speaker 0: Welcome to our AI podcast!\nSpeaker 1: Thanks! This is amazing technology.",
    "num_speakers": 2,
    "cfg_scale": 1.3
}

response = requests.post(
    f"{API_BASE_URL}/generate",
    headers=HEADERS,
    json=request_data
)

if response.status_code == 200:
    data = response.json()
    
    # Save audio file
    audio_bytes = base64.b64decode(data['audio_base64'])
    with open("podcast.wav", "wb") as f:
        f.write(audio_bytes)
    
    print(f"Generated {data['duration']:.1f}s of audio")
else:
    print(f"Error: {response.status_code}")
```

### cURL Example

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Authorization: Bearer 1234" \
  -H "Content-Type: application/json" \
  -d '{
    "script": "Speaker 0: Hello world!\nSpeaker 1: Hi there!",
    "num_speakers": 2,
    "cfg_scale": 1.3
  }'
```

## Script Formatting

The API supports two script formats:

### Format 1: Auto-assignment
```
Welcome to our podcast!
Thanks for having me.
Let's discuss AI technology.
```
Lines are automatically assigned to speakers in rotation.

### Format 2: Explicit speakers
```
Speaker 0: Welcome to our podcast!
Speaker 1: Thanks for having me.
Speaker 0: Let's discuss AI technology.
```
Explicitly specify which speaker says each line.

## Voice Selection

- Get available voices with `GET /voices`
- Default speakers: `en-Alice_woman`, `en-Carter_man`, `en-Frank_man`, `en-Maya_woman`
- Voices are loaded from `demo/voices/` directory
- Supported formats: WAV, MP3, FLAC, OGG, M4A, AAC

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `400`: Bad request (invalid parameters)
- `401`: Unauthorized (invalid API key)
- `422`: Validation error
- `500`: Server error
- `503`: Service unavailable (model not loaded)

Error responses include details:
```json
{
    "detail": "Script cannot be empty"
}
```

## Performance Notes

- First request may take longer due to model initialization
- Generation time scales with script length
- Typical generation: 30-60 seconds for short podcasts
- Server processes one generation at a time

## Logging

The server logs all activities to console:
- Request/response logging
- Generation progress
- Error details with stack traces
- Performance metrics

Log level can be configured via `LOG_LEVEL` environment variable.

## Development

### Running Tests
```bash
python test_api.py
```

### Development Mode
Set `RELOAD=true` in `.env` for auto-reload during development.

### Custom Voices
Add voice samples to `demo/voices/` directory and restart the server.