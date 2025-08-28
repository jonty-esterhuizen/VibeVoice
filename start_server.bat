@echo off
echo Starting VibeVoice API Server...
echo.
echo Configuration:
echo - API Key: 1234
echo - Port: 8000
echo - Model: models/microsoft--VibeVoice-1.5B
echo.
echo Server will be available at: http://localhost:8000
echo API Documentation: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

python server.py

pause