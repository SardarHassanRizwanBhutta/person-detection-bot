# Person Detection Welcome System

A production-ready YOLOv8-based person detection system that plays welcome audio when people are detected.

## Features

- 🎯 Real-time person detection using YOLOv8
- 🎵 Smart audio playback with multiple strategies
- 🔄 Detection gap strategy (no repeated welcomes)
- 📊 Performance monitoring and logging
- ⚙️ Configurable settings via JSON
- 🎥 Camera feed with optional bounding boxes
- 🔄 Auto-recovery from camera/audio failures

## Quick Setup (New Machine)

### Option 1: Automated Setup
```bash
# Run the setup script
setup.bat
```

### Option 2: Manual Setup
```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Running the System

### Option 1: Using Batch File
```bash
start_system.bat
```

### Option 2: Manual Start
```bash
# Activate virtual environment
venv\Scripts\activate

# Run the system
python production_welcome_system.py
```

## Configuration

Edit `config.json` to customize:

```json
{
  "detection": {
    "confidence_threshold": 0.6,
    "camera_index": 1,
    "frame_width": 640,
    "frame_height": 480
  },
  "audio": {
    "playback_strategy": "smart_rotation",
    "cooldown_seconds": 3.0,
    "detection_gap_threshold": 2.0,
    "volume": 0.8
  }
}
```

## Audio Files

Place your audio files (MP3, WAV, OGG) in the `audio_files/` folder. The system will automatically discover and use them.

## Key Features

### Detection Gap Strategy
- Welcomes each person only once per visit
- No repeated audio when someone lingers
- Configurable gap threshold (default: 2 seconds)

### Audio Strategies
- **smart_rotation**: Plays all sounds before repeating
- **random**: Random selection
- **sequential**: Plays in order

### Monitoring
- Real-time FPS tracking
- Memory usage monitoring
- Error recovery and logging
- Performance statistics

## Controls

- **Q**: Quit the system
- **Ctrl+C**: Emergency stop

## Troubleshooting

### Camera Issues
- Try different `camera_index` values (0, 1, 2...)
- Check camera permissions
- Ensure no other apps are using the camera

### Audio Issues
- Check audio files are in `audio_files/` folder
- Verify file formats (MP3, WAV, OGG supported)
- Check system volume settings

### Dependencies
If installation fails, try:
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## System Requirements

- Python 3.8+
- Windows 10/11
- Webcam
- Audio output device
- 4GB+ RAM recommended

## Project Structure

```
person_detection/
├── production_welcome_system.py  # Main system
├── config.json                   # Configuration
├── requirements.txt              # Dependencies
├── setup.bat                     # Setup script
├── start_system.bat             # Run script
├── audio_files/                 # Audio files folder
├── logs/                        # System logs
└── venv/                        # Virtual environment (not in git)
```
