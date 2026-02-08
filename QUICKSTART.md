# World2Data - Quick Start Guide

Get World2Data running in under 5 minutes!

## Prerequisites

- Python 3.8+ (we used Python 3.11)
- pip package manager
- 2GB free disk space (for models)
- Optional: GPU for faster processing

## Installation

```bash
# Clone or navigate to the project directory
cd hacknation

# Install dependencies (takes ~2 minutes)
pip install -r requirements.txt
```

**What gets installed**:
- opencv-python (video processing)
- ultralytics (YOLOv8)
- torch (deep learning backend)
- pillow, numpy (image processing)

## Usage

### Option 1: Use Our Demo Video

```bash
# Generate synthetic demo video
python create_demo_video.py

# Run the enhanced pipeline
python world2data_enhanced.py demo_video.mp4

# View results
open output/side_by_side_demo.mp4
cat output/ground_truth_enhanced.json
```

**Expected output**:
```
✓ Extracted 60 frames from 300 total frames at 30.00 FPS
✓ Generated 60 ground truth entries with temporal tracking
✓ Ground truth saved to output/ground_truth_enhanced.json
  State changes: 2
  Unique actions: ['walking_toward_door', 'opening_door', 'passing_through']
✓ Side-by-side demo saved to output/side_by_side_demo.mp4
```

### Option 2: Use Your Own Video

```bash
# Use any MP4 video of someone walking/navigating
python world2data_enhanced.py your_video.mp4

# Adjust sampling rate for longer videos (default: every 5 frames)
# Edit sample_rate in world2data_enhanced.py line 234
```

## Output Files

After running the pipeline, check the `output/` directory:

### 1. ground_truth_enhanced.json
Structured JSON with:
- Object positions and types
- State labels (closed, opening, open)
- Affordances (traversable, blocked, dynamic_agent)
- Human actions (walking_toward_door, opening_door, etc.)
- Interaction events (state changes)

**Example entry**:
```json
{
  "timestamp": 5.0,
  "frame_index": 150,
  "objects": [
    {
      "type": "door",
      "state": "opening",
      "affordance": "traversable",
      "bbox": [400, 100, 480, 350],
      "confidence": 0.95
    }
  ],
  "human_action": "opening_door",
  "interaction_event": {
    "type": "door_state_change",
    "from_state": "closed",
    "to_state": "opening"
  }
}
```

### 2. side_by_side_demo.mp4
Video showing:
- **Left**: Raw input video
- **Right**: AI-annotated version with bounding boxes, state labels, and action text

## Understanding the Pipeline

### What Happens Under the Hood

1. **Frame Extraction** (1-2s)
   - Reads video at original FPS
   - Samples every Nth frame (default: 5)
   - Extracts as numpy arrays

2. **Object Detection** (20-30s)
   - YOLOv8 Nano model downloads automatically on first run
   - Detects people, doors, furniture
   - Returns bounding boxes + confidence scores

3. **State Analysis** (< 1s)
   - Analyzes door state using color/edge detection
   - For production: integrate VL models here

4. **Temporal Tracking** (< 1s)
   - Tracks state changes across frames
   - Infers human actions from position + context
   - Generates interaction events

5. **Output Generation** (10-15s)
   - Saves JSON ground truth
   - Renders annotated video
   - Creates side-by-side comparison

**Total time**: ~40-60 seconds for 10-second video

## Customization

### Adjust Sampling Rate

Edit `world2data_enhanced.py` line 234:
```python
w2d.run_enhanced_pipeline(sample_rate=5)  # Change 5 to any number
```

- `sample_rate=1`: Process every frame (slower, more accurate)
- `sample_rate=10`: Process every 10th frame (faster, less detail)

### Add More Object Types

Edit `detect_door_state()` and `infer_human_action()` functions to handle:
- Stairs
- Appliances
- Furniture
- Outdoor obstacles

### Integrate Better Models

Replace rule-based detection with VL models:

```python
# Add to requirements.txt
transformers
accelerate

# In world2data_enhanced.py
from transformers import AutoModel, AutoProcessor

model = AutoModel.from_pretrained("LiquidAI/LFM2.5-VL-1.6B")
# Use for scene understanding instead of color detection
```

## Troubleshooting

### Issue: "YOLO model download fails"
**Solution**: Check internet connection. Model downloads automatically from Ultralytics on first run.

### Issue: "Video codec error"
**Solution**: Install ffmpeg:
```bash
# macOS
brew install ffmpeg

# Ubuntu
sudo apt install ffmpeg
```

### Issue: "Out of memory"
**Solution**: Increase sampling rate (process fewer frames):
```python
sample_rate=10  # or higher
```

### Issue: "Detections are inaccurate"
**Solution**: YOLOv8n is optimized for speed. For accuracy, use YOLOv8m:
```python
model = YOLO('yolov8m.pt')  # Medium model
```

## Performance Tips

### Speed Optimization
- Use GPU if available (10x faster YOLO inference)
- Increase `sample_rate` for long videos
- Use YOLOv8n (nano) for demos, YOLOv8s (small) for production

### Quality Optimization
- Decrease `sample_rate` for temporal precision
- Integrate VL models for semantic understanding
- Add SAM3 for object segmentation

## Next Steps

1. **Test on real videos**: Try doorbell camera footage, home videos
2. **Integrate VL models**: Add LiquidAI LFM2.5-VL for better semantics
3. **Extend environments**: Add kitchen, office, outdoor scenarios
4. **Build datasets**: Generate ground truth for robot navigation training

## Support

For issues, questions, or contributions:
- Check [README.md](README.md) for full documentation
- Review [TECHNICAL_SUMMARY.md](TECHNICAL_SUMMARY.md) for architecture details
- See [PROJECT_DESCRIPTION.md](PROJECT_DESCRIPTION.md) for vision and roadmap

---

**Happy ground truth generation! 🤖**

Built for HackNation AI Hackathon 2026
