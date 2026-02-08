# World2Data - AI-Powered Ground Truth Generation for Humanoid Navigation

**Turning the Physical World into Labeled Training Data for Robots**

## 🎯 The Problem

Humanoid robots fail in real-world environments not because they can't see or move, but because **the physical world isn't labeled**. To navigate homes and offices, robots need ground truth about:
- What objects are
- Which ones can be interacted with
- What state they're in (open/closed, traversable/blocked)
- How humans move through and manipulate space

Today, this ground truth is created manually—slowly, expensively, at small scale.

## 💡 Our Solution

**World2Data** is an AI-powered pipeline that automatically converts raw video of humans navigating spaces into structured, navigation-relevant ground truth labels.

### Core Innovation
We combine multiple AI models to create a **ground truth generation engine**:
- **YOLO** for fast object detection
- **Vision-Language Models** for semantic understanding
- **Temporal tracking** for state change detection
- **Affordance labeling** for interaction understanding

## 🚀 Demo: Door Navigation Ground Truth

### Input
Raw video of a person approaching and opening a door.

### Output
Structured JSON ground truth:
```json
{
  "timestamp": 2.4,
  "objects": [
    {
      "type": "door",
      "state": "closed",
      "affordance": "traversable",
      "bbox": [245, 120, 380, 450]
    },
    {
      "type": "person",
      "affordance": "dynamic_obstacle",
      "bbox": [150, 200, 280, 470]
    }
  ],
  "human_action": "approaching_door",
  "interaction_event": null
}
```

Plus annotated video with bounding boxes and state labels overlaid.

## 🏗️ Architecture

```
Raw Video → Frame Extraction → Object Detection (YOLO)
                              ↓
                    Scene Understanding (VL Model)
                              ↓
                    Ground Truth Generation
                              ↓
                    Structured JSON + Annotated Video
```

## 🎬 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run pipeline on your video
python world2data.py demo_video.mp4

# Output:
# - output/ground_truth.json (structured labels)
# - output/annotated_output.mp4 (visualized)
```

## 📊 Key Features

✅ **Zero-Shot Baseline**: Works out-of-the-box with pre-trained models
✅ **Temporal Precision**: Tracks state changes over time
✅ **Affordance Labeling**: Not just "door" but "traversable door in closed state"
✅ **Human-as-Signal**: Uses human motion as implicit supervision
✅ **Scalable**: Turn any home video into training data

## 🎯 Evaluation Metrics

- **Ground Truth Accuracy**: Objects, states, interactions correctly identified
- **Temporal Precision**: Accurate boundaries for actions and transitions
- **Scalability**: Automated labeling vs manual effort reduction
- **Demo Clarity**: Visual before/after transformation

## 🌟 Impact

If AI can generate ground truth at scale:
- Homes become training environments
- Navigation systems adapt continuously
- Humanoids improve safely in the wild
- Physical intelligence compounds over time

**Before robots can navigate the human world, they must first understand what the world means.**

World2Data builds that understanding.

## 🏆 HackNation AI Hackathon - VC Track
Built in <2 hours to demonstrate the future of humanoid robot training data.

---

**Team**: [Your Team Name]
**Track**: Venture Capital
**Challenge**: World2Data - Physical World Ground Truth
