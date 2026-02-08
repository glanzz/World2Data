#!/usr/bin/env python3
"""
World2Data - AI-Powered Ground Truth Generation for Humanoid Navigation
Converts raw video of human navigation into structured ground truth labels.
"""

import cv2
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from datetime import datetime

class World2Data:
    def __init__(self, video_path: str, output_dir: str = "output"):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.ground_truth_data = []
        self.frame_count = 0

    def extract_frames(self, sample_rate: int = 5):
        """Extract frames from video at specified sample rate (every N frames)."""
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        frames = []
        frame_indices = []

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate == 0:
                frames.append(frame)
                frame_indices.append(frame_idx)

            frame_idx += 1

        cap.release()
        self.fps = fps
        self.total_frames = frame_idx

        print(f"✓ Extracted {len(frames)} frames from {frame_idx} total frames at {fps:.2f} FPS")
        return frames, frame_indices

    def detect_objects(self, frames: List[np.ndarray]):
        """Run YOLO object detection on frames."""
        from ultralytics import YOLO

        print("Loading YOLO model...")
        model = YOLO('yolov8n.pt')  # Nano model for speed

        detections = []
        for idx, frame in enumerate(frames):
            results = model(frame, verbose=False)
            frame_detections = []

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    bbox = box.xyxy[0].tolist()

                    frame_detections.append({
                        "class_id": cls_id,
                        "class_name": model.names[cls_id],
                        "confidence": conf,
                        "bbox": bbox
                    })

            detections.append(frame_detections)
            if idx % 10 == 0:
                print(f"  Processed {idx+1}/{len(frames)} frames")

        print(f"✓ Object detection complete")
        return detections, model

    def generate_descriptions(self, frames: List[np.ndarray]):
        """Generate scene descriptions using VL model (placeholder for now)."""
        # For MVP: Simple rule-based descriptions
        # In production: Use LiquidAI VL model
        descriptions = []

        for idx, frame in enumerate(frames):
            # Placeholder description
            desc = {
                "scene": "indoor_environment",
                "activity": "navigation_task",
                "frame_index": idx
            }
            descriptions.append(desc)

        return descriptions

    def generate_ground_truth(self, frames, frame_indices, detections, descriptions):
        """Generate structured ground truth JSON."""
        ground_truth = []

        for idx, (frame_idx, dets, desc) in enumerate(zip(frame_indices, detections, descriptions)):
            timestamp = frame_idx / self.fps

            # Identify navigation-relevant objects
            doors = [d for d in dets if d['class_name'] in ['door']]
            people = [d for d in dets if d['class_name'] == 'person']

            # Determine affordances and states
            entry = {
                "timestamp": round(timestamp, 2),
                "frame_index": frame_idx,
                "objects": [],
                "human_action": "unknown",
                "interaction_event": None
            }

            # Add detected objects with affordances
            for det in dets:
                obj = {
                    "type": det['class_name'],
                    "bbox": [round(x, 2) for x in det['bbox']],
                    "confidence": round(det['confidence'], 3)
                }

                # Add affordance labels for navigation-relevant objects
                if det['class_name'] == 'door':
                    obj['affordance'] = 'traversable'
                    obj['state'] = 'unknown'  # Would be determined by VL model
                elif det['class_name'] in ['chair', 'couch', 'bench']:
                    obj['affordance'] = 'obstacle'
                elif det['class_name'] == 'person':
                    obj['affordance'] = 'dynamic_obstacle'

                entry['objects'].append(obj)

            # Infer human action from context
            if people:
                entry['human_action'] = 'present'

            ground_truth.append(entry)

        self.ground_truth_data = ground_truth
        return ground_truth

    def save_ground_truth(self, filename: str = "ground_truth.json"):
        """Save ground truth to JSON file."""
        output_path = self.output_dir / filename

        metadata = {
            "video_source": str(self.video_path),
            "fps": self.fps,
            "total_frames": self.total_frames,
            "sampled_frames": len(self.ground_truth_data),
            "generated_at": datetime.now().isoformat()
        }

        output = {
            "metadata": metadata,
            "ground_truth": self.ground_truth_data
        }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"✓ Ground truth saved to {output_path}")
        return output_path

    def visualize(self, frames, frame_indices, detections, output_video: str = "annotated_output.mp4"):
        """Create annotated video with bounding boxes and labels."""
        if not frames:
            print("No frames to visualize")
            return

        height, width = frames[0].shape[:2]
        output_path = self.output_dir / output_video

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, self.fps / 5, (width, height))

        for idx, (frame, dets) in enumerate(zip(frames, detections)):
            annotated = frame.copy()

            # Draw bounding boxes
            for det in dets:
                bbox = det['bbox']
                x1, y1, x2, y2 = map(int, bbox)

                # Color code by object type
                if det['class_name'] == 'person':
                    color = (0, 255, 0)  # Green
                elif det['class_name'] == 'door':
                    color = (255, 0, 0)  # Blue
                else:
                    color = (0, 165, 255)  # Orange

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                # Label
                label = f"{det['class_name']}: {det['confidence']:.2f}"
                cv2.putText(annotated, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Add timestamp
            timestamp = frame_indices[idx] / self.fps
            cv2.putText(annotated, f"t={timestamp:.2f}s", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            out.write(annotated)

        out.release()
        print(f"✓ Annotated video saved to {output_path}")
        return output_path

    def run_pipeline(self, sample_rate: int = 5):
        """Run the complete World2Data pipeline."""
        print("=" * 60)
        print("WORLD2DATA - Ground Truth Generation Pipeline")
        print("=" * 60)

        # Step 1: Extract frames
        print("\n[1/5] Extracting video frames...")
        frames, frame_indices = self.extract_frames(sample_rate)

        # Step 2: Object detection
        print("\n[2/5] Running object detection...")
        detections, yolo_model = self.detect_objects(frames)

        # Step 3: Generate descriptions
        print("\n[3/5] Generating scene descriptions...")
        descriptions = self.generate_descriptions(frames)

        # Step 4: Generate ground truth
        print("\n[4/5] Generating structured ground truth...")
        ground_truth = self.generate_ground_truth(frames, frame_indices, detections, descriptions)

        # Step 5: Save and visualize
        print("\n[5/5] Saving outputs...")
        self.save_ground_truth()
        self.visualize(frames, frame_indices, detections)

        print("\n" + "=" * 60)
        print("✓ PIPELINE COMPLETE")
        print(f"  Generated {len(ground_truth)} ground truth annotations")
        print(f"  Output directory: {self.output_dir}")
        print("=" * 60)

        return ground_truth


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python world2data.py <video_path>")
        print("Example: python world2data.py demo_video.mp4")
        sys.exit(1)

    video_path = sys.argv[1]

    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found")
        sys.exit(1)

    # Run pipeline
    w2d = World2Data(video_path)
    w2d.run_pipeline(sample_rate=5)


if __name__ == "__main__":
    main()
