#!/usr/bin/env python3
"""
World2Data Enhanced - AI-Powered Ground Truth with Temporal State Tracking
Demonstrates temporal precision and state change detection for humanoid navigation.
"""

import cv2
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from datetime import datetime

class World2DataEnhanced:
    def __init__(self, video_path: str, output_dir: str = "output"):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.ground_truth_data = []
        self.frame_count = 0

    def extract_frames(self, sample_rate: int = 5):
        """Extract frames from video at specified sample rate."""
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

    def detect_door_state(self, frame: np.ndarray):
        """Detect door state using color/edge analysis (for our synthetic video)."""
        # For real videos, this would use a VL model
        # For our demo: detect based on brown pixels (door color)

        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Brown door color range
        lower_brown = np.array([5, 50, 50])
        upper_brown = np.array([20, 255, 200])

        mask = cv2.inRange(hsv, lower_brown, upper_brown)
        door_pixels = cv2.countNonZero(mask)

        # Determine state based on door pixel count
        if door_pixels > 15000:
            return "closed"
        elif door_pixels > 5000:
            return "opening"
        elif door_pixels > 1000:
            return "open"
        else:
            return "not_visible"

    def detect_person_position(self, frame: np.ndarray):
        """Detect person position (simple color-based for our demo)."""
        # Red body color (BGR)
        lower_red = np.array([0, 0, 200])
        upper_red = np.array([100, 100, 255])

        mask = cv2.inRange(frame, lower_red, upper_red)

        # Find centroid
        moments = cv2.moments(mask)
        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])

            # Find bounding box
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                x, y, w, h = cv2.boundingRect(contours[0])
                return {
                    "detected": True,
                    "centroid": (cx, cy),
                    "bbox": [x, y, x+w, y+h]
                }

        return {"detected": False}

    def infer_human_action(self, person_pos, door_state, prev_person_pos=None):
        """Infer human action from position and door state."""
        if not person_pos["detected"]:
            return "not_visible"

        cx, cy = person_pos["centroid"]

        # Door is around x=400-480
        door_region = (350, 500)

        if door_region[0] < cx < door_region[1]:
            if door_state == "closed":
                return "approaching_door"
            elif door_state == "opening":
                return "opening_door"
            elif door_state == "open":
                return "passing_through_door"
        elif cx < door_region[0]:
            return "walking_toward_door"
        else:
            return "passed_through_door"

        return "navigating"

    def generate_ground_truth_enhanced(self, frames, frame_indices):
        """Generate enhanced ground truth with temporal tracking."""
        ground_truth = []
        prev_person_pos = None
        prev_door_state = None

        print("Analyzing frames for temporal ground truth...")

        for idx, (frame_idx, frame) in enumerate(zip(frame_indices, frames)):
            timestamp = frame_idx / self.fps

            # Detect door state
            door_state = self.detect_door_state(frame)

            # Detect person
            person_pos = self.detect_person_position(frame)

            # Infer action
            human_action = self.infer_human_action(person_pos, door_state, prev_person_pos)

            # Detect state change event
            interaction_event = None
            if prev_door_state and door_state != prev_door_state:
                interaction_event = {
                    "type": "door_state_change",
                    "from_state": prev_door_state,
                    "to_state": door_state,
                    "timestamp": timestamp
                }

            # Build ground truth entry
            entry = {
                "timestamp": round(timestamp, 2),
                "frame_index": frame_idx,
                "objects": [],
                "human_action": human_action,
                "interaction_event": interaction_event
            }

            # Add door object
            if door_state != "not_visible":
                entry["objects"].append({
                    "type": "door",
                    "state": door_state,
                    "affordance": "traversable" if door_state == "open" else "blocked",
                    "bbox": [400, 100, 480, 350],  # Approximate door region
                    "confidence": 0.95
                })

            # Add person object
            if person_pos["detected"]:
                entry["objects"].append({
                    "type": "person",
                    "affordance": "dynamic_agent",
                    "bbox": person_pos["bbox"],
                    "confidence": 0.90
                })

            ground_truth.append(entry)
            prev_person_pos = person_pos
            prev_door_state = door_state

        self.ground_truth_data = ground_truth
        print(f"✓ Generated {len(ground_truth)} ground truth entries with temporal tracking")
        return ground_truth

    def save_ground_truth(self, filename: str = "ground_truth_enhanced.json"):
        """Save enhanced ground truth to JSON."""
        output_path = self.output_dir / filename

        # Calculate statistics
        state_changes = sum(1 for entry in self.ground_truth_data if entry['interaction_event'])
        unique_actions = set(entry['human_action'] for entry in self.ground_truth_data)

        metadata = {
            "video_source": str(self.video_path),
            "fps": self.fps,
            "total_frames": self.total_frames,
            "sampled_frames": len(self.ground_truth_data),
            "state_changes_detected": state_changes,
            "unique_actions_detected": len(unique_actions),
            "actions": list(unique_actions),
            "generated_at": datetime.now().isoformat(),
            "model": "World2Data_Enhanced_v1.0"
        }

        output = {
            "metadata": metadata,
            "ground_truth": self.ground_truth_data
        }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"✓ Ground truth saved to {output_path}")
        print(f"  State changes: {state_changes}")
        print(f"  Unique actions: {list(unique_actions)}")
        return output_path

    def create_side_by_side_visualization(self, frames, frame_indices,
                                         output_video: str = "side_by_side_demo.mp4"):
        """Create side-by-side comparison: raw vs annotated."""
        if not frames:
            return

        height, width = frames[0].shape[:2]
        output_path = self.output_dir / output_video

        # Double width for side-by-side
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, self.fps / 5, (width * 2, height))

        print("Creating side-by-side visualization...")

        for idx, (frame_idx, frame) in enumerate(zip(frame_indices, frames)):
            # Left: Raw video
            raw = frame.copy()
            cv2.putText(raw, "RAW INPUT", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Right: Annotated
            annotated = frame.copy()

            gt_entry = self.ground_truth_data[idx]

            # Draw bounding boxes
            for obj in gt_entry['objects']:
                bbox = obj['bbox']
                x1, y1, x2, y2 = map(int, bbox)

                if obj['type'] == 'person':
                    color = (0, 255, 0)  # Green
                elif obj['type'] == 'door':
                    color = (255, 0, 0)  # Blue
                else:
                    color = (0, 165, 255)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                # Label with state
                if 'state' in obj:
                    label = f"{obj['type']}: {obj['state']}"
                else:
                    label = obj['type']

                cv2.putText(annotated, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Add action label
            action_text = f"Action: {gt_entry['human_action']}"
            cv2.putText(annotated, action_text, (10, height - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Add timestamp
            timestamp = gt_entry['timestamp']
            cv2.putText(annotated, f"t={timestamp:.2f}s", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Add "GROUND TRUTH" title
            cv2.putText(annotated, "AI GROUND TRUTH", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Event indicator
            if gt_entry['interaction_event']:
                cv2.putText(annotated, "STATE CHANGE!", (10, height - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Combine side-by-side
            combined = np.hstack([raw, annotated])
            out.write(combined)

        out.release()
        print(f"✓ Side-by-side demo saved to {output_path}")
        return output_path

    def run_enhanced_pipeline(self, sample_rate: int = 5):
        """Run the enhanced World2Data pipeline with temporal tracking."""
        print("=" * 70)
        print("WORLD2DATA ENHANCED - Temporal Ground Truth Generation")
        print("=" * 70)

        print("\n[1/4] Extracting video frames...")
        frames, frame_indices = self.extract_frames(sample_rate)

        print("\n[2/4] Generating temporal ground truth with state tracking...")
        self.generate_ground_truth_enhanced(frames, frame_indices)

        print("\n[3/4] Saving structured ground truth...")
        self.save_ground_truth()

        print("\n[4/4] Creating side-by-side visualization...")
        self.create_side_by_side_visualization(frames, frame_indices)

        print("\n" + "=" * 70)
        print("✓ ENHANCED PIPELINE COMPLETE")
        print(f"  Output directory: {self.output_dir}")
        print("  Files generated:")
        print("    - ground_truth_enhanced.json (structured labels)")
        print("    - side_by_side_demo.mp4 (visual comparison)")
        print("=" * 70)


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python world2data_enhanced.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found")
        sys.exit(1)

    w2d = World2DataEnhanced(video_path)
    w2d.run_enhanced_pipeline(sample_rate=5)


if __name__ == "__main__":
    main()
