#!/usr/bin/env python3
"""
World2Data Real-World - Enhanced for actual door navigation videos
Uses YOLO for door detection + edge-based state analysis
"""

import cv2
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from datetime import datetime

class World2DataRealWorld:
    def __init__(self, video_path: str, output_dir: str = "output"):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.ground_truth_data = []
        self.frame_count = 0

    def extract_frames(self, sample_rate: int = 5):
        """Extract frames from video."""
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

    def detect_objects_yolo(self, frames: List[np.ndarray]):
        """Use YOLO to detect objects including doors."""
        from ultralytics import YOLO

        print("Loading YOLO model for object detection...")
        model = YOLO('yolov8n.pt')

        all_detections = []
        for idx, frame in enumerate(frames):
            results = model(frame, verbose=False)
            frame_detections = {
                'doors': [],
                'people': [],
                'other': []
            }

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]
                    conf = float(box.conf[0])
                    bbox = box.xyxy[0].tolist()

                    detection = {
                        "class_name": cls_name,
                        "confidence": conf,
                        "bbox": bbox
                    }

                    # Categorize detections
                    if cls_name == 'door':
                        frame_detections['doors'].append(detection)
                    elif cls_name == 'person':
                        frame_detections['people'].append(detection)
                    else:
                        frame_detections['other'].append(detection)

            all_detections.append(frame_detections)

            if idx % 20 == 0:
                print(f"  Processed {idx+1}/{len(frames)} frames")

        print(f"✓ Object detection complete")
        return all_detections, model

    def detect_door_from_edges(self, frame: np.ndarray, roi: Tuple[int, int, int, int] = None) -> Dict:
        """
        Detect door using edge detection and geometric analysis.
        Works better for real-world doors than color detection.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Look for door-like rectangular contours
        door_candidates = []
        height, width = frame.shape[:2]

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 5000:  # Too small to be a door
                continue

            # Approximate contour to polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Door heuristics: vertical rectangle, reasonable size
            aspect_ratio = h / w if w > 0 else 0
            size_ratio = (w * h) / (width * height)

            if (1.5 < aspect_ratio < 4.0 and  # Vertical rectangle
                0.05 < size_ratio < 0.6 and    # Reasonable size
                w > 50 and h > 100):           # Minimum dimensions

                door_candidates.append({
                    'bbox': [x, y, x+w, y+h],
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'confidence': min(0.95, area / 50000)  # Heuristic confidence
                })

        if door_candidates:
            # Return largest door candidate
            best_door = max(door_candidates, key=lambda d: d['area'])
            return best_door

        return None

    def analyze_door_state(self, frame: np.ndarray, door_bbox: List[float]) -> str:
        """
        Analyze door state using region analysis.
        Detects: closed, partially_open, open
        """
        x1, y1, x2, y2 = map(int, door_bbox)

        # Extract door region
        door_region = frame[y1:y2, x1:x2]

        if door_region.size == 0:
            return "unknown"

        # Convert to grayscale
        gray = cv2.cvtColor(door_region, cv2.COLOR_BGR2GRAY)

        # Analyze vertical edges (strong vertical edges = closed door)
        edges = cv2.Canny(gray, 50, 150)
        vertical_edges = np.sum(edges, axis=0)  # Sum along columns

        # Standard deviation of edge intensity
        edge_variance = np.std(vertical_edges)

        # Analyze door region complexity
        # Closed door: uniform, low variance
        # Open door: see through, high variance
        mean_intensity = np.mean(gray)
        intensity_variance = np.std(gray)

        # Heuristic state detection
        if edge_variance < 50 and intensity_variance < 40:
            return "closed"
        elif edge_variance > 150 or intensity_variance > 80:
            return "open"
        else:
            return "partially_open"

    def detect_person_and_action(self, frame: np.ndarray, person_detections: List[Dict],
                                 door_bbox: List[float] = None) -> Tuple[List[Dict], str]:
        """Detect people and infer their actions relative to door."""
        if not person_detections:
            return [], "no_person_visible"

        action = "person_present"

        if door_bbox and person_detections:
            # Get door center
            door_x_center = (door_bbox[0] + door_bbox[2]) / 2

            # Get person center
            person = person_detections[0]  # Primary person
            person_x_center = (person['bbox'][0] + person['bbox'][2]) / 2

            # Calculate distance to door
            distance = abs(person_x_center - door_x_center)
            frame_width = frame.shape[1]

            if distance < frame_width * 0.15:  # Within 15% of frame width
                action = "at_door"
            elif person_x_center < door_x_center:
                action = "approaching_door"
            else:
                action = "passed_door"

        return person_detections, action

    def generate_ground_truth_realworld(self, frames, frame_indices, yolo_detections):
        """Generate ground truth with real-world door and person detection."""
        ground_truth = []
        prev_door_state = None

        print("Analyzing frames for real-world ground truth...")

        for idx, (frame_idx, frame, yolo_det) in enumerate(zip(frame_indices, frames, yolo_detections)):
            timestamp = frame_idx / self.fps

            # Start with empty entry
            entry = {
                "timestamp": round(timestamp, 2),
                "frame_index": frame_idx,
                "objects": [],
                "human_action": "unknown",
                "interaction_event": None
            }

            # Detect door (try YOLO first, fallback to edge detection)
            door_bbox = None
            door_state = "not_visible"

            if yolo_det['doors']:
                # Use YOLO door detection
                door = yolo_det['doors'][0]  # Take first door
                door_bbox = door['bbox']
                door_state = self.analyze_door_state(frame, door_bbox)

                entry['objects'].append({
                    "type": "door",
                    "state": door_state,
                    "affordance": "traversable" if door_state == "open" else "blocked",
                    "bbox": [round(x, 2) for x in door_bbox],
                    "confidence": round(door['confidence'], 3),
                    "detection_method": "yolo"
                })
            else:
                # Fallback: edge-based door detection
                door_edge = self.detect_door_from_edges(frame)
                if door_edge:
                    door_bbox = door_edge['bbox']
                    door_state = self.analyze_door_state(frame, door_bbox)

                    entry['objects'].append({
                        "type": "door",
                        "state": door_state,
                        "affordance": "traversable" if door_state == "open" else "blocked",
                        "bbox": [round(x, 2) for x in door_bbox],
                        "confidence": round(door_edge['confidence'], 3),
                        "detection_method": "edge_detection"
                    })

            # Detect people and infer action
            people, action = self.detect_person_and_action(frame, yolo_det['people'], door_bbox)

            for person in people:
                entry['objects'].append({
                    "type": "person",
                    "affordance": "dynamic_agent",
                    "bbox": [round(x, 2) for x in person['bbox']],
                    "confidence": round(person['confidence'], 3)
                })

            entry['human_action'] = action

            # Detect state change events
            if prev_door_state and door_state != prev_door_state and door_state != "not_visible":
                entry['interaction_event'] = {
                    "type": "door_state_change",
                    "from_state": prev_door_state,
                    "to_state": door_state,
                    "timestamp": timestamp
                }

            ground_truth.append(entry)
            prev_door_state = door_state if door_state != "not_visible" else prev_door_state

            if idx % 20 == 0:
                print(f"  Analyzed {idx+1}/{len(frames)} frames")

        self.ground_truth_data = ground_truth
        print(f"✓ Generated {len(ground_truth)} ground truth entries")
        return ground_truth

    def save_ground_truth(self, filename: str = "ground_truth_realworld.json"):
        """Save ground truth to JSON."""
        output_path = self.output_dir / filename

        state_changes = sum(1 for entry in self.ground_truth_data if entry['interaction_event'])
        unique_actions = set(entry['human_action'] for entry in self.ground_truth_data)
        door_detections = sum(1 for entry in self.ground_truth_data
                            if any(obj['type'] == 'door' for obj in entry['objects']))

        metadata = {
            "video_source": str(self.video_path),
            "fps": self.fps,
            "total_frames": self.total_frames,
            "sampled_frames": len(self.ground_truth_data),
            "door_detections": door_detections,
            "state_changes_detected": state_changes,
            "unique_actions_detected": len(unique_actions),
            "actions": list(unique_actions),
            "generated_at": datetime.now().isoformat(),
            "model": "World2Data_RealWorld_v1.0"
        }

        output = {
            "metadata": metadata,
            "ground_truth": self.ground_truth_data
        }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"✓ Ground truth saved to {output_path}")
        print(f"  Door detections: {door_detections}/{len(self.ground_truth_data)} frames")
        print(f"  State changes: {state_changes}")
        print(f"  Actions: {list(unique_actions)}")
        return output_path

    def create_visualization(self, frames, frame_indices, output_video: str = "realworld_demo.mp4"):
        """Create side-by-side visualization for real-world footage."""
        if not frames:
            return

        height, width = frames[0].shape[:2]
        output_path = self.output_dir / output_video

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, self.fps / 5, (width * 2, height))

        print("Creating visualization...")

        for idx, (frame_idx, frame) in enumerate(zip(frame_indices, frames)):
            raw = frame.copy()
            cv2.putText(raw, "RAW INPUT", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            annotated = frame.copy()
            gt_entry = self.ground_truth_data[idx]

            # Draw bounding boxes and labels
            for obj in gt_entry['objects']:
                bbox = obj['bbox']
                x1, y1, x2, y2 = map(int, bbox)

                if obj['type'] == 'person':
                    color = (0, 255, 0)  # Green
                    label = f"Person ({obj['confidence']:.2f})"
                elif obj['type'] == 'door':
                    color = (255, 0, 0)  # Blue
                    state = obj.get('state', 'unknown')
                    method = obj.get('detection_method', 'unknown')
                    label = f"Door: {state} ({obj['confidence']:.2f})"
                else:
                    color = (0, 165, 255)
                    label = obj['type']

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                cv2.putText(annotated, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Add action label
            action_text = f"Action: {gt_entry['human_action']}"
            cv2.putText(annotated, action_text, (10, height - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Timestamp
            timestamp = gt_entry['timestamp']
            cv2.putText(annotated, f"t={timestamp:.2f}s", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Title
            cv2.putText(annotated, "AI GROUND TRUTH", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Event indicator
            if gt_entry['interaction_event']:
                event = gt_entry['interaction_event']
                event_text = f"STATE CHANGE: {event['from_state']} -> {event['to_state']}"
                cv2.putText(annotated, event_text, (10, height - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Combine
            combined = np.hstack([raw, annotated])
            out.write(combined)

            if idx % 20 == 0:
                print(f"  Rendered {idx+1}/{len(frames)} frames")

        out.release()
        print(f"✓ Visualization saved to {output_path}")
        return output_path

    def run_pipeline(self, sample_rate: int = 5):
        """Run the complete real-world pipeline."""
        print("=" * 70)
        print("WORLD2DATA REAL-WORLD - Door Navigation Ground Truth")
        print("=" * 70)

        print("\n[1/5] Extracting video frames...")
        frames, frame_indices = self.extract_frames(sample_rate)

        print("\n[2/5] Running YOLO object detection...")
        yolo_detections, model = self.detect_objects_yolo(frames)

        print("\n[3/5] Generating real-world ground truth...")
        self.generate_ground_truth_realworld(frames, frame_indices, yolo_detections)

        print("\n[4/5] Saving structured ground truth...")
        self.save_ground_truth()

        print("\n[5/5] Creating visualization...")
        self.create_visualization(frames, frame_indices)

        print("\n" + "=" * 70)
        print("✓ PIPELINE COMPLETE")
        print(f"  Output: {self.output_dir}")
        print("=" * 70)


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python world2data_realworld.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found")
        sys.exit(1)

    w2d = World2DataRealWorld(video_path)
    w2d.run_pipeline(sample_rate=5)


if __name__ == "__main__":
    main()
