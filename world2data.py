#!/usr/bin/env python3
"""
World2Data V2 - Real-World Door Detection
Uses advanced CV techniques: edge detection, Hough lines, contour analysis
"""

import cv2
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from datetime import datetime

class DoorDetector:
    """Advanced door detection using computer vision"""

    def __init__(self):
        self.prev_door_bbox = None

    def detect_door(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Detect door using multiple CV techniques:
        1. Edge detection (Canny)
        2. Line detection (Hough)
        3. Contour analysis
        4. Geometric validation
        """
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # Edge detection
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)

        # Morphological operations to connect door edges
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        door_candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by area (doors are typically large)
            if area < 3000 or area > width * height * 0.7:
                continue

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Door shape validation
            aspect_ratio = h / w if w > 0 else 0
            size_ratio = area / (width * height)

            # Heuristics for door-like shapes:
            # - Aspect ratio 1.5-5.0 (vertical rectangle)
            # - Size ratio 0.03-0.5 (significant but not full frame)
            # - Minimum dimensions
            # - Positioned in reasonable location (not at extreme edges)

            is_vertical = 1.5 < aspect_ratio < 5.0
            reasonable_size = 0.03 < size_ratio < 0.5
            min_dimensions = w > 40 and h > 80
            reasonable_position = 0.1 * width < x < 0.9 * width

            if is_vertical and reasonable_size and min_dimensions and reasonable_position:
                # Calculate confidence score
                confidence = self._calculate_door_confidence(frame, x, y, w, h, aspect_ratio, size_ratio)

                door_candidates.append({
                    'bbox': [x, y, x+w, y+h],
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'confidence': confidence,
                    'center': (x + w//2, y + h//2)
                })

        if not door_candidates:
            # Try tracking from previous frame if available
            if self.prev_door_bbox:
                # Return previous bbox with lower confidence
                prev_bbox = self.prev_door_bbox.copy()
                prev_bbox['confidence'] *= 0.8  # Decay confidence
                if prev_bbox['confidence'] > 0.3:
                    return prev_bbox
            return None

        # Select best candidate
        # Prefer larger, more centered doors with good aspect ratio
        best_door = max(door_candidates, key=lambda d: d['confidence'])

        # Smooth with previous detection
        if self.prev_door_bbox:
            prev_bbox = self.prev_door_bbox['bbox']
            curr_bbox = best_door['bbox']

            # If very close to previous detection, smooth the bbox
            iou = self._calculate_iou(prev_bbox, curr_bbox)
            if iou > 0.5:
                # Weighted average
                alpha = 0.7
                best_door['bbox'] = [
                    alpha * curr_bbox[i] + (1-alpha) * prev_bbox[i]
                    for i in range(4)
                ]

        self.prev_door_bbox = best_door
        return best_door

    def _calculate_door_confidence(self, frame, x, y, w, h, aspect_ratio, size_ratio):
        """Calculate confidence score for door detection"""
        # Base confidence from aspect ratio (peak at 2.5)
        aspect_score = 1.0 - abs(aspect_ratio - 2.5) / 2.5
        aspect_score = max(0, min(1, aspect_score))

        # Size score (peak at 0.15)
        size_score = 1.0 - abs(size_ratio - 0.15) / 0.15
        size_score = max(0, min(1, size_score))

        # Extract door region
        door_region = frame[y:y+h, x:x+w]
        if door_region.size == 0:
            return 0.5

        # Analyze vertical structure (doors have strong vertical lines)
        gray_region = cv2.cvtColor(door_region, cv2.COLOR_BGR2GRAY)
        edges_region = cv2.Canny(gray_region, 50, 150)

        # Count vertical vs horizontal edges
        vertical_edges = np.sum(edges_region, axis=0)
        horizontal_edges = np.sum(edges_region, axis=1)

        v_strength = np.std(vertical_edges)
        h_strength = np.std(horizontal_edges)

        # Doors have stronger vertical structure
        structure_score = v_strength / (v_strength + h_strength + 1) if (v_strength + h_strength) > 0 else 0.5

        # Combine scores
        confidence = (0.4 * aspect_score + 0.3 * size_score + 0.3 * structure_score)
        return min(0.95, max(0.3, confidence))

    def _calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # Intersection
        x_min = max(x1_min, x2_min)
        y_min = max(y1_min, y2_min)
        x_max = min(x1_max, x2_max)
        y_max = min(y1_max, y2_max)

        if x_max < x_min or y_max < y_min:
            return 0.0

        intersection = (x_max - x_min) * (y_max - y_min)

        # Union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def analyze_door_state(self, frame: np.ndarray, door_bbox: List[float]) -> str:
        """Analyze whether door is closed, partially open, or open"""
        x1, y1, x2, y2 = map(int, door_bbox)
        door_region = frame[y1:y2, x1:x2]

        if door_region.size == 0:
            return "unknown"

        # Convert to grayscale
        gray = cv2.cvtColor(door_region, cv2.COLOR_BGR2GRAY)

        # Detect edges
        edges = cv2.Canny(gray, 50, 150)

        # Analyze edge density and distribution
        edge_density = np.sum(edges > 0) / edges.size

        # Analyze vertical lines (closed door has consistent vertical line)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=10)

        vertical_lines = 0
        if lines is not None:
            for line in lines:
                x1_l, y1_l, x2_l, y2_l = line[0]
                angle = np.abs(np.arctan2(y2_l - y1_l, x2_l - x1_l) * 180 / np.pi)
                if angle > 75 and angle < 105:  # Vertical-ish
                    vertical_lines += 1

        # Analyze color variance (open door shows background, more variance)
        color_variance = np.std(gray)

        # Decision logic
        if vertical_lines >= 2 and edge_density < 0.15 and color_variance < 50:
            return "closed"
        elif vertical_lines < 1 or edge_density > 0.25 or color_variance > 80:
            return "open"
        else:
            return "partially_open"


class World2DataV2:
    def __init__(self, video_path: str, output_dir: str = "output"):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.door_detector = DoorDetector()
        self.ground_truth_data = []

    def extract_frames(self, sample_rate: int = 5):
        """Extract frames from video"""
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

        print(f"✓ Extracted {len(frames)} frames from {frame_idx} total ({fps:.1f} FPS)")
        return frames, frame_indices

    def detect_people_yolo(self, frames: List[np.ndarray]):
        """Use YOLO only for person detection"""
        from ultralytics import YOLO

        print("Loading YOLO for person detection...")
        model = YOLO('yolov8n.pt')

        all_people = []
        for idx, frame in enumerate(frames):
            results = model(frame, verbose=False, classes=[0])  # class 0 = person

            people = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    conf = float(box.conf[0])
                    bbox = box.xyxy[0].tolist()
                    people.append({'bbox': bbox, 'confidence': conf})

            all_people.append(people)

            if (idx + 1) % 20 == 0:
                print(f"  Processed {idx+1}/{len(frames)} frames")

        print(f"✓ Person detection complete")
        return all_people

    def generate_ground_truth(self, frames, frame_indices, people_detections):
        """Generate ground truth with door detection"""
        ground_truth = []
        prev_door_state = None

        print("Detecting doors and generating ground truth...")

        for idx, (frame_idx, frame, people) in enumerate(zip(frame_indices, frames, people_detections)):
            timestamp = frame_idx / self.fps

            entry = {
                "timestamp": round(timestamp, 2),
                "frame_index": frame_idx,
                "objects": [],
                "human_action": "unknown",
                "interaction_event": None
            }

            # Detect door
            door = self.door_detector.detect_door(frame)

            if door:
                door_state = self.door_detector.analyze_door_state(frame, door['bbox'])

                entry['objects'].append({
                    "type": "door",
                    "state": door_state,
                    "affordance": "traversable" if door_state == "open" else "blocked",
                    "bbox": [round(x, 2) for x in door['bbox']],
                    "confidence": round(door['confidence'], 3)
                })

                # Detect state changes
                if prev_door_state and door_state != prev_door_state:
                    entry['interaction_event'] = {
                        "type": "door_state_change",
                        "from_state": prev_door_state,
                        "to_state": door_state
                    }

                prev_door_state = door_state

            # Add people
            for person in people:
                entry['objects'].append({
                    "type": "person",
                    "affordance": "dynamic_agent",
                    "bbox": [round(x, 2) for x in person['bbox']],
                    "confidence": round(person['confidence'], 3)
                })

            # Infer action
            if door and people:
                person_x = (people[0]['bbox'][0] + people[0]['bbox'][2]) / 2
                door_x = (door['bbox'][0] + door['bbox'][2]) / 2
                distance = abs(person_x - door_x)

                if distance < frame.shape[1] * 0.15:
                    entry['human_action'] = "at_door"
                elif person_x < door_x:
                    entry['human_action'] = "approaching_door"
                else:
                    entry['human_action'] = "passed_door"
            elif people:
                entry['human_action'] = "person_present"

            ground_truth.append(entry)

            if (idx + 1) % 20 == 0:
                print(f"  Analyzed {idx+1}/{len(frames)} frames")

        self.ground_truth_data = ground_truth
        print(f"✓ Generated {len(ground_truth)} annotations")
        return ground_truth

    def save_and_visualize(self, frames, frame_indices):
        """Save JSON and create video"""
        # Save JSON
        door_count = sum(1 for e in self.ground_truth_data if any(o['type']=='door' for o in e['objects']))
        state_changes = sum(1 for e in self.ground_truth_data if e['interaction_event'])

        output = {
            "metadata": {
                "video_source": str(self.video_path),
                "fps": self.fps,
                "frames_analyzed": len(self.ground_truth_data),
                "doors_detected": door_count,
                "state_changes": state_changes,
                "generated_at": datetime.now().isoformat()
            },
            "ground_truth": self.ground_truth_data
        }

        json_path = self.output_dir / "ground_truth_v2.json"
        with open(json_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"✓ Saved: {json_path}")
        print(f"  Doors detected: {door_count}/{len(self.ground_truth_data)} frames")
        print(f"  State changes: {state_changes}")

        # Create video
        print("Creating visualization...")
        height, width = frames[0].shape[:2]
        video_path = self.output_dir / "demo_v2.mp4"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, self.fps / 5, (width * 2, height))

        for idx, (frame, gt) in enumerate(zip(frames, self.ground_truth_data)):
            raw = frame.copy()
            annotated = frame.copy()

            # Draw annotations
            for obj in gt['objects']:
                x1, y1, x2, y2 = map(int, obj['bbox'])
                color = (255, 0, 0) if obj['type'] == 'door' else (0, 255, 0)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

                label = obj['type']
                if obj['type'] == 'door':
                    label = f"Door: {obj.get('state', 'unknown')} ({obj['confidence']:.2f})"

                cv2.putText(annotated, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Add labels
            cv2.putText(raw, "RAW INPUT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(annotated, "AI GROUND TRUTH", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            cv2.putText(annotated, f"Action: {gt['human_action']}", (10, height-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            if gt['interaction_event']:
                cv2.putText(annotated, "STATE CHANGE!", (10, height-50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            combined = np.hstack([raw, annotated])
            out.write(combined)

        out.release()
        print(f"✓ Saved: {video_path}")

    def run(self, sample_rate=5):
        """Run complete pipeline"""
        print("="*70)
        print("WORLD2DATA V2 - Real-World Door Detection")
        print("="*70)

        print("\n[1/4] Extracting frames...")
        frames, frame_indices = self.extract_frames(sample_rate)

        print("\n[2/4] Detecting people...")
        people = self.detect_people_yolo(frames)

        print("\n[3/4] Detecting doors and generating ground truth...")
        self.generate_ground_truth(frames, frame_indices, people)

        print("\n[4/4] Saving outputs...")
        self.save_and_visualize(frames, frame_indices)

        print("\n" + "="*70)
        print("✓ COMPLETE!")
        print("="*70)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python world2data_v2.py <video_path>")
        sys.exit(1)

    w2d = World2DataV2(sys.argv[1])
    w2d.run(sample_rate=5)
