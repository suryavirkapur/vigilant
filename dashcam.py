import cv2
import os
import time
from datetime import datetime
import numpy as np
import argparse
from tqdm import tqdm
from roboflow import Roboflow
from dotenv import load_dotenv

class TrafficDetector:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize Roboflow
        rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
        
        # Load traffic light model
        self.traffic_project = rf.workspace().project("traffic-light-detection-h8cvg")
        self.traffic_model = self.traffic_project.version(2).model
        
        # Load stop sign model
        self.stop_project = rf.workspace().project("stop-sign-detection-1")
        self.stop_model = self.stop_project.version(1).model

        # Color ranges for verification
        self.color_ranges = {
            'red': [(0, 100, 100), (10, 255, 255), (160, 100, 100), (180, 255, 255)],
            'yellow': [(15, 100, 100), (35, 255, 255)],
            'green': [(35, 50, 50), (85, 255, 255)]
        }

    def detect_color(self, img, bbox):
        """Verify traffic light color in the bounding box"""
        x1, y1, x2, y2 = map(int, bbox)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        for color, ranges in self.color_ranges.items():
            if color == 'red':
                mask1 = cv2.inRange(hsv, np.array(ranges[0]), np.array(ranges[1]))
                mask2 = cv2.inRange(hsv, np.array(ranges[2]), np.array(ranges[3]))
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = cv2.inRange(hsv, np.array(ranges[0]), np.array(ranges[1]))
            
            if np.sum(mask) > 500:
                return color
        return None

    def process_frame(self, frame):
        """Process a frame using Roboflow models"""
        # Save frame temporarily
        temp_path = "temp_frame.jpg"
        cv2.imwrite(temp_path, frame)
        
        detections = []
        
        try:
            # Get traffic light predictions
            traffic_results = self.traffic_model.predict(temp_path, confidence=40, overlap=30).json()
            
            # Process traffic light detections
            for pred in traffic_results['predictions']:
                x1 = pred['x'] - pred['width']/2
                y1 = pred['y'] - pred['height']/2
                x2 = x1 + pred['width']
                y2 = y1 + pred['height']
                
                # Verify color
                color = self.detect_color(frame, (x1, y1, x2, y2))
                if color:
                    detections.append({
                        'type': 'traffic_light',
                        'color': color,
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': pred['confidence']
                    })
            
            # Get stop sign predictions
            stop_results = self.stop_model.predict(temp_path, confidence=40, overlap=30).json()
            
            # Process stop sign detections
            for pred in stop_results['predictions']:
                if 'stop-sign-vandalized' not in pred['class'] and 'stop-sign-fake' not in pred['class']:
                    x1 = pred['x'] - pred['width']/2
                    y1 = pred['y'] - pred['height']/2
                    x2 = x1 + pred['width']
                    y2 = y1 + pred['height']
                    
                    detections.append({
                        'type': 'stop_sign',
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': pred['confidence']
                    })
                    
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        return detections

class VideoProcessor:
    def __init__(self, input_path, output_dir='processed_footage', frame_dir='processed_frames'):
        self.input_path = input_path
        self.output_dir = output_dir
        self.frame_dir = frame_dir
        self.detector = TrafficDetector()
        self.frame_interval = 3  # Save frame every 3 seconds
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(frame_dir, exist_ok=True)

    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels for detections"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            if det['type'] == 'traffic_light':
                color = (0, 255, 0) if det['color'] == 'green' else \
                        (0, 255, 255) if det['color'] == 'yellow' else \
                        (0, 0, 255)
                label = f"Traffic Light ({det['color']}) {det['confidence']:.2f}"
            else:  # stop sign
                color = (255, 0, 0)
                label = f"Stop Sign {det['confidence']:.2f}"
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with background
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            y1 = max(y1, label_size[1])
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame

    def process_video(self, display=True):
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise Exception("Error: Could not open video file")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = os.path.join(self.output_dir, 
                                 f"processed_{os.path.basename(self.input_path)}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        frames_to_save = set(range(0, total_frames, fps * self.frame_interval))

        try:
            pbar = tqdm(total=total_frames, desc="Processing video")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Only process every nth frame for efficiency
                if frame_count % 3 == 0:  # Process every 3rd frame
                    detections = self.detector.process_frame(frame)
                    frame_with_detections = self.draw_detections(frame.copy(), detections)
                else:
                    frame_with_detections = frame

                if frame_count in frames_to_save:
                    frame_path = os.path.join(self.frame_dir, f"frame_{frame_count:04d}.jpg")
                    cv2.imwrite(frame_path, frame_with_detections)

                out.write(frame_with_detections)

                if display:
                    cv2.imshow('Processing Video', frame_with_detections)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                frame_count += 1
                pbar.update(1)

        finally:
            pbar.close()
            cap.release()
            out.release()
            cv2.destroyAllWindows()

        print(f"\nProcessed video saved to: {output_path}")
        print(f"Processed frames saved to: {self.frame_dir}")

def main():
    parser = argparse.ArgumentParser(description='Process dashcam footage for traffic detection')
    parser.add_argument('input_video', type=str, help='Path to input video file')
    parser.add_argument('--no-display', action='store_true', help='Disable video display during processing')
    parser.add_argument('--output-dir', type=str, default='processed_footage', 
                        help='Directory for processed videos')
    parser.add_argument('--frame-dir', type=str, default='processed_frames',
                        help='Directory for saved frames')
    
    args = parser.parse_args()

    processor = VideoProcessor(args.input_video, args.output_dir, args.frame_dir)
    
    try:
        processor.process_video(display=not args.no_display)
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()