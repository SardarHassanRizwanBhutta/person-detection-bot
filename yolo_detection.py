# YOLOv8 Person Detection with Audio Welcome System
# More robust and accurate than HOG descriptor approach

import cv2
import pygame
import time
import os
import random
import glob
from ultralytics import YOLO

class PersonWelcomeSystem:
    def __init__(self, sound_files=None, confidence_threshold=0.5, cooldown=3.0, playback_strategy="random"):
        """
        Initialize the YOLOv8 person detection welcome system with multiple audio files
        
        Args:
            sound_files (list): List of MP3 file paths, or None to auto-detect all MP3s
            confidence_threshold (float): Minimum confidence for person detection (0.0-1.0)
            cooldown (float): Seconds between welcome sounds to prevent spam
            playback_strategy (str): "random", "sequential", "weighted", or "smart_rotation"
        """
        # Initialize pygame mixer for audio
        pygame.mixer.init()
        
        # Auto-detect MP3 files if none provided
        if sound_files is None:
            sound_files = glob.glob("*.mp3")
            if not sound_files:
                print("⚠️  No MP3 files found in current directory")
        
        # Load all welcome sounds
        self.sound_files = sound_files if sound_files else []
        self.welcome_sounds = []
        self.sound_names = []
        
        for sound_file in self.sound_files:
            if os.path.exists(sound_file):
                try:
                    sound = pygame.mixer.Sound(sound_file)
                    self.welcome_sounds.append(sound)
                    self.sound_names.append(os.path.basename(sound_file))
                    print(f"✅ Loaded: {os.path.basename(sound_file)}")
                except Exception as e:
                    print(f"⚠️  Failed to load {sound_file}: {e}")
            else:
                print(f"⚠️  File not found: {sound_file}")
        
        if not self.welcome_sounds:
            print("❌ No valid audio files loaded!")
        else:
            print(f"🎵 Total audio files loaded: {len(self.welcome_sounds)}")
        
        # Detection settings
        self.confidence_threshold = confidence_threshold
        self.cooldown = cooldown
        self.last_welcome_time = 0
        
        # Playback strategy settings
        self.playback_strategy = playback_strategy
        self.current_index = 0  # For sequential playback
        self.played_sounds = set()  # For smart rotation
        self.sound_weights = [1.0] * len(self.welcome_sounds)  # For weighted random
        
        # Load YOLOv8 model (will download automatically on first run)
        print("🔄 Loading YOLOv8 model...")
        self.model = YOLO('yolov8n.pt')  # nano version for speed, use 'yolov8s.pt' for better accuracy
        print("✅ YOLOv8 model loaded successfully!")
        
        # Initialize camera
        self.cap = cv2.VideoCapture(1)  # Use camera 0, change if needed
        if not self.cap.isOpened():
            # Try camera 1 if camera 0 fails
            self.cap = cv2.VideoCapture(1)
        
        if not self.cap.isOpened():
            raise Exception("❌ Could not open camera")
        
        print("✅ Camera initialized successfully!")
        print(f"🎯 Detection confidence threshold: {confidence_threshold}")
        print(f"⏰ Welcome cooldown: {cooldown} seconds")
        print(f"🎵 Playback strategy: {playback_strategy}")
        print("🚀 System ready! Press 'q' to quit.")
    
    def detect_persons(self, frame):
        """
        Detect persons in the frame using YOLOv8
        
        Args:
            frame: Input video frame
            
        Returns:
            list: List of person detections with bounding boxes and confidence
        """
        # Run YOLOv8 inference
        results = self.model(frame, verbose=False)
        
        persons = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Class 0 is 'person' in COCO dataset
                    if int(box.cls[0]) == 0 and float(box.conf[0]) >= self.confidence_threshold:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0])
                        
                        persons.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': confidence
                        })
        
        return persons
    
    def select_next_sound(self):
        """Select the next sound based on the playback strategy"""
        if not self.welcome_sounds:
            return None, "No sounds available"
        
        if self.playback_strategy == "random":
            # Pure random selection
            index = random.randint(0, len(self.welcome_sounds) - 1)
            return self.welcome_sounds[index], self.sound_names[index]
        
        elif self.playback_strategy == "sequential":
            # Play in order, loop back to start
            sound = self.welcome_sounds[self.current_index]
            name = self.sound_names[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.welcome_sounds)
            return sound, name
        
        elif self.playback_strategy == "weighted":
            # Weighted random selection (can be customized)
            index = random.choices(range(len(self.welcome_sounds)), 
                                 weights=self.sound_weights, k=1)[0]
            return self.welcome_sounds[index], self.sound_names[index]
        
        elif self.playback_strategy == "smart_rotation":
            # Ensure all sounds are played before repeating
            available_indices = [i for i in range(len(self.welcome_sounds)) 
                               if i not in self.played_sounds]
            
            if not available_indices:
                # All sounds played, reset and start over
                self.played_sounds.clear()
                available_indices = list(range(len(self.welcome_sounds)))
            
            index = random.choice(available_indices)
            self.played_sounds.add(index)
            return self.welcome_sounds[index], self.sound_names[index]
        
        else:
            # Default to random if unknown strategy
            index = random.randint(0, len(self.welcome_sounds) - 1)
            return self.welcome_sounds[index], self.sound_names[index]
    
    def play_welcome_sound(self):
        """Play a welcome sound if available and cooldown has passed"""
        current_time = time.time()
        if current_time - self.last_welcome_time > self.cooldown:
            sound, sound_name = self.select_next_sound()
            if sound:
                sound.play()
                self.last_welcome_time = current_time
                return True, sound_name
        return False, None
    
    def draw_detections(self, frame, persons):
        """
        Draw bounding boxes and labels on the frame
        
        Args:
            frame: Input video frame
            persons: List of person detections
            
        Returns:
            frame: Frame with drawn detections
        """
        for person in persons:
            x1, y1, x2, y2 = person['bbox']
            confidence = person['confidence']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence label
            label = f"Person: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return frame
    
    def run(self):
        """Main detection loop"""
        print("\n🎥 Starting person detection...")
        print("👋 System will welcome detected persons with audio!")
        
        frame_count = 0
        fps_start_time = time.time()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to read from camera")
                    break
                
                # Resize frame for faster processing (optional)
                frame = cv2.resize(frame, (640, 480))
                
                # Detect persons
                persons = self.detect_persons(frame)
                
                # Play welcome sound if persons detected
                if persons:
                    welcomed, sound_name = self.play_welcome_sound()
                    if welcomed:
                        print(f"👋 Welcome! Playing: {sound_name}")
                        print(f"   Detected {len(persons)} person(s) with confidence: "
                              f"{[f'{p['confidence']:.2f}' for p in persons]}")
                
                # Draw detections on frame
                frame = self.draw_detections(frame, persons)
                
                # Add status text
                status_text = f"Persons: {len(persons)} | Confidence: >{self.confidence_threshold}"
                cv2.putText(frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Calculate and display FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    fps = 30 / (time.time() - fps_start_time)
                    fps_start_time = time.time()
                    print(f"📊 FPS: {fps:.1f} | Persons detected: {len(persons)}")
                
                # Display frame
                cv2.imshow('YOLOv8 Person Detection - Welcome System', frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("👋 Goodbye!")
                    break
                    
        except KeyboardInterrupt:
            print("\n⏹️  Detection stopped by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()
        print("🧹 Cleanup completed")

def main():
    """Main function to run the person welcome system"""
    try:
        # Create and run the welcome system
        welcome_system = PersonWelcomeSystem(
            sound_files=None,  # Auto-detect all MP3 files, or specify: ["sound1.mp3", "sound2.mp3"]
            confidence_threshold=0.6,  # Adjust for accuracy vs sensitivity
            cooldown=3.0,  # 3 seconds between welcomes
            playback_strategy="smart_rotation"  # "random", "sequential", "weighted", "smart_rotation"
        )
        welcome_system.run()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n📋 Requirements:")
        print("   pip install ultralytics opencv-python pygame")
        print("   Make sure growth_plan.mp3 is in the current directory")
        print("   Ensure camera is connected and accessible")

if __name__ == "__main__":
    main()
