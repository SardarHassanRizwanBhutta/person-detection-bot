#!/usr/bin/env python3
"""
Production-Ready YOLOv8 Person Detection Welcome System
Features: Logging, Configuration Management, Error Recovery, Performance Monitoring
"""

import cv2
import pygame
import time
import os
import random
import glob
import json
import logging
import psutil
from logging.handlers import RotatingFileHandler
from ultralytics import YOLO
from mutagen import File as MutagenFile

class ProductionWelcomeSystem:
    def __init__(self, config_file="config.json"):
        """Initialize the production-ready welcome system"""
        
        # Load configuration
        self.config = self.load_config(config_file)
        
        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        self.logger.info("🚀 Starting Production Welcome System")
        
        # Initialize system components
        self.running = False
        self.model = None
        self.cap = None
        self.welcome_sounds = []
        self.sound_names = []
        
        # Performance tracking
        self.stats = {
            'detections_today': 0,
            'total_detections': 0,
            'system_start_time': time.time(),
            'last_detection_time': None,
            'fps_history': [],
            'error_count': 0,
            'camera_failures': 0,
            'audio_failures': 0
        }
        
        # Playback strategy state
        self.current_index = 0
        self.played_sounds = set()
        self.sound_weights = []
        self.last_welcome_time = 0
        
        # Audio state management
        self.audio_playing = False
        self.current_audio_start_time = 0
        self.audio_duration_estimate = 3.0  # Default audio length estimate
        self.audio_channel = None
        
        # Detection gap strategy state
        self.person_present = False
        self.last_detection_frame_time = 0
        self.detection_gap_threshold = 2.0  # Will be set from config in initialize_system
        
        # Initialize components
        self.initialize_system()
    
    def load_config(self, config_file):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"✅ Configuration loaded from {config_file}")
            return config
        except FileNotFoundError:
            print(f"⚠️  Config file {config_file} not found, using defaults")
            return self.get_default_config()
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in config file: {e}")
            return self.get_default_config()
    
    def get_default_config(self):
        """Return default configuration"""
        return {
            "detection": {
                "confidence_threshold": 0.6,
                "model_name": "yolov8n.pt",
                "camera_index": 1,
                "frame_width": 640,
                "frame_height": 480
            },
            "audio": {
                "sound_files": None,
                "playback_strategy": "smart_rotation",
                "cooldown_seconds": 3.0,
                "volume": 0.8,
                "prevent_overlap": True,
                "max_audio_duration": 10.0,
                "interrupt_on_new_detection": False,
                "detection_gap_threshold": 2.0
            },
            "system": {
                "fps_report_interval": 30,
                "log_level": "INFO",
                "log_file": "welcome_system.log",
                "max_log_size_mb": 10,
                "backup_log_count": 5
            },
            "monitoring": {
                "enable_performance_tracking": True,
                "memory_alert_threshold_mb": 500,
                "fps_alert_threshold": 5,
                "detection_timeout_seconds": 30
            }
        }
    
    def setup_logging(self):
        """Setup comprehensive logging system"""
        log_config = self.config['system']
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
        # Setup rotating file handler
        log_file = os.path.join('logs', log_config['log_file'])
        handler = RotatingFileHandler(
            log_file,
            maxBytes=log_config['max_log_size_mb'] * 1024 * 1024,
            backupCount=log_config['backup_log_count']
        )
        
        # Setup console handler
        console_handler = logging.StreamHandler()
        
        # Setup formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Configure logger
        logger = logging.getLogger()
        logger.setLevel(getattr(logging, log_config['log_level']))
        logger.addHandler(handler)
        logger.addHandler(console_handler)
    
    def initialize_system(self):
        """Initialize all system components with error handling"""
        try:
            # Load detection gap threshold from config
            self.detection_gap_threshold = self.config['audio'].get('detection_gap_threshold', 2.0)
            
            self.initialize_audio()
            self.initialize_model()
            self.initialize_camera()
            self.logger.info("✅ All system components initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ System initialization failed: {e}")
            raise
    
    def initialize_audio(self):
        """Initialize audio system with error handling"""
        try:
            # Initialize pygame mixer with specific settings for better control
            pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
            pygame.mixer.init()
            
            # Reserve a dedicated channel for welcome sounds
            pygame.mixer.set_num_channels(8)  # Ensure we have enough channels
            self.audio_channel = pygame.mixer.Channel(0)  # Dedicated channel for welcomes
            
            # Load audio files
            sound_files = self.config['audio']['sound_files']
            if sound_files is None:
                # Look for audio files in the audio_files folder
                audio_folder = "audio_files"
                if os.path.exists(audio_folder):
                    sound_files = glob.glob(os.path.join(audio_folder, "*.mp3"))
                    # Also check for other common audio formats
                    sound_files.extend(glob.glob(os.path.join(audio_folder, "*.wav")))
                    sound_files.extend(glob.glob(os.path.join(audio_folder, "*.ogg")))
                else:
                    self.logger.warning(f"⚠️  Audio folder '{audio_folder}' not found, creating it...")
                    os.makedirs(audio_folder, exist_ok=True)
                    sound_files = []
            
            self.welcome_sounds = []
            self.sound_names = []
            self.audio_durations = []  # Track estimated durations
            
            for sound_file in sound_files:
                if os.path.exists(sound_file):
                    try:
                        # Load pygame sound
                        sound = pygame.mixer.Sound(sound_file)
                        sound.set_volume(self.config['audio']['volume'])
                        self.welcome_sounds.append(sound)
                        self.sound_names.append(os.path.basename(sound_file))
                        
                        # Get precise audio duration using mutagen
                        duration = self.get_audio_duration(sound_file)
                        self.audio_durations.append(duration)
                        
                        self.logger.info(f"✅ Loaded audio: {os.path.basename(sound_file)} ({duration:.2f}s)")
                    except Exception as e:
                        self.logger.warning(f"⚠️  Failed to load {sound_file}: {e}")
                        self.stats['audio_failures'] += 1
            
            if not self.welcome_sounds:
                raise Exception("No valid audio files loaded")
            
            self.sound_weights = [1.0] * len(self.welcome_sounds)
            self.logger.info(f"🎵 Audio system initialized with {len(self.welcome_sounds)} files")
            
        except Exception as e:
            self.logger.error(f"❌ Audio initialization failed: {e}")
            raise
    
    def get_audio_duration(self, audio_file):
        """Get precise audio duration using mutagen library"""
        try:
            # Use mutagen to get exact audio duration
            audio_info = MutagenFile(audio_file)
            if audio_info is not None and hasattr(audio_info, 'info') and hasattr(audio_info.info, 'length'):
                duration = float(audio_info.info.length)
                self.logger.debug(f"📏 Precise duration for {os.path.basename(audio_file)}: {duration:.2f}s")
                return duration
            else:
                # Fallback: try to get duration from file info
                if audio_info is not None and audio_info.info:
                    # Some formats store duration differently
                    if hasattr(audio_info.info, 'duration'):
                        duration = float(audio_info.info.duration)
                        return duration
                
                self.logger.warning(f"⚠️  Could not get duration for {audio_file}, using fallback")
                return self.get_fallback_duration(audio_file)
                
        except Exception as e:
            self.logger.warning(f"⚠️  Mutagen failed for {audio_file}: {e}, using fallback")
            return self.get_fallback_duration(audio_file)
    
    def get_fallback_duration(self, audio_file):
        """Fallback method to estimate audio duration"""
        try:
            # Method 1: Use file size estimation (very rough)
            file_size = os.path.getsize(audio_file)
            # Rough estimate: MP3 at 128kbps ≈ 16KB per second
            estimated_duration = file_size / (16 * 1024)  # Very rough estimate
            
            # Clamp to reasonable bounds
            estimated_duration = max(1.0, min(30.0, estimated_duration))
            
            self.logger.debug(f"📏 Fallback duration estimate for {os.path.basename(audio_file)}: {estimated_duration:.1f}s")
            return estimated_duration
            
        except Exception as e:
            self.logger.warning(f"⚠️  All duration methods failed for {audio_file}: {e}")
            # Ultimate fallback - use config default
            return self.config['audio'].get('max_audio_duration', 5.0)
    
    def initialize_model(self):
        """Initialize YOLO model with error handling"""
        try:
            model_name = self.config['detection']['model_name']
            self.logger.info(f"🔄 Loading YOLO model: {model_name}")
            self.model = YOLO(model_name)
            self.logger.info("✅ YOLO model loaded successfully")
        except Exception as e:
            self.logger.error(f"❌ Model initialization failed: {e}")
            raise
    
    def initialize_camera(self):
        """Initialize camera with error handling and retry logic"""
        camera_index = self.config['detection']['camera_index']
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"🔄 Attempting to initialize camera {camera_index} (attempt {attempt + 1})")
                self.cap = cv2.VideoCapture(camera_index)
                
                if not self.cap.isOpened():
                    # Try alternative camera index
                    alt_camera = 0 if camera_index == 1 else 1
                    self.logger.warning(f"⚠️  Camera {camera_index} failed, trying camera {alt_camera}")
                    self.cap = cv2.VideoCapture(alt_camera)
                
                if not self.cap.isOpened():
                    raise Exception(f"Could not open camera {camera_index} or {alt_camera}")
                
                # Test camera by reading a frame
                ret, frame = self.cap.read()
                if not ret:
                    raise Exception("Camera opened but cannot read frames")
                
                self.logger.info("✅ Camera initialized successfully")
                return
                
            except Exception as e:
                self.logger.warning(f"⚠️  Camera initialization attempt {attempt + 1} failed: {e}")
                self.stats['camera_failures'] += 1
                if self.cap:
                    self.cap.release()
                    self.cap = None
                
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retry
                else:
                    self.logger.error("❌ All camera initialization attempts failed")
                    raise
    
    def detect_persons(self, frame):
        """Detect persons with error handling"""
        try:
            results = self.model(frame, verbose=False)
            persons = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        if int(box.cls[0]) == 0 and float(box.conf[0]) >= self.config['detection']['confidence_threshold']:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = float(box.conf[0])
                            
                            persons.append({
                                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'confidence': confidence
                            })
            
            return persons
            
        except Exception as e:
            self.logger.error(f"❌ Person detection failed: {e}")
            return []
    
    def select_next_sound(self):
        """Select next sound based on strategy"""
        if not self.welcome_sounds:
            return None, "No sounds available"
        
        strategy = self.config['audio']['playback_strategy']
        
        if strategy == "random":
            index = random.randint(0, len(self.welcome_sounds) - 1)
            return self.welcome_sounds[index], self.sound_names[index]
        
        elif strategy == "sequential":
            sound = self.welcome_sounds[self.current_index]
            name = self.sound_names[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.welcome_sounds)
            return sound, name
        
        elif strategy == "smart_rotation":
            available_indices = [i for i in range(len(self.welcome_sounds)) 
                               if i not in self.played_sounds]
            
            if not available_indices:
                self.played_sounds.clear()
                available_indices = list(range(len(self.welcome_sounds)))
            
            index = random.choice(available_indices)
            self.played_sounds.add(index)
            return self.welcome_sounds[index], self.sound_names[index]
        
        else:
            index = random.randint(0, len(self.welcome_sounds) - 1)
            return self.welcome_sounds[index], self.sound_names[index]
    
    def is_audio_currently_playing(self):
        """Check if audio is currently playing with precise timing"""
        current_time = time.time()
        
        # Method 1: Check pygame channel status (most reliable)
        if self.audio_channel and self.audio_channel.get_busy():
            return True
        
        # Method 2: Check precise time-based calculation
        if self.audio_playing:
            elapsed_time = current_time - self.current_audio_start_time
            # Add small buffer (0.1s) to account for system delays
            if elapsed_time < (self.audio_duration_estimate + 0.1):
                return True
            else:
                # Audio should have finished by now
                self.audio_playing = False
                self.logger.debug(f"🕐 Audio finished after {elapsed_time:.2f}s (expected {self.audio_duration_estimate:.2f}s)")
        
        # Method 3: Check if any mixer channel is busy (fallback)
        if pygame.mixer.get_busy():
            # Some other audio might be playing, be cautious
            return True
        
        # If we reach here, no audio is playing
        self.audio_playing = False
        return False
    
    def get_remaining_audio_time(self):
        """Get remaining time for current audio playback"""
        if not self.audio_playing:
            return 0.0
        
        current_time = time.time()
        elapsed_time = current_time - self.current_audio_start_time
        remaining_time = max(0.0, self.audio_duration_estimate - elapsed_time)
        
        return remaining_time
    
    def stop_current_audio(self):
        """Stop any currently playing audio"""
        try:
            if self.audio_channel:
                self.audio_channel.stop()
            pygame.mixer.stop()  # Stop all sounds as fallback
            self.audio_playing = False
            self.logger.info("🔇 Stopped current audio playback")
        except Exception as e:
            self.logger.warning(f"⚠️  Failed to stop audio: {e}")
    
    def should_welcome_person(self, persons_detected):
        """Determine if we should welcome based on detection gap strategy"""
        current_time = time.time()
        
        if persons_detected:
            # Person(s) detected
            if not self.person_present:
                # Person just arrived - welcome them!
                self.person_present = True
                self.last_detection_frame_time = current_time
                self.logger.info("👋 New person arrival detected - triggering welcome")
                return True
            else:
                # Person still present - update last seen time but don't welcome again
                self.last_detection_frame_time = current_time
                return False
        else:
            # No person detected
            if self.person_present:
                # Check if person has been gone long enough
                time_since_last_detection = current_time - self.last_detection_frame_time
                if time_since_last_detection > self.detection_gap_threshold:
                    self.person_present = False
                    self.logger.info(f"👋 Person left (gap: {time_since_last_detection:.1f}s) - ready for next welcome")
            return False
    
    def play_welcome_sound(self):
        """Play welcome sound with robust overlap prevention"""
        current_time = time.time()
        cooldown = self.config['audio']['cooldown_seconds']
        prevent_overlap = self.config['audio'].get('prevent_overlap', True)
        interrupt_on_new = self.config['audio'].get('interrupt_on_new_detection', False)
        
        # Check 1: Cooldown period
        if current_time - self.last_welcome_time <= cooldown:
            return False, None
        
        # Check 2: Audio overlap prevention
        if prevent_overlap and self.is_audio_currently_playing():
            remaining_time = self.get_remaining_audio_time()
            if interrupt_on_new:
                self.logger.info(f"🔄 Interrupting current audio ({remaining_time:.1f}s remaining) for new detection")
                self.stop_current_audio()
            else:
                self.logger.debug(f"🔇 Audio already playing ({remaining_time:.1f}s remaining), skipping new welcome")
                return False, None
        
        try:
            sound, sound_name = self.select_next_sound()
            if sound:
                # Get the index of the selected sound for duration estimation
                sound_index = self.sound_names.index(sound_name)
                estimated_duration = self.audio_durations[sound_index]
                
                # Stop any residual audio (safety measure)
                self.stop_current_audio()
                
                # Play the new sound on dedicated channel
                if self.audio_channel:
                    self.audio_channel.play(sound)
                else:
                    sound.play()  # Fallback
                
                # Update state tracking
                self.audio_playing = True
                self.current_audio_start_time = current_time
                self.audio_duration_estimate = estimated_duration
                self.last_welcome_time = current_time
                
                # Update statistics
                self.stats['detections_today'] += 1
                self.stats['total_detections'] += 1
                self.stats['last_detection_time'] = current_time
                
                self.logger.info(f"👋 Welcome played: {sound_name} ({estimated_duration:.2f}s)")
                return True, sound_name
                
        except Exception as e:
            self.logger.error(f"❌ Audio playback failed: {e}")
            self.stats['audio_failures'] += 1
            self.audio_playing = False
        
        return False, None
    
    def monitor_performance(self):
        """Monitor system performance and log alerts"""
        try:
            # Memory usage
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            threshold = self.config['monitoring']['memory_alert_threshold_mb']
            
            if memory_mb > threshold:
                self.logger.warning(f"⚠️  High memory usage: {memory_mb:.1f}MB (threshold: {threshold}MB)")
            
            # Detection timeout check
            if self.stats['last_detection_time']:
                timeout = self.config['monitoring']['detection_timeout_seconds']
                time_since_detection = time.time() - self.stats['last_detection_time']
                
                if time_since_detection > timeout:
                    self.logger.warning(f"⚠️  No detections for {time_since_detection:.1f}s")
            
            # Log system stats
            uptime = time.time() - self.stats['system_start_time']
            self.logger.info(f"📊 Stats - Uptime: {uptime/3600:.1f}h | "
                           f"Detections: {self.stats['total_detections']} | "
                           f"Memory: {memory_mb:.1f}MB | "
                           f"Errors: {self.stats['error_count']}")
            
        except Exception as e:
            self.logger.error(f"❌ Performance monitoring failed: {e}")
    
    def draw_detections(self, frame, persons):
        """Draw detection boxes and info"""
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
        
        # Add system info
        person_status = "Present" if self.person_present else "Absent"
        info_text = f"Detections: {self.stats['total_detections']} | Person: {person_status} | Strategy: {self.config['audio']['playback_strategy']}"
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def run(self):
        """Main production system loop with comprehensive error handling"""
        self.running = True
        self.logger.info("🎥 Starting production welcome system")
        
        frame_count = 0
        fps_start_time = time.time()
        last_performance_check = time.time()
        
        try:
            while self.running:
                try:
                    # Read frame with timeout
                    ret, frame = self.cap.read()
                    if not ret:
                        self.logger.error("❌ Failed to read camera frame")
                        self.reinitialize_camera()
                        continue
                    
                    # Resize frame
                    width = self.config['detection']['frame_width']
                    height = self.config['detection']['frame_height']
                    frame = cv2.resize(frame, (width, height))
                    
                    # Flip frame vertically (upside down)
                    frame = cv2.flip(frame, 0)
                    
                    # Detect persons
                    persons = self.detect_persons(frame)
                    
                    # Check if we should welcome using detection gap strategy
                    should_welcome = self.should_welcome_person(len(persons) > 0)
                    
                    # Play welcome sound if detection gap strategy says we should
                    if should_welcome and persons:
                        welcomed, sound_name = self.play_welcome_sound()
                        if welcomed:
                            self.logger.info(f"👋 Detection event - Count: {len(persons)} | "
                                           f"Audio: {sound_name} | "
                                           f"Confidence: {[f'{p['confidence']:.2f}' for p in persons]}")
                    
                    # Always update detection gap strategy (even when no persons detected)
                    if not persons:
                        self.should_welcome_person(False)  # Update state for no detection
                    
                    # Draw detections
                    frame = self.draw_detections(frame, persons)
                    
                    # Calculate and track FPS
                    frame_count += 1
                    if frame_count % self.config['system']['fps_report_interval'] == 0:
                        fps = self.config['system']['fps_report_interval'] / (time.time() - fps_start_time)
                        self.stats['fps_history'].append(fps)
                        fps_start_time = time.time()
                        
                        # Check FPS alert
                        if fps < self.config['monitoring']['fps_alert_threshold']:
                            self.logger.warning(f"⚠️  Low FPS detected: {fps:.1f}")
                        
                        self.logger.info(f"📊 FPS: {fps:.1f} | Persons: {len(persons)}")
                    
                    # Performance monitoring
                    if time.time() - last_performance_check > 60:  # Every minute
                        self.monitor_performance()
                        last_performance_check = time.time()
                    
                    # Display frame
                    cv2.imshow('Production Welcome System', frame)
                    
                    # Check for quit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.logger.info("👋 System shutdown requested by user")
                        break
                        
                except KeyboardInterrupt:
                    self.logger.info("⏹️  System interrupted by user")
                    break
                    
                except Exception as e:
                    self.logger.error(f"❌ Runtime error: {e}")
                    self.stats['error_count'] += 1
                    
                    # Auto-recovery attempt
                    if self.stats['error_count'] % 10 == 0:
                        self.logger.info("🔄 Attempting system recovery...")
                        self.reinitialize_camera()
                    
                    time.sleep(1)  # Prevent rapid error loops
                    
        except Exception as e:
            self.logger.critical(f"💥 Critical system failure: {e}")
            
        finally:
            self.cleanup()
    
    def reinitialize_camera(self):
        """Reinitialize camera on failure"""
        try:
            self.logger.info("🔄 Reinitializing camera...")
            if self.cap:
                self.cap.release()
            self.initialize_camera()
            self.logger.info("✅ Camera reinitialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Camera reinitialization failed: {e}")
    
    def cleanup(self):
        """Clean shutdown of all system components"""
        self.logger.info("🧹 Starting system cleanup...")
        self.running = False
        
        try:
            # Stop any playing audio
            self.stop_current_audio()
            
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            pygame.mixer.quit()
            
            # Log final statistics
            uptime = time.time() - self.stats['system_start_time']
            self.logger.info(f"📊 Final Stats - Uptime: {uptime/3600:.2f}h | "
                           f"Total Detections: {self.stats['total_detections']} | "
                           f"Audio Failures: {self.stats['audio_failures']} | "
                           f"Errors: {self.stats['error_count']}")
            
            self.logger.info("✅ System cleanup completed")
            
        except Exception as e:
            self.logger.error(f"❌ Cleanup error: {e}")

def main():
    """Main function with production error handling"""
    try:
        system = ProductionWelcomeSystem()
        system.run()
        
    except KeyboardInterrupt:
        print("\n⏹️  System interrupted")
    except Exception as e:
        print(f"💥 Critical error: {e}")
        logging.error(f"Critical startup error: {e}")

if __name__ == "__main__":
    main()
