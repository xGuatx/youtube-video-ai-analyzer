"""
Local AI recognition module optimized for local processing
Prioritizes Ollama and local models over external APIs
"""

import cv2
import numpy as np
import requests
import json
import logging
import os
import subprocess
from typing import List, Dict, Any, Optional
from pathlib import Path
import base64

logger = logging.getLogger(__name__)

class LocalAIDetector:
    """
    Local AI detector using Ollama and local models
    """

    # Vision models in priority order (moondream first - lighter and more stable)
    VISION_MODELS = ['moondream', 'moondream:latest', 'llava', 'llava:7b', 'llava:13b', 'bakllava']

    def __init__(self):
        """
        Initialize local AI detector
        """
        self.ollama_host = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
        self.ollama_available = self._check_ollama()
        self.local_models = self._discover_local_models()
        self.vision_model = self._detect_vision_model()
        self.using_fallback = self.vision_model is None

        if self.using_fallback and self.ollama_available:
            logger.warning("WARNING: No vision model found (llava, moondream). Using OpenCV fallback.")
            logger.warning("Install a vision model: ollama pull llava")

    def _detect_vision_model(self) -> Optional[str]:
        """
        Detect available vision model in priority order: llava > moondream > bakllava

        Returns:
            Model name if found, None otherwise
        """
        if not self.ollama_available:
            return None

        installed_models = [m.replace('ollama:', '') for m in self.local_models if m.startswith('ollama:')]

        for vision_model in self.VISION_MODELS:
            for installed in installed_models:
                if vision_model in installed or installed.startswith(vision_model.split(':')[0]):
                    logger.info(f"Vision model detected: {installed}")
                    return installed

        logger.warning("No vision model found. Available models: " + ", ".join(installed_models))
        return None

    def get_status(self) -> dict:
        """Get detector status for API responses"""
        return {
            "ollama_available": self.ollama_available,
            "ollama_host": self.ollama_host,
            "vision_model": self.vision_model,
            "using_fallback": self.using_fallback,
            "fallback_reason": "No vision model (llava/moondream) installed" if self.using_fallback else None,
            "recommendation": "Run: ollama pull llava" if self.using_fallback and self.ollama_available else None
        }

    def _check_ollama(self) -> bool:
        """
        Check if Ollama is available

        Returns:
            True if Ollama is available
        """
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info("Ollama detected and available")
                return True
        except Exception as e:
            logger.info(f"Ollama not available: {e}")

        return False

    def _discover_local_models(self) -> List[str]:
        """
        Discover available local models

        Returns:
            List of available models
        """
        models = []

        # Ollama models
        if self.ollama_available:
            try:
                response = requests.get(f"{self.ollama_host}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    ollama_models = [model['name'] for model in data.get('models', [])]
                    models.extend([f"ollama:{model}" for model in ollama_models])
                    logger.info(f"Ollama models found: {ollama_models}")
            except Exception as e:
                logger.warning(f"Error retrieving Ollama models: {e}")

        # Local OpenCV models (always available)
        opencv_models = ["opencv:haarcascade", "opencv:dnn"]
        models.extend(opencv_models)

        # Local YOLO models (if available)
        yolo_models = self._check_yolo_models()
        models.extend(yolo_models)

        logger.info(f"Local models available: {models}")
        return models

    def _check_yolo_models(self) -> List[str]:
        """
        Check locally available YOLO models

        Returns:
            List of YOLO models
        """
        models = []

        # Check if ultralytics is available
        try:
            import ultralytics
            models.append("yolo:yolov8n")
            models.append("yolo:yolov8s")
            models.append("yolo:yolov8m")
            logger.info("YOLO ultralytics models available")
        except ImportError:
            logger.info("Ultralytics not available")

        return models

    def detect_with_ollama(self, image_path: str, target_object: str,
                          model_name: str = "llava") -> List[Dict]:
        """
        Object detection with Ollama

        Args:
            image_path: Path to image
            target_object: Object to detect
            model_name: Ollama model name

        Returns:
            List of detections
        """
        if not self.ollama_available:
            return []

        try:
            # Encode image in base64
            with open(image_path, 'rb') as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

            # Prepare detection prompt
            prompt = f"""
            Analyze this image and detect if there are objects of type "{target_object}".
            Respond only with JSON in this structure:
            {{
                "detections": [
                    {{
                        "object": "{target_object}",
                        "confidence": 0.95,
                        "bbox": [x, y, width, height],
                        "description": "description of detected object"
                    }}
                ],
                "found": true/false
            }}

            If no object is found, respond: {{"detections": [], "found": false}}
            """

            # Request to Ollama
            payload = {
                "model": model_name,
                "prompt": prompt,
                "images": [img_base64],
                "stream": False
            }

            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '')

                # Parse JSON response
                try:
                    # Extract JSON from response
                    start_idx = response_text.find('{')
                    end_idx = response_text.rfind('}') + 1

                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = response_text[start_idx:end_idx]
                        detection_result = json.loads(json_str)

                        detections = detection_result.get('detections', [])
                        logger.info(f"Ollama detections: {len(detections)} objects found")
                        return detections

                except json.JSONDecodeError as e:
                    logger.warning(f"Ollama JSON parsing error: {e}")
                    logger.debug(f"Raw response: {response_text}")

        except Exception as e:
            logger.error(f"Ollama detection error: {e}")

        return []

    def describe_characters(self, image_path: str, model_name: str = None) -> Dict:
        """
        Describe WHO the characters/people are in the image.
        Auto-detects best vision model: llava > moondream > bakllava

        Args:
            image_path: Path to image
            model_name: Ollama model to use (auto-detect if None)

        Returns:
            Dict with character descriptions
        """
        # Auto-detect vision model if not specified
        if model_name is None:
            model_name = self.vision_model

        # If no vision model available, return with warning
        if model_name is None or not self.ollama_available:
            return {
                "characters": [],
                "character_count": 0,
                "warning": "No vision model available. Install: ollama pull llava",
                "using_fallback": True,
                "status": self.get_status()
            }

        try:
            with open(image_path, 'rb') as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

            prompt = """Describe this image in detail. Focus on any people or characters visible.
How many people are there? What do they look like? What are they doing? What is the setting?"""

            payload = {
                "model": model_name,
                "prompt": prompt,
                "images": [img_base64],
                "stream": False
            }

            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '').strip()
                logger.info(f"Vision model response: {response_text[:200]}...")

                # Count people mentioned in response
                import re
                text_lower = response_text.lower()

                # Try to extract number of people
                char_count = 0
                num_match = re.search(r'(\d+)\s*(?:people|person|characters|individuals|men|women|children)', text_lower)
                if num_match:
                    char_count = int(num_match.group(1))
                elif 'no people' in text_lower or 'no one' in text_lower or 'empty' in text_lower:
                    char_count = 0
                elif 'three' in text_lower or '3' in response_text:
                    char_count = 3
                elif 'two' in text_lower or '2' in response_text:
                    char_count = 2
                elif 'one' in text_lower or 'a person' in text_lower or 'a man' in text_lower or 'a woman' in text_lower:
                    char_count = 1

                return {
                    "character_count": char_count,
                    "characters": [],
                    "scene_description": response_text,
                    "model_used": model_name
                }
            else:
                logger.error(f"Ollama error: {response.status_code} - {response.text[:200]}")
                return {"characters": [], "character_count": 0, "error": f"Ollama returned {response.status_code}"}

        except Exception as e:
            logger.error(f"Error describing characters: {e}")
            return {"characters": [], "character_count": 0, "error": str(e)}

    def ask_about_video(self, frame_descriptions: List[Dict], user_question: str,
                        text_model: str = None) -> Dict:
        """
        Answer user questions about the video based on frame descriptions.
        Uses a text LLM to synthesize information from all frames.

        Args:
            frame_descriptions: List of frame descriptions from vision model
            user_question: User's question about the video
            text_model: Text model to use (auto-detect if None)

        Returns:
            Dict with the answer
        """
        if not self.ollama_available:
            return {
                "answer": "Cannot answer - Ollama not available",
                "error": "Ollama service is not running",
                "confidence": "none"
            }

        # Auto-detect text model
        if text_model is None:
            text_model = self._detect_text_model()

        if text_model is None:
            return {
                "answer": "Cannot answer - no text model available. Install: ollama pull mistral",
                "error": "No text model found",
                "confidence": "none"
            }

        try:
            # Build context from frame descriptions (limit to avoid token overflow)
            context_parts = []
            for frame in frame_descriptions:
                time_fmt = frame.get('time_formatted', 'unknown')
                desc = frame.get('scene_description', '')
                if desc:
                    # Limit each description to 300 chars
                    context_parts.append(f"[{time_fmt}] {desc[:300]}")

            # Limit total context (keep most important frames spread across video)
            if len(context_parts) > 30:
                step = len(context_parts) // 30
                context_parts = context_parts[::step][:30]

            context = "\n".join(context_parts)

            # Build prompt that encourages honesty
            prompt = f"""Analyze the following video frame descriptions and answer the user's question.
Be honest: if you cannot determine something from the frames, say so clearly.

FRAME DESCRIPTIONS:
{context}

QUESTION: {user_question}

Give a direct, concise answer (under 300 words). Be honest about uncertainties."""

            payload = {
                "model": text_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 800,  # Limit output tokens
                    "temperature": 0.7
                }
            }

            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json=payload,
                timeout=180
            )

            if response.status_code == 200:
                result = response.json()
                # Some models (qwen3) put response in 'thinking' when using think mode
                answer = result.get('response', '').strip()
                if not answer and 'thinking' in result:
                    # Extract the actual answer from thinking (take last paragraph)
                    thinking = result.get('thinking', '')
                    # For qwen3, use the thinking as the answer if response is empty
                    answer = thinking.strip()

                # Determine confidence based on answer content
                confidence = "high"
                uncertainty_markers = ["cannot determine", "not clear", "uncertain",
                                       "may be", "possibly", "i don't know", "not sure",
                                       "cannot see", "not visible", "unclear"]
                for marker in uncertainty_markers:
                    if marker in answer.lower():
                        confidence = "low"
                        break

                return {
                    "answer": answer,
                    "model_used": text_model,
                    "frames_analyzed": len(frame_descriptions),
                    "confidence": confidence
                }
            else:
                return {
                    "answer": f"Error from LLM: {response.status_code}",
                    "error": response.text[:200],
                    "confidence": "none"
                }

        except Exception as e:
            logger.error(f"Error asking about video: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "error": str(e),
                "confidence": "none"
            }

    def _detect_text_model(self) -> Optional[str]:
        """
        Detect available text model for Q&A.
        Priority: mistral > llama > phi > any other
        """
        if not self.ollama_available:
            return None

        installed_models = [m.replace('ollama:', '') for m in self.local_models if m.startswith('ollama:')]

        # Priority order for text models (avoid qwen3 thinking mode)
        text_models_priority = ['mistral', 'llama3', 'llama2', 'phi', 'gemma', 'llava']

        for priority_model in text_models_priority:
            for installed in installed_models:
                if priority_model in installed.lower():
                    logger.info(f"Text model detected: {installed}")
                    return installed

        # If no priority model found, use first non-vision model
        for installed in installed_models:
            if not any(v in installed.lower() for v in ['llava', 'moondream', 'bakllava']):
                logger.info(f"Using text model: {installed}")
                return installed

        return None

    def describe_characters_fallback(self, image_path: str) -> Dict:
        """
        Fallback character description using OpenCV when Ollama is not available.
        Detects faces/bodies and provides basic descriptions.
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {"characters": [], "character_count": 0}

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width = img.shape[:2]

            characters = []

            # Detect faces
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for i, (x, y, w, h) in enumerate(faces):
                # Determine position
                center_x = x + w // 2
                if center_x < width // 3:
                    position = "left"
                elif center_x > 2 * width // 3:
                    position = "right"
                else:
                    position = "center"

                # Estimate if child or adult based on face size ratio
                face_ratio = h / height
                if face_ratio > 0.3:
                    age_group = "adult (close-up)"
                elif face_ratio > 0.15:
                    age_group = "adult"
                else:
                    age_group = "person (distant)"

                characters.append({
                    "id": i + 1,
                    "age_group": age_group,
                    "appearance": f"Person #{i+1} visible in frame",
                    "position": position,
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "detection_method": "opencv_haar"
                })

            # Detect full bodies
            body_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_fullbody.xml'
            )
            bodies = body_cascade.detectMultiScale(gray, 1.1, 4)

            for i, (x, y, w, h) in enumerate(bodies):
                # Check if this body overlaps with a detected face
                is_new = True
                for char in characters:
                    if 'bbox' in char:
                        fx, fy, fw, fh = char['bbox']
                        if x < fx + fw and x + w > fx and y < fy + fh and y + h > fy:
                            is_new = False
                            break

                if is_new:
                    center_x = x + w // 2
                    position = "left" if center_x < width // 3 else "right" if center_x > 2 * width // 3 else "center"

                    characters.append({
                        "id": len(characters) + 1,
                        "age_group": "person",
                        "appearance": f"Person #{len(characters) + 1} (full body visible)",
                        "position": position,
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "detection_method": "opencv_haar"
                    })

            scene_desc = f"Scene with {len(characters)} person(s) detected" if characters else "No persons detected in frame"

            return {
                "character_count": len(characters),
                "characters": characters,
                "scene_description": scene_desc,
                "note": "Using OpenCV detection (Ollama not available for detailed descriptions)"
            }

        except Exception as e:
            logger.error(f"Fallback description error: {e}")
            return {"characters": [], "character_count": 0, "error": str(e)}

    def detect_with_yolo_local(self, image_path: str, target_object: str,
                              confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Detection with local YOLO

        Args:
            image_path: Path to image
            target_object: Object to detect
            confidence_threshold: Confidence threshold

        Returns:
            List of detections
        """
        try:
            from ultralytics import YOLO

            # Load YOLO model
            model = YOLO('yolov8n.pt')  # Lightweight model

            # Perform detection
            results = model(image_path, verbose=False)

            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box information
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        class_name = model.names[cls]

                        # Check if it matches target object and confidence threshold
                        if conf >= confidence_threshold and self._match_object(class_name, target_object):
                            x1, y1, x2, y2 = box.xyxy[0].tolist()

                            detections.append({
                                'object': class_name,
                                'confidence': conf,
                                'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                                'description': f"{class_name} detected with {conf:.2f} confidence"
                            })

            logger.info(f"YOLO local: {len(detections)} detections")
            return detections

        except Exception as e:
            logger.error(f"YOLO local error: {e}")
            return []

    def detect_with_opencv_local(self, image_path: str, target_object: str) -> List[Dict]:
        """
        Detection with local OpenCV

        Args:
            image_path: Path to image
            target_object: Object to detect

        Returns:
            List of detections
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return []

            detections = []

            # Face/person detection with Haar Cascades
            if target_object.lower() in ['person', 'face', 'child', 'adult']:
                # Face cascade
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)

                for (x, y, w, h) in faces:
                    detections.append({
                        'object': 'face',
                        'confidence': 0.8,
                        'bbox': [int(x), int(y), int(w), int(h)],
                        'description': "Face detected with OpenCV Haar Cascade"
                    })

                # Cascade for full body
                body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
                bodies = body_cascade.detectMultiScale(gray, 1.1, 4)

                for (x, y, w, h) in bodies:
                    detections.append({
                        'object': 'person',
                        'confidence': 0.7,
                        'bbox': [int(x), int(y), int(w), int(h)],
                        'description': "Person detected with OpenCV Haar Cascade"
                    })

            logger.info(f"OpenCV local: {len(detections)} detections")
            return detections

        except Exception as e:
            logger.error(f"OpenCV local error: {e}")
            return []

    def _match_object(self, detected_class: str, target_object: str) -> bool:
        """
        Check if detected class matches target object

        Args:
            detected_class: Detected class
            target_object: Target object

        Returns:
            True if match
        """
        # Normalize strings
        detected = detected_class.lower()
        target = target_object.lower()

        # Direct match
        if detected == target:
            return True

        # Synonym matching
        synonyms = {
            'person': ['human', 'people', 'child', 'adult', 'man', 'woman'],
            'car': ['vehicle', 'automobile', 'truck', 'bus'],
            'animal': ['dog', 'cat', 'bird', 'horse'],
        }

        for key, values in synonyms.items():
            if detected == key and target in values:
                return True
            if target == key and detected in values:
                return True

        return False

class LocalAIRecognitionEngine:
    """
    AI recognition engine optimized for local processing
    """

    def __init__(self):
        """
        Initialize local recognition engine
        """
        self.local_detector = LocalAIDetector()
        self.detector_priority = self._get_detector_priority()

    def _get_detector_priority(self) -> List[str]:
        """
        Define detector priority order (local first)

        Returns:
            Ordered list of detectors
        """
        priority = []

        # 1. Ollama (use actual installed models)
        if self.local_detector.ollama_available:
            for model in self.local_detector.local_models:
                if model.startswith("ollama:"):
                    priority.append(model)

        # 2. YOLO local
        priority.extend([
            "yolo:yolov8n",
            "yolo:yolov8s"
        ])

        # 3. OpenCV local
        priority.extend([
            "opencv:haarcascade",
            "opencv:dnn"
        ])

        # 4. External APIs (last resort only)
        priority.extend([
            "openai:gpt-4-vision",
            "google:vision",
            "aws:rekognition"
        ])

        return priority

    def get_available_detectors(self) -> List[str]:
        """
        Get list of available detectors (local priority)

        Returns:
            List of available detectors
        """
        available = []

        # Local detectors
        for detector in self.detector_priority:
            if detector.startswith("ollama:") and self.local_detector.ollama_available:
                available.append(detector)
            elif detector.startswith("yolo:"):
                try:
                    import ultralytics
                    available.append(detector)
                except ImportError:
                    pass
            elif detector.startswith("opencv:"):
                available.append(detector)

        # Add external detectors only if explicitly requested
        # (not by default to respect local preference)

        return available

    def detect_in_frame(self, frame_path: str, target_object: str,
                       detector_name: str = "auto", confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Detect objects in frame with local priority

        Args:
            frame_path: Path to frame
            target_object: Object to detect
            detector_name: Detector name ("auto" for automatic selection)
            confidence_threshold: Confidence threshold

        Returns:
            List of detections
        """
        if detector_name == "auto":
            # Auto-select best local detector
            detector_name = self._select_best_local_detector(target_object)

        logger.info(f"Detection with {detector_name} for '{target_object}' in {frame_path}")

        try:
            # Ollama detectors
            if detector_name.startswith("ollama:"):
                model_name = detector_name.split(":", 1)[1]
                return self.local_detector.detect_with_ollama(frame_path, target_object, model_name)

            # YOLO detectors
            elif detector_name.startswith("yolo:"):
                return self.local_detector.detect_with_yolo_local(frame_path, target_object, confidence_threshold)

            # OpenCV detectors
            elif detector_name.startswith("opencv:"):
                return self.local_detector.detect_with_opencv_local(frame_path, target_object)

            # Fallback to simulation if no local detector available
            else:
                logger.warning(f"Detector {detector_name} not available, using simulation")
                return self._simulate_detection(frame_path, target_object, confidence_threshold)

        except Exception as e:
            logger.error(f"Detection error with {detector_name}: {e}")
            return []

    def _select_best_local_detector(self, target_object: str) -> str:
        """
        Select best local detector for target object

        Args:
            target_object: Object to detect

        Returns:
            Optimal detector name
        """
        # For people/faces, prioritize Ollama then OpenCV
        if target_object.lower() in ['person', 'child', 'face', 'human']:
            if self.local_detector.ollama_available:
                return "ollama:llava"
            else:
                return "opencv:haarcascade"

        # For general objects, prioritize Ollama then YOLO
        else:
            if self.local_detector.ollama_available:
                return "ollama:llava"
            else:
                try:
                    import ultralytics
                    return "yolo:yolov8n"
                except ImportError:
                    return "opencv:haarcascade"

    def _simulate_detection(self, frame_path: str, target_object: str,
                           confidence_threshold: float) -> List[Dict]:
        """
        Simulated detection for testing

        Args:
            frame_path: Path to frame
            target_object: Object to detect
            confidence_threshold: Confidence threshold

        Returns:
            Simulated detections
        """
        import random

        # Simulate random detection
        if random.random() > 0.7:  # 30% chance of detection
            return [{
                'object': target_object,
                'confidence': random.uniform(confidence_threshold, 1.0),
                'bbox': [
                    random.randint(0, 200),
                    random.randint(0, 200),
                    random.randint(50, 150),
                    random.randint(50, 150)
                ],
                'description': f"Simulated detection of {target_object}"
            }]

        return []

    def benchmark_detectors(self, test_image_path: str, target_object: str) -> Dict[str, Dict]:
        """
        Compare local detector performance

        Args:
            test_image_path: Test image
            target_object: Object to detect

        Returns:
            Benchmark results
        """
        import time
        results = {}
        available_detectors = self.get_available_detectors()

        for detector in available_detectors:
            start_time = time.time()

            try:
                detections = self.detect_in_frame(test_image_path, target_object, detector)
                end_time = time.time()

                results[detector] = {
                    'detections_count': len(detections),
                    'processing_time': end_time - start_time,
                    'success': True,
                    'detections': detections
                }

            except Exception as e:
                results[detector] = {
                    'detections_count': 0,
                    'processing_time': 0,
                    'success': False,
                    'error': str(e)
                }

        return results

def test_local_ai():
    """
    Test local AI functionality
    """
    print("=== Local AI Test ===")

    # Initialize engine
    engine = LocalAIRecognitionEngine()

    # Display available detectors
    detectors = engine.get_available_detectors()
    print(f"Available local detectors: {detectors}")

    # Test with dummy image
    import cv2
    import numpy as np

    # Create test image
    test_img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    test_path = "test_local_ai.jpg"
    cv2.imwrite(test_path, test_img)

    # Detection test
    for detector in detectors[:2]:  # Test first 2
        print(f"\nTest with {detector}:")
        detections = engine.detect_in_frame(test_path, "person", detector)
        print(f"  Detections: {len(detections)}")

        for detection in detections:
            print(f"    - {detection['object']}: {detection['confidence']:.2f}")

    # Cleanup
    try:
        os.remove(test_path)
    except:
        pass

    print("\nLocal AI test complete!")

if __name__ == "__main__":
    test_local_ai()
