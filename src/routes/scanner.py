"""
Routes API pour le scanner video YouTube
"""

from flask import Blueprint, request, jsonify, current_app
import os
import sys
import json
import hashlib
import shutil
import threading
from datetime import datetime, timedelta
import logging

# Ajouter le repertoire src au path pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from local_ai_recognition import LocalAIRecognitionEngine

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Creer le blueprint
scanner_bp = Blueprint('scanner', __name__)

# Instance globale du scanner
scanner = None

# Cache pour les videos (5 minutes)
VIDEO_CACHE_TTL = 300  # 5 minutes en secondes
video_cache = {}  # {url_hash: {'video_info': ..., 'frames': [...], 'expires': datetime}}
cache_lock = threading.Lock()


def get_url_hash(url):
    """Generate a hash for the URL"""
    return hashlib.md5(url.encode()).hexdigest()


def cleanup_expired_cache():
    """Remove expired cache entries and their files"""
    now = datetime.now()
    expired = []

    with cache_lock:
        for url_hash, data in video_cache.items():
            if now > data['expires']:
                expired.append(url_hash)

        for url_hash in expired:
            data = video_cache.pop(url_hash)
            # Delete video and frame files
            try:
                if 'video_path' in data.get('video_info', {}):
                    video_path = data['video_info']['video_path']
                    if os.path.exists(video_path):
                        os.remove(video_path)
                        logger.info(f"Deleted cached video: {video_path}")

                # Delete frames directory
                if data.get('frames') and len(data['frames']) > 0:
                    frames_dir = os.path.dirname(data['frames'][0])
                    if os.path.exists(frames_dir):
                        shutil.rmtree(frames_dir)
                        logger.info(f"Deleted cached frames: {frames_dir}")
            except Exception as e:
                logger.error(f"Error cleaning cache: {e}")

    return len(expired)


def get_cached_video(url):
    """Get video from cache if available and not expired"""
    cleanup_expired_cache()
    url_hash = get_url_hash(url)

    with cache_lock:
        if url_hash in video_cache:
            data = video_cache[url_hash]
            if datetime.now() < data['expires']:
                # Extend TTL on access
                data['expires'] = datetime.now() + timedelta(seconds=VIDEO_CACHE_TTL)
                logger.info(f"Cache hit for {url[:50]}...")
                return data
    return None


def cache_video(url, video_info, frames):
    """Cache video info and frames"""
    url_hash = get_url_hash(url)

    with cache_lock:
        video_cache[url_hash] = {
            'video_info': video_info,
            'frames': frames,
            'expires': datetime.now() + timedelta(seconds=VIDEO_CACHE_TTL)
        }
    logger.info(f"Cached video: {url[:50]}... (expires in {VIDEO_CACHE_TTL}s)")

def get_scanner():
    """
    Obtient l'instance du scanner (singleton) optimisé pour l'IA locale
    """
    global scanner
    if scanner is None:
        output_dir = os.environ.get('SCANNER_DATA_DIR', '/app/data')
        
        # Créer les composants avec priorité locale
        from video_processor import YouTubeVideoProcessor
        from database import TimecodeDatabase
        
        video_processor = YouTubeVideoProcessor(
            output_dir=os.path.join(output_dir, "videos"),
            frames_dir=os.path.join(output_dir, "frames")
        )
        ai_engine = LocalAIRecognitionEngine()  # Moteur IA local
        database = TimecodeDatabase(os.path.join(output_dir, "timecodes.db"))
        
        # Créer un objet scanner optimisé pour le local
        class LocalScanner:
            def __init__(self):
                self.video_processor = video_processor
                self.ai_engine = ai_engine
                self.database = database
                
            def get_available_detectors(self):
                return self.ai_engine.get_available_detectors()
            
            def scan_video(self, url, target_object, detector="auto",
                          confidence_threshold=0.5, interval_seconds=5, max_frames=None):
                """
                Scan video avec priorite aux detecteurs locaux et cache
                """
                logger.info(f"Demarrage scan local: {url}")

                # Check cache first
                cached = get_cached_video(url)
                if cached:
                    video_info = cached['video_info']
                    frames = cached['frames']
                    logger.info("Using cached video and frames")
                else:
                    # Download and extract frames
                    video_info = self.video_processor.download_video(url)
                    frames = self.video_processor.extract_frames(
                        video_info['video_path'],
                        interval_seconds=interval_seconds,
                        max_frames=max_frames
                    )
                    # Cache for future queries
                    cache_video(url, video_info, frames)
                
                # Add video to database
                video_info['url'] = url
                video_id = self.database.add_video(video_info)

                # Create scan in database
                scan_id = self.database.add_scan(
                    video_id=video_id,
                    target_object=target_object,
                    model_used=detector,
                    confidence_threshold=confidence_threshold,
                    interval_seconds=interval_seconds,
                    total_frames=len(frames)
                )
                
                # Analyser les frames avec l'IA locale
                total_detections = 0
                for i, frame_path in enumerate(frames):
                    timestamp = i * interval_seconds
                    
                    detections = self.ai_engine.detect_in_frame(
                        frame_path, target_object, detector, confidence_threshold
                    )
                    
                    # Enregistrer les detections
                    for detection in detections:
                        self.database.add_detection(
                            scan_id=scan_id,
                            timestamp=timestamp,
                            confidence=detection.get('confidence', 0),
                            frame_path=frame_path,
                            bounding_box=detection.get('box'),
                            metadata=detection
                        )
                        total_detections += 1
                
                # Retourner les résultats
                return {
                    'scan_info': {'scan_id': scan_id},
                    'results': {
                        'total_detections': total_detections,
                        'detection_rate': total_detections / len(frames) if frames else 0
                    },
                    'video_info': {
                        'duration': video_info['duration'],
                        'total_frames_analyzed': len(frames)
                    },
                    'timecodes': self.database.get_detections_for_scan(scan_id)
                }
        
        scanner = LocalScanner()
    return scanner

@scanner_bp.route('/health', methods=['GET'])
def health_check():
    """
    Vérification de l'état de l'API
    """
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@scanner_bp.route('/detectors', methods=['GET'])
def get_detectors():
    """
    Récupère la liste des détecteurs disponibles
    """
    try:
        scanner_instance = get_scanner()
        detectors = scanner_instance.get_available_detectors()
        
        return jsonify({
            'success': True,
            'detectors': detectors,
            'count': len(detectors)
        })
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des détecteurs: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@scanner_bp.route('/scan', methods=['POST'])
def start_scan():
    """
    Démarre un scan de vidéo YouTube
    """
    try:
        data = request.get_json()
        
        # Validation des paramètres
        required_fields = ['url', 'target_object']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Champ requis manquant: {field}'
                }), 400
        
        url = data['url']
        target_object = data['target_object']
        detector = data.get('detector', 'yolo')
        confidence_threshold = float(data.get('confidence_threshold', 0.5))
        interval_seconds = int(data.get('interval_seconds', 5))
        max_frames = data.get('max_frames')
        
        if max_frames is not None:
            max_frames = int(max_frames)
        
        # Validation des valeurs
        if confidence_threshold < 0 or confidence_threshold > 1:
            return jsonify({
                'success': False,
                'error': 'confidence_threshold doit être entre 0 et 1'
            }), 400
        
        if interval_seconds < 1:
            return jsonify({
                'success': False,
                'error': 'interval_seconds doit être >= 1'
            }), 400
        
        logger.info(f"Démarrage du scan: {url}, objet: {target_object}")
        
        # Démarrer le scan
        scanner_instance = get_scanner()
        
        # Vérifier que le détecteur existe
        available_detectors = scanner_instance.get_available_detectors()
        if detector not in available_detectors:
            return jsonify({
                'success': False,
                'error': f'Détecteur non disponible: {detector}. Disponibles: {available_detectors}'
            }), 400
        
        # Pour cette version, on fait le scan de manière synchrone
        # Dans une version production, on utiliserait Celery ou similar pour l'asynchrone
        try:
            result = scanner_instance.scan_video(
                url=url,
                target_object=target_object,
                detector=detector,
                confidence_threshold=confidence_threshold,
                interval_seconds=interval_seconds,
                max_frames=max_frames
            )
            
            return jsonify({
                'success': True,
                'scan_id': result['scan_info']['scan_id'],
                'message': 'Scan terminé avec succès',
                'results': {
                    'total_detections': result['results']['total_detections'],
                    'detection_rate': result['results']['detection_rate'],
                    'video_duration': result['video_info']['duration'],
                    'frames_analyzed': result['video_info']['total_frames_analyzed']
                },
                'timecodes': result['timecodes'][:10]  # Limiter à 10 premiers pour l'API
            })
            
        except Exception as scan_error:
            logger.error(f"Erreur lors du scan: {str(scan_error)}")
            return jsonify({
                'success': False,
                'error': f'Erreur lors du scan: {str(scan_error)}'
            }), 500
        
    except Exception as e:
        logger.error(f"Erreur dans start_scan: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@scanner_bp.route('/scan/<int:scan_id>', methods=['GET'])
def get_scan_results(scan_id):
    """
    Récupère les résultats d'un scan
    """
    try:
        scanner_instance = get_scanner()
        
        # Récupérer les informations du scan
        summary = scanner_instance.database.get_timecodes_summary(scan_id)
        
        if not summary or summary.get('total_detections', 0) == 0:
            return jsonify({
                'success': False,
                'error': 'Scan non trouvé ou aucune détection'
            }), 404
        
        # Récupérer les détections
        detections = scanner_instance.database.get_detections_for_scan(scan_id)
        
        # Formater les timecodes
        timecodes = []
        for detection in detections:
            timestamp = detection['timestamp']
            hours = int(timestamp // 3600)
            minutes = int((timestamp % 3600) // 60)
            seconds = int(timestamp % 60)
            time_formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            timecodes.append({
                'timestamp': timestamp,
                'time_formatted': time_formatted,
                'confidence': detection['confidence'],
                'frame_path': detection['frame_path']
            })
        
        return jsonify({
            'success': True,
            'scan_info': summary,
            'timecodes': timecodes,
            'total_detections': len(timecodes)
        })
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du scan {scan_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@scanner_bp.route('/scan/<int:scan_id>/export', methods=['GET'])
def export_scan_csv(scan_id):
    """
    Exporte les résultats d'un scan en CSV
    """
    try:
        scanner_instance = get_scanner()
        
        # Vérifier que le scan existe
        summary = scanner_instance.database.get_timecodes_summary(scan_id)
        if not summary:
            return jsonify({
                'success': False,
                'error': 'Scan non trouvé'
            }), 404
        
        # Générer le fichier CSV
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            scanner_instance.database.export_timecodes_csv(scan_id, tmp_file.name)
            csv_path = tmp_file.name
        
        # Lire le contenu du CSV
        with open(csv_path, 'r', encoding='utf-8') as f:
            csv_content = f.read()
        
        # Nettoyer le fichier temporaire
        os.unlink(csv_path)
        
        return jsonify({
            'success': True,
            'csv_content': csv_content,
            'filename': f'timecodes_scan_{scan_id}.csv'
        })
        
    except Exception as e:
        logger.error(f"Erreur lors de l'export CSV du scan {scan_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@scanner_bp.route('/test', methods=['POST'])
def test_detection():
    """
    Test de détection sur une image uploadée
    """
    try:
        # Cette route pourrait être utilisée pour tester les détecteurs
        # sur une image uploadée par l'utilisateur
        
        return jsonify({
            'success': True,
            'message': 'Fonctionnalité de test en développement'
        })
        
    except Exception as e:
        logger.error(f"Erreur lors du test: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@scanner_bp.route('/stats', methods=['GET'])
def get_stats():
    """
    Récupère les statistiques générales
    """
    try:
        # Cette route pourrait retourner des statistiques sur l'utilisation
        return jsonify({
            'success': True,
            'stats': {
                'total_scans': 0,
                'total_detections': 0,
                'available_detectors': len(get_scanner().get_available_detectors())
            }
        })
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des stats: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def _calculate_sampling(duration: float, user_interval: int = None, user_max: int = None) -> tuple:
    """
    Calculate optimal frame sampling based on video duration.
    Returns (interval_seconds, max_frames)
    """
    if user_interval and user_max:
        return (user_interval, user_max)

    if duration < 60:  # < 1 min
        interval = 2
        max_frames = 30
    elif duration < 600:  # 1-10 min
        interval = 5
        max_frames = 60
    elif duration < 3600:  # 10-60 min
        interval = 10
        max_frames = 100
    else:  # > 1h
        interval = 30
        max_frames = 150

    # Override with user values if provided
    if user_interval:
        interval = user_interval
    if user_max:
        max_frames = user_max

    return (interval, max_frames)


@scanner_bp.route('/describe-characters', methods=['POST'])
def describe_characters():
    """
    Describe WHO the characters are in a video (not just detect them).
    Uses Ollama vision models to provide detailed character descriptions.
    Auto-adjusts sampling based on video duration for optimal coverage.
    """
    try:
        data = request.get_json()

        if 'url' not in data:
            return jsonify({
                'success': False,
                'error': 'URL required'
            }), 400

        url = data['url']
        user_interval = data.get('interval_seconds')
        user_max = data.get('max_frames')

        logger.info(f"Describing characters in: {url}")

        scanner_instance = get_scanner()

        # Check cache or download
        cached = get_cached_video(url)
        if cached:
            video_info = cached['video_info']
            duration = video_info.get('duration', 60)
            interval_seconds, max_frames = _calculate_sampling(duration, user_interval, user_max)
            frames = cached['frames'][:max_frames]
            logger.info(f"Using cached video (duration: {duration}s, interval: {interval_seconds}s)")
        else:
            video_info = scanner_instance.video_processor.download_video(url)
            duration = video_info.get('duration', 60)
            interval_seconds, max_frames = _calculate_sampling(duration, user_interval, user_max)
            logger.info(f"Auto sampling: {duration}s video -> {interval_seconds}s interval, max {max_frames} frames")

            frames = scanner_instance.video_processor.extract_frames(
                video_info['video_path'],
                interval_seconds=interval_seconds,
                max_frames=max_frames
            )
            video_info['url'] = url
            cache_video(url, video_info, frames)

        # Describe characters in each frame
        all_characters = []
        frame_descriptions = []

        for i, frame_path in enumerate(frames):
            timestamp = i * interval_seconds
            hours = int(timestamp // 3600)
            minutes = int((timestamp % 3600) // 60)
            seconds = int(timestamp % 60)
            time_fmt = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            # Get character description (vision model if available, fallback to OpenCV)
            detector = scanner_instance.ai_engine.local_detector
            if detector.vision_model:
                char_info = detector.describe_characters(frame_path)
            else:
                char_info = detector.describe_characters_fallback(frame_path)

            frame_descriptions.append({
                'timestamp': timestamp,
                'time_formatted': time_fmt,
                'frame_path': frame_path,
                'character_count': char_info.get('character_count', 0),
                'characters': char_info.get('characters', []),
                'scene_description': char_info.get('scene_description', '')
            })

            # Collect unique characters
            for char in char_info.get('characters', []):
                all_characters.append({
                    'first_seen': time_fmt,
                    **char
                })

        # Summarize character descriptions
        summary = _summarize_characters(frame_descriptions)

        # Get detector status for warnings
        detector_status = scanner_instance.ai_engine.local_detector.get_status()

        response = {
            'success': True,
            'video_info': {
                'title': video_info.get('title'),
                'duration': video_info.get('duration'),
                'url': url
            },
            'sampling': {
                'interval_seconds': interval_seconds,
                'max_frames': max_frames,
                'frames_analyzed': len(frames),
                'auto_adjusted': user_interval is None
            },
            'character_summary': summary,
            'frame_by_frame': frame_descriptions,
            'total_frames_analyzed': len(frames),
            'detector_status': detector_status
        }

        # Add warning if using fallback
        if detector_status.get('using_fallback'):
            response['warning'] = "No vision model detected. Using OpenCV fallback (limited accuracy)."
            response['recommendation'] = detector_status.get('recommendation')

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error describing characters: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@scanner_bp.route('/ask-video', methods=['POST'])
def ask_video():
    """
    Ask any question about a video. The LLM will analyze the video frames
    and answer based on what it can see.

    Request body:
        - url: YouTube video URL (required)
        - question: User's question about the video (required)
        - interval_seconds: Frame sampling interval (optional, auto-calculated)
        - max_frames: Maximum frames to analyze (optional, auto-calculated)

    The LLM will be honest about what it cannot determine from the frames.
    """
    try:
        data = request.get_json()

        # Validate required fields
        if 'url' not in data:
            return jsonify({
                'success': False,
                'error': 'URL required'
            }), 400

        if 'question' not in data:
            return jsonify({
                'success': False,
                'error': 'Question required. What do you want to know about the video?'
            }), 400

        url = data['url']
        question = data['question']
        user_interval = data.get('interval_seconds')
        user_max = data.get('max_frames')

        logger.info(f"Video Q&A: {url}")
        logger.info(f"Question: {question}")

        scanner_instance = get_scanner()

        # Step 1: Get video and frames (from cache or download)
        cached = get_cached_video(url)
        if cached:
            video_info = cached['video_info']
            duration = video_info.get('duration', 60)
            interval_seconds, max_frames = _calculate_sampling(duration, user_interval, user_max)
            frames = cached['frames'][:max_frames]
            logger.info(f"Using cached video (duration: {duration}s)")
        else:
            video_info = scanner_instance.video_processor.download_video(url)
            duration = video_info.get('duration', 60)
            interval_seconds, max_frames = _calculate_sampling(duration, user_interval, user_max)
            logger.info(f"Auto sampling: {duration}s video -> {interval_seconds}s interval, max {max_frames} frames")

            frames = scanner_instance.video_processor.extract_frames(
                video_info['video_path'],
                interval_seconds=interval_seconds,
                max_frames=max_frames
            )
            video_info['url'] = url
            cache_video(url, video_info, frames)

        # Step 2: Analyze frames with vision model
        frame_descriptions = []
        detector = scanner_instance.ai_engine.local_detector

        for i, frame_path in enumerate(frames):
            timestamp = i * interval_seconds
            hours = int(timestamp // 3600)
            minutes = int((timestamp % 3600) // 60)
            seconds = int(timestamp % 60)
            time_fmt = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            # Get frame description
            if detector.vision_model:
                char_info = detector.describe_characters(frame_path)
            else:
                char_info = detector.describe_characters_fallback(frame_path)

            frame_descriptions.append({
                'timestamp': timestamp,
                'time_formatted': time_fmt,
                'scene_description': char_info.get('scene_description', ''),
                'character_count': char_info.get('character_count', 0)
            })

        # Step 3: Send to text LLM with user question
        answer_result = detector.ask_about_video(frame_descriptions, question)

        # Build response
        response = {
            'success': True,
            'video_info': {
                'title': video_info.get('title'),
                'duration': video_info.get('duration'),
                'url': url
            },
            'question': question,
            'answer': answer_result.get('answer', 'No answer available'),
            'confidence': answer_result.get('confidence', 'unknown'),
            'analysis_info': {
                'frames_analyzed': len(frames),
                'interval_seconds': interval_seconds,
                'vision_model': detector.vision_model,
                'text_model': answer_result.get('model_used')
            }
        }

        # Add warnings if applicable
        if detector.using_fallback:
            response['warning'] = "Using OpenCV fallback (limited accuracy). Install: ollama pull llava"

        if answer_result.get('error'):
            response['llm_error'] = answer_result.get('error')

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in ask-video: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def _summarize_characters(frame_descriptions: list) -> dict:
    """Summarize character descriptions across frames"""
    descriptions = []
    total_count = 0

    for frame in frame_descriptions:
        count = frame.get('character_count', 0)
        desc = frame.get('scene_description', '')
        if count > 0 and desc:
            total_count = max(total_count, count)
            descriptions.append({
                'time': frame.get('time_formatted'),
                'count': count,
                'description': desc[:200]
            })

    if not descriptions:
        return {
            'total_detected': 0,
            'descriptions': [],
            'summary': 'No characters detected in analyzed frames'
        }

    # Build summary
    summary_text = f"Detected up to {total_count} character(s) across {len(descriptions)} frames:\n"
    for d in descriptions[:3]:
        summary_text += f"\n[{d['time']}] {d['count']} character(s): {d['description'][:100]}..."

    return {
        'total_detected': total_count,
        'frames_with_characters': len(descriptions),
        'descriptions': descriptions,
        'summary': summary_text
    }
