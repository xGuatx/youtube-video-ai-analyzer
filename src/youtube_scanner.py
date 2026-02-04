"""
Module principal intégrant tous les composants du scanner vidéo YouTube
"""

import os
import json
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Optional

from video_processor import YouTubeVideoProcessor
from ai_recognition import AIRecognitionEngine
from database import TimecodeDatabase

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class YouTubeScanner:
    """
    Classe principale du scanner vidéo YouTube
    Intègre le téléchargement, l'extraction de frames, la reconnaissance IA et le stockage
    """
    
    def __init__(self, output_dir: str = "scanner_output"):
        """
        Initialise le scanner YouTube
        
        Args:
            output_dir: Répertoire de sortie principal
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialiser les composants
        self.video_processor = YouTubeVideoProcessor(
            output_dir=str(self.output_dir / "videos"),
            frames_dir=str(self.output_dir / "frames")
        )
        
        self.ai_engine = AIRecognitionEngine()
        self.database = TimecodeDatabase(str(self.output_dir / "timecodes.db"))
        
        logger.info(f"Scanner YouTube initialisé dans: {self.output_dir}")
    
    def scan_video(self, url: str, target_object: str, 
                  detector: str = 'yolo', confidence_threshold: float = 0.5,
                  interval_seconds: int = 5, max_frames: int = None) -> Dict:
        """
        Scanne une vidéo YouTube pour détecter un objet spécifique
        
        Args:
            url: URL de la vidéo YouTube
            target_object: Objet à détecter (ex: 'person', 'enfant', 'child')
            detector: Détecteur IA à utiliser ('yolo', 'opencv', 'cloud_openai')
            confidence_threshold: Seuil de confiance minimum (0.0 à 1.0)
            interval_seconds: Intervalle entre les frames à analyser
            max_frames: Nombre maximum de frames à traiter (None = pas de limite)
            
        Returns:
            Dictionnaire avec les résultats du scan
        """
        try:
            logger.info(f"Début du scan de: {url}")
            logger.info(f"Objet cible: {target_object}, Détecteur: {detector}")
            
            # Étape 1: Télécharger et extraire les frames
            logger.info("Étape 1: Téléchargement et extraction des frames...")
            
            # Vérifier si la vidéo existe déjà en base
            existing_video = self.database.get_video_by_url(url)
            
            if existing_video:
                logger.info("Vidéo déjà en base de données")
                video_info = existing_video
                video_id = existing_video['id']
                
                # Récupérer les frames existantes
                video_name = Path(existing_video['video_path']).stem
                frames_dir = self.output_dir / "frames" / video_name
                frames = sorted(list(frames_dir.glob("*.jpg")))
                frames = [str(f) for f in frames]
            else:
                # Télécharger et extraire
                video_info = self.video_processor.process_youtube_video(
                    url=url,
                    interval_seconds=interval_seconds,
                    use_ffmpeg=True,
                    max_frames=max_frames
                )
                
                # Ajouter à la base de données
                video_id = self.database.add_video(video_info)
                frames = video_info['frames']
            
            logger.info(f"Frames disponibles: {len(frames)}")
            
            # Étape 2: Créer un scan en base
            scan_id = self.database.add_scan(
                video_id=video_id,
                target_object=target_object,
                model_used=detector,
                confidence_threshold=confidence_threshold,
                interval_seconds=interval_seconds,
                total_frames=len(frames)
            )
            
            # Étape 3: Reconnaissance IA sur les frames
            logger.info("Étape 2: Reconnaissance IA...")
            
            # Traitement par lots pour optimiser les performances
            batch_size = 10
            total_detections = 0
            processed_frames = 0
            
            for i in range(0, len(frames), batch_size):
                batch_frames = frames[i:i + batch_size]
                
                # Détection en lot
                batch_results = self.ai_engine.batch_detect(
                    frame_paths=batch_frames,
                    target_object=target_object,
                    detector_name=detector,
                    confidence_threshold=confidence_threshold
                )
                
                # Traiter les résultats
                for frame_path, detections in batch_results.items():
                    processed_frames += 1
                    
                    if detections:
                        # Calculer le timestamp à partir du nom de fichier
                        frame_name = Path(frame_path).name
                        
                        # Extraire le timestamp du nom de fichier (format: frame_XXXXXX_XXXs.jpg)
                        try:
                            if '_' in frame_name and 's.jpg' in frame_name:
                                timestamp_str = frame_name.split('_')[-1].replace('s.jpg', '')
                                timestamp = float(timestamp_str)
                            else:
                                # Fallback: calculer à partir de l'index
                                frame_index = processed_frames - 1
                                timestamp = frame_index * interval_seconds
                        except:
                            timestamp = processed_frames * interval_seconds
                        
                        # Ajouter chaque détection à la base
                        for detection in detections:
                            self.database.add_detection(
                                scan_id=scan_id,
                                timestamp=timestamp,
                                frame_path=frame_path,
                                confidence=detection['confidence'],
                                bounding_box=detection['bbox'],
                                metadata={'detector': detector, 'class': detection['class']}
                            )
                            total_detections += 1
                
                # Afficher le progrès
                progress = (processed_frames / len(frames)) * 100
                logger.info(f"Progrès: {progress:.1f}% ({processed_frames}/{len(frames)} frames)")
            
            # Étape 4: Générer le rapport final
            logger.info("Étape 3: Génération du rapport...")
            
            summary = self.database.get_timecodes_summary(scan_id)
            detections = self.database.get_detections_for_scan(scan_id)
            
            # Exporter les timecodes en CSV
            csv_path = self.output_dir / f"timecodes_scan_{scan_id}.csv"
            self.database.export_timecodes_csv(scan_id, str(csv_path))
            
            # Créer le rapport JSON
            report = {
                'scan_info': {
                    'scan_id': scan_id,
                    'video_url': url,
                    'video_title': video_info.get('title', 'Titre inconnu'),
                    'target_object': target_object,
                    'detector_used': detector,
                    'confidence_threshold': confidence_threshold,
                    'scan_date': datetime.now().isoformat()
                },
                'video_info': {
                    'duration': video_info.get('duration', 0),
                    'total_frames_analyzed': len(frames),
                    'interval_seconds': interval_seconds
                },
                'results': {
                    'total_detections': total_detections,
                    'detection_rate': total_detections / len(frames) if frames else 0,
                    'first_detection': summary.get('first_detection'),
                    'last_detection': summary.get('last_detection'),
                    'avg_confidence': summary.get('avg_confidence')
                },
                'timecodes': [
                    {
                        'timestamp': d['timestamp'],
                        'time_formatted': self._format_timestamp(d['timestamp']),
                        'confidence': d['confidence'],
                        'frame_path': d['frame_path']
                    }
                    for d in detections
                ],
                'files': {
                    'csv_export': str(csv_path),
                    'database': str(self.database.db_path)
                }
            }
            
            # Sauvegarder le rapport
            report_path = self.output_dir / f"report_scan_{scan_id}.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Scan terminé! {total_detections} détections trouvées")
            logger.info(f"Rapport sauvegardé: {report_path}")
            
            return report
            
        except Exception as e:
            logger.error(f"Erreur lors du scan: {str(e)}")
            raise
    
    def _format_timestamp(self, timestamp: float) -> str:
        """
        Formate un timestamp en HH:MM:SS
        
        Args:
            timestamp: Timestamp en secondes
            
        Returns:
            Timestamp formaté
        """
        hours = int(timestamp // 3600)
        minutes = int((timestamp % 3600) // 60)
        seconds = int(timestamp % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def get_scan_history(self) -> List[Dict]:
        """
        Récupère l'historique des scans
        
        Returns:
            Liste des scans effectués
        """
        # Cette méthode nécessiterait une requête SQL plus complexe
        # Pour l'instant, on retourne une liste vide
        return []
    
    def add_cloud_detector(self, provider: str, api_key: str):
        """
        Ajoute un détecteur cloud
        
        Args:
            provider: Fournisseur (openai, google, aws)
            api_key: Clé API
        """
        self.ai_engine.add_cloud_detector(provider, api_key)
        logger.info(f"Détecteur cloud ajouté: {provider}")
    
    def get_available_detectors(self) -> List[str]:
        """
        Retourne la liste des détecteurs disponibles
        """
        return self.ai_engine.get_available_detectors()
    
    def cleanup_old_data(self, days: int = 30):
        """
        Nettoie les anciennes données
        
        Args:
            days: Nombre de jours à conserver
        """
        self.database.cleanup_old_data(days)
        logger.info(f"Nettoyage effectué: données de plus de {days} jours supprimées")

def demo_scan():
    """
    Démonstration du scanner avec une vidéo de test
    """
    # Initialiser le scanner
    scanner = YouTubeScanner("demo_output")
    
    print("=== Démonstration du Scanner Vidéo YouTube ===")
    print(f"Détecteurs disponibles: {scanner.get_available_detectors()}")
    
    # Créer une vidéo de test locale pour la démo
    print("\\nCréation d'une vidéo de test...")
    
    import cv2
    import numpy as np
    
    # Créer une vidéo de test avec des "personnes" simulées
    width, height = 640, 480
    fps = 30
    duration = 30  # 30 secondes
    total_frames = fps * duration
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = "demo_video.mp4"
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    for i in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (50, 100, 150)  # Fond bleu
        
        # Ajouter des "personnes" à certains moments
        if i % 90 == 0:  # Toutes les 3 secondes
            # Dessiner un rectangle représentant une personne
            x = 200 + (i // 90) * 50
            y = 150
            cv2.rectangle(frame, (x, y), (x + 100, y + 200), (255, 255, 255), -1)
            cv2.putText(frame, 'Person', (x + 10, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Texte avec le temps
        time_text = f"Time: {i/fps:.1f}s"
        cv2.putText(frame, time_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Vidéo de test créée: {video_path}")
    
    # Simuler un scan (sans vraie URL YouTube)
    print("\\nSimulation d'un scan...")
    
    try:
        # Pour la démo, on va traiter directement la vidéo locale
        # En réalité, on passerait une URL YouTube
        
        # Extraire les frames de la vidéo de test
        frames = scanner.video_processor.extract_frames(video_path, interval_seconds=3)
        print(f"Frames extraites: {len(frames)}")
        
        # Simuler la détection IA
        target_object = "person"
        detector = "yolo"
        
        print(f"Recherche de '{target_object}' avec le détecteur '{detector}'...")
        
        detections_found = 0
        for i, frame_path in enumerate(frames):
            # Simulation de détection
            detections = scanner.ai_engine.detect_in_frame(frame_path, target_object, detector)
            if detections:
                timestamp = i * 3  # 3 secondes d'intervalle
                print(f"  Détection à {timestamp}s (confiance: {detections[0]['confidence']:.2f})")
                detections_found += 1
        
        print(f"\\nRésumé: {detections_found} détections trouvées sur {len(frames)} frames analysées")
        
    except Exception as e:
        print(f"Erreur lors de la démo: {str(e)}")

if __name__ == "__main__":
    demo_scan()

