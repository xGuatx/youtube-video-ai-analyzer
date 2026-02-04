"""
Module de base de données pour stocker les timecodes et métadonnées
Utilise SQLite pour la simplicité et la portabilité
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class TimecodeDatabase:
    """
    Classe pour gérer la base de données des timecodes détectés
    """
    
    def __init__(self, db_path: str = "timecodes.db"):
        """
        Initialise la base de données
        
        Args:
            db_path: Chemin vers le fichier de base de données SQLite
        """
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """
        Initialise les tables de la base de données
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Table des vidéos
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS videos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT UNIQUE NOT NULL,
                    title TEXT,
                    duration REAL,
                    video_path TEXT,
                    upload_date TEXT,
                    uploader TEXT,
                    view_count INTEGER,
                    description TEXT,
                    processing_date TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Table des scans (sessions de détection)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS scans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id INTEGER,
                    target_object TEXT NOT NULL,
                    model_used TEXT,
                    confidence_threshold REAL,
                    interval_seconds INTEGER,
                    total_frames INTEGER,
                    detections_count INTEGER,
                    scan_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (video_id) REFERENCES videos (id)
                )
            ''')
            
            # Table des détections (timecodes)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_id INTEGER,
                    timestamp REAL NOT NULL,
                    frame_path TEXT,
                    confidence REAL,
                    bounding_box TEXT,  -- JSON: {"x": 0, "y": 0, "width": 100, "height": 100}
                    metadata TEXT,      -- JSON pour données supplémentaires
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (scan_id) REFERENCES scans (id)
                )
            ''')
            
            # Index pour optimiser les requêtes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_videos_url ON videos (url)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON detections (timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_detections_scan_id ON detections (scan_id)')
            
            conn.commit()
            logger.info("Base de données initialisée")
    
    def add_video(self, video_info: Dict) -> int:
        """
        Ajoute une vidéo à la base de données
        
        Args:
            video_info: Dictionnaire contenant les informations de la vidéo
            
        Returns:
            ID de la vidéo ajoutée
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO videos 
                (url, title, duration, video_path, upload_date, uploader, 
                 view_count, description, processing_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                video_info.get('url'),
                video_info.get('title'),
                video_info.get('duration'),
                video_info.get('video_path'),
                video_info.get('upload_date'),
                video_info.get('uploader'),
                video_info.get('view_count'),
                video_info.get('description'),
                video_info.get('processing_date', datetime.now().isoformat())
            ))
            
            video_id = cursor.lastrowid
            conn.commit()
            
            logger.info(f"Vidéo ajoutée avec l'ID: {video_id}")
            return video_id
    
    def add_scan(self, video_id: int, target_object: str, model_used: str = None,
                 confidence_threshold: float = 0.5, interval_seconds: int = 5,
                 total_frames: int = 0) -> int:
        """
        Ajoute une session de scan à la base de données
        
        Args:
            video_id: ID de la vidéo
            target_object: Objet cible à détecter
            model_used: Modèle IA utilisé
            confidence_threshold: Seuil de confiance
            interval_seconds: Intervalle entre les frames
            total_frames: Nombre total de frames analysées
            
        Returns:
            ID du scan ajouté
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO scans 
                (video_id, target_object, model_used, confidence_threshold, 
                 interval_seconds, total_frames, detections_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                video_id, target_object, model_used, confidence_threshold,
                interval_seconds, total_frames, 0  # detections_count sera mis à jour
            ))
            
            scan_id = cursor.lastrowid
            conn.commit()
            
            logger.info(f"Scan ajouté avec l'ID: {scan_id}")
            return scan_id
    
    def add_detection(self, scan_id: int, timestamp: float, frame_path: str = None,
                     confidence: float = None, bounding_box: Dict = None,
                     metadata: Dict = None) -> int:
        """
        Ajoute une détection (timecode) à la base de données
        
        Args:
            scan_id: ID du scan
            timestamp: Timestamp en secondes
            frame_path: Chemin vers la frame
            confidence: Score de confiance
            bounding_box: Boîte englobante {"x": 0, "y": 0, "width": 100, "height": 100}
            metadata: Métadonnées supplémentaires
            
        Returns:
            ID de la détection ajoutée
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO detections 
                (scan_id, timestamp, frame_path, confidence, bounding_box, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                scan_id, timestamp, frame_path, confidence,
                json.dumps(bounding_box) if bounding_box else None,
                json.dumps(metadata) if metadata else None
            ))
            
            detection_id = cursor.lastrowid
            
            # Mettre à jour le compteur de détections du scan
            cursor.execute('''
                UPDATE scans 
                SET detections_count = (
                    SELECT COUNT(*) FROM detections WHERE scan_id = ?
                )
                WHERE id = ?
            ''', (scan_id, scan_id))
            
            conn.commit()
            
            logger.info(f"Détection ajoutée: timestamp={timestamp:.2f}s, confidence={confidence}")
            return detection_id
    
    def get_video_by_url(self, url: str) -> Optional[Dict]:
        """
        Récupère une vidéo par son URL
        
        Args:
            url: URL de la vidéo
            
        Returns:
            Dictionnaire contenant les informations de la vidéo ou None
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM videos WHERE url = ?', (url,))
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return None
    
    def get_scans_for_video(self, video_id: int) -> List[Dict]:
        """
        Récupère tous les scans pour une vidéo
        
        Args:
            video_id: ID de la vidéo
            
        Returns:
            Liste des scans
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM scans WHERE video_id = ? ORDER BY scan_date DESC', (video_id,))
            rows = cursor.fetchall()
            
            return [dict(row) for row in rows]
    
    def get_detections_for_scan(self, scan_id: int) -> List[Dict]:
        """
        Récupère toutes les détections pour un scan
        
        Args:
            scan_id: ID du scan
            
        Returns:
            Liste des détections avec parsing JSON
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM detections WHERE scan_id = ? ORDER BY timestamp', (scan_id,))
            rows = cursor.fetchall()
            
            detections = []
            for row in rows:
                detection = dict(row)
                
                # Parser les champs JSON
                if detection['bounding_box']:
                    detection['bounding_box'] = json.loads(detection['bounding_box'])
                if detection['metadata']:
                    detection['metadata'] = json.loads(detection['metadata'])
                
                detections.append(detection)
            
            return detections
    
    def get_timecodes_summary(self, scan_id: int) -> Dict:
        """
        Récupère un résumé des timecodes pour un scan
        
        Args:
            scan_id: ID du scan
            
        Returns:
            Dictionnaire avec le résumé
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Informations du scan
            cursor.execute('SELECT * FROM scans WHERE id = ?', (scan_id,))
            scan_info = cursor.fetchone()
            
            if not scan_info:
                return {}
            
            # Statistiques des détections
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_detections,
                    MIN(timestamp) as first_detection,
                    MAX(timestamp) as last_detection,
                    AVG(confidence) as avg_confidence,
                    MIN(confidence) as min_confidence,
                    MAX(confidence) as max_confidence
                FROM detections 
                WHERE scan_id = ?
            ''', (scan_id,))
            
            stats = cursor.fetchone()
            
            return {
                'scan_id': scan_id,
                'target_object': scan_info[2],  # target_object
                'model_used': scan_info[3],     # model_used
                'total_detections': stats[0] if stats[0] else 0,
                'first_detection': stats[1],
                'last_detection': stats[2],
                'avg_confidence': stats[3],
                'min_confidence': stats[4],
                'max_confidence': stats[5],
                'scan_date': scan_info[8]       # scan_date
            }
    
    def export_timecodes_csv(self, scan_id: int, output_path: str):
        """
        Exporte les timecodes en format CSV
        
        Args:
            scan_id: ID du scan
            output_path: Chemin du fichier CSV de sortie
        """
        import csv
        
        detections = self.get_detections_for_scan(scan_id)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['timestamp', 'time_formatted', 'confidence', 'frame_path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for detection in detections:
                # Formater le timestamp en HH:MM:SS
                timestamp = detection['timestamp']
                hours = int(timestamp // 3600)
                minutes = int((timestamp % 3600) // 60)
                seconds = int(timestamp % 60)
                time_formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                
                writer.writerow({
                    'timestamp': timestamp,
                    'time_formatted': time_formatted,
                    'confidence': detection['confidence'],
                    'frame_path': detection['frame_path']
                })
        
        logger.info(f"Timecodes exportés vers: {output_path}")
    
    def cleanup_old_data(self, days: int = 30):
        """
        Nettoie les données anciennes
        
        Args:
            days: Nombre de jours à conserver
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Supprimer les anciens scans et leurs détections
            cursor.execute('''
                DELETE FROM detections 
                WHERE scan_id IN (
                    SELECT id FROM scans 
                    WHERE scan_date < datetime('now', '-{} days')
                )
            '''.format(days))
            
            cursor.execute('''
                DELETE FROM scans 
                WHERE scan_date < datetime('now', '-{} days')
            '''.format(days))
            
            # Supprimer les vidéos sans scans
            cursor.execute('''
                DELETE FROM videos 
                WHERE id NOT IN (SELECT DISTINCT video_id FROM scans)
                AND created_at < datetime('now', '-{} days')
            '''.format(days))
            
            conn.commit()
            logger.info(f"Nettoyage effectué: données de plus de {days} jours supprimées")


def test_database():
    """
    Test de la base de données
    """
    # Créer une base de données de test
    db = TimecodeDatabase("test_timecodes.db")
    
    # Ajouter une vidéo de test
    video_info = {
        'url': 'https://www.youtube.com/watch?v=test123',
        'title': 'Vidéo de test',
        'duration': 300.0,
        'video_path': '/path/to/video.mp4',
        'upload_date': '20231201',
        'uploader': 'Test Channel',
        'view_count': 1000,
        'description': 'Description de test'
    }
    
    video_id = db.add_video(video_info)
    print(f"Vidéo ajoutée avec l'ID: {video_id}")
    
    # Ajouter un scan
    scan_id = db.add_scan(
        video_id=video_id,
        target_object='enfant',
        model_used='YOLO',
        confidence_threshold=0.7,
        interval_seconds=5,
        total_frames=60
    )
    print(f"Scan ajouté avec l'ID: {scan_id}")
    
    # Ajouter quelques détections
    detections_data = [
        (15.5, 0.85, {'x': 100, 'y': 50, 'width': 200, 'height': 300}),
        (45.2, 0.92, {'x': 150, 'y': 75, 'width': 180, 'height': 280}),
        (120.8, 0.78, {'x': 200, 'y': 100, 'width': 160, 'height': 250})
    ]
    
    for timestamp, confidence, bbox in detections_data:
        detection_id = db.add_detection(
            scan_id=scan_id,
            timestamp=timestamp,
            frame_path=f'/path/to/frame_{timestamp:.1f}s.jpg',
            confidence=confidence,
            bounding_box=bbox
        )
        print(f"Détection ajoutée: ID={detection_id}, timestamp={timestamp}s")
    
    # Tester les requêtes
    print("\\n--- Résumé des timecodes ---")
    summary = db.get_timecodes_summary(scan_id)
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\\n--- Détections ---")
    detections = db.get_detections_for_scan(scan_id)
    for detection in detections:
        print(f"Timestamp: {detection['timestamp']:.2f}s, Confiance: {detection['confidence']:.2f}")
    
    # Exporter en CSV
    db.export_timecodes_csv(scan_id, "test_timecodes.csv")
    print("\\nTimecodes exportés vers test_timecodes.csv")


if __name__ == "__main__":
    test_database()

