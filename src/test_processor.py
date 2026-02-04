"""
Module de test pour le processeur vidéo avec une vidéo locale
"""

import cv2
import numpy as np
from video_processor import YouTubeVideoProcessor
import os

def create_test_video():
    """
    Crée une vidéo de test simple pour tester l'extraction de frames
    """
    # Paramètres de la vidéo
    width, height = 640, 480
    fps = 30
    duration = 10  # 10 secondes
    total_frames = fps * duration
    
    # Créer le writer vidéo
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('test_video.mp4', fourcc, fps, (width, height))
    
    for i in range(total_frames):
        # Créer une frame avec un carré qui se déplace
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Couleur de fond qui change
        frame[:, :] = (i % 255, (i * 2) % 255, (i * 3) % 255)
        
        # Carré qui se déplace
        x = int((i / total_frames) * (width - 100))
        y = int(height / 2 - 50)
        cv2.rectangle(frame, (x, y), (x + 100, y + 100), (255, 255, 255), -1)
        
        # Texte avec le numéro de frame
        cv2.putText(frame, f'Frame {i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Vidéo de test créée: test_video.mp4 ({duration}s, {total_frames} frames)")

def test_video_processor():
    """
    Test le processeur vidéo avec une vidéo locale
    """
    # Créer une vidéo de test
    create_test_video()
    
    # Initialiser le processeur
    processor = YouTubeVideoProcessor()
    
    # Tester l'extraction de frames
    try:
        print("Test d'extraction de frames avec OpenCV...")
        frames_opencv = processor.extract_frames('test_video.mp4', interval_seconds=2)
        print(f"OpenCV: {len(frames_opencv)} frames extraites")
        
        print("\\nTest d'extraction de frames avec FFmpeg...")
        frames_ffmpeg = processor.extract_frames_ffmpeg('test_video.mp4', interval_seconds=2)
        print(f"FFmpeg: {len(frames_ffmpeg)} frames extraites")
        
        print("\\nInformations de la vidéo:")
        info = processor.get_video_info('test_video.mp4')
        for key, value in info.items():
            print(f"- {key}: {value}")
        
        print("\\nTest réussi!")
        
    except Exception as e:
        print(f"Erreur lors du test: {str(e)}")

if __name__ == "__main__":
    test_video_processor()

