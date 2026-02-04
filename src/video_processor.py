"""
Video processing module for YouTube videos
Uses yt-dlp for downloading and OpenCV/FFmpeg for frame extraction
Supports streaming mode for long videos
"""

import os
import subprocess
import cv2
import yt_dlp
from pathlib import Path
import json
import logging
import tempfile
from typing import Dict, List, Optional, Tuple, Generator, Callable
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YouTubeVideoProcessor:
    """
    Classe principale pour le téléchargement et l'extraction de frames de vidéos YouTube
    """
    
    def __init__(self, output_dir: str = "downloads", frames_dir: str = "frames"):
        """
        Initialise le processeur de vidéos YouTube
        
        Args:
            output_dir: Répertoire de sortie pour les vidéos téléchargées
            frames_dir: Répertoire de sortie pour les frames extraites
        """
        self.output_dir = Path(output_dir)
        self.frames_dir = Path(frames_dir)
        
        # Créer les répertoires s'ils n'existent pas
        self.output_dir.mkdir(exist_ok=True)
        self.frames_dir.mkdir(exist_ok=True)
        
        # Configuration yt-dlp
        self.ydl_opts = {
            'format': 'bestvideo[height<=720]+bestaudio/best[height<=720]/best',  # Flexible format selection
            'outtmpl': str(self.output_dir / '%(title)s.%(ext)s'),
            'writeinfojson': True,  # Sauvegarder les métadonnées
            'writesubtitles': False,  # Pas de sous-titres pour l'instant
            'writeautomaticsub': False,
        }
    
    def download_video(self, url: str) -> Dict:
        """
        Télécharge une vidéo YouTube
        
        Args:
            url: URL de la vidéo YouTube
            
        Returns:
            Dict contenant les informations de la vidéo téléchargée
        """
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                # Extraire les informations de la vidéo
                info = ydl.extract_info(url, download=False)
                
                # Vérifier la durée (limiter à 8 heures max pour éviter les abus)
                duration = info.get('duration', 0)
                if duration > 28800:  # 8 heures en secondes
                    raise ValueError(f"Vidéo trop longue: {duration/3600:.1f}h (max 8h)")
                
                logger.info(f"Téléchargement de: {info.get('title', 'Titre inconnu')}")
                logger.info(f"Durée: {duration//3600}h {(duration%3600)//60}m {duration%60}s")
                
                # Télécharger la vidéo
                ydl.download([url])
                
                # Construire le chemin du fichier téléchargé
                video_filename = ydl.prepare_filename(info)
                
                return {
                    'title': info.get('title', 'Titre inconnu'),
                    'duration': duration,
                    'video_path': video_filename,
                    'url': url,
                    'upload_date': info.get('upload_date'),
                    'uploader': info.get('uploader'),
                    'view_count': info.get('view_count'),
                    'description': info.get('description', '')[:500]  # Limiter la description
                }
                
        except Exception as e:
            logger.error(f"Erreur lors du téléchargement: {str(e)}")
            raise
    
    def extract_frames(self, video_path: str, interval_seconds: int = 5, 
                      max_frames: int = None) -> List[str]:
        """
        Extrait des frames d'une vidéo à intervalles réguliers
        
        Args:
            video_path: Chemin vers le fichier vidéo
            interval_seconds: Intervalle en secondes entre les frames
            max_frames: Nombre maximum de frames à extraire (None = pas de limite)
            
        Returns:
            Liste des chemins vers les frames extraites
        """
        try:
            # Ouvrir la vidéo avec OpenCV
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Impossible d'ouvrir la vidéo: {video_path}")
            
            # Obtenir les propriétés de la vidéo
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            logger.info(f"Vidéo: {fps:.2f} FPS, {total_frames} frames, {duration:.2f}s")
            
            # Calculer l'intervalle en frames
            frame_interval = int(fps * interval_seconds)
            
            # Créer un sous-répertoire pour cette vidéo
            video_name = Path(video_path).stem
            video_frames_dir = self.frames_dir / video_name
            video_frames_dir.mkdir(exist_ok=True)
            
            extracted_frames = []
            frame_count = 0
            extracted_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extraire une frame à l'intervalle spécifié
                if frame_count % frame_interval == 0:
                    if max_frames and extracted_count >= max_frames:
                        break
                    
                    # Calculer le timestamp
                    timestamp = frame_count / fps
                    
                    # Nom du fichier frame
                    frame_filename = f"frame_{extracted_count:06d}_{timestamp:.2f}s.jpg"
                    frame_path = video_frames_dir / frame_filename
                    
                    # Sauvegarder la frame
                    cv2.imwrite(str(frame_path), frame)
                    extracted_frames.append(str(frame_path))
                    extracted_count += 1
                    
                    if extracted_count % 100 == 0:
                        logger.info(f"Frames extraites: {extracted_count}")
                
                frame_count += 1
            
            cap.release()
            logger.info(f"Extraction terminée: {extracted_count} frames extraites")
            
            return extracted_frames
            
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction des frames: {str(e)}")
            raise
    
    def extract_frames_ffmpeg(self, video_path: str, interval_seconds: int = 5) -> List[str]:
        """
        Alternative: extraction de frames avec FFmpeg (plus rapide pour certains cas)
        
        Args:
            video_path: Chemin vers le fichier vidéo
            interval_seconds: Intervalle en secondes entre les frames
            
        Returns:
            Liste des chemins vers les frames extraites
        """
        try:
            # Créer un sous-répertoire pour cette vidéo
            video_name = Path(video_path).stem
            video_frames_dir = self.frames_dir / video_name
            video_frames_dir.mkdir(exist_ok=True)
            
            # Commande FFmpeg pour extraire les frames
            output_pattern = str(video_frames_dir / "frame_%06d.jpg")
            
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vf', f'fps=1/{interval_seconds}',  # Une frame toutes les N secondes
                '-y',  # Écraser les fichiers existants
                output_pattern
            ]
            
            logger.info(f"Exécution de FFmpeg: {' '.join(cmd)}")
            
            # Exécuter la commande
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"Erreur FFmpeg: {result.stderr}")
            
            # Lister les frames extraites
            extracted_frames = sorted(list(video_frames_dir.glob("frame_*.jpg")))
            extracted_frames = [str(f) for f in extracted_frames]
            
            logger.info(f"FFmpeg: {len(extracted_frames)} frames extraites")
            
            return extracted_frames
            
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction FFmpeg: {str(e)}")
            raise
    
    def get_video_info(self, video_path: str) -> Dict:
        """
        Obtient les informations d'une vidéo
        
        Args:
            video_path: Chemin vers le fichier vidéo
            
        Returns:
            Dict contenant les informations de la vidéo
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Impossible d'ouvrir la vidéo: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps
            
            cap.release()
            
            return {
                'fps': fps,
                'total_frames': total_frames,
                'width': width,
                'height': height,
                'duration': duration,
                'file_size': os.path.getsize(video_path)
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de l'obtention des infos vidéo: {str(e)}")
            raise
    
    def process_youtube_video(self, url: str, interval_seconds: int = 5, 
                            use_ffmpeg: bool = True, max_frames: int = None) -> Dict:
        """
        Traite complètement une vidéo YouTube: téléchargement + extraction de frames
        
        Args:
            url: URL de la vidéo YouTube
            interval_seconds: Intervalle en secondes entre les frames
            use_ffmpeg: Utiliser FFmpeg au lieu d'OpenCV pour l'extraction
            max_frames: Nombre maximum de frames à extraire
            
        Returns:
            Dict contenant toutes les informations du traitement
        """
        try:
            # Télécharger la vidéo
            video_info = self.download_video(url)
            video_path = video_info['video_path']
            
            # Obtenir les informations techniques de la vidéo
            tech_info = self.get_video_info(video_path)
            
            # Extraire les frames
            if use_ffmpeg:
                frames = self.extract_frames_ffmpeg(video_path, interval_seconds)
            else:
                frames = self.extract_frames(video_path, interval_seconds, max_frames)
            
            # Combiner toutes les informations
            result = {
                **video_info,
                **tech_info,
                'frames': frames,
                'frame_count': len(frames),
                'interval_seconds': interval_seconds,
                'processing_date': datetime.now().isoformat()
            }
            
            # Sauvegarder les métadonnées
            metadata_path = self.output_dir / f"{Path(video_path).stem}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Traitement terminé: {len(frames)} frames extraites")
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement de la vidéo: {str(e)}")
            raise

    def stream_frames_from_url(self, url: str, interval_seconds: int = 5,
                               max_frames: int = None,
                               callback: Callable = None) -> Generator[Tuple[str, float], None, None]:
        """
        Stream frames from YouTube URL without downloading entire video.
        Processes frames on-the-fly for long videos.

        Args:
            url: YouTube URL
            interval_seconds: Interval between frames
            max_frames: Maximum frames to process
            callback: Optional callback(frame_path, timestamp) for each frame

        Yields:
            Tuple of (frame_path, timestamp)
        """
        try:
            # Get video info first
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                duration = info.get('duration', 0)
                title = info.get('title', 'video')
                video_url = info.get('url') or info.get('formats', [{}])[-1].get('url')

            logger.info(f"Streaming: {title} ({duration}s)")

            # Create temp directory for frames
            video_name = "".join(c for c in title if c.isalnum() or c in " -_")[:50]
            frames_dir = self.frames_dir / video_name
            frames_dir.mkdir(exist_ok=True)

            # Use ffmpeg to stream and extract frames directly from URL
            # This avoids downloading the entire video
            stream_url = self._get_stream_url(url)
            if not stream_url:
                logger.warning("Could not get stream URL, falling back to download")
                yield from self._fallback_extract(url, interval_seconds, max_frames, callback)
                return

            cmd = [
                'ffmpeg',
                '-i', stream_url,
                '-vf', f'fps=1/{interval_seconds}',
                '-f', 'image2pipe',
                '-vcodec', 'mjpeg',
                '-'
            ]

            frame_count = 0
            timestamp = 0.0

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8
            )

            buffer = b''
            while True:
                if max_frames and frame_count >= max_frames:
                    break

                chunk = process.stdout.read(4096)
                if not chunk:
                    break

                buffer += chunk

                # Look for JPEG markers
                while True:
                    start = buffer.find(b'\xff\xd8')
                    end = buffer.find(b'\xff\xd9')

                    if start != -1 and end != -1 and end > start:
                        # Extract complete JPEG
                        jpeg_data = buffer[start:end+2]
                        buffer = buffer[end+2:]

                        # Save frame
                        frame_path = str(frames_dir / f"frame_{frame_count:06d}_{timestamp:.2f}s.jpg")
                        with open(frame_path, 'wb') as f:
                            f.write(jpeg_data)

                        if callback:
                            callback(frame_path, timestamp)

                        yield (frame_path, timestamp)

                        frame_count += 1
                        timestamp += interval_seconds

                        if frame_count % 10 == 0:
                            logger.info(f"Streamed {frame_count} frames...")
                    else:
                        break

            process.terminate()
            logger.info(f"Streaming complete: {frame_count} frames")

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise

    def _get_stream_url(self, url: str) -> Optional[str]:
        """Get direct stream URL from YouTube"""
        try:
            opts = {
                'format': 'best[height<=480]',
                'quiet': True,
            }
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return info.get('url')
        except Exception as e:
            logger.error(f"Could not get stream URL: {e}")
            return None

    def _fallback_extract(self, url: str, interval_seconds: int,
                         max_frames: int, callback: Callable) -> Generator:
        """Fallback to download mode if streaming fails"""
        video_info = self.download_video(url)
        frames = self.extract_frames(
            video_info['video_path'],
            interval_seconds=interval_seconds,
            max_frames=max_frames
        )
        for i, frame_path in enumerate(frames):
            timestamp = i * interval_seconds
            if callback:
                callback(frame_path, timestamp)
            yield (frame_path, timestamp)


def main():
    """
    Test function
    """
    processor = YouTubeVideoProcessor()
    
    # URL de test (vidéo courte)
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Roll (3:32)
    
    try:
        result = processor.process_youtube_video(
            url=test_url,
            interval_seconds=10,  # Une frame toutes les 10 secondes
            use_ffmpeg=True,
            max_frames=50  # Limiter pour le test
        )
        
        print(f"Traitement réussi:")
        print(f"- Titre: {result['title']}")
        print(f"- Durée: {result['duration']:.2f}s")
        print(f"- Frames extraites: {result['frame_count']}")
        print(f"- Chemin vidéo: {result['video_path']}")
        
    except Exception as e:
        print(f"Erreur: {str(e)}")


if __name__ == "__main__":
    main()

