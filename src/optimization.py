"""
Module d'optimisation et de traitement parallèle pour les vidéos longues
Implémente des stratégies avancées pour traiter efficacement des vidéos de 7+ heures
"""

import multiprocessing as mp
import concurrent.futures
import threading
import queue
import time
import psutil
import os
import logging
from typing import List, Dict, Callable, Any, Optional
from pathlib import Path
import json
import hashlib

logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """
    Classe pour optimiser les performances du traitement vidéo
    """
    
    def __init__(self):
        """
        Initialise l'optimiseur de performances
        """
        self.cpu_count = mp.cpu_count()
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        logger.info(f"Optimiseur initialisé: {self.cpu_count} CPUs, {self.memory_gb:.1f}GB RAM")
    
    def get_optimal_workers(self, task_type: str = "cpu") -> int:
        """
        Détermine le nombre optimal de workers selon le type de tâche
        
        Args:
            task_type: Type de tâche ("cpu", "io", "mixed")
            
        Returns:
            Nombre optimal de workers
        """
        if task_type == "cpu":
            # Pour les tâches CPU-intensives (IA/ML)
            return max(1, self.cpu_count - 1)
        elif task_type == "io":
            # Pour les tâches I/O (téléchargement, lecture fichiers)
            return min(32, self.cpu_count * 4)
        else:  # mixed
            # Pour les tâches mixtes
            return max(2, self.cpu_count // 2)
    
    def get_optimal_batch_size(self, total_items: int, memory_per_item_mb: float = 10) -> int:
        """
        Calcule la taille optimale des lots selon la mémoire disponible
        
        Args:
            total_items: Nombre total d'éléments à traiter
            memory_per_item_mb: Mémoire estimée par élément en MB
            
        Returns:
            Taille optimale du lot
        """
        available_memory_mb = (self.memory_gb * 1024) * 0.7  # 70% de la RAM disponible
        max_batch_size = int(available_memory_mb / memory_per_item_mb)
        
        # Limiter entre 1 et 100 éléments par lot
        optimal_batch_size = max(1, min(100, max_batch_size))
        
        # Ajuster selon le nombre total d'éléments
        if total_items < optimal_batch_size:
            return total_items
        
        return optimal_batch_size
    
    def create_cache_key(self, data: Dict) -> str:
        """
        Crée une clé de cache unique pour les données
        
        Args:
            data: Données à hasher
            
        Returns:
            Clé de cache
        """
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def get_from_cache(self, cache_key: str) -> Optional[Any]:
        """
        Récupère des données du cache
        
        Args:
            cache_key: Clé de cache
            
        Returns:
            Données cachées ou None
        """
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Erreur lecture cache {cache_key}: {e}")
        
        return None
    
    def save_to_cache(self, cache_key: str, data: Any):
        """
        Sauvegarde des données dans le cache
        
        Args:
            cache_key: Clé de cache
            data: Données à sauvegarder
        """
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Erreur sauvegarde cache {cache_key}: {e}")

class ParallelFrameProcessor:
    """
    Processeur parallèle pour l'analyse de frames
    """
    
    def __init__(self, ai_engine, optimizer: PerformanceOptimizer = None):
        """
        Initialise le processeur parallèle
        
        Args:
            ai_engine: Moteur de reconnaissance IA
            optimizer: Optimiseur de performances
        """
        self.ai_engine = ai_engine
        self.optimizer = optimizer or PerformanceOptimizer()
        self.progress_queue = queue.Queue()
        self.results_queue = queue.Queue()
    
    def process_frame_batch(self, frame_paths: List[str], target_object: str,
                           detector_name: str, confidence_threshold: float) -> List[Dict]:
        """
        Traite un lot de frames
        
        Args:
            frame_paths: Liste des chemins vers les frames
            target_object: Objet à détecter
            detector_name: Nom du détecteur
            confidence_threshold: Seuil de confiance
            
        Returns:
            Liste des résultats de détection
        """
        results = []
        
        for frame_path in frame_paths:
            try:
                # Vérifier le cache
                cache_data = {
                    'frame_path': frame_path,
                    'target_object': target_object,
                    'detector_name': detector_name,
                    'confidence_threshold': confidence_threshold
                }
                cache_key = self.optimizer.create_cache_key(cache_data)
                cached_result = self.optimizer.get_from_cache(cache_key)
                
                if cached_result:
                    results.append({
                        'frame_path': frame_path,
                        'detections': cached_result,
                        'cached': True
                    })
                else:
                    # Traitement IA
                    detections = self.ai_engine.detect_in_frame(
                        frame_path, target_object, detector_name, confidence_threshold
                    )
                    
                    # Sauvegarder en cache
                    self.optimizer.save_to_cache(cache_key, detections)
                    
                    results.append({
                        'frame_path': frame_path,
                        'detections': detections,
                        'cached': False
                    })
                
                # Signaler le progrès
                self.progress_queue.put(1)
                
            except Exception as e:
                logger.error(f"Erreur traitement frame {frame_path}: {e}")
                results.append({
                    'frame_path': frame_path,
                    'detections': [],
                    'error': str(e)
                })
        
        return results
    
    def process_frames_parallel(self, frame_paths: List[str], target_object: str,
                               detector_name: str, confidence_threshold: float,
                               progress_callback: Callable = None) -> Dict[str, List[Dict]]:
        """
        Traite les frames en parallèle avec optimisations
        
        Args:
            frame_paths: Liste des chemins vers les frames
            target_object: Objet à détecter
            detector_name: Nom du détecteur
            confidence_threshold: Seuil de confiance
            progress_callback: Fonction de callback pour le progrès
            
        Returns:
            Dictionnaire des résultats {frame_path: [détections]}
        """
        total_frames = len(frame_paths)
        
        if total_frames == 0:
            return {}
        
        # Optimiser les paramètres
        num_workers = self.optimizer.get_optimal_workers("mixed")
        batch_size = self.optimizer.get_optimal_batch_size(total_frames, memory_per_item_mb=5)
        
        logger.info(f"Traitement parallèle: {total_frames} frames, {num_workers} workers, lots de {batch_size}")
        
        # Créer les lots
        batches = []
        for i in range(0, total_frames, batch_size):
            batch = frame_paths[i:i + batch_size]
            batches.append(batch)
        
        # Traitement parallèle
        results = {}
        processed_frames = 0
        
        # Thread pour surveiller le progrès
        def progress_monitor():
            nonlocal processed_frames
            while processed_frames < total_frames:
                try:
                    self.progress_queue.get(timeout=1)
                    processed_frames += 1
                    
                    if progress_callback:
                        progress_callback(processed_frames, total_frames)
                        
                except queue.Empty:
                    continue
        
        progress_thread = threading.Thread(target=progress_monitor)
        progress_thread.daemon = True
        progress_thread.start()
        
        # Traitement des lots en parallèle
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_batch = {
                executor.submit(
                    self.process_frame_batch,
                    batch, target_object, detector_name, confidence_threshold
                ): batch
                for batch in batches
            }
            
            for future in concurrent.futures.as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    
                    for result in batch_results:
                        frame_path = result['frame_path']
                        results[frame_path] = result['detections']
                        
                except Exception as e:
                    batch = future_to_batch[future]
                    logger.error(f"Erreur traitement lot {len(batch)} frames: {e}")
                    
                    # Ajouter des résultats vides pour ce lot
                    for frame_path in batch:
                        results[frame_path] = []
        
        progress_thread.join(timeout=1)
        
        return results

class SmartFrameSampler:
    """
    Échantillonneur intelligent de frames pour optimiser le traitement
    """
    
    def __init__(self):
        """
        Initialise l'échantillonneur
        """
        pass
    
    def detect_scene_changes(self, frame_paths: List[str], threshold: float = 0.3) -> List[int]:
        """
        Détecte les changements de scène pour optimiser l'échantillonnage
        
        Args:
            frame_paths: Liste des chemins vers les frames
            threshold: Seuil de détection des changements
            
        Returns:
            Liste des indices des frames avec changements de scène
        """
        import cv2
        import numpy as np
        
        scene_changes = [0]  # Toujours inclure la première frame
        
        if len(frame_paths) < 2:
            return scene_changes
        
        prev_hist = None
        
        for i, frame_path in enumerate(frame_paths):
            try:
                # Charger l'image
                img = cv2.imread(frame_path)
                if img is None:
                    continue
                
                # Calculer l'histogramme
                hist = cv2.calcHist([img], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                
                if prev_hist is not None:
                    # Calculer la corrélation
                    correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                    
                    # Si la corrélation est faible, c'est un changement de scène
                    if correlation < (1 - threshold):
                        scene_changes.append(i)
                
                prev_hist = hist
                
            except Exception as e:
                logger.warning(f"Erreur analyse frame {frame_path}: {e}")
                continue
        
        return scene_changes
    
    def adaptive_sampling(self, frame_paths: List[str], target_frames: int = 100) -> List[str]:
        """
        Échantillonnage adaptatif basé sur les changements de scène
        
        Args:
            frame_paths: Liste des chemins vers les frames
            target_frames: Nombre cible de frames à analyser
            
        Returns:
            Liste des frames sélectionnées
        """
        total_frames = len(frame_paths)
        
        if total_frames <= target_frames:
            return frame_paths
        
        # Détecter les changements de scène
        scene_changes = self.detect_scene_changes(frame_paths)
        
        # Si peu de changements de scène, échantillonnage uniforme
        if len(scene_changes) < target_frames // 4:
            step = total_frames // target_frames
            return [frame_paths[i] for i in range(0, total_frames, step)][:target_frames]
        
        # Échantillonnage basé sur les scènes
        selected_frames = []
        
        for i in range(len(scene_changes)):
            start_idx = scene_changes[i]
            end_idx = scene_changes[i + 1] if i + 1 < len(scene_changes) else total_frames
            
            # Nombre de frames à prendre dans cette scène
            scene_length = end_idx - start_idx
            frames_per_scene = max(1, (target_frames * scene_length) // total_frames)
            
            # Échantillonner dans la scène
            if scene_length <= frames_per_scene:
                selected_frames.extend(frame_paths[start_idx:end_idx])
            else:
                step = scene_length // frames_per_scene
                for j in range(frames_per_scene):
                    idx = start_idx + j * step
                    if idx < end_idx:
                        selected_frames.append(frame_paths[idx])
        
        # Limiter au nombre cible
        return selected_frames[:target_frames]

class ProgressTracker:
    """
    Suivi du progrès pour les opérations longues
    """
    
    def __init__(self, total_items: int, description: str = "Traitement"):
        """
        Initialise le tracker de progrès
        
        Args:
            total_items: Nombre total d'éléments
            description: Description de l'opération
        """
        self.total_items = total_items
        self.description = description
        self.processed_items = 0
        self.start_time = time.time()
        self.last_update = 0
    
    def update(self, increment: int = 1):
        """
        Met à jour le progrès
        
        Args:
            increment: Nombre d'éléments traités
        """
        self.processed_items += increment
        current_time = time.time()
        
        # Mettre à jour toutes les 2 secondes
        if current_time - self.last_update >= 2:
            self.last_update = current_time
            self._log_progress()
    
    def _log_progress(self):
        """
        Affiche le progrès
        """
        if self.total_items == 0:
            return
        
        percentage = (self.processed_items / self.total_items) * 100
        elapsed_time = time.time() - self.start_time
        
        if self.processed_items > 0:
            estimated_total_time = elapsed_time * (self.total_items / self.processed_items)
            remaining_time = estimated_total_time - elapsed_time
            
            logger.info(
                f"{self.description}: {percentage:.1f}% "
                f"({self.processed_items}/{self.total_items}) - "
                f"Temps restant: {remaining_time:.0f}s"
            )
        else:
            logger.info(f"{self.description}: {percentage:.1f}% ({self.processed_items}/{self.total_items})")
    
    def finish(self):
        """
        Termine le suivi
        """
        total_time = time.time() - self.start_time
        logger.info(f"{self.description} terminé en {total_time:.1f}s")

def test_optimizations():
    """
    Test des optimisations de performance
    """
    print("=== Test des optimisations de performance ===")
    
    # Test de l'optimiseur
    optimizer = PerformanceOptimizer()
    print(f"CPUs disponibles: {optimizer.cpu_count}")
    print(f"RAM disponible: {optimizer.memory_gb:.1f}GB")
    print(f"Workers optimaux (CPU): {optimizer.get_optimal_workers('cpu')}")
    print(f"Workers optimaux (I/O): {optimizer.get_optimal_workers('io')}")
    print(f"Taille de lot optimale (1000 items): {optimizer.get_optimal_batch_size(1000)}")
    
    # Test du cache
    test_data = {"test": "data", "number": 42}
    cache_key = optimizer.create_cache_key(test_data)
    print(f"Clé de cache: {cache_key}")
    
    optimizer.save_to_cache(cache_key, test_data)
    cached_data = optimizer.get_from_cache(cache_key)
    print(f"Données récupérées du cache: {cached_data}")
    
    # Test de l'échantillonneur
    sampler = SmartFrameSampler()
    
    # Créer des frames de test
    import cv2
    import numpy as np
    
    test_frames = []
    for i in range(20):
        # Créer une image de test
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        frame_path = f"test_frame_{i:03d}.jpg"
        cv2.imwrite(frame_path, img)
        test_frames.append(frame_path)
    
    # Test d'échantillonnage adaptatif
    sampled_frames = sampler.adaptive_sampling(test_frames, target_frames=10)
    print(f"Frames échantillonnées: {len(sampled_frames)}/{len(test_frames)}")
    
    # Nettoyer les fichiers de test
    for frame_path in test_frames:
        try:
            os.remove(frame_path)
        except:
            pass
    
    # Test du tracker de progrès
    print("\\nTest du tracker de progrès:")
    tracker = ProgressTracker(100, "Test de progression")
    
    for i in range(100):
        time.sleep(0.01)  # Simuler du travail
        tracker.update()
        
        if i % 20 == 0:  # Forcer l'affichage
            tracker._log_progress()
    
    tracker.finish()

if __name__ == "__main__":
    test_optimizations()

