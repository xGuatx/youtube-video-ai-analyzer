"""
Module de conformité RGPD et sécurité pour le scanner vidéo YouTube
Implémente les mesures de protection des données et de sécurité
"""

import os
import json
import hashlib
import secrets
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import sqlite3
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

logger = logging.getLogger(__name__)

class GDPRCompliance:
    """
    Gestionnaire de conformité RGPD
    """
    
    def __init__(self, db_path: str = "gdpr_compliance.db"):
        """
        Initialise le gestionnaire RGPD
        
        Args:
            db_path: Chemin vers la base de données de conformité
        """
        self.db_path = db_path
        self.init_compliance_db()
    
    def init_compliance_db(self):
        """
        Initialise la base de données de conformité RGPD
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Table des consentements
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS consents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    consent_type TEXT NOT NULL,
                    consent_given BOOLEAN NOT NULL,
                    consent_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ip_address TEXT,
                    user_agent TEXT,
                    purpose TEXT,
                    data_retention_days INTEGER DEFAULT 30,
                    withdrawn_date TIMESTAMP NULL
                )
            ''')
            
            # Table des traitements de données
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS data_processing (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    processing_type TEXT NOT NULL,
                    data_types TEXT,  -- JSON array
                    purpose TEXT NOT NULL,
                    legal_basis TEXT NOT NULL,
                    start_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    end_date TIMESTAMP NULL,
                    retention_period_days INTEGER,
                    data_location TEXT DEFAULT 'local'
                )
            ''')
            
            # Table des demandes d'accès/suppression
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS data_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    request_type TEXT NOT NULL,  -- 'access', 'deletion', 'portability', 'rectification'
                    request_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'pending',  -- 'pending', 'processing', 'completed', 'rejected'
                    completion_date TIMESTAMP NULL,
                    notes TEXT
                )
            ''')
            
            # Table des violations de données
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS data_breaches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    breach_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    description TEXT NOT NULL,
                    affected_data_types TEXT,  -- JSON array
                    affected_users_count INTEGER,
                    severity TEXT,  -- 'low', 'medium', 'high', 'critical'
                    reported_to_authority BOOLEAN DEFAULT FALSE,
                    users_notified BOOLEAN DEFAULT FALSE,
                    mitigation_actions TEXT
                )
            ''')
            
            conn.commit()
            logger.info("Base de données de conformité RGPD initialisée")
    
    def record_consent(self, user_id: str, consent_type: str, consent_given: bool,
                      purpose: str, ip_address: str = None, user_agent: str = None,
                      retention_days: int = 30) -> int:
        """
        Enregistre un consentement utilisateur
        
        Args:
            user_id: Identifiant utilisateur
            consent_type: Type de consentement
            consent_given: Consentement accordé ou non
            purpose: Finalité du traitement
            ip_address: Adresse IP
            user_agent: User agent
            retention_days: Durée de conservation
            
        Returns:
            ID du consentement enregistré
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO consents 
                (user_id, consent_type, consent_given, purpose, ip_address, user_agent, data_retention_days)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, consent_type, consent_given, purpose, ip_address, user_agent, retention_days))
            
            consent_id = cursor.lastrowid
            conn.commit()
            
            logger.info(f"Consentement enregistré: {consent_id} pour {user_id}")
            return consent_id
    
    def check_consent(self, user_id: str, consent_type: str) -> bool:
        """
        Vérifie si un utilisateur a donné son consentement
        
        Args:
            user_id: Identifiant utilisateur
            consent_type: Type de consentement
            
        Returns:
            True si le consentement est valide
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT consent_given, consent_date, data_retention_days, withdrawn_date
                FROM consents 
                WHERE user_id = ? AND consent_type = ? 
                ORDER BY consent_date DESC 
                LIMIT 1
            ''', (user_id, consent_type))
            
            result = cursor.fetchone()
            
            if not result:
                return False
            
            consent_given, consent_date, retention_days, withdrawn_date = result
            
            # Vérifier si le consentement a été retiré
            if withdrawn_date:
                return False
            
            # Vérifier si le consentement n'a pas expiré
            consent_datetime = datetime.fromisoformat(consent_date.replace('Z', '+00:00'))
            expiry_date = consent_datetime + timedelta(days=retention_days)
            
            if datetime.now() > expiry_date:
                return False
            
            return consent_given
    
    def withdraw_consent(self, user_id: str, consent_type: str) -> bool:
        """
        Retire un consentement
        
        Args:
            user_id: Identifiant utilisateur
            consent_type: Type de consentement
            
        Returns:
            True si le retrait a été effectué
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE consents 
                SET withdrawn_date = CURRENT_TIMESTAMP
                WHERE user_id = ? AND consent_type = ? AND withdrawn_date IS NULL
            ''', (user_id, consent_type))
            
            affected_rows = cursor.rowcount
            conn.commit()
            
            if affected_rows > 0:
                logger.info(f"Consentement retiré: {consent_type} pour {user_id}")
                return True
            
            return False
    
    def record_data_processing(self, user_id: str, processing_type: str,
                              data_types: List[str], purpose: str, legal_basis: str,
                              retention_days: int = 30) -> int:
        """
        Enregistre un traitement de données
        
        Args:
            user_id: Identifiant utilisateur
            processing_type: Type de traitement
            data_types: Types de données traitées
            purpose: Finalité du traitement
            legal_basis: Base légale
            retention_days: Durée de conservation
            
        Returns:
            ID du traitement enregistré
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO data_processing 
                (user_id, processing_type, data_types, purpose, legal_basis, retention_period_days)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, processing_type, json.dumps(data_types), purpose, legal_basis, retention_days))
            
            processing_id = cursor.lastrowid
            conn.commit()
            
            logger.info(f"Traitement de données enregistré: {processing_id}")
            return processing_id
    
    def handle_data_request(self, user_id: str, request_type: str) -> Dict:
        """
        Traite une demande d'accès/suppression de données
        
        Args:
            user_id: Identifiant utilisateur
            request_type: Type de demande
            
        Returns:
            Résultat de la demande
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Enregistrer la demande
            cursor.execute('''
                INSERT INTO data_requests (user_id, request_type)
                VALUES (?, ?)
            ''', (user_id, request_type))
            
            request_id = cursor.lastrowid
            
            if request_type == 'access':
                # Collecter toutes les données de l'utilisateur
                user_data = self._collect_user_data(user_id)
                
                cursor.execute('''
                    UPDATE data_requests 
                    SET status = 'completed', completion_date = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (request_id,))
                
                conn.commit()
                
                return {
                    'request_id': request_id,
                    'status': 'completed',
                    'data': user_data
                }
            
            elif request_type == 'deletion':
                # Supprimer toutes les données de l'utilisateur
                deleted_count = self._delete_user_data(user_id)
                
                cursor.execute('''
                    UPDATE data_requests 
                    SET status = 'completed', completion_date = CURRENT_TIMESTAMP,
                        notes = ?
                    WHERE id = ?
                ''', (f"Supprimé {deleted_count} enregistrements", request_id))
                
                conn.commit()
                
                return {
                    'request_id': request_id,
                    'status': 'completed',
                    'deleted_records': deleted_count
                }
            
            else:
                cursor.execute('''
                    UPDATE data_requests 
                    SET status = 'pending'
                    WHERE id = ?
                ''', (request_id,))
                
                conn.commit()
                
                return {
                    'request_id': request_id,
                    'status': 'pending',
                    'message': f"Demande {request_type} en cours de traitement"
                }
    
    def _collect_user_data(self, user_id: str) -> Dict:
        """
        Collecte toutes les données d'un utilisateur
        
        Args:
            user_id: Identifiant utilisateur
            
        Returns:
            Données de l'utilisateur
        """
        # Cette méthode devrait collecter les données depuis toutes les tables
        # Pour cette démo, on retourne un exemple
        return {
            'user_id': user_id,
            'consents': [],
            'processing_records': [],
            'scan_history': [],
            'export_date': datetime.now().isoformat()
        }
    
    def _delete_user_data(self, user_id: str) -> int:
        """
        Supprime toutes les données d'un utilisateur
        
        Args:
            user_id: Identifiant utilisateur
            
        Returns:
            Nombre d'enregistrements supprimés
        """
        # Cette méthode devrait supprimer les données depuis toutes les tables
        # Pour cette démo, on retourne un nombre fictif
        return 0

class DataEncryption:
    """
    Gestionnaire de chiffrement des données sensibles
    """
    
    def __init__(self, password: str = None):
        """
        Initialise le gestionnaire de chiffrement
        
        Args:
            password: Mot de passe pour la clé de chiffrement
        """
        if password is None:
            password = os.environ.get('ENCRYPTION_PASSWORD', 'default_password_change_me')
        
        self.key = self._derive_key(password)
        self.cipher = Fernet(self.key)
    
    def _derive_key(self, password: str) -> bytes:
        """
        Dérive une clé de chiffrement à partir d'un mot de passe
        
        Args:
            password: Mot de passe
            
        Returns:
            Clé de chiffrement
        """
        password_bytes = password.encode()
        salt = b'youtube_scanner_salt'  # En production, utiliser un salt aléatoire
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
        return key
    
    def encrypt_data(self, data: str) -> str:
        """
        Chiffre des données
        
        Args:
            data: Données à chiffrer
            
        Returns:
            Données chiffrées (base64)
        """
        encrypted_data = self.cipher.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted_data).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """
        Déchiffre des données
        
        Args:
            encrypted_data: Données chiffrées (base64)
            
        Returns:
            Données déchiffrées
        """
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
        decrypted_data = self.cipher.decrypt(encrypted_bytes)
        return decrypted_data.decode()
    
    def hash_data(self, data: str) -> str:
        """
        Hash des données (irréversible)
        
        Args:
            data: Données à hasher
            
        Returns:
            Hash des données
        """
        return hashlib.sha256(data.encode()).hexdigest()

class SecurityManager:
    """
    Gestionnaire de sécurité général
    """
    
    def __init__(self):
        """
        Initialise le gestionnaire de sécurité
        """
        self.gdpr = GDPRCompliance()
        self.encryption = DataEncryption()
        self.session_tokens = {}
    
    def generate_session_token(self, user_id: str) -> str:
        """
        Génère un token de session sécurisé
        
        Args:
            user_id: Identifiant utilisateur
            
        Returns:
            Token de session
        """
        token = secrets.token_urlsafe(32)
        expiry = datetime.now() + timedelta(hours=24)
        
        self.session_tokens[token] = {
            'user_id': user_id,
            'expiry': expiry,
            'created': datetime.now()
        }
        
        return token
    
    def validate_session_token(self, token: str) -> Optional[str]:
        """
        Valide un token de session
        
        Args:
            token: Token de session
            
        Returns:
            User ID si valide, None sinon
        """
        if token not in self.session_tokens:
            return None
        
        session_data = self.session_tokens[token]
        
        if datetime.now() > session_data['expiry']:
            del self.session_tokens[token]
            return None
        
        return session_data['user_id']
    
    def sanitize_input(self, user_input: str) -> str:
        """
        Nettoie les entrées utilisateur
        
        Args:
            user_input: Entrée utilisateur
            
        Returns:
            Entrée nettoyée
        """
        # Supprimer les caractères dangereux
        dangerous_chars = ['<', '>', '"', "'", '&', ';', '(', ')', '|', '`']
        
        sanitized = user_input
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        # Limiter la longueur
        return sanitized[:1000]
    
    def validate_youtube_url(self, url: str) -> bool:
        """
        Valide une URL YouTube
        
        Args:
            url: URL à valider
            
        Returns:
            True si l'URL est valide
        """
        import re
        
        youtube_patterns = [
            r'https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+',
            r'https?://(?:www\.)?youtu\.be/[\w-]+',
            r'https?://(?:www\.)?youtube\.com/embed/[\w-]+',
        ]
        
        for pattern in youtube_patterns:
            if re.match(pattern, url):
                return True
        
        return False
    
    def log_security_event(self, event_type: str, details: Dict):
        """
        Enregistre un événement de sécurité
        
        Args:
            event_type: Type d'événement
            details: Détails de l'événement
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'details': details
        }
        
        # En production, envoyer vers un système de logging sécurisé
        logger.warning(f"Événement de sécurité: {json.dumps(log_entry)}")

def test_gdpr_security():
    """
    Test des fonctionnalités RGPD et sécurité
    """
    print("=== Test RGPD et Sécurité ===")
    
    # Test RGPD
    gdpr = GDPRCompliance("test_gdpr.db")
    
    # Enregistrer un consentement
    user_id = "user123"
    consent_id = gdpr.record_consent(
        user_id=user_id,
        consent_type="video_analysis",
        consent_given=True,
        purpose="Analyse de vidéo YouTube pour détection d'objets",
        ip_address="192.168.1.1"
    )
    print(f"Consentement enregistré: {consent_id}")
    
    # Vérifier le consentement
    has_consent = gdpr.check_consent(user_id, "video_analysis")
    print(f"Consentement valide: {has_consent}")
    
    # Enregistrer un traitement
    processing_id = gdpr.record_data_processing(
        user_id=user_id,
        processing_type="video_scan",
        data_types=["video_frames", "detection_results"],
        purpose="Détection d'objets dans vidéo",
        legal_basis="consent"
    )
    print(f"Traitement enregistré: {processing_id}")
    
    # Test de chiffrement
    encryption = DataEncryption()
    
    sensitive_data = "Données sensibles à protéger"
    encrypted = encryption.encrypt_data(sensitive_data)
    print(f"Données chiffrées: {encrypted[:50]}...")
    
    decrypted = encryption.decrypt_data(encrypted)
    print(f"Données déchiffrées: {decrypted}")
    
    # Test de hash
    hashed = encryption.hash_data(sensitive_data)
    print(f"Hash: {hashed}")
    
    # Test du gestionnaire de sécurité
    security = SecurityManager()
    
    # Générer un token de session
    token = security.generate_session_token(user_id)
    print(f"Token de session: {token[:20]}...")
    
    # Valider le token
    validated_user = security.validate_session_token(token)
    print(f"Utilisateur validé: {validated_user}")
    
    # Test de validation d'URL
    test_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://malicious-site.com/fake",
        "not_a_url"
    ]
    
    for url in test_urls:
        is_valid = security.validate_youtube_url(url)
        print(f"URL '{url[:30]}...': {'Valide' if is_valid else 'Invalide'}")
    
    # Test de nettoyage d'entrée
    dangerous_input = "<script>alert('xss')</script>Hello World"
    sanitized = security.sanitize_input(dangerous_input)
    print(f"Entrée nettoyée: '{sanitized}'")
    
    # Test de demande d'accès aux données
    access_result = gdpr.handle_data_request(user_id, "access")
    print(f"Demande d'accès: {access_result['status']}")
    
    print("\\nTests RGPD et sécurité terminés!")

if __name__ == "__main__":
    test_gdpr_security()

