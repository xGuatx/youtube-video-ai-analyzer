# YouTube Video Scanner - Application Locale avec IA Prioritaire

## [TARGET] Description

Outil de scan video YouTube avec detection automatique d'objets utilisant **prioritairement l'IA locale**. L'application privilegie Ollama et les modeles locaux par rapport aux APIs externes, garantissant confidentialite et performance.

## [FEATURES] Fonctionnalites

- **[AI] IA Locale Prioritaire** : Ollama (llava, moondream) en premier choix
- **[SECURE] Confidentialite Maximale** : Traitement 100% local, aucune donnee envoyee vers des APIs externes
- **[FAST] Performance Optimisee** : YOLO local, OpenCV, traitement parallele
- **[TARGET] Detection Personnalisee** : Recherche de n'importe quel objet (enfant, personne, voiture, etc.)
- **[VIDEO] Videos Longues** : Optimise pour videos de 7+ heures
- **[WEB] Interface Moderne** : Application web responsive
- **[EXPORT] Export Complet** : Timecodes exportables en CSV

## [ARCH] Architecture IA Locale

### Ordre de Priorite des Detecteurs

1. **Ollama** (Priorite maximale)
   - `llava:7b` - Modele de vision multimodal
   - `moondream` - Modele de vision leger
   - `bakllava` - Alternative de vision

2. **YOLO Local**
   - `yolov8n` - Modele leger et rapide
   - `yolov8s` - Modele standard

3. **OpenCV Local**
   - Haar Cascades pour visages/personnes
   - DNN pour objets generaux

4. **APIs Externes** (Dernier recours uniquement)
   - Desactivees par defaut
   - Activables manuellement si necessaire

## [START] Installation et Demarrage

### Prerequis

- **Docker et Docker Compose** installes
- **Au moins 4GB de RAM** (8GB recommandes pour Ollama)
- **10GB d'espace disque** libre
- **Connexion internet** pour telechargement initial des modeles

### Demarrage Rapide - Mode IA Locale

```bash
# Demarrer avec IA locale (Ollama + modeles locaux)
./start-local.sh
```

### Demarrage Standard (sans Ollama)

```bash
# Demarrer sans Ollama (YOLO + OpenCV uniquement)
./start.sh
```

### Commandes de Gestion

```bash
# Voir les logs
docker-compose -f docker-compose.local.yml logs -f

# Arreter l'application
docker-compose -f docker-compose.local.yml down

# Redemarrer
docker-compose -f docker-compose.local.yml restart

# Nettoyer completement
./clean.sh
```

## [GUIDE] Guide d'utilisation

### 1. Interface Web Optimisee IA Locale

1. **Acceder** : http://localhost:5000
2. **URL Video** : Coller l'URL YouTube
3. **Objet Cible** : Specifier l'objet (ex: "person", "child", "car")
4. **Detecteur** : 
   - **Auto** (recommande) : Selection automatique du meilleur detecteur local
   - **Ollama** : Force l'utilisation d'Ollama si disponible
   - **YOLO** : Detection rapide locale
   - **OpenCV** : Detection classique
5. **Parametres** :
   - Seuil de confiance : 30% a 90%
   - Intervalle : 1 a 10 secondes
   - Limite de frames : optionnel

### 2. Verification des Modeles IA

```bash
# Verifier les modeles Ollama installes
curl http://localhost:11434/api/tags

# Installer un nouveau modele Ollama
docker-compose -f docker-compose.local.yml exec youtube-scanner-local ollama pull llava:13b

# Lister tous les detecteurs disponibles
curl http://localhost:5000/api/scanner/detectors
```

### 3. Interface Ollama (Optionnelle)

Si activee lors du demarrage :
- **Interface Ollama** : http://localhost:3000
- Permet de tester et gerer les modeles directement

## [CONFIG] Configuration IA Locale

### Variables d'Environnement

```yaml
environment:
  - PREFER_LOCAL_AI=true           # Priorite IA locale
  - OLLAMA_HOST=0.0.0.0:11434     # Serveur Ollama
  - OLLAMA_MODELS=/app/data/ollama # Stockage modeles
  - DISABLE_EXTERNAL_APIS=true     # Desactiver APIs externes
```

### Modeles Recommandes

```bash
# Modeles de vision legers (recommandes)
ollama pull llava:7b        # 4.5GB - Vision multimodale
ollama pull moondream       # 1.7GB - Vision ultra-legere

# Modeles plus puissants (si RAM suffisante)
ollama pull llava:13b       # 8GB - Vision haute qualite
ollama pull bakllava        # 4.5GB - Alternative vision
```

## [PERF] Optimisations Performance

### Configuration Systeme

```yaml
# docker-compose.local.yml
deploy:
  resources:
    limits:
      memory: 4G      # Limite memoire
    reservations:
      memory: 2G      # Memoire reservee
```

### Parametres Recommandes

| Duree Video | Intervalle | Modele | RAM Min |
|-------------|------------|--------|---------|
| < 30 min    | 1-3 sec    | llava  | 4GB     |
| 30min-2h    | 3-5 sec    | moondream | 4GB  |
| 2h-7h       | 5-10 sec   | yolo   | 2GB     |
| > 7h        | 10+ sec    | opencv | 1GB     |

## [FILES] Structure des Donnees

```
data/
|---- videos/          # Videos telechargees
|---- frames/          # Frames extraites
|---- cache/           # Cache des detections
|---- scanner_data/    # Base de donnees
\---- ollama/          # Modeles Ollama (persistants)
```

## [SECURITY] Confidentialite et Securite

### Avantages IA Locale

-  **Aucune donnee externe** : Tout traitement en local
-  **Pas de tracking** : Aucune telemetrie vers des services tiers
-  **Controle total** : Modeles et donnees sous votre controle
-  **Offline capable** : Fonctionne sans internet (apres installation)
-  **RGPD compliant** : Conformite automatique par design

### Mesures de Securite

- Validation stricte des URLs YouTube
- Chiffrement des donnees sensibles
- Isolation par conteneurs Docker
- Logs de securite detailles

## [EXPORT] Comparaison des Detecteurs

| Detecteur | Vitesse | Precision | RAM | Confidentialite |
|-----------|---------|-----------|-----|-----------------|
| Ollama llava |  |  | 4GB |  |
| Ollama moondream |  |  | 2GB |  |
| YOLO local |  |  | 1GB |  |
| OpenCV |  |  | 512MB |  |
| APIs externes |  |  | 0GB |  |

## [DEBUG] Depannage IA Locale

### Problemes Ollama

```bash
# Verifier le statut d'Ollama
curl http://localhost:11434/api/tags

# Redemarrer Ollama
docker-compose -f docker-compose.local.yml restart youtube-scanner-local

# Voir les logs Ollama
docker-compose -f docker-compose.local.yml logs youtube-scanner-local | grep ollama
```

### Problemes de Memoire

```bash
# Verifier l'utilisation memoire
docker stats

# Utiliser un modele plus leger
# Remplacer llava:7b par moondream dans la configuration
```

### Modeles Non Disponibles

```bash
# Telecharger manuellement un modele
docker-compose -f docker-compose.local.yml exec youtube-scanner-local ollama pull llava:7b

# Verifier l'espace disque
df -h
```

## [START] Modes de Deploiement

### Mode 1: IA Locale Complete (Recommande)
```bash
./start-local.sh
# Ollama + YOLO + OpenCV
# Confidentialite maximale
```

### Mode 2: IA Locale Legere
```bash
./start.sh
# YOLO + OpenCV uniquement
# Moins de RAM requise
```

### Mode 3: Developpement
```bash
docker-compose -f docker-compose.local.yml -f docker-compose.dev.yml up
# Mode developpement avec rechargement automatique
```

## [LICENSE] Licence et Ethique

- **Respect de la vie privee** : Aucune donnee n'est envoyee vers des services externes
- **Utilisation responsable** : Respecter les conditions d'utilisation de YouTube
- **Conformite legale** : Respecter les reglementations locales sur la protection des donnees

---

**[TARGET] Developpe pour privilegier la confidentialite et les processus locaux**

**[AI] Ollama + IA locale = Confidentialite maximale + Performance optimale**

