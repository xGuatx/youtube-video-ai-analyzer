#!/bin/bash

# Script de démarrage optimisé pour l'IA locale avec Ollama
# Usage: ./start-local.sh

echo "🎥 YouTube Video Scanner - Mode IA Locale"
echo "=========================================="

# Vérifier que Docker est installé
if ! command -v docker &> /dev/null; then
    echo "❌ Docker n'est pas installé. Veuillez installer Docker d'abord."
    echo "   https://docs.docker.com/get-docker/"
    exit 1
fi

# Vérifier que Docker Compose est installé
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose n'est pas installé. Veuillez installer Docker Compose d'abord."
    echo "   https://docs.docker.com/compose/install/"
    exit 1
fi

# Créer les répertoires de données
echo "📁 Création des répertoires de données..."
mkdir -p data/videos data/frames data/cache data/scanner_data data/ollama logs

# Vérifier les ressources système
echo "🔍 Vérification des ressources système..."
TOTAL_RAM=$(free -g | awk '/^Mem:/{print $2}')
if [ "$TOTAL_RAM" -lt 4 ]; then
    echo "⚠️  Attention: Moins de 4GB de RAM détectés. L'IA locale peut être lente."
    echo "   Recommandé: Au moins 4GB de RAM pour Ollama + modèles de vision"
fi

# Construire et démarrer avec la configuration locale
echo "🔨 Construction et démarrage avec IA locale (Ollama)..."
echo "   Cela peut prendre plusieurs minutes au premier démarrage..."
echo "   Les modèles IA seront téléchargés automatiquement."

docker-compose -f docker-compose.local.yml up -d --build

# Attendre que l'application soit prête
echo "⏳ Attente du démarrage des services..."
echo "   - Ollama (IA locale)"
echo "   - Flask (API + Interface web)"

# Attendre plus longtemps pour le premier démarrage
sleep 30

# Vérifier que l'application fonctionne
echo "🔍 Vérification des services..."

# Vérifier Flask
if curl -s http://localhost:5000/api/scanner/health > /dev/null; then
    echo "✅ Application Flask démarrée avec succès!"
else
    echo "⚠️  Application Flask en cours de démarrage..."
fi

# Vérifier Ollama
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "✅ Ollama (IA locale) démarré avec succès!"
    
    # Afficher les modèles disponibles
    echo "📋 Modèles IA disponibles:"
    curl -s http://localhost:11434/api/tags | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    models = data.get('models', [])
    if models:
        for model in models:
            print(f'   - {model[\"name\"]}')
    else:
        print('   Aucun modèle installé encore (téléchargement en cours...)')
except:
    print('   Vérification des modèles en cours...')
"
else
    echo "⚠️  Ollama en cours de démarrage..."
fi

echo ""
echo "🌐 Accès à l'application:"
echo "   Interface web : http://localhost:5000"
echo "   API REST      : http://localhost:5000/api/scanner/"
echo "   Ollama API    : http://localhost:11434"
echo ""
echo "📋 Commandes utiles :"
echo "   docker-compose -f docker-compose.local.yml logs -f    # Voir les logs"
echo "   docker-compose -f docker-compose.local.yml down       # Arrêter"
echo "   docker-compose -f docker-compose.local.yml restart    # Redémarrer"
echo ""
echo "🤖 Détecteurs IA prioritaires :"
echo "   1. Ollama (llava, moondream) - Vision IA locale"
echo "   2. YOLO local - Détection d'objets rapide"
echo "   3. OpenCV - Détection classique"
echo ""
echo "🎯 L'application privilégie automatiquement les processus locaux!"

# Optionnel: Démarrer l'interface Ollama WebUI
read -p "🖥️  Voulez-vous aussi démarrer l'interface web Ollama ? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Démarrage de l'interface Ollama WebUI..."
    docker-compose -f docker-compose.local.yml --profile webui up -d
    echo "✅ Interface Ollama disponible sur: http://localhost:3000"
fi

