#!/bin/bash

# Script de démarrage rapide pour YouTube Scanner
# Usage: ./start.sh

echo "🎥 YouTube Video Scanner - Démarrage"
echo "=================================="

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
mkdir -p data/videos data/frames data/cache data/scanner_data logs

# Construire et démarrer les conteneurs
echo "🔨 Construction et démarrage des conteneurs..."
docker-compose up -d --build

# Attendre que l'application soit prête
echo "⏳ Attente du démarrage de l'application..."
sleep 10

# Vérifier que l'application fonctionne
if curl -s http://localhost:5000/api/scanner/health > /dev/null; then
    echo "✅ Application démarrée avec succès!"
    echo ""
    echo "🌐 Interface web : http://localhost:5000"
    echo "📡 API REST : http://localhost:5000/api/scanner/"
    echo ""
    echo "📋 Commandes utiles :"
    echo "   docker-compose logs -f          # Voir les logs"
    echo "   docker-compose down             # Arrêter l'application"
    echo "   docker-compose restart          # Redémarrer"
    echo ""
    echo "🎯 L'application est prête à analyser vos vidéos YouTube!"
else
    echo "❌ Erreur lors du démarrage. Vérifiez les logs :"
    echo "   docker-compose logs youtube-scanner"
fi

