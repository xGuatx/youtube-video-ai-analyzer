#!/bin/bash

# Script d'arrêt pour YouTube Scanner
# Usage: ./stop.sh

echo "🛑 YouTube Video Scanner - Arrêt"
echo "==============================="

# Arrêter les conteneurs
echo "⏹️  Arrêt des conteneurs..."
docker-compose down

echo "✅ Application arrêtée avec succès!"
echo ""
echo "💡 Pour redémarrer : ./start.sh"
echo "🗑️  Pour nettoyer les données : ./clean.sh"

