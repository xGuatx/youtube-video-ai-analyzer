#!/bin/bash

# Script de nettoyage pour YouTube Scanner
# Usage: ./clean.sh

echo "🧹 YouTube Video Scanner - Nettoyage"
echo "===================================="

read -p "⚠️  Voulez-vous supprimer TOUTES les données (vidéos, scans, cache) ? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🗑️  Suppression des conteneurs et volumes..."
    docker-compose down -v
    
    echo "🗑️  Suppression des données locales..."
    rm -rf data/ logs/
    
    echo "🗑️  Suppression des images Docker..."
    docker-compose down --rmi all
    
    echo "✅ Nettoyage terminé!"
    echo "💡 Pour redémarrer : ./start.sh"
else
    echo "❌ Nettoyage annulé."
fi

