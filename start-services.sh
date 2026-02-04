#!/bin/bash

# Script de démarrage pour les services locaux (Ollama + Flask)

echo "🚀 Démarrage des services locaux YouTube Scanner"
echo "================================================"

# Démarrer Ollama en arrière-plan
echo "📡 Démarrage d'Ollama..."
ollama serve &
OLLAMA_PID=$!

# Attendre qu'Ollama soit prêt
echo "⏳ Attente du démarrage d'Ollama..."
sleep 10

# Vérifier si Ollama fonctionne
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "✅ Ollama démarré avec succès"
    
    # Télécharger les modèles essentiels si pas déjà présents
    echo "📥 Vérification des modèles Ollama..."
    
    # Modèle de vision léger
    if ! ollama list | grep -q "llava"; then
        echo "📥 Téléchargement du modèle llava (vision)..."
        ollama pull llava:7b &
    fi
    
    # Modèle de vision ultra-léger
    if ! ollama list | grep -q "moondream"; then
        echo "📥 Téléchargement du modèle moondream (vision légère)..."
        ollama pull moondream &
    fi
    
else
    echo "⚠️  Ollama n'a pas pu démarrer, l'application fonctionnera avec les détecteurs locaux uniquement"
fi

# Démarrer Flask
echo "🌐 Démarrage de l'application Flask..."
python src/main.py &
FLASK_PID=$!

# Fonction de nettoyage
cleanup() {
    echo "🛑 Arrêt des services..."
    kill $FLASK_PID 2>/dev/null
    kill $OLLAMA_PID 2>/dev/null
    exit 0
}

# Capturer les signaux d'arrêt
trap cleanup SIGTERM SIGINT

# Attendre que les processus se terminent
wait

