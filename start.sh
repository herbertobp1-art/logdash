#!/bin/bash

# LogDash - Script de Inicialização Unificada

echo "🚀 Iniciando LogDash Platform..."

# 1. Verificar dependências
if ! command -v python3 &> /dev/null; then
    echo "❌ Erro: Python3 não encontrado. Por favor, instale o Python3."
    exit 1
fi

# 2. Instalar dependências do backend
echo "📦 Instalando dependências..."
cd backend
pip install -r requirements.txt -q

# 3. Iniciar o servidor unificado
echo "🌐 Servidor rodando em http://localhost:8000"
python3 server.py
