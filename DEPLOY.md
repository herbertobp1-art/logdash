# Guia de Implantação Permanente - LogDash

Este pacote contém a plataforma **LogDash** completa, consolidada e pronta para ser hospedada de forma permanente em qualquer servidor (VPS, Heroku, Railway, etc.).

## 📋 Pré-requisitos
- Python 3.9 ou superior instalado.
- Acesso à internet (para sincronização com o Google Drive).

## 🚀 Como Iniciar (Rápido)
Basta executar o script de inicialização na raiz do projeto:
```bash
./start.sh
```
O site estará disponível em: `http://localhost:8000`

## 📁 Estrutura do Pacote
- `backend/`: Contém o servidor FastAPI e a lógica de processamento.
- `frontend_build/`: Contém a versão otimizada de produção do site (React).
- `start.sh`: Script para instalar dependências e rodar tudo automaticamente.

## ☁️ Configuração do Google Drive
O sistema está configurado para salvar dados no Google Drive. Para que isso funcione no seu servidor permanente:
1. Instale o **rclone** no seu servidor.
2. Configure um remote chamado `manus_google_drive`.
3. O sistema criará automaticamente a pasta `LogDash_Data` no seu Drive.

## 🔑 Credenciais Padrão
- **Usuário:** `batlog` | **Senha:** `123`
- **Usuário:** `editorajuspodivm` | **Senha:** `123`

## 🛠️ Hospedagem Permanente (Dicas)
Para manter o site online 24/7, recomendamos usar o **PM2** ou criar um serviço no **Systemd** do Linux para rodar o `python3 backend/server.py`.
