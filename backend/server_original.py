from fastapi import FastAPI, APIRouter, HTTPException, Depends, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime, timezone, timedelta
import jwt
import pandas as pd
import io
from collections import defaultdict

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# JWT Configuration
SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 hours

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

security = HTTPBearer()

# Mock users database
USERS_DB = {
    "batlog": "123",
    "editorajuspodivm": "123"
}

# Pydantic Models
class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    token: str
    username: str

class AnalysisResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    upload_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_records: int
    date_range: Dict[str, str]
    analysis_data: Dict[str, Any]

class AnalysisResultCreate(BaseModel):
    filename: str
    total_records: int
    date_range: Dict[str, str]
    analysis_data: Dict[str, Any]

# JWT Helper Functions
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Data Processing Functions
def process_excel_data(file_content: bytes, filename: str) -> Dict[str, Any]:
    """Process Excel file and generate all 8 required analyses"""
    try:
        # Read Excel file
        df = pd.read_excel(io.BytesIO(file_content))
        
        # Normalize column names to handle encoding issues
        df.columns = df.columns.str.replace('Ã§Ã£', 'ção').str.replace('Ã¡', 'á').str.replace('Ã©', 'é').str.replace('Ãº', 'ú')
        
        # Find situation/status column
        situacao_col = None
        for col in ['Situação', 'Situacao', 'Status', 'Status Transportador']:
            if col in df.columns:
                situacao_col = col
                break
        
        # Normalize status values
        if situacao_col:
            # Create normalized 'Situação' column
            df['Situação'] = df[situacao_col].copy()
            
            # Normalize: OnTime -> No prazo, Atraso -> Atrasado, etc.
            df['Situação'] = df['Situação'].replace({
                'OnTime': 'No prazo',
                'Atraso': 'Atrasado',
                'Entregue no Prazo': 'No prazo',
                'Entregue em Atraso': 'Atrasado',
                'Pendente': 'Pendente',
                'Em Trânsito': 'Pendente',
                'Em Transito': 'Pendente',
                'Aguardando Coleta': 'Pendente',
                'Em Separação': 'Pendente',
                'Saiu para Entrega': 'Pendente'
            })
        else:
            # If no status column, try to infer from 'Data Entrega'
            if 'Data Entrega' in df.columns:
                df['Data Entrega'] = pd.to_datetime(df['Data Entrega'], errors='coerce')
                df['Situação'] = df['Data Entrega'].apply(lambda x: 'No prazo' if pd.notna(x) else 'Pendente')
            else:
                df['Situação'] = 'No prazo'  # Default if no info available
        
        # Convert date columns
        if 'Data Despacho' in df.columns:
            df['Data Despacho'] = pd.to_datetime(df['Data Despacho'], errors='coerce')
            df['Mes'] = df['Data Despacho'].dt.strftime('%b').str.lower()
            df['Ano'] = df['Data Despacho'].dt.year
            df['Mes_Ano'] = df['Data Despacho'].dt.strftime('%b/%y')
        
        # Calculate date range
        date_range = {
            "start": df['Data Despacho'].min().strftime('%Y-%m-%d') if pd.notna(df['Data Despacho'].min()) else "N/A",
            "end": df['Data Despacho'].max().strftime('%Y-%m-%d') if pd.notna(df['Data Despacho'].max()) else "N/A"
        }
        
        # 1. Total de pedidos por Região e Mês
        analise_1 = generate_pedidos_regiao_mes(df)
        
        # 2. Total de pedidos por Mês
        analise_2 = generate_pedidos_mes(df)
        
        # 3. % de pedidos por Transportadora e Mês
        analise_3 = generate_percentual_transportadora_mes(df)
        
        # 4. Participação percentual por transportadora
        analise_4 = generate_participacao_transportadora(df)
        
        # 5. Pedidos Atrasados x No Prazo por Transportadora
        analise_5 = generate_atrasados_no_prazo(df)
        
        # 6. SLA por Transportadora
        analise_6 = generate_sla_transportadora(df)
        
        # 7. SLA por Transportadora e Região
        analise_7 = generate_sla_transportadora_regiao(df)
        
        # 8. SLA por Região
        analise_8 = generate_sla_regiao(df)
        
        # KPIs principais
        kpis = generate_kpis(df)
        
        # 9. Indicadores Financeiros
        indicadores_financeiros = generate_indicadores_financeiros(df)
        
        # 10. Evolução Temporal do On Time Delivery
        evolucao_on_time = generate_evolucao_on_time(df)
        
        # 11. Insights Inteligentes
        insights = generate_insights(df, kpis, analise_5, analise_6, analise_8, indicadores_financeiros)
        
        return {
            "total_records": len(df),
            "date_range": date_range,
            "kpis": kpis,
            "analise_1_regiao_mes": analise_1,
            "analise_2_pedidos_mes": analise_2,
            "analise_3_percentual_transportadora_mes": analise_3,
            "analise_4_participacao_transportadora": analise_4,
            "analise_5_atrasados_no_prazo": analise_5,
            "analise_6_sla_transportadora": analise_6,
            "analise_7_sla_transportadora_regiao": analise_7,
            "analise_8_sla_regiao": analise_8,
            "indicadores_financeiros": indicadores_financeiros,
            "evolucao_on_time": evolucao_on_time,
            "insights": insights
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

def generate_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate main KPIs"""
    total_pedidos = len(df)
    
    # Performance SLA - normalize situation values
    if 'Situação' in df.columns:
        no_prazo = len(df[df['Situação'] == 'No prazo'])
        atrasado = len(df[df['Situação'] == 'Atrasado'])
        pendente = len(df[df['Situação'] == 'Pendente'])
        sla_geral = (no_prazo / total_pedidos * 100) if total_pedidos > 0 else 0
    else:
        no_prazo = 0
        atrasado = 0
        pendente = 0
        sla_geral = 0
    
    # Entregas (considerando entregues = no prazo + atrasado)
    entregas_concluidas = no_prazo + atrasado
    
    # Regi\u00f5es
    regioes_atendidas = df['Região'].nunique() if 'Região' in df.columns else 0
    
    # Transportadoras
    transportadoras = df['Transportadora'].nunique() if 'Transportadora' in df.columns else 0
    
    return {
        "total_pedidos": total_pedidos,
        "pedidos_no_prazo": no_prazo,
        "pedidos_atrasados": atrasado,
        "pedidos_pendentes": pendente,
        "sla_geral": round(sla_geral, 2),
        "entregas_concluidas": entregas_concluidas,
        "regioes_atendidas": regioes_atendidas,
        "transportadoras_ativas": transportadoras
    }
    
    # Entregas
    if 'Status Transportador' in df.columns:
        entregues = len(df[df['Status Transportador'] == 'Entregue'])
    else:
        entregues = 0
    
    # Regiões
    regioes_atendidas = df['Região'].nunique() if 'Região' in df.columns else 0
    
    # Transportadoras
    transportadoras = df['Transportadora'].nunique() if 'Transportadora' in df.columns else 0
    
    return {
        "total_pedidos": total_pedidos,
        "pedidos_no_prazo": no_prazo,
        "pedidos_atrasados": atrasado,
        "sla_geral": round(sla_geral, 2),
        "entregas_concluidas": entregues,
        "regioes_atendidas": regioes_atendidas,
        "transportadoras_ativas": transportadoras
    }

def generate_pedidos_regiao_mes(df: pd.DataFrame) -> Dict[str, Any]:
    """1. Total de pedidos por Região e Mês"""
    if 'Região' not in df.columns or 'Mes_Ano' not in df.columns:
        return {"data": [], "meses": []}
    
    pivot = df.pivot_table(
        index='Região',
        columns='Mes_Ano',
        values='Pedido',
        aggfunc='count',
        fill_value=0
    )
    
    # Calculate totals
    pivot['Total'] = pivot.sum(axis=1)
    pivot.loc['Total'] = pivot.sum(axis=0)
    
    result = []
    for regiao in pivot.index:
        row = {"regiao": str(regiao)}
        for mes in pivot.columns:
            row[str(mes)] = int(float(pivot.loc[regiao, mes]))
        result.append(row)
    
    return {
        "data": result,
        "meses": [str(col) for col in pivot.columns]
    }

def generate_pedidos_mes(df: pd.DataFrame) -> Dict[str, Any]:
    """2. Total de pedidos por Mês"""
    if 'Mes_Ano' not in df.columns:
        return {"data": []}
    
    monthly = df.groupby('Mes_Ano')['Pedido'].count().reset_index()
    monthly.columns = ['mes', 'total']
    monthly['total'] = monthly['total'].astype(int)
    
    total_geral = int(monthly['total'].sum())
    monthly_dict = monthly.to_dict('records')
    monthly_dict.append({"mes": "Total Geral", "total": total_geral})
    
    return {"data": monthly_dict}

def generate_percentual_transportadora_mes(df: pd.DataFrame) -> Dict[str, Any]:
    """3. % de pedidos por Transportadora e Mês"""
    if 'Transportadora' not in df.columns or 'Mes_Ano' not in df.columns:
        return {"data": [], "meses": []}
    
    pivot = df.pivot_table(
        index='Transportadora',
        columns='Mes_Ano',
        values='Pedido',
        aggfunc='count',
        fill_value=0
    )
    
    total_por_mes = pivot.sum(axis=0)
    total_geral = float(pivot.sum().sum())
    
    result = []
    for transp in pivot.index:
        row = {
            "transportadora": str(transp),
            "total_geral": int(float(pivot.loc[transp].sum()))
        }
        
        for mes in pivot.columns:
            valor = int(float(pivot.loc[transp, mes]))
            pct_mes = (valor / float(total_por_mes[mes]) * 100) if float(total_por_mes[mes]) > 0 else 0
            row[f"{mes}"] = valor
            row[f"{mes}_pct"] = round(float(pct_mes), 2)
        
        pct_total = (row['total_geral'] / total_geral * 100) if total_geral > 0 else 0
        row["total_geral_pct"] = round(float(pct_total), 2)
        result.append(row)
    
    return {
        "data": result,
        "meses": [str(col) for col in pivot.columns]
    }

def generate_participacao_transportadora(df: pd.DataFrame) -> Dict[str, Any]:
    """4. Participação percentual por transportadora"""
    if 'Transportadora' not in df.columns or 'Mes_Ano' not in df.columns:
        return {"data": [], "meses": []}
    
    pivot = df.pivot_table(
        index='Transportadora',
        columns='Mes_Ano',
        values='Pedido',
        aggfunc='count',
        fill_value=0
    )
    
    total_por_mes = pivot.sum(axis=0)
    total_geral = float(pivot.sum().sum())
    
    result = []
    for transp in pivot.index:
        row = {"transportadora": str(transp)}
        
        for mes in pivot.columns:
            pct = (float(pivot.loc[transp, mes]) / float(total_por_mes[mes]) * 100) if float(total_por_mes[mes]) > 0 else 0
            row[f"pct_{mes}"] = round(float(pct), 2)
        
        total = float(pivot.loc[transp].sum())
        pct_total = (total / total_geral * 100) if total_geral > 0 else 0
        row["pct_total"] = round(float(pct_total), 2)
        result.append(row)
    
    return {
        "data": result,
        "meses": [str(col) for col in pivot.columns]
    }

def generate_atrasados_no_prazo(df: pd.DataFrame) -> Dict[str, Any]:
    """5. Pedidos Atrasados x No Prazo por Transportadora"""
    if 'Transportadora' not in df.columns or 'Situação' not in df.columns:
        return {"data": []}
    
    pivot = df.pivot_table(
        index='Transportadora',
        columns='Situação',
        values='Pedido',
        aggfunc='count',
        fill_value=0
    )
    
    result = []
    for transp in pivot.index:
        atrasado = int(float(pivot.loc[transp].get('Atrasado', 0)))
        no_prazo = int(float(pivot.loc[transp].get('No prazo', 0)))
        total = atrasado + no_prazo
        
        pct_atrasado = (atrasado / total * 100) if total > 0 else 0
        pct_no_prazo = (no_prazo / total * 100) if total > 0 else 0
        
        result.append({
            "transportadora": str(transp),
            "atrasado": atrasado,
            "no_prazo": no_prazo,
            "total": total,
            "pct_atrasado": round(float(pct_atrasado), 2),
            "pct_no_prazo": round(float(pct_no_prazo), 2)
        })
    
    return {"data": result}

def generate_sla_transportadora(df: pd.DataFrame) -> Dict[str, Any]:
    """6. SLA por Transportadora"""
    if 'Transportadora' not in df.columns or 'Situação' not in df.columns:
        return {"data": []}
    
    result = []
    for transp in df['Transportadora'].unique():
        df_transp = df[df['Transportadora'] == transp]
        no_prazo = int(len(df_transp[df_transp['Situação'] == 'No prazo']))
        total = int(len(df_transp))
        sla = (no_prazo / total * 100) if total > 0 else 0
        
        result.append({
            "transportadora": str(transp),
            "no_prazo": no_prazo,
            "total": total,
            "sla": round(float(sla), 2)
        })
    
    # Sort by SLA descending
    result.sort(key=lambda x: x['sla'], reverse=True)
    
    return {"data": result}

def generate_sla_transportadora_regiao(df: pd.DataFrame) -> Dict[str, Any]:
    """7. SLA por Transportadora e Região"""
    if 'Transportadora' not in df.columns or 'Região' not in df.columns or 'Situação' not in df.columns:
        return {"data": []}
    
    result = []
    grouped = df.groupby(['Transportadora', 'Região'])
    
    for (transp, regiao), group in grouped:
        atrasado = int(len(group[group['Situação'] == 'Atrasado']))
        no_prazo = int(len(group[group['Situação'] == 'No prazo']))
        total = int(len(group))
        sla = (no_prazo / total * 100) if total > 0 else 0
        
        # Determine color based on SLA
        if sla >= 95:
            color = "green"
        elif sla >= 90:
            color = "yellow"
        else:
            color = "red"
        
        result.append({
            "transportadora": str(transp),
            "regiao": str(regiao),
            "atrasado": atrasado,
            "no_prazo": no_prazo,
            "total": total,
            "sla": round(float(sla), 2),
            "color": color
        })
    
    # Sort by transportadora and SLA
    result.sort(key=lambda x: (x['transportadora'], -x['sla']))
    
    return {"data": result}

def generate_sla_regiao(df: pd.DataFrame) -> Dict[str, Any]:
    """8. SLA por Região"""
    if 'Região' not in df.columns or 'Situação' not in df.columns:
        return {"data": []}
    
    result = []
    for regiao in df['Região'].unique():
        df_regiao = df[df['Região'] == regiao]
        atrasado = int(len(df_regiao[df_regiao['Situação'] == 'Atrasado']))
        no_prazo = int(len(df_regiao[df_regiao['Situação'] == 'No prazo']))
        total = int(len(df_regiao))
        sla = (no_prazo / total * 100) if total > 0 else 0
        
        result.append({
            "regiao": str(regiao),
            "atrasado": atrasado,
            "no_prazo": no_prazo,
            "total": total,
            "sla": round(float(sla), 2)
        })
    
    # Add total
    total_atrasado = sum(r['atrasado'] for r in result)
    total_no_prazo = sum(r['no_prazo'] for r in result)
    total_geral = sum(r['total'] for r in result)
    sla_geral = (total_no_prazo / total_geral * 100) if total_geral > 0 else 0
    
    result.append({
        "regiao": "Total Geral",
        "atrasado": total_atrasado,
        "no_prazo": total_no_prazo,
        "total": total_geral,
        "sla": round(float(sla_geral), 2)
    })
    
    return {"data": result}

def generate_indicadores_financeiros(df: pd.DataFrame) -> Dict[str, Any]:
    """9. Indicadores Financeiros"""
    
    # Verificar se as colunas necessárias existem
    custo_frete_col = None
    for col in ['Custo Frete', 'Custo de Frete', 'Valor do Frete', 'Custo']:
        if col in df.columns:
            custo_frete_col = col
            break
    
    if custo_frete_col is None or 'Transportadora' not in df.columns:
        return {
            "kpis": {
                "custo_total_frete": 0,
                "custo_medio_pedido": 0,
                "numero_pedidos": 0
            },
            "custo_por_transportadora": {"data": []},
            "representatividade_custos": {"data": []},
            "resumo_financeiro": {
                "custo_total": 0,
                "custo_medio": 0,
                "total_pedidos": 0,
                "numero_transportadoras": 0
            }
        }
    
    # Limpar e converter valores de frete
    df_financeiro = df.copy()
    df_financeiro[custo_frete_col] = pd.to_numeric(df_financeiro[custo_frete_col], errors='coerce').fillna(0)
    
    # KPIs Financeiros
    custo_total = float(df_financeiro[custo_frete_col].sum())
    numero_pedidos = int(df_financeiro['Pedido'].nunique())
    custo_medio = (custo_total / numero_pedidos) if numero_pedidos > 0 else 0
    
    # Custo por Transportadora
    custo_transportadora = []
    for transp in df_financeiro['Transportadora'].unique():
        df_transp = df_financeiro[df_financeiro['Transportadora'] == transp]
        custo_transp = float(df_transp[custo_frete_col].sum())
        pedidos_transp = int(df_transp['Pedido'].nunique())
        custo_medio_transp = (custo_transp / pedidos_transp) if pedidos_transp > 0 else 0
        
        custo_transportadora.append({
            "transportadora": str(transp),
            "custo_total": round(custo_transp, 2),
            "custo_medio": round(custo_medio_transp, 2),
            "numero_pedidos": pedidos_transp
        })
    
    # Ordenar por custo médio decrescente
    custo_transportadora.sort(key=lambda x: x['custo_medio'], reverse=True)
    
    # Representatividade de Custos
    representatividade = []
    for item in custo_transportadora:
        percentual = (item['custo_total'] / custo_total * 100) if custo_total > 0 else 0
        representatividade.append({
            "transportadora": item['transportadora'],
            "custo_total": item['custo_total'],
            "percentual": round(float(percentual), 2)
        })
    
    # Resumo Financeiro
    numero_transportadoras = int(df_financeiro['Transportadora'].nunique())
    
    return {
        "kpis": {
            "custo_total_frete": round(custo_total, 2),
            "custo_medio_pedido": round(custo_medio, 2),
            "numero_pedidos": numero_pedidos
        },
        "custo_por_transportadora": {"data": custo_transportadora},
        "representatividade_custos": {"data": representatividade},
        "resumo_financeiro": {
            "custo_total": round(custo_total, 2),
            "custo_medio": round(custo_medio, 2),
            "total_pedidos": numero_pedidos,
            "numero_transportadoras": numero_transportadoras
        }
    }

def generate_evolucao_on_time(df: pd.DataFrame) -> Dict[str, Any]:
    """10. Evolução do On Time Delivery ao longo do tempo"""
    
    if 'Data Despacho' not in df.columns or 'Situação' not in df.columns:
        return {"data": [], "periodos": []}
    
    # Criar cópia para não modificar o original
    df_tempo = df.copy()
    df_tempo['Data Despacho'] = pd.to_datetime(df_tempo['Data Despacho'], errors='coerce')
    df_tempo = df_tempo.dropna(subset=['Data Despacho'])
    
    # Agrupar por mês/ano
    df_tempo['periodo'] = df_tempo['Data Despacho'].dt.strftime('%b/%y')
    df_tempo['mes_ano_order'] = df_tempo['Data Despacho'].dt.to_period('M')
    
    # Calcular On Time por período
    resultado = []
    for periodo in sorted(df_tempo['mes_ano_order'].unique()):
        df_periodo = df_tempo[df_tempo['mes_ano_order'] == periodo]
        total = len(df_periodo)
        no_prazo = len(df_periodo[df_periodo['Situação'] == 'No prazo'])
        on_time_pct = (no_prazo / total * 100) if total > 0 else 0
        
        resultado.append({
            "periodo": str(periodo).replace('-', '/'),
            "periodo_label": df_periodo['periodo'].iloc[0],
            "on_time_pct": round(float(on_time_pct), 2),
            "total_entregas": int(total),
            "entregas_no_prazo": int(no_prazo)
        })
    
    # Ordenar por período
    resultado.sort(key=lambda x: x['periodo'])
    
    return {
        "data": resultado,
        "periodos": [r['periodo_label'] for r in resultado]
    }

def generate_insights(
    df: pd.DataFrame,
    kpis: Dict[str, Any],
    analise_atrasados: Dict[str, Any],
    analise_sla_transportadora: Dict[str, Any],
    analise_sla_regiao: Dict[str, Any],
    indicadores_financeiros: Dict[str, Any]
) -> Dict[str, Any]:
    """11. Gerar Insights Inteligentes baseados nos dados"""
    
    insights = {
        "visao_geral": [],
        "regiao": [],
        "transportadora": [],
        "sla": [],
        "financeiro": []
    }
    
    # Insights - Visão Geral
    total_pedidos = kpis['total_pedidos']
    sla_geral = kpis['sla_geral']
    taxa_atraso = 100 - sla_geral
    
    # Top transportadora por volume
    if 'Transportadora' in df.columns:
        transp_volume = df.groupby('Transportadora')['Pedido'].count().sort_values(ascending=False)
        top_transp = transp_volume.index[0]
        top_transp_volume = int(transp_volume.iloc[0])
        top_transp_pct = (top_transp_volume / total_pedidos * 100)
        
        # SLA da top transportadora
        top_transp_sla = next((t['sla'] for t in analise_sla_transportadora['data'] if t['transportadora'] == top_transp), 0)
        
        insights['visao_geral'].append({
            "titulo": "📦 Volume & SLA",
            "texto": f"{top_transp} concentra {top_transp_pct:.1f}% das entregas ({top_transp_volume:,} pedidos) e apresenta SLA de {top_transp_sla:.1f}%.",
            "acao": f"{'Manter como parceiro estratégico e avaliar aumento de volume.' if top_transp_sla >= 95 else 'Revisar performance e considerar redistribuição de volume.'}"
        })
    
    insights['visao_geral'].append({
        "titulo": "⏱️ Performance Geral",
        "texto": f"Taxa de atraso de {taxa_atraso:.1f}%, {'indicando operação eficiente' if taxa_atraso < 10 else 'requerendo atenção imediata'}.",
        "acao": f"{'Atuar preventivamente para atingir excelência (>98% On Time).' if taxa_atraso < 10 else 'Implementar plano de ação urgente para reduzir atrasos.'}"
    })
    
    # Insights - Região
    regioes_data = analise_sla_regiao['data']
    if len(regioes_data) > 1:
        regioes_validas = [r for r in regioes_data if r['regiao'] != 'Total Geral']
        melhor_regiao = max(regioes_validas, key=lambda x: x['sla'])
        pior_regiao = min(regioes_validas, key=lambda x: x['sla'])
        
        insights['regiao'].append({
            "titulo": "🌎 Performance Regional",
            "texto": f"Região {melhor_regiao['regiao']} possui o melhor SLA ({melhor_regiao['sla']:.1f}%) com {melhor_regiao['total']:,} entregas.",
            "acao": f"Usar {melhor_regiao['regiao']} como benchmark operacional para outras regiões."
        })
        
        if pior_regiao['sla'] < 90:
            insights['regiao'].append({
                "titulo": "🚨 Atenção Regional",
                "texto": f"Região {pior_regiao['regiao']} apresenta SLA crítico de {pior_regiao['sla']:.1f}%.",
                "acao": "Investigar causas raiz: infraestrutura, transportadoras ou processos locais."
            })
    
    # Insights - Transportadora
    transportadoras_data = analise_sla_transportadora['data']
    if transportadoras_data:
        melhor_transp_sla = transportadoras_data[0]  # Já ordenado por SLA
        pior_transp_sla = transportadoras_data[-1]
        
        insights['transportadora'].append({
            "titulo": "🏆 Melhor Performance",
            "texto": f"{melhor_transp_sla['transportadora']} lidera com SLA de {melhor_transp_sla['sla']:.1f}% ({melhor_transp_sla['total']:,} entregas).",
            "acao": "Considerar como transportadora preferencial para rotas críticas."
        })
        
        if pior_transp_sla['sla'] < 85:
            insights['transportadora'].append({
                "titulo": "🚨 Risco Operacional",
                "texto": f"{pior_transp_sla['transportadora']} possui SLA crítico de {pior_transp_sla['sla']:.1f}%, impactando negativamente a operação.",
                "acao": "Revisar SLA contratual, aplicar penalidades ou reduzir participação."
            })
    
    # Insights - SLA Detalhado
    atrasados_data = analise_atrasados['data']
    if atrasados_data:
        total_atrasados = sum(t['atrasado'] for t in atrasados_data)
        transp_mais_atrasada = max(atrasados_data, key=lambda x: x['pct_atrasado'])
        
        insights['sla'].append({
            "titulo": "📊 Análise de Atrasos",
            "texto": f"Total de {total_atrasados:,} pedidos atrasados no período. {transp_mais_atrasada['transportadora']} apresenta maior taxa de atraso ({transp_mais_atrasada['pct_atrasado']:.1f}%).",
            "acao": "Focar esforços de melhoria nas transportadoras com maior índice de atraso."
        })
    
    # Insights - Financeiro
    if indicadores_financeiros['kpis']['custo_total_frete'] > 0:
        custo_medio = indicadores_financeiros['kpis']['custo_medio_pedido']
        custos_transp = indicadores_financeiros['custo_por_transportadora']['data']
        
        if custos_transp:
            transp_mais_cara = max(custos_transp, key=lambda x: x['custo_medio'])
            transp_mais_barata = min(custos_transp, key=lambda x: x['custo_medio'])
            
            insights['financeiro'].append({
                "titulo": "💰 Análise de Custos",
                "texto": f"Custo médio de frete é R$ {custo_medio:.2f}. {transp_mais_barata['transportadora']} oferece o melhor custo (R$ {transp_mais_barata['custo_medio']:.2f}).",
                "acao": "Renegociar contratos com transportadoras acima da média ou redistribuir volume."
            })
            
            diferenca_custo = transp_mais_cara['custo_medio'] - transp_mais_barata['custo_medio']
            if diferenca_custo > custo_medio * 0.3:  # Diferença > 30%
                insights['financeiro'].append({
                    "titulo": "💡 Oportunidade de Economia",
                    "texto": f"Diferença de R$ {diferenca_custo:.2f} entre {transp_mais_cara['transportadora']} (R$ {transp_mais_cara['custo_medio']:.2f}) e {transp_mais_barata['transportadora']} (R$ {transp_mais_barata['custo_medio']:.2f}).",
                    "acao": f"Potencial economia de {(diferenca_custo / transp_mais_cara['custo_medio'] * 100):.1f}% migrando volume para transportadora mais econômica."
                })
    
    return insights

# API Routes
@api_router.post("/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Login endpoint"""
    if request.username not in USERS_DB or USERS_DB[request.username] != request.password:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    access_token = create_access_token(data={"sub": request.username})
    return LoginResponse(token=access_token, username=request.username)

@api_router.get("/auth/verify")
async def verify(username: str = Depends(verify_token)):
    """Verify token endpoint"""
    return {"username": username, "valid": True}

@api_router.post("/upload", response_model=AnalysisResult)
async def upload_file(
    file: UploadFile = File(...),
    username: str = Depends(verify_token)
):
    """Upload and process Excel file"""
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Only Excel files are allowed")
    
    try:
        # Read file content
        content = await file.read()
        
        # Process data
        analysis_data = process_excel_data(content, file.filename)
        
        # Create result object
        result_create = AnalysisResultCreate(
            filename=file.filename,
            total_records=analysis_data['total_records'],
            date_range=analysis_data['date_range'],
            analysis_data=analysis_data
        )
        
        result_obj = AnalysisResult(**result_create.model_dump())
        
        # Save to MongoDB
        doc = result_obj.model_dump()
        doc['upload_date'] = doc['upload_date'].isoformat()
        doc['username'] = username
        
        await db.analysis_results.insert_one(doc)
        
        return result_obj
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@api_router.get("/analysis/latest")
async def get_latest_analysis(username: str = Depends(verify_token)):
    """Get the latest analysis result for the user"""
    result = await db.analysis_results.find_one(
        {"username": username},
        {"_id": 0},
        sort=[("upload_date", -1)]
    )
    
    if not result:
        raise HTTPException(status_code=404, detail="No analysis found")
    
    return result

@api_router.get("/analysis/history")
async def get_analysis_history(username: str = Depends(verify_token)):
    """Get analysis history for the user"""
    results = await db.analysis_results.find(
        {"username": username},
        {"_id": 0, "id": 1, "filename": 1, "upload_date": 1, "total_records": 1, "date_range": 1}
    ).sort("upload_date", -1).limit(10).to_list(10)
    
    return {"history": results}

@api_router.get("/analysis/{analysis_id}")
async def get_analysis_by_id(analysis_id: str, username: str = Depends(verify_token)):
    """Get specific analysis by ID"""
    result = await db.analysis_results.find_one(
        {"id": analysis_id, "username": username},
        {"_id": 0}
    )
    
    if not result:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return result

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
