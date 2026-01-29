from fastapi import FastAPI, APIRouter, HTTPException, Depends, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware
import os
import logging
import json
import subprocess
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime, timezone, timedelta
import jwt
import pandas as pd
import io

# JWT Configuration
SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 hours

# Google Drive Configuration
GDRIVE_SYNC_DIR = Path("/home/ubuntu/logdash_gdrive_sync")
GDRIVE_REMOTE = "manus_google_drive:LogDash_Data"
RCLONE_CONFIG = "/home/ubuntu/.gdrive-rclone.ini"
STORAGE_FILE = GDRIVE_SYNC_DIR / "analysis_storage.json"

os.makedirs(GDRIVE_SYNC_DIR, exist_ok=True)

def sync_from_gdrive():
    try:
        subprocess.run([
            "rclone", "copy", GDRIVE_REMOTE, str(GDRIVE_SYNC_DIR),
            "--config", RCLONE_CONFIG
        ], check=True)
    except Exception as e:
        print(f"Error syncing from GDrive: {e}")

def sync_to_gdrive():
    try:
        subprocess.run([
            "rclone", "copy", str(GDRIVE_SYNC_DIR), GDRIVE_REMOTE,
            "--config", RCLONE_CONFIG
        ], check=True)
    except Exception as e:
        print(f"Error syncing to GDrive: {e}")

def load_storage():
    sync_from_gdrive()
    if STORAGE_FILE.exists():
        try:
            with open(STORAGE_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_storage(data):
    with open(STORAGE_FILE, 'w') as f:
        json.dump(data, f)
    sync_to_gdrive()

# Initial load
ANALYSIS_STORAGE = load_storage()

# Mock databases
USERS_DB = {
    "batlog": "123",
    "editorajuspodivm": "123"
}

app = FastAPI()
api_router = APIRouter(prefix="/api")
security = HTTPBearer()

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

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

@api_router.post("/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    if request.username not in USERS_DB or USERS_DB[request.username] != request.password:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    access_token = create_access_token(data={"sub": request.username})
    return LoginResponse(token=access_token, username=request.username)

@api_router.get("/auth/verify")
async def verify(username: str = Depends(verify_token)):
    return {"username": username, "valid": True}

@api_router.get("/analysis/latest")
async def get_latest_analysis(username: str = Depends(verify_token)):
    global ANALYSIS_STORAGE
    ANALYSIS_STORAGE = load_storage()
    user_results = [r for r in ANALYSIS_STORAGE.values() if r.get('username') == username]
    if not user_results:
        raise HTTPException(status_code=404, detail="No analysis found")
    latest = max(user_results, key=lambda x: x['upload_date'])
    return latest

@api_router.get("/analysis/history")
async def get_analysis_history(username: str = Depends(verify_token)):
    global ANALYSIS_STORAGE
    ANALYSIS_STORAGE = load_storage()
    user_results = [
        {"id": r['id'], "filename": r['filename'], "upload_date": r['upload_date'], 
         "total_records": r['total_records'], "date_range": r['date_range']}
        for r in ANALYSIS_STORAGE.values() if r.get('username') == username
    ]
    user_results.sort(key=lambda x: x['upload_date'], reverse=True)
    return {"history": user_results[:10]}

# Import the original processing functions
import sys
sys.path.append(os.path.dirname(__file__))
from server_original import process_excel_data, AnalysisResultCreate

@api_router.post("/upload", response_model=AnalysisResult)
async def upload_file(file: UploadFile = File(...), username: str = Depends(verify_token)):
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Only Excel files are allowed")
    try:
        content = await file.read()
        analysis_data = process_excel_data(content, file.filename)
        result_obj = AnalysisResult(
            filename=file.filename,
            total_records=analysis_data['total_records'],
            date_range=analysis_data['date_range'],
            analysis_data=analysis_data
        )
        doc = result_obj.model_dump()
        doc['upload_date'] = doc['upload_date'].isoformat()
        doc['username'] = username
        
        global ANALYSIS_STORAGE
        ANALYSIS_STORAGE = load_storage()
        ANALYSIS_STORAGE[result_obj.id] = doc
        save_storage(ANALYSIS_STORAGE)
        
        return result_obj
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

app.include_router(api_router)
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files from the React build folder
BUILD_DIR = Path(__file__).parent.parent / "frontend_build"
if BUILD_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(BUILD_DIR / "static")), name="static")

    @app.get("/{full_path:path}")
    async def serve_react_app(full_path: str):
        if full_path.startswith("api"):
            raise HTTPException(status_code=404)
        file_path = BUILD_DIR / full_path
        if full_path != "" and file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(BUILD_DIR / "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
