"""
API FastAPI para o sistema RAG de análise de documentos
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import logging
import os
from pathlib import Path
import asyncio
from rag_system import get_rag_system

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Criar diretórios necessários
os.makedirs('logs', exist_ok=True)
os.makedirs('src/static', exist_ok=True)

app = FastAPI(
    title="Sistema RAG de Análise de Documentos",
    description="API para análise de documentos de projetos de energia solar",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir arquivos estáticos
app.mount("/static", StaticFiles(directory="src/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve a página principal"""
    try:
        html_path = Path("src/static/index.html")
        if html_path.exists():
            return HTMLResponse(content=html_path.read_text(encoding='utf-8'))
        else:
            return HTMLResponse(content="""
            <html>
                <body>
                    <h1>Sistema RAG</h1>
                    <p>Arquivo index.html não encontrado em src/static/</p>
                </body>
            </html>
            """)
    except Exception as e:
        logger.error(f"Erro ao servir página principal: {e}")
        return HTMLResponse(content=f"<html><body><h1>Erro: {str(e)}</h1></body></html>")

@app.post("/api/analyze")
async def analyze_documents(files: List[UploadFile] = File(...)):
    """Endpoint para análise de documentos"""
    try:
        logger.info(f"Recebidos {len(files)} arquivos para análise")
        
        # Validações
        if len(files) > 10:
            raise HTTPException(status_code=400, detail="Máximo de 10 arquivos permitidos")
        
        if len(files) == 0:
            raise HTTPException(status_code=400, detail="Nenhum arquivo enviado")
        
        # Validar tipos de arquivo
        allowed_types = {
            'application/pdf',
            'application/msword',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'text/plain'
        }
        
        processed_files = []
        
        for file in files:
            # Verificar tipo de arquivo
            if file.content_type not in allowed_types:
                logger.warning(f"Tipo de arquivo não suportado: {file.filename} ({file.content_type})")
                continue
            
            # Verificar tamanho (máximo 10MB por arquivo)
            content = await file.read()
            if len(content) > 10 * 1024 * 1024:  # 10MB
                logger.warning(f"Arquivo muito grande: {file.filename} ({len(content)} bytes)")
                continue
            
            processed_files.append((file.filename, content))
            logger.info(f"Arquivo processado: {file.filename} ({len(content)} bytes)")
        
        if not processed_files:
            raise HTTPException(status_code=400, detail="Nenhum arquivo válido encontrado")
        
        # Obter sistema RAG
        rag = get_rag_system()
        
        # Analisar documentos
        result = rag.analyze_documents(processed_files)
        
        logger.info("Análise concluída com sucesso")
        return JSONResponse(content=result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro na análise de documentos: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno do servidor: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Endpoint de health check"""
    try:
        # Verificar se o sistema RAG está funcionando
        rag = get_rag_system()
        available_providers = rag.ai_manager.get_available_providers()
        
        return {
            "status": "healthy",
            "ai_providers": available_providers,
            "collection_count": rag.collection.count()
        }
    except Exception as e:
        logger.error(f"Erro no health check: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/api/status")
async def get_status():
    """Endpoint para obter status do sistema"""
    try:
        rag = get_rag_system()
        
        return {
            "ai_providers": rag.ai_manager.get_available_providers(),
            "collection_name": os.getenv('COLLECTION_NAME', 'documents'),
            "total_documents": rag.collection.count(),
            "vector_db_path": os.getenv('VECTOR_DB_PATH', './data/embeddings')
        }
    except Exception as e:
        logger.error(f"Erro ao obter status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(500)
async def internal_server_error_handler(request: Request, exc: Exception):
    """Handler para erros internos do servidor"""
    logger.error(f"Erro interno do servidor: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Erro interno do servidor"}
    )

@app.exception_handler(404)
async def not_found_handler(request: Request, exc: Exception):
    """Handler para recursos não encontrados"""
    return JSONResponse(
        status_code=404,
        content={"detail": "Recurso não encontrado"}
    )

if __name__ == "__main__":
    import uvicorn
    
    # Configurações do servidor
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"Iniciando servidor em {host}:{port}")
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
