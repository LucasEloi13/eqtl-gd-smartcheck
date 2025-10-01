"""
Sistema RAG para análise de documentos de microgeração distribuída.
Mantém embeddings da norma NT.00020 no ChromaDB e processa documentos do projeto em memória.
"""

import os
import json
import yaml
import logging
import re
from io import BytesIO
from typing import Dict, List, Any, Optional
from datetime import datetime

import chromadb
from chromadb.config import Settings
import openai
from openai import OpenAI
from dotenv import load_dotenv

# Processamento de documentos
import PyPDF2
from docx import Document

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIProviderManager:
    """Gerencia provedores de AI de forma agnóstica"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.providers = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Inicializa provedores disponíveis"""
        ai_config = self.config.get('ai_providers', {})
        
        # OpenAI
        if 'openai' in ai_config:
            try:
                openai_config = ai_config['openai']
                api_key = os.getenv(openai_config['api_key_env'])
                if api_key:
                    self.providers['openai'] = OpenAI(api_key=api_key)
                    logger.info("Provedor openai inicializado com sucesso")
            except Exception as e:
                logger.error(f"Erro ao inicializar OpenAI: {e}")
        
        # DeepSeek
        if 'deepseek' in ai_config:
            try:
                deepseek_config = ai_config['deepseek']
                api_key = os.getenv(deepseek_config['api_key_env'])
                if api_key:
                    self.providers['deepseek'] = OpenAI(
                        api_key=api_key,
                        base_url=deepseek_config['base_url']
                    )
                    logger.info("Provedor deepseek inicializado com sucesso")
            except Exception as e:
                logger.error(f"Erro ao inicializar DeepSeek: {e}")
    
    def get_available_providers(self) -> List[str]:
        """Retorna lista de provedores disponíveis"""
        return list(self.providers.keys())
    
    def generate_embedding(self, text: str, provider: str = None) -> List[float]:
        """Gera embedding usando o provedor especificado ou o primeiro disponível"""
        if not provider:
            provider = list(self.providers.keys())[0] if self.providers else None
        
        if not provider or provider not in self.providers:
            raise ValueError(f"Provedor {provider} não disponível")
        
        client = self.providers[provider]
        
        try:
            if provider == 'openai':
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text
                )
                return response.data[0].embedding
            
            elif provider == 'deepseek':
                # DeepSeek usa o mesmo formato da OpenAI
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text
                )
                return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Erro ao gerar embedding com {provider}: {e}")
            raise
    
    def generate_chat_response(self, messages: List[Dict], provider: str = None) -> str:
        """Gera resposta de chat usando o provedor especificado"""
        if not provider:
            provider = list(self.providers.keys())[0] if self.providers else None
        
        if not provider or provider not in self.providers:
            raise ValueError(f"Provedor {provider} não disponível")
        
        client = self.providers[provider]
        
        try:
            if provider == 'openai':
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.1
                )
                return response.choices[0].message.content
            
            elif provider == 'deepseek':
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    temperature=0.1
                )
                return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Erro ao gerar resposta com {provider}: {e}")
            raise
    
    def generate_analysis(self, context: str, prompt_config: Dict) -> Dict[str, Any]:
        """Gera análise usando configuração de prompt"""
        try:
            # Usar primeiro provedor disponível
            provider = list(self.providers.keys())[0] if self.providers else None
            if not provider:
                raise ValueError("Nenhum provedor AI disponível")
            
            persona = prompt_config.get('persona', '')
            task = prompt_config.get('task', '')
            
            system_message = f"""
{persona}

{task}

IMPORTANTE: Você DEVE retornar APENAS um JSON válido, sem texto adicional antes ou depois.

{context}

FORMATO DE SAÍDA OBRIGATÓRIO:
{{
    "Empresa responsável": "string",
    "Cliente": "string", 
    "Local": "string",
    "Potência do sistema": "string",
    "Potência nominal do inversor": "string",
    "Modelo do inversor": "string",
    "Quantidade de inversores": "string",
    "Potência dos módulos": "string",
    "Quantidade de módulos": "string",
    "Modelo dos módulos": "string",
    "Tensão da UC": "string",
    "Disjuntor de entrada da UC": "string",
    "Disjuntor de proteção do sistema FV (saída do inversor)": "string",
    "Potência disponibilizada (PD)": "string",
    "Incongruências encontradas": ["string1", "string2"]
}}
"""
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": "Extraia as informações e retorne APENAS o JSON:"}
            ]
            
            response = self.generate_chat_response(messages, provider)
            
            # Tentar parsear JSON
            try:
                response_clean = response.strip()
                if response_clean.startswith('```'):
                    lines = response_clean.split('\n')
                    if len(lines) > 2:
                        response_clean = '\n'.join(lines[1:-1])
                
                result = json.loads(response_clean)
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"Erro JSON: {e}")
                logger.error(f"Resposta: {response}")
                raise ValueError("Resposta não é JSON válido")
                
        except Exception as e:
            logger.error(f"Erro na análise: {e}")
            raise

class DocumentProcessor:
    """Processa diferentes tipos de documentos"""
    
    def extract_text_from_pdf(self, file_content: bytes) -> str:
        """Extrai texto de PDF"""
        try:
            pdf_file = BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Erro ao processar PDF: {e}")
            raise
    
    def extract_text_from_docx(self, file_content: bytes) -> str:
        """Extrai texto de DOCX"""
        try:
            docx_file = BytesIO(file_content)
            doc = Document(docx_file)
            
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Erro ao processar DOCX: {e}")
            raise
    
    def extract_text_from_txt(self, file_content: bytes) -> str:
        """Extrai texto de TXT"""
        try:
            return file_content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                return file_content.decode('latin-1')
            except Exception as e:
                logger.error(f"Erro ao processar TXT: {e}")
                raise
    
    def process_file(self, filename: str, file_content: bytes) -> str:
        """Processa arquivo baseado na extensão"""
        filename_lower = filename.lower()
        
        if filename_lower.endswith('.pdf'):
            return self.extract_text_from_pdf(file_content)
        elif filename_lower.endswith('.docx'):
            return self.extract_text_from_docx(file_content)
        elif filename_lower.endswith('.txt'):
            return self.extract_text_from_txt(file_content)
        else:
            raise ValueError(f"Tipo de arquivo não suportado: {filename}")

class RAGSystem:
    """Sistema RAG completo"""
    
    def __init__(self):
        # Carregar configurações
        load_dotenv()
        self.config = self._load_config()
        
        # Inicializar componentes
        self.ai_manager = AIProviderManager(self.config)
        self.document_processor = DocumentProcessor()
        self.chroma_client = self._setup_chromadb()
        self.collection = self._get_or_create_collection()
        
        # Configurações
        self.chunk_size = 1000
        self.chunk_overlap = 200
    
    def _load_config(self) -> Dict:
        """Carrega configuração do arquivo YAML"""
        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Erro ao carregar config.yaml: {e}")
            return {}
    
    def _setup_chromadb(self):
        """Configura cliente ChromaDB"""
        db_path = self.config.get('vector_db', {}).get('path', './data/embeddings')
        os.makedirs(db_path, exist_ok=True)
        
        client = chromadb.PersistentClient(path=db_path)
        logger.info(f"ChromaDB configurado em: {db_path}")
        
        return client
    
    def _get_or_create_collection(self):
        """Obtém ou cria coleção no ChromaDB"""
        collection_name = self.config.get('vector_db', {}).get('collection_name', 'documents')
        
        try:
            collection = self.chroma_client.get_collection(collection_name)
            logger.info(f"Coleção '{collection_name}' carregada")
        except:
            collection = self.chroma_client.create_collection(collection_name)
            logger.info(f"Coleção '{collection_name}' criada")
        
        return collection
    
    def split_text_into_chunks(self, text: str) -> List[str]:
        """Divide texto em chunks inteligentes"""
        chunks = []
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) < self.chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                if len(paragraph) > self.chunk_size:
                    sentences = paragraph.split('. ')
                    temp_chunk = ""
                    
                    for sentence in sentences:
                        if len(temp_chunk) + len(sentence) < self.chunk_size:
                            temp_chunk += sentence + ". "
                        else:
                            if temp_chunk.strip():
                                chunks.append(temp_chunk.strip())
                            temp_chunk = sentence + ". "
                    
                    current_chunk = temp_chunk
                else:
                    current_chunk = paragraph + "\n\n"
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def process_documents_in_memory(self, files: List[tuple]) -> List[Dict]:
        """Processa documentos e mantém embeddings em memória (não salva no ChromaDB)"""
        document_chunks = []
        
        for filename, file_content in files:
            try:
                logger.info(f"Processando arquivo: {filename}")
                
                # Extrair texto do documento
                text = self.document_processor.process_file(filename, file_content)
                
                if not text.strip():
                    logger.warning(f"Nenhum texto extraído de {filename}")
                    continue
                
                # Dividir em chunks
                chunks = self.split_text_into_chunks(text)
                
                # Processar cada chunk
                for i, chunk in enumerate(chunks):
                    try:
                        # Gerar embedding
                        embedding = self.ai_manager.generate_embedding(chunk)
                        
                        # Armazenar em memória (não no ChromaDB)
                        chunk_data = {
                            "id": f"{filename}_{i}_{len(chunk)}",
                            "text": chunk,
                            "embedding": embedding,
                            "metadata": {
                                "source": filename,
                                "chunk_index": i,
                                "chunk_size": len(chunk),
                                "total_chunks": len(chunks),
                                "document_type": "project_upload"
                            }
                        }
                        
                        document_chunks.append(chunk_data)
                        
                    except Exception as e:
                        logger.error(f"Erro ao processar chunk {i} de {filename}: {e}")
                        continue
                
                logger.info(f"Arquivo {filename} processado: {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Erro ao processar arquivo {filename}: {e}")
                continue
        
        logger.info(f"Total de chunks processados em memória: {len(document_chunks)}")
        return document_chunks
    
    def search_relevant_context_from_norm(self, query: str, n_results: int = 10) -> str:
        """Busca contexto relevante apenas na norma NT.00020 (ChromaDB)"""
        try:
            # Gerar embedding da consulta
            query_embedding = self.ai_manager.generate_embedding(query)
            
            # Buscar documentos similares apenas na norma (ChromaDB)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Construir contexto da norma
            context_parts = []
            for i, doc in enumerate(results['documents'][0]):
                similarity = 1 - results['distances'][0][i]
                if similarity > 0.3:  # Filtro de similaridade mínima
                    source = results['metadatas'][0][i].get('source', 'NT.00020')
                    context_parts.append(f"[Norma: {source}]\n{doc}")
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Erro ao buscar contexto da norma: {e}")
            return ""
    
    def build_project_context(self, document_chunks: List[Dict]) -> str:
        """Constrói contexto dos documentos do projeto (em memória)"""
        try:
            context_parts = []
            for chunk in document_chunks:
                source = chunk['metadata']['source']
                text = chunk['text']
                context_parts.append(f"[Documento do Projeto: {source}]\n{text}")
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Erro ao construir contexto do projeto: {e}")
            return ""
    
    def analyze_documents(self, files: List[tuple]) -> Dict[str, Any]:
        """Analisa documentos usando RAG - documentos em memória + norma no ChromaDB"""
        try:
            # 1. Processar documentos do projeto em memória (não salvar no ChromaDB)
            document_chunks = self.process_documents_in_memory(files)
            
            if not document_chunks:
                raise ValueError("Nenhum conteúdo extraído dos documentos")
            
            # 2. Construir contexto dos documentos do projeto
            project_context = self.build_project_context(document_chunks)
            
            # 3. Buscar contexto relevante da norma NT.00020 (ChromaDB)
            norm_context = self.search_relevant_context_from_norm(
                "análise projeto microgeração minigeração distribuída conexão sistema distribuição",
                n_results=15
            )
            
            # 4. Combinar contextos
            combined_context = f"""
CONTEXTO DA NORMA TÉCNICA:
{norm_context}

CONTEXTO DOS DOCUMENTOS DO PROJETO:
{project_context}
"""
            
            # 5. Gerar análise usando a IA
            analysis_result = self.ai_manager.generate_analysis(
                context=combined_context,
                prompt_config=self.config.get('prompt_config', {})
            )
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Erro na análise de documentos: {e}")
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """Retorna status do sistema"""
        try:
            return {
                "ai_providers": self.ai_manager.get_available_providers(),
                "collection_name": self.collection.name,
                "total_documents": self.collection.count(),
                "vector_db_path": self.config.get('vector_db', {}).get('path', './data/embeddings')
            }
        except Exception as e:
            logger.error(f"Erro ao obter status: {e}")
            return {}
    
    def health_check(self) -> Dict[str, Any]:
        """Verifica saúde do sistema"""
        try:
            providers = self.ai_manager.get_available_providers()
            collection_count = self.collection.count()
            
            return {
                "status": "healthy",
                "ai_providers": providers,
                "collection_count": collection_count
            }
        except Exception as e:
            logger.error(f"Erro no health check: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }

# Instância global do sistema RAG
rag_system = None

def get_rag_system():
    """Retorna instância singleton do sistema RAG"""
    global rag_system
    if rag_system is None:
        rag_system = RAGSystem()
    return rag_system
