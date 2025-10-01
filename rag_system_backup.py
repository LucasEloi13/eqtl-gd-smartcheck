"""
Sistema RAG agnóstico de provedor para análise de documentos
"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from openai import OpenAI
import tempfile
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Document processing libraries
import PyPDF2
import docx
import io

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/rag_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AIProviderManager:
    """Gerenciador agnóstico de provedores de AI"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.providers = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Inicializa todos os provedores disponíveis"""
        ai_providers = self.config.get('ai_providers', {})
        
        for provider_name, provider_config in ai_providers.items():
            try:
                if provider_name == 'openai':
                    self.providers[provider_name] = {
                        'client': OpenAI(
                            api_key=os.getenv('OPENAI_API_KEY'),
                            base_url=provider_config['base_url']
                        ),
                        'config': provider_config
                    }
                elif provider_name == 'deepseek':
                    self.providers[provider_name] = {
                        'client': OpenAI(
                            api_key=os.getenv('DEEPSEEK_API_KEY'),
                            base_url=provider_config['base_url']
                        ),
                        'config': provider_config
                    }
                
                logger.info(f"Provedor {provider_name} inicializado com sucesso")
                
            except Exception as e:
                logger.warning(f"Erro ao inicializar provedor {provider_name}: {e}")
    
    def get_available_providers(self) -> List[str]:
        """Retorna lista de provedores disponíveis"""
        return list(self.providers.keys())
    
    def generate_embedding(self, text: str, provider: str = None) -> List[float]:
        """Gera embedding usando o provedor especificado ou o primeiro disponível"""
        if not provider:
            provider = os.getenv('AI_PROVIDER', 'openai')
        
        if provider not in self.providers:
            available = self.get_available_providers()
            if not available:
                raise ValueError("Nenhum provedor de AI disponível")
            provider = available[0]
            logger.warning(f"Provedor {provider} não disponível, usando {available[0]}")
        
        try:
            client = self.providers[provider]['client']
            config = self.providers[provider]['config']
            
            if provider in ['openai', 'deepseek']:
                response = client.embeddings.create(
                    model=config['models']['embedding'],
                    input=text
                )
                return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Erro ao gerar embedding com {provider}: {e}")
            raise
    
    def generate_chat_response(self, messages: List[Dict], provider: str = None) -> str:
        """Gera resposta usando o provedor especificado"""
        if not provider:
            provider = os.getenv('AI_PROVIDER', 'openai')
        
        if provider not in self.providers:
            available = self.get_available_providers()
            if not available:
                raise ValueError("Nenhum provedor de AI disponível")
            provider = available[0]
        
        try:
            client = self.providers[provider]['client']
            config = self.providers[provider]['config']
            
            response = client.chat.completions.create(
                model=config['models']['chat'],
                messages=messages,
                max_tokens=config.get('max_tokens', 4096),
                temperature=config.get('temperature', 0.5)
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Erro ao gerar resposta com {provider}: {e}")
            raise

class DocumentProcessor:
    """Processador de documentos para extração de texto"""
    
    @staticmethod
    def extract_text_from_pdf(file_content: bytes) -> str:
        """Extrai texto de PDF"""
        try:
            pdf_file = io.BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    text += f"\n--- Página {page_num + 1} ---\n"
                    text += page_text
                except Exception as e:
                    logger.warning(f"Erro ao extrair texto da página {page_num + 1}: {e}")
                    continue
            
            return text
            
        except Exception as e:
            logger.error(f"Erro ao processar PDF: {e}")
            raise
    
    @staticmethod
    def extract_text_from_docx(file_content: bytes) -> str:
        """Extrai texto de DOCX"""
        try:
            doc_file = io.BytesIO(file_content)
            doc = docx.Document(doc_file)
            text = ""
            
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            return text
            
        except Exception as e:
            logger.error(f"Erro ao processar DOCX: {e}")
            raise
    
    @staticmethod
    def extract_text_from_txt(file_content: bytes) -> str:
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
        elif filename_lower.endswith('.doc'):
            # Para .doc seria necessário python-docx2txt ou similar
            raise ValueError("Arquivos .doc não são suportados. Use .docx")
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
        try:
            with open('config.yaml', 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Erro ao carregar config.yaml: {e}")
            raise
    
    def _setup_chromadb(self) -> chromadb.Client:
        """Configura ChromaDB"""
        try:
            db_path = Path(os.getenv('VECTOR_DB_PATH', './data/embeddings'))
            db_path.mkdir(parents=True, exist_ok=True)
            
            client = chromadb.PersistentClient(
                path=str(db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            logger.info(f"ChromaDB configurado em: {db_path}")
            return client
            
        except Exception as e:
            logger.error(f"Erro ao configurar ChromaDB: {e}")
            raise
    
    def _get_or_create_collection(self):
        """Obtém ou cria coleção no ChromaDB"""
        collection_name = os.getenv('COLLECTION_NAME', 'documents')
        
        try:
            return self.chroma_client.get_collection(name=collection_name)
        except:
            return self.chroma_client.create_collection(name=collection_name)
    
    def split_text_into_chunks(self, text: str) -> List[str]:
        """Divide texto em chunks"""
        chunks = []
        paragraphs = text.split('\n\n')
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) <= self.chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                if len(paragraph) > self.chunk_size:
                    sentences = paragraph.split('. ')
                    temp_chunk = ""
                    
                    for sentence in sentences:
                        if len(temp_chunk) + len(sentence) <= self.chunk_size:
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
            
            # 4. Construir prompt usando configuração
            prompt_config = self.config.get('prompt_config', {})
            
            persona = prompt_config.get('persona', '')
            task = prompt_config.get('task', '')
            output_instructions = prompt_config.get('output_format', {}).get('instructions', '')
            
            system_message = f"""
{persona}

{task}

IMPORTANTE: Você DEVE retornar APENAS um JSON válido, sem texto adicional antes ou depois. Não inclua explicações, comentários ou formatação markdown.

            combined_context = f"""
CONTEXTO DA NORMA TÉCNICA:
{norm_context}

CONTEXTO DOS DOCUMENTOS DO PROJETO:
{project_context}

FORMATO DE SAÍDA OBRIGATÓRIO:
Retorne EXCLUSIVAMENTE um JSON no seguinte formato exato:
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

Se não encontrar algum campo, use string vazia "". Para incongruências, use array vazio [] se não houver."""

            user_message = "Extraia as informações dos documentos e retorne APENAS o JSON solicitado:"
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
            
            # 5. Gerar análise usando AI
            response = self.ai_manager.generate_chat_response(messages)
            
            # 6. Processar resposta JSON
            try:
                # Tentar extrair JSON da resposta
                response_clean = response.strip()
                
                # Remover possíveis wrappers de markdown
                if response_clean.startswith('```json'):
                    response_clean = response_clean[7:]
                elif response_clean.startswith('```'):
                    response_clean = response_clean[3:]
                
                if response_clean.endswith('```'):
                    response_clean = response_clean[:-3]
                
                # Procurar por JSON dentro da resposta se não começar com {
                if not response_clean.strip().startswith('{'):
                    import re
                    json_match = re.search(r'\{.*\}', response_clean, re.DOTALL)
                    if json_match:
                        response_clean = json_match.group(0)
                    else:
                        # Se não encontrar JSON, tentar novamente com prompt mais específico
                        logger.warning("Primeira tentativa não retornou JSON, tentando novamente...")
                        
                        simple_messages = [
                            {
                                "role": "system", 
                                "content": "Você deve retornar APENAS um JSON válido. Não adicione texto antes ou depois do JSON."
                            },
                            {
                                "role": "user", 
                                "content": f"""Com base nestes documentos, extraia as informações e retorne APENAS o JSON:

DOCUMENTOS:
{docs_context[:2000]}...

JSON obrigatório:
{{
    "Empresa responsável": "",
    "Cliente": "",
    "Local": "",
    "Potência do sistema": "",
    "Potência nominal do inversor": "",
    "Modelo do inversor": "",
    "Quantidade de inversores": "",
    "Potência dos módulos": "",
    "Quantidade de módulos": "",
    "Modelo dos módulos": "",
    "Tensão da UC": "",
    "Disjuntor de entrada da UC": "",
    "Disjuntor de proteção do sistema FV (saída do inversor)": "",
    "Potência disponibilizada (PD)": "",
    "Incongruências encontradas": []
}}"""
                            }
                        ]
                        
                        response = self.ai_manager.generate_chat_response(simple_messages)
                        response_clean = response.strip()
                        
                        # Limpar novamente
                        if response_clean.startswith('```json'):
                            response_clean = response_clean[7:]
                        elif response_clean.startswith('```'):
                            response_clean = response_clean[3:]
                        
                        if response_clean.endswith('```'):
                            response_clean = response_clean[:-3]
                        
                        # Procurar JSON novamente
                        if not response_clean.strip().startswith('{'):
                            json_match = re.search(r'\{.*\}', response_clean, re.DOTALL)
                            if json_match:
                                response_clean = json_match.group(0)
                
                logger.info(f"JSON extraído: {response_clean[:200]}...")
                result = json.loads(response_clean.strip())
                
                # Garantir que todos os campos obrigatórios existam
                required_fields = [
                    "Empresa responsável", "Cliente", "Local", "Potência do sistema",
                    "Potência nominal do inversor", "Modelo do inversor", "Quantidade de inversores",
                    "Potência dos módulos", "Quantidade de módulos", "Modelo dos módulos",
                    "Tensão da UC", "Disjuntor de entrada da UC", 
                    "Disjuntor de proteção do sistema FV (saída do inversor)",
                    "Potência disponibilizada (PD)", "Incongruências encontradas"
                ]
                
                for field in required_fields:
                    if field not in result:
                        result[field] = ""
                
                # Garantir que incongruências seja uma lista
                if not isinstance(result.get("Incongruências encontradas"), list):
                    result["Incongruências encontradas"] = []
            except json.JSONDecodeError as e:
                logger.error(f"Erro ao decodificar JSON: {e}")
                logger.error(f"Resposta recebida: {response}")
                
                # Fallback: criar JSON básico com informações extraídas por regex
                logger.warning("Tentando extrair informações com regex como fallback...")
                result = self._extract_info_with_regex(docs_context)
                
            except Exception as e:
                logger.error(f"Erro inesperado no processamento JSON: {e}")
                result = self._create_empty_result()
                
            # Garantir que todos os campos obrigatórios existam
            required_fields = [
                "Empresa responsável", "Cliente", "Local", "Potência do sistema",
                "Potência nominal do inversor", "Modelo do inversor", "Quantidade de inversores",
                "Potência dos módulos", "Quantidade de módulos", "Modelo dos módulos",
                "Tensão da UC", "Disjuntor de entrada da UC", 
                "Disjuntor de proteção do sistema FV (saída do inversor)",
                "Potência disponibilizada (PD)", "Incongruências encontradas"
            ]
            
            for field in required_fields:
                if field not in result:
                    result[field] = "" if field != "Incongruências encontradas" else []
            
            # Garantir que incongruências seja uma lista
            if not isinstance(result.get("Incongruências encontradas"), list):
                result["Incongruências encontradas"] = []
            
            logger.info("Análise concluída com sucesso")
            return result
        
        except Exception as e:
            logger.error(f"Erro na análise de documentos: {e}")
            raise
    
    def _extract_info_with_regex(self, text: str) -> Dict[str, Any]:
        """Extrai informações usando regex como fallback"""
        import re
        
        result = self._create_empty_result()
        
        try:
            # Padrões regex para extração básica
            patterns = {
                "Empresa responsável": r"(?:empresa|responsável|projetista)[:\s]*([^\n]{1,100})",
                "Cliente": r"(?:cliente|proprietário|titular)[:\s]*([^\n]{1,100})",
                "Local": r"(?:local|endereço|localização)[:\s]*([^\n]{1,150})",
                "Potência do sistema": r"(?:potência.*sistema|sistema.*potência)[:\s]*([0-9,.\s]*k?[Ww]p?)",
                "Potência nominal do inversor": r"(?:potência.*inversor|inversor.*potência)[:\s]*([0-9,.\s]*k?[Ww])",
                "Modelo do inversor": r"(?:modelo.*inversor|inversor.*modelo)[:\s]*([^\n]{1,50})",
                "Quantidade de inversores": r"(?:quantidade.*inversor|inversor.*quantidade)[:\s]*([0-9]+)",
                "Potência dos módulos": r"(?:potência.*módulo|módulo.*potência)[:\s]*([0-9,.\s]*[Ww])",
                "Quantidade de módulos": r"(?:quantidade.*módulo|módulo.*quantidade)[:\s]*([0-9]+)",
                "Modelo dos módulos": r"(?:modelo.*módulo|módulo.*modelo)[:\s]*([^\n]{1,50})",
                "Tensão da UC": r"(?:tensão)[:\s]*([0-9]+\s*[Vv])",
                "Disjuntor de entrada da UC": r"(?:disjuntor.*entrada)[:\s]*([0-9]+\s*[Aa])",
                "Disjuntor de proteção do sistema FV (saída do inversor)": r"(?:disjuntor.*proteção|disjuntor.*saída)[:\s]*([0-9]+\s*[Aa])",
                "Potência disponibilizada (PD)": r"(?:potência disponibilizada|PD)[:\s]*([0-9,.\s]*k?[Ww])"
            }
            
            for field, pattern in patterns.items():
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    result[field] = matches[0].strip()
            
            logger.info("Informações extraídas com regex")
            
        except Exception as e:
            logger.error(f"Erro na extração com regex: {e}")
        
        return result
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """Cria resultado vazio com estrutura correta"""
        return {
            "Empresa responsável": "",
            "Cliente": "",
            "Local": "",
            "Potência do sistema": "",
            "Potência nominal do inversor": "",
            "Modelo do inversor": "",
            "Quantidade de inversores": "",
            "Potência dos módulos": "",
            "Quantidade de módulos": "",
            "Modelo dos módulos": "",
            "Tensão da UC": "",
            "Disjuntor de entrada da UC": "",
            "Disjuntor de proteção do sistema FV (saída do inversor)": "",
            "Potência disponibilizada (PD)": "",
            "Incongruências encontradas": []
        }

# Instância global do sistema RAG
rag_system = None

def get_rag_system():
    """Obtém instância do sistema RAG (singleton)"""
    global rag_system
    if rag_system is None:
        rag_system = RAGSystem()
    return rag_system
