"""
Script para gerar embeddings de PDF usando OpenAI e armazenar no ChromaDB
"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
import PyPDF2
from openai import OpenAI
import logging
from typing import List, Dict
import hashlib

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/embeddings.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PDFEmbeddingGenerator:
    def __init__(self):
        """Inicializa o gerador de embeddings"""
        # Carregar variáveis de ambiente
        load_dotenv()
        
        # Carregar configuração
        self.config = self._load_config()
        
        # Configurar cliente OpenAI
        self.openai_client = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url=self.config['ai_providers']['openai']['base_url']
        )
        
        # Configurar ChromaDB
        self.chroma_client = self._setup_chromadb()
        
        # Configurações
        self.embedding_model = self.config['ai_providers']['openai']['models']['embedding']
        self.collection_name = os.getenv('COLLECTION_NAME', 'documents')
        self.chunk_size = 1000  # Tamanho do chunk de texto
        self.chunk_overlap = 200  # Sobreposição entre chunks
        
    def _load_config(self) -> Dict:
        """Carrega configuração do arquivo YAML"""
        try:
            with open('config.yaml', 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Erro ao carregar config.yaml: {e}")
            raise
    
    def _setup_chromadb(self) -> chromadb.Client:
        """Configura e retorna cliente ChromaDB"""
        try:
            # Criar diretório para o banco de dados se não existir
            db_path = Path(os.getenv('VECTOR_DB_PATH', './data/embeddings'))
            db_path.mkdir(parents=True, exist_ok=True)
            
            # Configurar cliente ChromaDB
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
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extrai texto do PDF"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        text += f"\n--- Página {page_num + 1} ---\n"
                        text += page_text
                    except Exception as e:
                        logger.warning(f"Erro ao extrair texto da página {page_num + 1}: {e}")
                        continue
                
                logger.info(f"Texto extraído do PDF: {len(text)} caracteres")
                return text
                
        except Exception as e:
            logger.error(f"Erro ao extrair texto do PDF {pdf_path}: {e}")
            raise
    
    def split_text_into_chunks(self, text: str) -> List[str]:
        """Divide o texto em chunks"""
        chunks = []
        
        # Dividir por parágrafos primeiro
        paragraphs = text.split('\n\n')
        current_chunk = ""
        
        for paragraph in paragraphs:
            # Se adicionar este parágrafo não exceder o tamanho do chunk
            if len(current_chunk) + len(paragraph) <= self.chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                # Se o chunk atual não está vazio, adiciona à lista
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # Se o parágrafo é muito grande, divide em sentenças
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
        
        # Adicionar o último chunk se não estiver vazio
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Aplicar sobreposição
        overlapped_chunks = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
            else:
                # Adicionar sobreposição do chunk anterior
                prev_chunk = chunks[i-1]
                overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk
                overlapped_chunk = overlap_text + "\n\n" + chunk
                overlapped_chunks.append(overlapped_chunk)
        
        logger.info(f"Texto dividido em {len(overlapped_chunks)} chunks")
        return overlapped_chunks
    
    def generate_embedding(self, text: str) -> List[float]:
        """Gera embedding para um texto usando OpenAI"""
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Erro ao gerar embedding: {e}")
            raise
    
    def create_document_id(self, text: str, chunk_index: int) -> str:
        """Cria um ID único para o documento baseado no hash do texto"""
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        return f"doc_{text_hash}_chunk_{chunk_index}"
    
    def process_pdf_and_store_embeddings(self, pdf_path: str) -> None:
        """Processa o PDF e armazena embeddings no ChromaDB"""
        try:
            logger.info(f"Iniciando processamento do PDF: {pdf_path}")
            
            # Verificar se o arquivo existe
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"Arquivo PDF não encontrado: {pdf_path}")
            
            # Extrair texto do PDF
            text = self.extract_text_from_pdf(pdf_path)
            if not text.strip():
                raise ValueError("Nenhum texto extraído do PDF")
            
            # Dividir em chunks
            chunks = self.split_text_into_chunks(text)
            if not chunks:
                raise ValueError("Nenhum chunk gerado do texto")
            
            # Obter ou criar coleção
            try:
                collection = self.chroma_client.get_collection(name=self.collection_name)
                logger.info(f"Coleção existente '{self.collection_name}' encontrada")
            except:
                collection = self.chroma_client.create_collection(name=self.collection_name)
                logger.info(f"Nova coleção '{self.collection_name}' criada")
            
            # Processar cada chunk
            successful_embeddings = 0
            failed_embeddings = 0
            
            for i, chunk in enumerate(chunks):
                try:
                    logger.info(f"Processando chunk {i+1}/{len(chunks)}")
                    
                    # Gerar embedding
                    embedding = self.generate_embedding(chunk)
                    
                    # Criar ID único
                    doc_id = self.create_document_id(chunk, i)
                    
                    # Metadados
                    metadata = {
                        "source": os.path.basename(pdf_path),
                        "chunk_index": i,
                        "chunk_size": len(chunk),
                        "total_chunks": len(chunks),
                        "file_path": pdf_path
                    }
                    
                    # Adicionar à coleção
                    collection.add(
                        embeddings=[embedding],
                        documents=[chunk],
                        metadatas=[metadata],
                        ids=[doc_id]
                    )
                    
                    successful_embeddings += 1
                    logger.info(f"Chunk {i+1} processado com sucesso")
                    
                except Exception as e:
                    failed_embeddings += 1
                    logger.error(f"Erro ao processar chunk {i+1}: {e}")
                    continue
            
            # Resumo do processamento
            logger.info(f"Processamento concluído!")
            logger.info(f"Embeddings criados com sucesso: {successful_embeddings}")
            logger.info(f"Embeddings com falha: {failed_embeddings}")
            logger.info(f"Total de chunks: {len(chunks)}")
            
            # Verificar coleção
            collection_count = collection.count()
            logger.info(f"Total de documentos na coleção '{self.collection_name}': {collection_count}")
            
        except Exception as e:
            logger.error(f"Erro no processamento do PDF: {e}")
            raise

def main():
    """Função principal"""
    try:
        # Caminho do PDF
        pdf_path = "data/documents/NT.00020.EQTL-05-Conexao-de-Micro-e-Minigeracao-Distribuida-ao-Sistema-de-Distribuicao.pdf"
        
        # Verificar se o arquivo existe
        if not os.path.exists(pdf_path):
            logger.error(f"Arquivo PDF não encontrado: {pdf_path}")
            return
        
        # Criar diretório de logs se não existir
        os.makedirs('logs', exist_ok=True)
        
        # Inicializar gerador de embeddings
        generator = PDFEmbeddingGenerator()
        
        # Processar PDF e gerar embeddings
        generator.process_pdf_and_store_embeddings(pdf_path)
        
        logger.info("Script executado com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro na execução do script: {e}")
        raise

if __name__ == "__main__":
    main()
