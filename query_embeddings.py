"""
Script para consultar embeddings armazenados no ChromaDB
"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from openai import OpenAI
import logging
from typing import List, Dict, Optional

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmbeddingQuerier:
    def __init__(self):
        """Inicializa o consultor de embeddings"""
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
        
        # Obter coleção
        try:
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
            logger.info(f"Coleção '{self.collection_name}' carregada com sucesso")
        except Exception as e:
            logger.error(f"Erro ao carregar coleção '{self.collection_name}': {e}")
            raise
    
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
            
            if not db_path.exists():
                raise FileNotFoundError(f"Banco de dados ChromaDB não encontrado em: {db_path}")
            
            # Configurar cliente ChromaDB
            client = chromadb.PersistentClient(
                path=str(db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            logger.info(f"ChromaDB carregado de: {db_path}")
            return client
            
        except Exception as e:
            logger.error(f"Erro ao configurar ChromaDB: {e}")
            raise
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """Gera embedding para uma consulta usando OpenAI"""
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=query
            )
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Erro ao gerar embedding da consulta: {e}")
            raise
    
    def search_similar_documents(self, query: str, n_results: int = 5) -> Dict:
        """Busca documentos similares baseado na consulta"""
        try:
            logger.info(f"Buscando documentos similares para: '{query}'")
            
            # Gerar embedding da consulta
            query_embedding = self.generate_query_embedding(query)
            
            # Buscar documentos similares
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Formatar resultados
            formatted_results = []
            for i in range(len(results['documents'][0])):
                result = {
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'similarity_score': 1 - results['distances'][0][i]  # Converter distância para similaridade
                }
                formatted_results.append(result)
            
            logger.info(f"Encontrados {len(formatted_results)} documentos similares")
            return {
                'query': query,
                'results': formatted_results,
                'total_found': len(formatted_results)
            }
            
        except Exception as e:
            logger.error(f"Erro na busca por documentos similares: {e}")
            raise
    
    def get_collection_stats(self) -> Dict:
        """Obtém estatísticas da coleção"""
        try:
            count = self.collection.count()
            
            # Obter alguns documentos para análise
            sample_results = self.collection.get(
                limit=5,
                include=['documents', 'metadatas']
            )
            
            # Estatísticas básicas
            stats = {
                'total_documents': count,
                'collection_name': self.collection_name,
                'sample_documents': []
            }
            
            # Adicionar documentos de exemplo
            for i in range(len(sample_results['documents'])):
                sample_doc = {
                    'id': sample_results['ids'][i],
                    'metadata': sample_results['metadatas'][i],
                    'document_preview': sample_results['documents'][i][:200] + "..." if len(sample_results['documents'][i]) > 200 else sample_results['documents'][i]
                }
                stats['sample_documents'].append(sample_doc)
            
            return stats
            
        except Exception as e:
            logger.error(f"Erro ao obter estatísticas da coleção: {e}")
            raise
    
    def search_by_metadata(self, metadata_filter: Dict, n_results: int = 10) -> List[Dict]:
        """Busca documentos baseado em metadados"""
        try:
            logger.info(f"Buscando documentos com filtro de metadados: {metadata_filter}")
            
            results = self.collection.get(
                where=metadata_filter,
                limit=n_results,
                include=['documents', 'metadatas']
            )
            
            # Formatar resultados
            formatted_results = []
            for i in range(len(results['documents'])):
                result = {
                    'id': results['ids'][i],
                    'document': results['documents'][i],
                    'metadata': results['metadatas'][i]
                }
                formatted_results.append(result)
            
            logger.info(f"Encontrados {len(formatted_results)} documentos com o filtro especificado")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Erro na busca por metadados: {e}")
            raise

def main():
    """Função principal para demonstrar o uso"""
    try:
        # Inicializar consultor
        querier = EmbeddingQuerier()
        
        # Obter estatísticas da coleção
        print("=== ESTATÍSTICAS DA COLEÇÃO ===")
        stats = querier.get_collection_stats()
        print(f"Nome da coleção: {stats['collection_name']}")
        print(f"Total de documentos: {stats['total_documents']}")
        print()
        
        # Exemplo de busca semântica
        print("=== BUSCA SEMÂNTICA ===")
        query = "conexão de microgeração distribuída"
        results = querier.search_similar_documents(query, n_results=3)
        
        print(f"Consulta: {results['query']}")
        print(f"Documentos encontrados: {results['total_found']}")
        print()
        
        for i, result in enumerate(results['results'], 1):
            print(f"--- Resultado {i} ---")
            print(f"Similaridade: {result['similarity_score']:.4f}")
            print(f"Chunk: {result['metadata']['chunk_index']}")
            print(f"Fonte: {result['metadata']['source']}")
            print(f"Texto: {result['document'][:300]}...")
            print()
        
        # Exemplo de busca por metadados
        print("=== BUSCA POR METADADOS ===")
        metadata_filter = {"source": "NT.00020.EQTL-05-Conexao-de-Micro-e-Minigeracao-Distribuida-ao-Sistema-de-Distribuicao.pdf"}
        metadata_results = querier.search_by_metadata(metadata_filter, n_results=3)
        
        print(f"Documentos encontrados com filtro de metadados: {len(metadata_results)}")
        for i, result in enumerate(metadata_results, 1):
            print(f"--- Documento {i} ---")
            print(f"ID: {result['id']}")
            print(f"Chunk: {result['metadata']['chunk_index']}")
            print(f"Texto: {result['document'][:200]}...")
            print()
        
        logger.info("Consulta executada com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro na execução da consulta: {e}")
        raise

if __name__ == "__main__":
    main()
