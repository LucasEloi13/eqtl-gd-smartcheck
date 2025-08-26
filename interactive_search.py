"""
Interface interativa para consultar embeddings do documento PDF
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

class InteractiveEmbeddingSearch:
    def __init__(self):
        """Inicializa o sistema de busca interativo"""
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
            logger.info(f"Total de documentos na coleção: {self.collection.count()}")
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
            db_path = Path(os.getenv('VECTOR_DB_PATH', './data/embeddings'))
            
            if not db_path.exists():
                raise FileNotFoundError(f"Banco de dados ChromaDB não encontrado em: {db_path}")
            
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
    
    def search_documents(self, query: str, n_results: int = 5, min_similarity: float = 0.0) -> List[Dict]:
        """Busca documentos similares baseado na consulta"""
        try:
            # Gerar embedding da consulta
            query_embedding = self.generate_query_embedding(query)
            
            # Buscar documentos similares
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Formatar e filtrar resultados
            formatted_results = []
            for i in range(len(results['documents'][0])):
                similarity_score = 1 - results['distances'][0][i]
                
                # Filtrar por similaridade mínima
                if similarity_score >= min_similarity:
                    result = {
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i],
                        'similarity_score': similarity_score
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Erro na busca por documentos similares: {e}")
            raise
    
    def get_context_for_question(self, question: str, max_results: int = 5) -> str:
        """Obtém contexto relevante para uma pergunta"""
        try:
            # Buscar documentos relevantes
            results = self.search_documents(question, n_results=max_results, min_similarity=0.2)
            
            if not results:
                return "Nenhum contexto relevante encontrado para sua pergunta."
            
            # Construir contexto
            context_parts = []
            for i, result in enumerate(results, 1):
                context_part = f"[Trecho {i} - Similaridade: {result['similarity_score']:.3f}]\n"
                context_part += f"Página/Chunk: {result['metadata'].get('chunk_index', 'N/A')}\n"
                context_part += f"Conteúdo: {result['document']}\n"
                context_parts.append(context_part)
            
            return "\n" + "="*80 + "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Erro ao obter contexto: {e}")
            return f"Erro ao buscar contexto: {str(e)}"
    
    def search_by_keywords(self, keywords: List[str], n_results: int = 10) -> List[Dict]:
        """Busca documentos que contêm palavras-chave específicas"""
        try:
            all_results = []
            
            for keyword in keywords:
                # Buscar por cada palavra-chave
                results = self.search_documents(keyword, n_results=n_results//len(keywords))
                all_results.extend(results)
            
            # Remover duplicatas e ordenar por similaridade
            seen_ids = set()
            unique_results = []
            
            for result in all_results:
                # Usar chunk_index e similaridade como identificador único
                result_id = f"{result['metadata']['chunk_index']}_{result['similarity_score']:.6f}"
                if result_id not in seen_ids:
                    seen_ids.add(result_id)
                    unique_results.append(result)
            
            # Ordenar por similaridade decrescente
            unique_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return unique_results[:n_results]
            
        except Exception as e:
            logger.error(f"Erro na busca por palavras-chave: {e}")
            raise
    
    def interactive_search(self):
        """Interface interativa para busca"""
        print("=== SISTEMA DE BUSCA INTERATIVO ===")
        print("Digite suas consultas ou 'quit' para sair")
        print("Comandos especiais:")
        print("  'stats' - Ver estatísticas da coleção")
        print("  'help' - Ver esta ajuda")
        print()
        
        while True:
            try:
                query = input("Consulta: ").strip()
                
                if query.lower() == 'quit':
                    print("Saindo...")
                    break
                
                elif query.lower() == 'stats':
                    count = self.collection.count()
                    print(f"Total de documentos na coleção: {count}")
                    continue
                
                elif query.lower() == 'help':
                    print("Comandos:")
                    print("  'stats' - Estatísticas da coleção")
                    print("  'quit' - Sair do programa")
                    print("  Qualquer texto - Buscar documentos similares")
                    continue
                
                elif not query:
                    continue
                
                # Realizar busca
                print(f"\nBuscando por: '{query}'")
                results = self.search_documents(query, n_results=3, min_similarity=0.2)
                
                if not results:
                    print("Nenhum resultado encontrado com similaridade suficiente.")
                    continue
                
                print(f"\nEncontrados {len(results)} resultados:")
                print("="*80)
                
                for i, result in enumerate(results, 1):
                    print(f"\n--- Resultado {i} ---")
                    print(f"Similaridade: {result['similarity_score']:.4f}")
                    print(f"Chunk: {result['metadata']['chunk_index']}")
                    print(f"Fonte: {result['metadata']['source']}")
                    print(f"Tamanho do chunk: {result['metadata']['chunk_size']} caracteres")
                    print(f"Total de chunks: {result['metadata']['total_chunks']}")
                    print(f"\nConteúdo:")
                    # Mostrar apenas os primeiros 500 caracteres
                    content = result['document']
                    if len(content) > 500:
                        print(content[:500] + "...")
                    else:
                        print(content)
                    print("-" * 80)
                
            except KeyboardInterrupt:
                print("\n\nSaindo...")
                break
            except Exception as e:
                print(f"Erro: {str(e)}")
                continue

def main():
    """Função principal"""
    try:
        # Inicializar sistema de busca
        search_system = InteractiveEmbeddingSearch()
        
        # Executar interface interativa
        search_system.interactive_search()
        
    except Exception as e:
        logger.error(f"Erro na execução: {e}")
        print(f"Erro ao inicializar o sistema: {str(e)}")

if __name__ == "__main__":
    main()
