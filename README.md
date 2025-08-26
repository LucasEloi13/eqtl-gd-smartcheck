# Sistema de Embeddings para Documentos PDF

Este projeto contém um sistema completo para gerar e consultar embeddings de documentos PDF usando OpenAI e ChromaDB.

## Arquivos Principais

### 1. `generate_embeddings.py`
Script principal para gerar embeddings do PDF e armazenar no ChromaDB.

**Funcionalidades:**
- Extrai texto do PDF localizado em `data/documents/`
- Divide o texto em chunks de 1000 caracteres com sobreposição de 200 caracteres
- Gera embeddings usando o modelo `text-embedding-3-small` da OpenAI
- Armazena os embeddings no ChromaDB com metadados

**Como executar:**
```bash
python generate_embeddings.py
```

### 2. `query_embeddings.py`
Script para consultar os embeddings armazenados.

**Funcionalidades:**
- Busca semântica por similaridade
- Busca por metadados
- Estatísticas da coleção

**Como executar:**
```bash
python query_embeddings.py
```

### 3. `interactive_search.py`
Interface interativa para realizar buscas nos embeddings.

**Funcionalidades:**
- Interface de linha de comando interativa
- Busca por palavras-chave
- Contexto para perguntas
- Comandos especiais (stats, help, quit)

**Como executar:**
```bash
python interactive_search.py
```

## Configuração

### Arquivos de Configuração

#### `.env`
Contém as variáveis de ambiente necessárias:
```
AI_PROVIDER=openai
OPENAI_API_KEY=sua_chave_aqui
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
VECTOR_DB_PATH=./data/embeddings
COLLECTION_NAME=documents
```

#### `config.yaml`
Configurações dos provedores de AI:
```yaml
ai_providers:
  openai:
    base_url: "https://api.openai.com/v1"
    models:
      chat: "gpt-4o-mini"
      embedding: "text-embedding-3-small"
    max_tokens: 4096
    temperature: 0.5
```

## Estrutura de Dados

### ChromaDB
Os embeddings são armazenados no diretório `data/embeddings/` com a seguinte estrutura:

**Metadados por chunk:**
- `source`: Nome do arquivo PDF
- `chunk_index`: Índice do chunk no documento
- `chunk_size`: Tamanho do chunk em caracteres
- `total_chunks`: Total de chunks do documento
- `file_path`: Caminho completo do arquivo

### Documento Processado
- **Arquivo:** `NT.00020.EQTL-05-Conexao-de-Micro-e-Minigeracao-Distribuida-ao-Sistema-de-Distribuicao.pdf`
- **Total de chunks:** 236
- **Caracteres extraídos:** 200,276
- **Modelo de embedding:** text-embedding-3-small
- **Dimensões:** 1536

## Logs

Os logs são salvos em `logs/embeddings.log` e incluem:
- Informações sobre o processamento
- Estatísticas de chunks criados
- Errors e warnings
- Requisições HTTP para a API da OpenAI

## Dependências

```
chromadb
PyPDF2
openai
python-dotenv
pyyaml
langchain
langchain-openai
langchain-community
```

## Exemplos de Uso

### Busca Semântica
```python
from query_embeddings import EmbeddingQuerier

querier = EmbeddingQuerier()
results = querier.search_similar_documents("conexão de microgeração distribuída", n_results=3)
```

### Busca por Metadados
```python
metadata_filter = {"source": "NT.00020.EQTL-05-Conexao-de-Micro-e-Minigeracao-Distribuida-ao-Sistema-de-Distribuicao.pdf"}
results = querier.search_by_metadata(metadata_filter, n_results=5)
```

### Interface Interativa
```bash
python interactive_search.py
# Digite consultas como:
# - "requisitos para conexão"
# - "proteção elétrica"
# - "normas técnicas"
```

## Estatísticas do Processamento

- ✅ **236 embeddings** criados com sucesso
- ✅ **0 falhas** no processamento
- ✅ **236 chunks** processados
- ✅ **Total de documentos** na coleção: 236

## Comandos de Terminal Úteis

### Gerar embeddings:
```bash
& "C:/Users/u12283/Documents/LUCAS ELOI/eqtl-gd-smartcheck/.venv/Scripts/python.exe" generate_embeddings.py
```

### Consultar embeddings:
```bash
& "C:/Users/u12283/Documents/LUCAS ELOI/eqtl-gd-smartcheck/.venv/Scripts/python.exe" query_embeddings.py
```

### Busca interativa:
```bash
& "C:/Users/u12283/Documents/LUCAS ELOI/eqtl-gd-smartcheck/.venv/Scripts/python.exe" interactive_search.py
```

## Observações

1. **API Key**: Certifique-se de que a chave da OpenAI está configurada corretamente no arquivo `.env`
2. **Modelo**: O sistema usa especificamente o modelo `text-embedding-3-small` conforme especificado
3. **Persistência**: Os embeddings são salvos permanentemente no ChromaDB e podem ser reutilizados
4. **Performance**: O processamento inicial pode demorar alguns minutos devido às chamadas da API
5. **Custos**: Cada embedding gera uma requisição para a API da OpenAI

## Próximos Passos

- Implementar cache para embeddings de consultas frequentes
- Adicionar suporte para múltiplos documentos
- Criar interface web para consultas
- Implementar sistema de RAG (Retrieval Augmented Generation)
