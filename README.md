# 🏗️ Sistema RAG para Análise de Projetos de Microgeração Distribuída

Sistema inteligente para análise automatizada de documentos de projetos de energia solar com base na norma técnica NT.00020.EQTL-05 da Equatorial Energia.

## 🎯 Objetivo

Automatizar a análise de conformidade de projetos de microgeração distribuída, identificando incongruências técnicas e extraindo informações estruturadas dos documentos.

## 🔧 Arquitetura do Sistema

```mermaid
graph TB
    subgraph "Frontend"
        UI[Interface Web] --> Upload[Upload de Documentos]
        Upload --> Display[Exibição de Resultados]
    end
    
    subgraph "API Layer"
        FastAPI[FastAPI Server]
        FastAPI --> Health[/api/health]
        FastAPI --> Status[/api/status] 
        FastAPI --> Analyze[/api/analyze]
    end
    
    subgraph "RAG System Core"
        RAG[Sistema RAG]
        RAG --> AIManager[AI Provider Manager]
        RAG --> DocProcessor[Document Processor]
        RAG --> Memory[Processamento em Memória]
    end
    
    subgraph "AI Providers"
        OpenAI[OpenAI API]
        DeepSeek[DeepSeek API]
        AIManager --> OpenAI
        AIManager --> DeepSeek
    end
    
    subgraph "Storage"
        ChromaDB[(ChromaDB<br/>NT.00020 Embeddings)]
        InMemory[Memória<br/>Project Embeddings]
    end
    
    subgraph "Documents"
        Norm[Norma NT.00020<br/>📋 Permanente]
        Project[Documentos do Projeto<br/>📄 Temporário]
    end
    
    UI --> FastAPI
    FastAPI --> RAG
    RAG --> ChromaDB
    RAG --> InMemory
    Norm --> ChromaDB
    Project --> Memory
    DocProcessor --> Memory
```

## 🧠 Fluxo de Processamento

### 1. **Inicialização do Sistema**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Load Config    │ →  │  Setup ChromaDB │ →  │ Initialize AI   │
│  (config.yaml)  │    │  (NT.00020)     │    │  Providers      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 2. **Processamento de Documentos**
```
📄 Upload de Documentos
    ↓
🔍 Extração de Texto (PDF/DOCX/TXT)
    ↓
✂️ Divisão em Chunks
    ↓
🤖 Geração de Embeddings (OpenAI/DeepSeek)
    ↓
💾 Armazenamento em Memória (Temporário)
```

### 3. **Análise RAG**
```
🔍 Query: "Análise de conformidade"
    ↓
┌─────────────────────────────────────────────────────────┐
│                    RAG Analysis                          │
├─────────────────────┬───────────────────────────────────┤
│  📋 Contexto da     │  📄 Contexto do Projeto          │
│     Norma NT.00020  │     (Documentos Upload)          │
│  (ChromaDB)         │  (Memória)                        │
└─────────────────────┴───────────────────────────────────┘
    ↓
🤖 Análise por IA (OpenAI/DeepSeek)
    ↓
📊 JSON Estruturado + Incongruências
```

## 🏛️ Estrutura do Projeto

```
eqtl-gd-smartcheck/
├── 📁 src/
│   └── 📁 static/
│       └── 📄 index.html          # Interface web
├── 📁 data/
│   ├── 📁 documents/
│   │   └── 📄 NT.00020.pdf        # Norma técnica
│   └── 📁 embeddings/             # ChromaDB storage
├── 📁 logs/                       # Logs do sistema
├── 📄 app.py                      # FastAPI server
├── 📄 rag_system.py              # Sistema RAG core
├── 📄 generate_embeddings.py     # Script para gerar embeddings da norma
├── 📄 test_system.py             # Testes automatizados
├── 📄 config.yaml                # Configurações
├── 📄 .env                       # Variáveis de ambiente
└── 📄 requirements.txt           # Dependências
```

## ⚙️ Componentes Técnicos

### 🤖 **AI Provider Manager**
- **Função**: Gerencia provedores de IA de forma agnóstica
- **Suporte**: OpenAI e DeepSeek APIs
- **Recursos**: 
  - Geração de embeddings
  - Análise de texto via chat completion
  - Fallback entre provedores

### 📄 **Document Processor**
- **Formatos Suportados**: PDF, DOCX, TXT
- **Bibliotecas**: PyPDF2, python-docx
- **Processamento**: Extração de texto, divisão em chunks

### 🗄️ **Storage Strategy**
- **Norma NT.00020**: ChromaDB (persistente)
- **Documentos do Projeto**: Memória (temporário)
- **Vantagem**: Performance + economia de storage

### 🔍 **RAG (Retrieval-Augmented Generation)**
- **Busca**: Similarity search no ChromaDB
- **Contexto**: Combinação norma + projeto
- **Output**: JSON estruturado com incongruências

## 🛠️ Instalação e Configuração

### 1. **Pré-requisitos**
```bash
Python 3.8+
Git
```

### 2. **Clonagem e Setup**
```bash
git clone <repository>
cd eqtl-gd-smartcheck
```

### 3. **Ambiente Virtual**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

### 4. **Instalação de Dependências**
```bash
pip install -r requirements.txt
```

### 5. **Configuração de Variáveis de Ambiente**
Crie/edite o arquivo `.env`:
```env
# AI Provider Configuration
OPENAI_API_KEY=sua_chave_openai_aqui
DEEPSEEK_API_KEY=sua_chave_deepseek_aqui

# Embedding Configuration
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
```

### 6. **Geração de Embeddings da Norma**
```bash
python generate_embeddings.py
```

### 7. **Inicialização do Servidor**
```bash
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

### 8. **Acesso ao Sistema**
```
http://127.0.0.1:8000
```

## 🧪 Testes

### **Execução de Testes**
```bash
python test_system.py
```

### **Testes Disponíveis**
- ✅ **Health Check**: Verificação de saúde do sistema
- ✅ **Status**: Estado dos provedores AI e base de dados
- ✅ **Análise**: Teste completo com documento exemplo

## 📋 Formato de Resposta

### **Estrutura JSON de Saída**
```json
{
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
  "Incongruências encontradas": [
    "Descrição da incongruência 1",
    "Descrição da incongruência 2"
  ]
}
```

## 🔧 APIs Disponíveis

### **GET /api/health**
Verifica saúde do sistema
```json
{
  "status": "healthy",
  "ai_providers": ["openai", "deepseek"],
  "collection_count": 308
}
```

### **GET /api/status**
Status detalhado do sistema
```json
{
  "ai_providers": ["openai", "deepseek"],
  "collection_name": "documents", 
  "total_documents": 308,
  "vector_db_path": "./data/embeddings"
}
```

### **POST /api/analyze**
Análise de documentos
- **Input**: Multipart form com arquivos
- **Output**: JSON estruturado + incongruências

## 🎯 Exemplos de Incongruências Detectadas

1. **Potência do Sistema vs Inversor**
   - Sistema: 13,09 kWp
   - Inversor: 10,0 kW
   - ❌ Sistema maior que capacidade do inversor

2. **Potência Disponibilizada vs Nominal**
   - PD: 12 kW
   - Nominal: 10,0 kW
   - ❌ PD incompatível com inversor

3. **Dados Inconsistentes**
   - Nomes diferentes entre documentos
   - Modelos incorretos
   - Valores conflitantes

## 🚀 Tecnologias Utilizadas

### **Backend**
- **FastAPI**: Framework web moderno
- **ChromaDB**: Vector database
- **OpenAI/DeepSeek**: APIs de IA
- **PyPDF2/python-docx**: Processamento de documentos

### **Frontend**
- **HTML5 + JavaScript**: Interface web
- **Tailwind CSS**: Estilização
- **Fetch API**: Comunicação com backend

### **AI/ML**
- **RAG (Retrieval-Augmented Generation)**: Arquitetura principal
- **Text Embeddings**: Busca semântica
- **Large Language Models**: Análise e extração

## 📊 Performance

### **Processamento**
- ⚡ Embeddings da norma: Pré-processados (uma vez)
- ⚡ Documentos do projeto: Processamento em memória
- ⚡ Análise: ~10-30 segundos por projeto

### **Capacidade**
- 📄 Até 10 documentos por análise
- 📋 Suporte a PDF, DOCX, TXT
- 🗄️ Base persistente com 308+ chunks da norma

## 🔐 Segurança

- 🔑 Chaves de API protegidas via variáveis de ambiente
- 🗃️ Documentos do projeto não armazenados permanentemente
- 🔒 Processamento local dos embeddings

## 🐛 Troubleshooting

### **Problema: Provedores AI vazios**
```bash
# Verificar variáveis de ambiente
python -c "import os; print('OpenAI:', bool(os.getenv('OPENAI_API_KEY')))"
```

### **Problema: ChromaDB não encontra documentos**
```bash
# Regenerar embeddings da norma
python generate_embeddings.py
```

### **Problema: Servidor não inicia**
```bash
# Verificar dependências
pip install -r requirements.txt

# Verificar porta
netstat -an | findstr :8000
```

## 🤝 Contribuição

1. Fork do projeto
2. Criar branch para feature (`git checkout -b feature/nova-funcionalidade`)
3. Commit das mudanças (`git commit -m 'Adiciona nova funcionalidade'`)
4. Push para branch (`git push origin feature/nova-funcionalidade`)
5. Criar Pull Request

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 📞 Suporte

Para dúvidas ou problemas:
- 📧 Email: suporte@equatorialenergia.com.br
- 📋 Issues: GitHub Issues
- 📖 Documentação: Este README

---

**Desenvolvido com ❤️ para Equatorial Energia - Geração Distribuída**
