"""
Script de teste para verificar a funcionalidade do sistema RAG
"""

import requests
import json
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_health_check():
    """Testa o health check da API"""
    try:
        response = requests.get("http://127.0.0.1:8000/api/health", timeout=10)
        print("=== HEALTH CHECK ===")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Erro no health check: {e}")
        return False

def test_status():
    """Testa o status da API"""
    try:
        response = requests.get("http://127.0.0.1:8000/api/status", timeout=10)
        print("\n=== STATUS ===")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Erro no status: {e}")
        return False

def test_analyze_with_sample():
    """Testa análise com arquivo de exemplo"""
    try:
        # Criar um arquivo de teste simples
        test_content = """
        PROJETO DE MICROGERAÇÃO DISTRIBUÍDA
        
        Empresa responsável: FELIX ASSESSORIA & SERVIÇOS INTEGRADOS LTDA
        Engenheiro: Matheus Pinheiro da Silva Félix – CREA-PA 151998978-4
        
        Cliente: Milton Ferreira de Sousa
        Local: Travessa João Pessoa, 400 – Centro – Olho D'Água das Cunhãs – MA
        
        ESPECIFICAÇÕES TÉCNICAS:
        Potência do sistema: 13,09 kWp
        Potência nominal do inversor: 10,0 kW
        Modelo do inversor: SAJ R6-10K-S3-18
        Quantidade de inversores: 1
        
        Potência dos módulos: 595 W
        Quantidade de módulos: 22
        Modelo dos módulos: TSUN POWER TS595S8E-144GANT
        
        Tensão da UC: 220 V monofásico
        Disjuntor de entrada da UC: 63 A
        Disjuntor de proteção do sistema FV: 50 A
        Potência disponibilizada (PD): 12 kW
        """
        
        files = {
            'files': ('projeto_teste.txt', test_content.encode('utf-8'), 'text/plain')
        }
        
        print("\n=== TESTE DE ANÁLISE ===")
        print("Enviando arquivo de teste...")
        
        response = requests.post("http://127.0.0.1:8000/api/analyze", files=files, timeout=30)
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Análise bem-sucedida!")
            print(f"Response: {json.dumps(result, indent=2, ensure_ascii=False)}")
            return True
        else:
            print(f"❌ Erro na análise: {response.text}")
            return False
            
    except Exception as e:
        print(f"Erro no teste de análise: {e}")
        return False

def main():
    """Executa todos os testes"""
    print("🧪 INICIANDO TESTES DO SISTEMA RAG")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health_check),
        ("Status", test_status),
        ("Análise de Documento", test_analyze_with_sample)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 Executando: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASSOU" if result else "❌ FALHOU"
            print(f"Resultado: {status}")
        except Exception as e:
            print(f"❌ ERRO: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 RESUMO DOS TESTES")
    print("=" * 50)
    
    for test_name, passed in results:
        status = "✅" if passed else "❌"
        print(f"{status} {test_name}")
    
    total_passed = sum(1 for _, passed in results if passed)
    print(f"\nTotal: {total_passed}/{len(results)} testes passaram")
    
    if total_passed == len(results):
        print("🎉 TODOS OS TESTES PASSARAM!")
    else:
        print("⚠️  Alguns testes falharam")

if __name__ == "__main__":
    main()
