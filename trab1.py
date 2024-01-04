import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Variáveis de entrada
retorno_anual = ctrl.Antecedent(np.arange(0, 11, 1), 'retorno_anual')
volatilidade_ativo = ctrl.Antecedent(np.arange(0, 11, 1), 'volatilidade_ativo')
relacao_lucro_prejuizo = ctrl.Antecedent(np.arange(0, 11, 1), 'relacao_lucro_prejuizo')
analise_analistas = ctrl.Antecedent(np.arange(0, 11, 1), 'analise_analistas')

# Variável de saída
decisao_investimento = ctrl.Consequent(np.arange(0, 101, 1), 'decisao_investimento')

# Funções de pertinência para as variáveis de entrada e saída
retorno_anual['baixo'] = fuzz.trimf(retorno_anual.universe, [0, 0, 5])
retorno_anual['medio'] = fuzz.trimf(retorno_anual.universe, [0, 5, 10])
retorno_anual['alto'] = fuzz.trimf(retorno_anual.universe, [5, 10, 10])

volatilidade_ativo['baixa'] = fuzz.trimf(volatilidade_ativo.universe, [0, 0, 5])
volatilidade_ativo['media'] = fuzz.trimf(volatilidade_ativo.universe, [0, 5, 10])
volatilidade_ativo['alta'] = fuzz.trimf(volatilidade_ativo.universe, [5, 10, 10])

relacao_lucro_prejuizo['baixa'] = fuzz.trimf(relacao_lucro_prejuizo.universe, [0, 0, 5])
relacao_lucro_prejuizo['media'] = fuzz.trimf(relacao_lucro_prejuizo.universe, [0, 5, 10])
relacao_lucro_prejuizo['alta'] = fuzz.trimf(relacao_lucro_prejuizo.universe, [5, 10, 10])

analise_analistas['baixa'] = fuzz.trimf(analise_analistas.universe, [0, 0, 5])
analise_analistas['media'] = fuzz.trimf(analise_analistas.universe, [0, 5, 10])
analise_analistas['alta'] = fuzz.trimf(analise_analistas.universe, [5, 10, 10])

decisao_investimento['nao_investir'] = fuzz.trimf(decisao_investimento.universe, [0, 0, 50])
decisao_investimento['investir'] = fuzz.trimf(decisao_investimento.universe, [50, 100, 100])

# Regras fuzzy
regra1 = ctrl.Rule(retorno_anual['baixo'] & (volatilidade_ativo['alta'] | relacao_lucro_prejuizo['baixa']), decisao_investimento['nao_investir'])
regra2 = ctrl.Rule((retorno_anual['alto'] | analise_analistas['alta']) & (volatilidade_ativo['baixa'] | relacao_lucro_prejuizo['alta']), decisao_investimento['investir'])
regra3 = ctrl.Rule(retorno_anual['medio'] & volatilidade_ativo['media'] & relacao_lucro_prejuizo['media'], decisao_investimento['investir'])

# Sistema de Controle
sistema_controle = ctrl.ControlSystem([regra1, regra2, regra3])
decisao_investimento_ctrl = ctrl.ControlSystemSimulation(sistema_controle)

# Casos de teste
test_cases = [
    (7.5, 3.5, 8.0, 9.0),
    # Adicione mais casos de teste conforme necessário
]

# Execute os testes
for test_case in test_cases:
    decisao_investimento_ctrl.input['retorno_anual'], decisao_investimento_ctrl.input['volatilidade_ativo'], \
    decisao_investimento_ctrl.input['relacao_lucro_prejuizo'], decisao_investimento_ctrl.input['analise_analistas'] = test_case
    decisao_investimento_ctrl.compute()
    print(f"Caso de Teste: {test_case}")
    print(f"Decisão de Investimento: {decisao_investimento_ctrl.output['decisao_investimento']:.2f}%")
    print("-" * 40)