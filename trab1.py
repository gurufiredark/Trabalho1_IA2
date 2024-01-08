import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Variáveis de entrada
retorno_anual = ctrl.Antecedent(np.arange(0, 101, 1), 'retorno_anual')
volatilidade_ativo = ctrl.Antecedent(np.arange(0, 101, 1), 'volatilidade_ativo')
relacao_lucro_prejuizo = ctrl.Antecedent(np.arange(0, 101, 1), 'relacao_lucro_prejuizo')
analise_analistas = ctrl.Antecedent(np.arange(0, 101, 1), 'analise_analistas')

# Variável de saída
decisao_investimento = ctrl.Consequent(np.arange(0, 101, 1), 'decisao_investimento')

# Funções de pertinência para as variáveis de entrada e saída
# Fuzzyficação
retorno_anual['baixa'] = fuzz.trimf(retorno_anual.universe, [0, 0, 50])
retorno_anual['medio'] = fuzz.trimf(retorno_anual.universe, [0, 50, 100])
retorno_anual['alto'] = fuzz.trimf(retorno_anual.universe, [50, 100, 100])

volatilidade_ativo['baixa'] = fuzz.trimf(volatilidade_ativo.universe, [0, 0, 50])
volatilidade_ativo['media'] = fuzz.trimf(volatilidade_ativo.universe, [0, 50, 100])
volatilidade_ativo['alta'] = fuzz.trimf(volatilidade_ativo.universe, [50, 100, 100])

relacao_lucro_prejuizo['baixa'] = fuzz.trimf(relacao_lucro_prejuizo.universe, [0, 0, 50])
relacao_lucro_prejuizo['media'] = fuzz.trimf(relacao_lucro_prejuizo.universe, [0, 50, 100])
relacao_lucro_prejuizo['alta'] = fuzz.trimf(relacao_lucro_prejuizo.universe, [50, 100, 100])

analise_analistas['baixa'] = fuzz.trimf(analise_analistas.universe, [0, 0, 50])
analise_analistas['media'] = fuzz.trimf(analise_analistas.universe, [0, 50, 100])
analise_analistas['alta'] = fuzz.trimf(analise_analistas.universe, [50, 100, 100])

decisao_investimento['nao_investir'] = fuzz.trimf(decisao_investimento.universe, [0, 0, 50])
decisao_investimento['investir'] = fuzz.trimf(decisao_investimento.universe, [50, 100, 100])

# Regras fuzzy
regra1 = ctrl.Rule(retorno_anual['baixa'] & (volatilidade_ativo['alta'] | relacao_lucro_prejuizo['baixa']), decisao_investimento['nao_investir'])
regra2 = ctrl.Rule((retorno_anual['alto'] | analise_analistas['alta']) & (volatilidade_ativo['baixa'] | relacao_lucro_prejuizo['alta']), decisao_investimento['investir'])
regra3 = ctrl.Rule(retorno_anual['medio'] & volatilidade_ativo['media'] & relacao_lucro_prejuizo['media'], decisao_investimento['investir'])
regra4 = ctrl.Rule(analise_analistas['baixa'] & volatilidade_ativo['alta'], decisao_investimento['nao_investir']) 
regra5 = ctrl.Rule(analise_analistas['alta'] & volatilidade_ativo['baixa'], decisao_investimento['investir']) 

# Criando o sistema de controle (construtor) e criando a instancia do sistema de controle (decisao_investimento_ctrl)
sistema_controle = ctrl.ControlSystem([regra1, regra2, regra3, regra4, regra5])
decisao_investimento_ctrl = ctrl.ControlSystemSimulation(sistema_controle)

# Casos de teste
test_cases = [
    (75, 35, 80, 90)
    # (35, 55, 25, 97)
    # (55, 55, 55, 55),
    # (15, 90, 15, 15),
    # (90, 15, 15, 15),
    # (15, 15, 90, 15),
    # (15, 15, 15, 90),
    # (15, 15, 15, 15),
    # (90, 90, 90, 90),
    # (0, 0, 0, 0),
    # (100, 100, 100, 5),
]

# Execução dos testes
for test_case in test_cases:
    decisao_investimento_ctrl.input['retorno_anual'], decisao_investimento_ctrl.input['volatilidade_ativo'], decisao_investimento_ctrl.input['relacao_lucro_prejuizo'], decisao_investimento_ctrl.input['analise_analistas'] = test_case
    decisao_investimento_ctrl.compute()

    # Exibição dos gráficos
    retorno_anual.view(sim=decisao_investimento_ctrl)
    volatilidade_ativo.view(sim=decisao_investimento_ctrl)
    relacao_lucro_prejuizo.view(sim=decisao_investimento_ctrl)
    analise_analistas.view(sim=decisao_investimento_ctrl)
    decisao_investimento.view(sim=decisao_investimento_ctrl)

    print(f"Caso de Teste: {test_case}")
    print(f"Decisao de Investimento: {decisao_investimento_ctrl.output['decisao_investimento']:.2f}%")
    print("-" * 40)

plt.show()    