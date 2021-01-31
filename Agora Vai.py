#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import operator as op
import random as rd
from copy import deepcopy
from anytree import Node, RenderTree, AsciiStyle, PreOrderIter
from anytree.cachedsearch import findall
from sklearn.metrics.cluster import v_measure_score
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils.metric import distance_metric, type_metric


# In[4]:


# Dataframe de treino e variáveis associadas
pd.set_option('display.max_rows', None)
df = pd.read_csv('glass_train.csv')
cluster_column = df.columns.values[-1]
cluster_count = df[cluster_column].nunique()
df_unclass = df.drop([cluster_column], axis=1)
df_attr = len(df_unclass.columns)


# In[5]:


# Definindo operadores e terminais
def div0(a, b): return 1 if b == 0 else a/b
ops = {
    '+': op.add,
    '-': op.sub,
    '*': op.mul,
    '/': div0,
    'max': max,
    'min': min,
}
nonterminals = list(ops.keys())
terminals = [str(i) + 'a' for i in range(df_attr)]
terminals += [str(i) + 'b' for i in range(df_attr)]
terminals += ['c']
def random_constant(min_=-1000, max_=1000): return rd.uniform(min_, max_)
def get_terminal_value(term, row1, row2):
    t = str(term[-1])
    # Constante
    if t != 'a' and t != 'b': return float(term)
    # Terminal comum
    index = int(term[:-1])
    return row1[index] if t == 'a' else row2[index]
def random_terminal():
    t = rd.choice(terminals)
    return random_constant() if t == 'c' else t
def random_nonterminal(): return rd.choice(nonterminals)


# In[109]:


# Criação de árvores para população inicial
max_h = 7

def create_tree(h = 0, full = False, max_depth = max_h):
    node = Node('')
    # Retorna um terminal se a árvore passar do tamanho máximo
    if h >= max_depth:
        node.name = random_terminal()
    else:
        # Método full: escolhe não terminais até não poder mais
        if full: value = random_nonterminal()
        # Método grow: escolhe entre terminais ou não terminais
        else: value = random_terminal() if rd.randint(0,1) == 0 else random_nonterminal()
        node.name = value
        if value in nonterminals:
            left_child = create_tree(h = h+1, full = full, max_depth = max_depth)
            left_child.parent = node
            right_child = create_tree(h = h+1, full = full, max_depth = max_depth)
            right_child.parent = node
    return node

# Imprime uma árvore
def print_tree(t, f = None):
    for pre, _, node in RenderTree(t):
        if f != None: print("%s%s" % (pre, node.name), file = f)
        else: print("%s%s" % (pre, node.name))
        
# Função que calcula a distância entre dois pontos do dataframe com a árvore
def evaluate_tree(t, row1, row2):
    try:
        v = str(t.name)
        if v in nonterminals:
            lhs = evaluate_tree(t.children[0], row1, row2)
            rhs = evaluate_tree(t.children[1], row1, row2)
            return ops[v](lhs, rhs)
        else:
            if v[-1] != 'a' and v[-1] != 'b':
                return float(v)
            else:
                index = int(v[:-1])
                return row1[index] if v[-1] == 'a' else row2[index]
    except IndexError:
        print('Index Error when evaluating operator: ' + str(t))
        print('Children (' + str(len(t.children)) + ') ')
        print('Full Tree: ')
        print_tree(t.root)
        raise KeyboardInterrupt

evaluated_tree = None
def evaluate(row1, row2): return evaluate_tree(evaluated_tree, row1, row2)

def same_tree(t1, t2):
    t1_list = [node.name for node in PreOrderIter(t1)]
    t2_list = [node.name for node in PreOrderIter(t2)]
    return t1_list == t2_list


# In[7]:


# Inicializando os centros uma única vez para resultados de fitness não mudarem entre chamadas
df_unclass = df_unclass.drop(['pred'], axis=1, errors='ignore')
init_centers = kmeans_plusplus_initializer(df_unclass, cluster_count).initialize()

# Calcula fitness usando kmeans e v measure em cima da função de um indivíduo
def fitness(t):
    global df_unclass
    global evaluated_tree
    evaluated_tree = t
    # Primeiro, descarta previsões passadas
    df_unclass = df_unclass.drop(['pred'], axis=1, errors='ignore')
    # Depois, roda o kmeans para o clustering
    mt = distance_metric(type_metric.USER_DEFINED, func = evaluate)
    #init_centers = kmeans_plusplus_initializer(df_unclass, cluster_count).initialize()
    kmeans_inst = kmeans(df_unclass, init_centers, metric = mt, itermax = 20)
    kmeans_inst.process()
    kmeans_clusters = kmeans_inst.get_clusters()
    # Para cada cluster, coloca os valores como previsões
    for i in range(len(kmeans_clusters)):
        df_unclass.loc[kmeans_clusters[i], 'pred'] = df.iloc[kmeans_clusters[i]].groupby(cluster_column).size().idxmax()
    # Compara as previsões com os valores reais com v measure
    fit = v_measure_score(df[cluster_column], df_unclass['pred'])
    t.fitness = fit
    return fit


# In[127]:


# Operadores genéticos e variáveis associadas
crossover_prob = 0.9
mutation_prob = 0.05
good_crossovers = 0
bad_crossovers = 0
good_mutations = 0
bad_mutations = 0

# Seleciona um nó aleatório de uma árvore
def random_node(t):
    nodes = [node for node in PreOrderIter(t)]
    return rd.choice(nodes)

def replace_left_child(t, c):
    if len(t.children) >= 2: t.children = (c,) + t.children[1:]
    else: t.children = (c,) + t.children
def replace_right_child(t, c):
    if len(t.children) >= 2: t.children = t.children[:-1] + (c,)
    else: t.children += (c,)

def is_left_child(t, c): return c == t.children[0]
def is_right_child(t, c): return c == t.children[1]

def get_relation(t, c):
    if t == None: return 'root'
    elif c == t.children[0]: return 'left_child'
    elif c == t.children[1]: return 'right_child'
    else:
        print(str(t) + ' and ' + str(c) + ' are not related')
        raise KeyboardInterrupt

# Crossover
def crossover(t1_original, t2_original):
    t1, t2 = deepcopy(t1_original), deepcopy(t2_original)
    node1, node2 = random_node(t1), random_node(t2)
    parent1, parent2 = node1.parent, node2.parent
    parent1_c, parent2_c = deepcopy(parent1), deepcopy(parent2)
    
    # Fazendo cruzamento
    relations = [get_relation(parent1, node1), get_relation(parent2, node2)]
    if relations[0] == 'root': t1 = node2
    elif relations[0] == 'left_child': replace_left_child(parent1, node2)
    elif relations[0] == 'right_child': replace_right_child(parent1, node2)
        
    if relations[1] == 'root': t2 = node1
    elif relations[1] == 'left_child': replace_left_child(parent2, node1)
    elif relations[1] == 'right_child': replace_right_child(parent2, node1)
    
    # Algum indivíduo novo passou a altura máxima, nesse caso substitui por um dos pais
    if t1.height > max_h:
        t1 = deepcopy(t1_original)
    if t2.height > max_h:
        t2 = deepcopy(t2_original)
    
    return t1, t2

# Mutação (ponto)
def mutation(t_original):
    t = deepcopy(t_original)
    node = random_node(t)
    if node.name in nonterminals:
        pool = [i for i in nonterminals if i != node.name]
        node.name = rd.choice(pool)
    else:
        if str(node.name)[-1] != 'a' and str(node.name)[-1] != 'b':
            node.name = 'c'
        pool = [i for i in terminals if i != node.name]
        node.name = rd.choice(pool)
        if node.name == 'c': node.name = random_constant()
    return t


# In[132]:


# Funções para algoritmo genético e variáveis relacionadas
pop_size = 50
generations = 20
tournament_k = 4
elitism_count = 2

# Gerando população inicial com ramped half-half
def initialize_population():
    pop = []
    group_count = len(range(2, max_h + 1))
    group_size = pop_size // group_count
    # Faz grupos de profundidade 2 até o máximo determinado
    for i in range(2, max_h + 1):
        half = False
        # Cria cada indivíduo de cada grupo e os coloca na população
        for j in range(group_size):
            ind = create_tree(full = half, max_depth = i)
            ind.fitness = fitness(ind)
            half = not half
            pop.append(ind)
    return pop

def elitism(pop):
    pop = sorted(pop, key=lambda n: n.fitness, reverse=True)
    elite = []
    for i in range(elitism_count):
        elite.append(pop[i])
    return elite

def tournament(pop):
    championship = [rd.choice(pop) for i in range(tournament_k)]
    championship = sorted(championship, key=lambda n: n.fitness, reverse=True)
    return championship[0]

def new_generation(past_pop):
    global good_crossovers, bad_crossovers, good_mutations, bad_mutations
    # Começa nova população com elitismo
    new_pop = elitism(past_pop)
    # Depois gera filhos
    while len(new_pop) < pop_size:
        parent1, parent2 = tournament(past_pop), tournament(past_pop)
        selected_op = rd.uniform(0, 1)
        
        # Faz operações genéticas de acordo com a probabilidade
        # Crossover
        if selected_op <= crossover_prob:
            child1, child2 = crossover(parent1, parent2)
            child1.fitness, child2.fitness = fitness(child1), fitness(child2)
            new_pop += [child1, child2]
            if child1.fitness > parent1.fitness and child1.fitness > parent2.fitness: good_crossovers += 1
            else: bad_crossovers += 1
            if child2.fitness > parent1.fitness and child2.fitness > parent2.fitness: good_crossovers += 1
            else: bad_crossovers += 1
        # Mutação
        elif selected_op <= crossover_prob + mutation_prob:
            child1 = mutation(parent1)
            child1.fitness = fitness(child1)
            new_pop += [child1]
            if child1.fitness > parent1.fitness: good_mutations += 1
            else: bad_mutations += 1
        # Reprodução
        else:
            new_pop += [parent1]
            
    #data = population_report(new_pop)
    #data.report()
    return new_pop


# In[141]:


class population_report:
    def __init__(self, pop):
        self.population = pop
        pop_sorted = sorted(pop, key=lambda n: n.fitness, reverse=True)
        self.best_fit = pop_sorted[0].fitness
        self.worst_fit = pop_sorted[-1].fitness
        self.avg_fit = sum([i.fitness for i in pop])/len(pop)
        self.good_crossovers = good_crossovers
        self.bad_crossovers = bad_crossovers
        self.good_mutations = good_mutations
        self.bad_mutations = bad_mutations
    def report(self):
        print('---------POPULATION REPORT---------')
        print('Size: ' + str(len(self.population)))
        print('Avg fitness: ' + str(self.avg_fit))
        print('Best fitness: ' + str(self.best_fit))
        print('Worst fitness: ' + str(self.worst_fit))
        print('Good/Bad Crossovers: ' + str(self.good_crossovers) + '/' +str(self.bad_crossovers))
        print('Good/Bad Mutations: ' + str(self.good_mutations) + '/' +str(self.bad_mutations))
    def get_same_individuals_count(self):
        pop_c = deepcopy(self.population)
        repeated_count = 0
        for ind in pop_c:
            count = 0
            for i in range(len(pop_c)):
                if same_tree(pop_c[i], ind):
                    count += 1
                    pop_c.pop(i)
            if count > 1: repeated_count += count
        return repeated_count


# In[117]:


def genetic_algorithm():
    pop = initialize_population()
    #init_data = population_report(pop)
    #init_data.report()

    for i in range(generations):
        pop = new_generation(pop)
    return pop


# In[108]:


# Dataframe de teste e variáveis associadas
df_test = pd.read_csv('glass_test.csv')
test_cluster_column = df_test.columns.values[-1]
test_cluster_count = df_test[cluster_column].nunique()
df_test_unclass = df_test.drop([cluster_column], axis=1)
df_test_attr = len(df_unclass.columns)


# In[152]:


# Calculando fitness do melhor indivíduo selecionado
#selected_tree = sorted(pop, key=lambda n: n.fitness, reverse=True)[0]
selected_tree = None
def evaluate_test(row1, row2): return evaluate_tree(selected_tree, row1, row2)
def fitness_test(t):
    global df_test_unclass
    global selected_tree
    selected_tree = t
    # Primeiro, descarta previsões passadas
    df_test_unclass = df_test_unclass.drop(['pred'], axis=1, errors='ignore')
    # Depois, roda o kmeans para o clustering
    mt = distance_metric(type_metric.USER_DEFINED, func = evaluate_test)
    init_centers = kmeans_plusplus_initializer(df_test_unclass, test_cluster_count).initialize()
    kmeans_inst = kmeans(df_test_unclass, init_centers, metric = mt, itermax = 20)
    kmeans_inst.process()
    kmeans_clusters = kmeans_inst.get_clusters()
    print('Generated ' + str(len(kmeans_clusters)))
    # Para cada cluster, coloca os valores como previsões
    for i in range(len(kmeans_clusters)):
        df_test_unclass.loc[kmeans_clusters[i], 'pred'] = df_test.iloc[kmeans_clusters[i]].groupby(test_cluster_column).size().idxmax()
    # Compara as previsões com os valores reais com v measure
    fit = v_measure_score(df_test[cluster_column], df_test_unclass['pred'])
    t.fitness = fit
    return fit


# In[151]:


def reset_parameters(size = 50, gens = 15, k = 3, prob_config = 1):
    print('Resetting parameters...')
    global pop_size, tournament_k, generations, crossover_prob, mutation_prob
    pop_size = size
    tournament_k = k
    generations = gens
    if prob_config == 1:
        crossover_prob = 0.9
        mutation_prob = 0.05
    else:
        crossover_prob = 0.6
        mutation_prob = 0.3

def reset_counters():
    global good_crossovers, bad_crossovers, good_mutations, bad_mutations
    good_crossovers = bad_crossovers = good_mutations = bad_mutations = 0

# Realizando testes com parâmetros
def test_pop_size(sizes):
    global pop_size
    reset_parameters()
    size_fitness = []
    test_data = []
    for size in sizes:
        reset_counters()
        pop_size = size
        pop = genetic_algorithm()
        pop_data = population_report(pop)
        pop_data.report()
        # Resultado para um dado tamanho
        result_tree = sorted(pop, key=lambda n: n.fitness, reverse=True)[0]
        fit = fitness_test(result_tree)
        size_fitness += [fit]
        test_data += [pop_data]
        print('Fitness result for size ' + str(size) + ': ' + str(fit))
    return size_fitness, test_data
    
def test_tournament(values):
    global tournament_k
    reset_paremeters()
    tournament_fitness = []
    test_data = []
    for i in values:
        reset_counters()
        tournament_k = i
        pop = genetic_algorithm()
        pop_data = population_report(pop)
        pop_data.report()
        # Resultado para um dado tamanho
        result_tree = sorted(pop, key=lambda n: n.fitness, reverse=True)[0]
        fit = fitness_test(result_tree)
        tournament_fitness += [fit]
        test_data += [pop_data]
        print('Fitness result for size ' + str(size) + ': ' + str(fit))
    return tournament_fitness, test_data

def test_genetic_operators():
    reset_paremeters(prob_config = config)
    config_fitness = []
    test_data = []
    for i in range(1,3):
        reset_counters()
        pop = genetic_algorithm()
        pop_data = population_report(pop)
        pop_data.report()
        # Resultado para um dado tamanho
        result_tree = sorted(pop, key=lambda n: n.fitness, reverse=True)[0]
        fit = fitness_test(result_tree)
        tournament_fitness += [fit]
        test_data += [pop_data]
        print('Fitness result for size ' + str(size) + ': ' + str(fit))
    return tournament_fitness, test_data


# In[150]:


# Rodando testes
size_test_fitness, size_test_data = test_pop_size([30, 50, 100])
tournament_test_fitness, tournament_test_data = test_tournament([2, 3, 5, 7])
gen_ops_test_fitness, gen_ops_test_data = test_genetic_operators()

