import random
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.utils import check_X_y

def init_population(population_size, c, top_number): 

    population = []                             
    for i in range(population_size):
        individual = [0]*c
        j = 0
        while(j < top_number):
            p = random.uniform(0, 1) 
            position = random.randrange(c) 
            if(p >= 0.5 and individual[position] == 0):
                individual[position] = 1
                j = j + 1

        if(sum(individual) == 0):
            position = random.randrange(c)
            individual[position] = 1

        population.append(individual)
    return population

iris = load_iris()
data = pd.DataFrame(iris.data) 
target = iris.target 

def calculate_fitness(features,target):
    model = MLPClassifier(max_iter=500)

    features, target = check_X_y(features, target)
    
    scores = cross_val_score(model, features, target, scoring='f1_macro', n_jobs=-1, cv=10)
    return scores.mean()

def get_fitness(population,data):
    fitness_values = []
    for individual in population:
        df = pd.DataFrame(data) 
        i=0
        for i, column in enumerate(df.columns):
            if individual[i] == 0:
                df = df.drop(column, axis=1)

        if not df.empty:
            features = df
            individual_fitness = calculate_fitness(features,target)
            fitness_values.append(individual_fitness)

    return fitness_values


def select_parents(population,fitness_values):
  parents = []
  total = sum(fitness_values)

  norm_fitness_values = [x/total for x in fitness_values]

  cumulative_fitness = []
  start = 0
  for norm_value in norm_fitness_values:
    start+=norm_value
    cumulative_fitness.append(start)

  population_size = len(population)
  for count in range(population_size):
    random_number = random.uniform(0, 1)
    individual_number = 0
    for score in cumulative_fitness:
      if(random_number<=score):
        parents.append(population[individual_number])
        break
      individual_number+=1

  return parents

def two_point_crossover(parents,probability):
    random.shuffle(parents) 
    no_of_pairs = round(len(parents)*probability/2) 
    chromosome_len = len(parents[0])
    crossover_population = []
  
    for num in range(no_of_pairs):
      length = len(parents) 
      parent1_index = random.randrange(length)
      parent2_index = random.randrange(length)
      while(parent1_index == parent2_index):
        parent2_index = random.randrange(length)

      start = random.randrange(chromosome_len)
      end = random.randrange(chromosome_len)
      if(start>end):
        start,end = end, start

      parent1 = parents[parent1_index]
      parent2 = parents[parent2_index]

      child1 =  parent1[0:start] 
      child1.extend(parent2[start:end])
      child1.extend(parent1[end:])

      child2 =  parent2[0:start]
      child2.extend(parent1[start:end])
      child2.extend(parent2[end:])

      parents.remove(parent1)
      parents.remove(parent2)
      crossover_population.append(child1)
      crossover_population.append(child2)

    if(len(parents)>0):
      for remaining_parents in parents:
        crossover_population.append(remaining_parents)

    return crossover_population
    
def mutation(crossover_population):
    
    for individual in crossover_population:
      index_1 = random.randrange(len(individual))
      index_2 = random.randrange(len(individual))

      while(index_2==index_1 and individual[index_1] != individual[index_2]):
        index_2 = random.randrange(len(individual))

      temp = individual[index_1]
      individual[index_1] = individual[index_2]
      individual[index_2] = temp

    return crossover_population

def ga(population_size, c, top_number, iterations):
    population = init_population(population_size, c, top_number)
    print(population)
    for i in range(iterations):
        fitness_values = get_fitness(population, data)
        print("The fitness values are")
        print(fitness_values)

        parents = select_parents(population, fitness_values)
        print("the parents are")       
        print(parents)

        crossover_population = two_point_crossover(parents, 0.8)
        print("the population after crossover is")
        print(crossover_population)

        mutated_population = mutation(crossover_population)
        print("the population after mutation is")
        print(mutated_population)
        population = mutated_population

    fitness_values = get_fitness(population, data)
    best_index = np.argmax(fitness_values)
    best_individual = population[best_index]
   
    return best_individual

best_solution = ga(10, 4, 2, 100)
print("the best set of features are")
print(best_solution)