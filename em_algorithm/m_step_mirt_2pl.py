from hashlib import new
from m_step import m_step
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from scipy.stats import bernoulli
import random
import sys
import os
sys.path.append(os.path.realpath("./models"))
# Custom modules, import violates pep8, so we have to declare an exeption
if True:  # noqa: E402
    from mirt_2pl import mirt_2pl


class m_step_ga_mml(m_step):

    def __init__(self, model: mirt_2pl) -> None:
        super().__init__()
        self.model = model

    def genetic_algorithm(self, item: int, fitness_function, x0: np.array, constraints=None,
                          population_size: int = 30, p_mutate: float = 0.5, p_crossover: float = 0.2):
        # Helping functions
        def mutate(individual): return individual+multivariate_normal.rvs(
            mean=np.zeros(len(individual)), cov=np.identity(len(individual)))

        def crossover(individual1, individual2):
            crossover_indices = random.choices(
                population=list(range(0, len(individual1))), k=int(len(individual1)/2))
            new_individual = [individual2[i] if i in crossover_indices else individual1[i]
                              for i in range(0, len(individual1))]
            return(new_individual)

        # Initialization with last M-Step
        population_base = [mutate(x0)
                           for i in range(0, int(population_size*0.85))]
        population_base = population_base + \
            [x0 for i in range(0, population_size-len(population_base))]
        fitness = [fitness_function(individual)
                   for individual in population_base]  # TODO: Multiprocessing ggf.
        population_base = list(zip(fitness, population_base))
        population_base.sort(reverse=True)
        converged = False
        while not converged:
            # Selection
            # (Len-rank)/gaussian_sum
            population = random.choices(population=population_base, weights=list(
                range(len(population_base)+1, 1, -1)), k=population_size)
            population = [individual[1] for individual in population]
            # Breeding
            for i in range(0, len(population)):
                if bernoulli.rvs(p=p_mutate, size=1):
                    population[i] = mutate(population[i])
                if bernoulli.rvs(p=p_crossover, size=1):
                    # TODO: prohibit self-crossover
                    partner = random.choices(population=population, k=1)
                    population[i] = crossover(population[i], partner[0])
            # Evaluation
            fitness = [fitness_function(individual)
                       for individual in population]  # TODO: Multiprocessing ggf.
            fitness.sort(reverse=True)
            highest_fitness = fitness[0]
            print("Highest Current Fitness:")
            print(max(highest_fitness, population_base[0][0]))
            # TODO: I could decrease mutation variance in case of lower fitness
            if (abs(highest_fitness - population_base[0][0]) < 0.001) or (highest_fitness < population_base[0][0]):
                converged = True
            population_base = population_base + list(zip(fitness, population))
            print("Length of Population = {0}".format(len(population_base)))
            population_base.sort(reverse=True)
        new_a_item = population_base[0][1][0:self.model.latent_dimension]
        new_delta_item = population_base[0][1][len(x0)-1]
        return(new_a_item, new_delta_item)

    def step(self, pe_functions: dict):
        # Find the new value for Sigma
        q_0 = pe_functions["q_0"]

        def func(sigma_vector): return q_0(np.reshape(
            sigma_vector, newshape=(
                self.model.latent_dimension, self.model.latent_dimension)))
        x0 = np.reshape(
            self.model.person_parameters["covariance"], newshape=self.model.latent_dimension**2)
        new_sigma = minimize(func, x0=x0, method='Nelder-Mead').x
        new_sigma = np.reshape(new_sigma, newshape=(
            self.model.latent_dimension, self.model.latent_dimension))
        # Find new values for A and delta
        new_A = np.empty(
            shape=self.model.item_parameters["discrimination_matrix"].shape)
        new_delta = np.empty(
            shape=self.model.item_parameters["intercept_vector"].shape)
        for item in range(0, self.model.item_dimension):
            a_init = self.model.item_parameters["discrimination_matrix"][item, :]
            delta_init = self.model.item_parameters["intercept_vector"][item]
            x0 = np.concatenate(
                (a_init, np.expand_dims(delta_init, 0)), axis=0)

            def q_item(input): return pe_functions["q_item_list"][item](
                a_item=input[0:len(input)-1], delta_item=input[len(input)-1])
            new_a_item, new_delta_item = self.genetic_algorithm(
                fitness_function=q_item, item=item, x0=x0)
            new_A[item] = new_a_item
            new_delta[item] = new_delta_item
        return({"item_parameters": {"discrimination_matrix": new_A, "intercept_vector": new_delta},
                "person_parameters": {"covariance": new_sigma}})
