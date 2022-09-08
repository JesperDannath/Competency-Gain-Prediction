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

    # TODO: Python package für ga ausprobieren: cmaes (https://github.com/CMA-ES/pycma)
    def genetic_algorithm(self, fitness_function, x0: np.array, constraint_function=lambda x: True,
                          population_size: int = 40, p_mutate: float = 0.5, p_crossover: float = 0.2, mutation_variance=0.3):
        # Helping functions
        def mutate(individual):
            valid_individual = False
            while not valid_individual:
                new_individual = individual+multivariate_normal.rvs(
                    mean=np.zeros(len(individual)), cov=mutation_variance*np.identity(len(individual)))
                try:
                    if not constraint_function(new_individual):
                        continue
                except Exception:
                    continue
                valid_individual = True
            return(new_individual)

        def crossover(individual1, individual2):
            crossover_indices = random.choices(
                population=list(range(0, len(individual1))), k=int(len(individual1)/2))
            new_individual = np.array([individual2[i] if i in crossover_indices else individual1[i]
                                       for i in range(0, len(individual1))])
            # TODO: Better hanle constraint case
            constraint_function(new_individual)
            return(new_individual)

        # Initialization with last M-Step
        population_base = [mutate(x0)
                           for i in range(0, int(population_size*0.85))]
        population_base = population_base + \
            [x0 for i in range(0, population_size-len(population_base))]
        fitness = [fitness_function(individual)
                   for individual in population_base]
        population_base = list(zip(fitness, population_base))
        population_base.sort(
            reverse=True, key=lambda individual: individual[0])
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
                       for individual in population]
            # fitness.sort(reverse=True)
            highest_fitness = max(fitness)
            #print("Highest Current Fitness:")
            #print(max(highest_fitness, population_base[0][0]))
            # TODO: I could decrease mutation variance in case of lower fitness
            if (abs(highest_fitness - population_base[0][0]) < 0.001) or (highest_fitness < population_base[0][0]):
                converged = True
            population_base = population_base + list(zip(fitness, population))
            #print("Length of Population = {0}".format(len(population_base)))
            population_base.sort(
                reverse=True, key=lambda individual: individual[0])
        return(population_base[0][1])

    def step(self, pe_functions: dict):
        # Find the new value for Sigma
        print("Maximize Q-0")
        # The only parameters we need to optimise are the correlations
        # TODO: Matrix X^tX = sigma benutzen um p.s.d zu enforcen
        log_likelihood = 0.0

        def q_0(corr_vector):
            sigma = self.model.corr_to_sigma(corr_vector)
            return pe_functions["q_0"](np.reshape(
                sigma, newshape=(
                    self.model.latent_dimension, self.model.latent_dimension)))
        x0 = self.model.person_parameters["covariance"][np.triu_indices(
            self.model.latent_dimension, k=1)]
        new_corr = self.genetic_algorithm(
            q_0, x0=x0, constraint_function=lambda corr: self.model.check_sigma(self.model.corr_to_sigma(corr)), p_crossover=0.0)
        #new_sigma = minimize(func, x0=x0, method='BFGS').x
        new_sigma = self.model.corr_to_sigma(new_corr)
        # Find new values for A and delta
        new_A = np.empty(
            shape=self.model.item_parameters["discrimination_matrix"].shape)
        new_delta = np.empty(
            shape=self.model.item_parameters["intercept_vector"].shape)
        log_likelihood += q_0(new_corr)
        print("Maximize the Q_i's")
        for item in range(0, self.model.item_dimension):
            a_init = self.model.item_parameters["discrimination_matrix"][item, :]
            delta_init = self.model.item_parameters["intercept_vector"][item]
            x0 = np.concatenate(
                (a_init, np.expand_dims(delta_init, 0)), axis=0)
            x0 = x0[x0 != 0]

            def q_item(input):
                delta_item = input[len(input)-1]
                a_item = self.model.fill_zero_discriminations(
                    input[0:len(input)-1], item=item)
                return pe_functions["q_item_list"][item](
                    a_item=a_item, delta_item=delta_item)

            new_item_parameters = self.genetic_algorithm(
                fitness_function=q_item, x0=x0, constraint_function=lambda arg: np.all(arg[0:len(arg)-1] > 0))
            log_likelihood += q_item(new_item_parameters)
            new_a_item = self.model.fill_zero_discriminations(
                new_item_parameters[0:self.model.latent_dimension], item=item)
            new_delta_item = new_item_parameters[len(x0)-1]
            new_A[item] = new_a_item
            new_delta[item] = new_delta_item
        return({"item_parameters": {"discrimination_matrix": new_A, "intercept_vector": new_delta},
                "person_parameters": {"covariance": new_sigma}}, log_likelihood)
