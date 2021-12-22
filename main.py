import random
import typing

import numpy as np
import matplotlib.pyplot as plt


class GenotypeDecoder:
    def decode(self, genotype: np.ndarray, x: np.ndarray, bias: typing.Union[int, np.ndarray]):
        return sum((genotype * x) + bias)


class FitnessEvaluator:
    def __init__(self, genotype_decoder: GenotypeDecoder):
        self.genotype_decoder = genotype_decoder

    def evaluate(self, genotype: np.ndarray, x: np.ndarray, bias: typing.Union[int, np.ndarray], goal_value: int):
        return abs(self.genotype_decoder.decode(genotype, x, bias) - goal_value)


class Individual:
    def __init__(self, genotype: np.ndarray, fitness: int):
        self.genotype = genotype
        self.fitness = fitness

    def __repr__(self):
        return "Individual/genotype = " + str(self.genotype) + " Fitness = " + str(self.fitness)


class IndividualFactory:
    def __init__(self, genotype_length: int, fitness_evaluator: FitnessEvaluator, x: np.ndarray,
                 bias: typing.Union[int, np.ndarray], goal_value: int):
        self.genotype_length = genotype_length
        self.fitness_evaluator = fitness_evaluator
        self.x = x
        self.bias = bias
        self.goal = goal_value

        self.binary_string_format = '{:0' + str(self.genotype_length) + 'b}'

    def with_random_genotype(self):
        """
        Creates an individual with a random genotype-used at the very beginning to create
         a random population as a starting point
        :return: Individual
        """
        random_genotype = np.random.random((self.genotype_length,))
        fitness = self.fitness_evaluator.evaluate(random_genotype, self.x, self.bias, self.goal)
        return Individual(random_genotype, fitness)

    def with_set_genotype(self, genotype: np.ndarray):
        """
        Creates an individual with a provided genotype-used when a new individual is created through the breeding of
        two individuals from the previous generation.
        :param genotype:
        :return:
        """
        fitness = self.fitness_evaluator.evaluate(genotype, self.x, self.bias, self.goal)
        return Individual(genotype, fitness)

    def with_minimal_fitness(self):
        """
        Creates an individual with a genotype consisting solely of zeros-used to create an alternative starting point
        with a population of individuals with fitness == 0
        :return:
        """
        minimal_fitness_genotype = self.binary_string_format.format(0)
        fitness = self.fitness_evaluator.evaluate(minimal_fitness_genotype)
        return Individual(minimal_fitness_genotype, fitness)


class Population:
    """
    Population class holds a collection of individuals. It provides a way of getting the fittest individuals through
    `get_the_fittest` method.
    """

    def __init__(self, individuals):
        self.individuals = individuals

    def get_the_fittest(self, n: int):
        self._sort_by_fitness()
        return self.individuals[:n]

    def _sort_by_fitness(self):
        self.individuals.sort(key=self._individual_fitness_sort_key, reverse=False)

    def _individual_fitness_sort_key(self, individual: Individual):
        return individual.fitness


class PopulatoinFactory:
    """
    PopulationFactory is a counterpart of `IndividualFactory` and provides methods of creating populations with random
    individuals, with given individuals, and with minimal-fitness individuals.
    """

    def __init__(self, individual_factory: IndividualFactory):
        self.individual_factory = individual_factory

    def with_random_individuals(self, size: int):
        individuals = []
        for i in range(size):
            individuals.append(self.individual_factory.with_random_genotype())
        return Population(individuals)

    def with_individuals(self, individuals):
        return Population(individuals)

    def with_minimal_fitness_individuals(self, size: int):
        individuals = []
        for i in range(size):
            individuals.append(self.individual_factory.with_minimal_fitness())
        return Population


class ParentSelector:
    """
    In this implementation, each new generation will completely replace the previous generation.
    This, combined with the fact that each pair of parents will produce two children, will lead to
    **Constant population size**.
    """
    def select_parents(self, population: Population):
        total_fitness = 0
        fitness_scale = []
        for index, individual in enumerate(population.individuals):
            total_fitness += individual.fitness
            if index == 0:
                fitness_scale.append(individual.fitness)
            else:
                fitness_scale.append(individual.fitness + fitness_scale[index - 1])

        # Store the selected parents
        mating_pool = []
        # Equal to the size of the population
        number_of_parents = len(population.individuals)
        # How fast we move along the fitness scale
        fitness_step = total_fitness / number_of_parents
        random_offset = random.uniform(0, fitness_step)

        # Iterate over the parents size range and for each:
        # - generate pointer position on the fitness scale
        # - pick the parent who corresponds to the current pointer position and add them to the mating pool
        current_fitness_pointer = random_offset
        last_fitness_scale_position = 0
        for index in range(len(population.individuals)):
            for fitness_scale_position in range(last_fitness_scale_position, len(fitness_scale)):
                if fitness_scale[fitness_scale_position] >= current_fitness_pointer:
                    mating_pool.append(population.individuals[fitness_scale_position])
                    last_fitness_scale_position = fitness_scale_position
                    break
            current_fitness_pointer += fitness_step

        return mating_pool


class SinglePointCrossover:
    def __init__(self, individual_factory: IndividualFactory):
        self.individual_factory = individual_factory

    def crossover(self, parent_1: Individual, parent_2: Individual):
        crossover_point = np.random.random((len(parent_1.genotype,))) > 0.5
        genotype_1 = self._new_genotype(crossover_point, parent_1, parent_2)
        genotype_2 = self._new_genotype(crossover_point, parent_2, parent_1)
        child_1 = self.individual_factory.with_set_genotype(genotype=genotype_1)
        child_2 = self.individual_factory.with_set_genotype(genotype=genotype_2)
        return child_1, child_2

    def _new_genotype(self, crossover_point: np.ndarray, parent_1: Individual, parent_2: Individual):
        return np.where(crossover_point, parent_1.genotype, parent_2.genotype)


class Mutator:
    def __init__(self, individual_factory: IndividualFactory):
        self.individual_factory = individual_factory

    def mutate(self, individual: Individual):
        mutation_probability = 1 / len(individual.genotype)

        mutated_genotype = individual.genotype
        mutated_genotype = np.where(np.random.random((len(mutated_genotype),)) > mutation_probability,
                                    np.random.random((len(mutated_genotype,))), mutated_genotype)

        return self.individual_factory.with_set_genotype(genotype=mutated_genotype)


class Breeder:
    def __init__(self, single_point_crossover: SinglePointCrossover, mutator: Mutator):
        self.single_point_crossover = single_point_crossover
        self.mutator = mutator

    def produce_offspring(self, parents):
        """
        With each iteration the algorithm:
        1. Picks two individuals from the pool at random.
        2. Creates two new individuals by crossing over the genotypes of the selected parents.
        3. Mutates the genotypes of the newly created offspring.
        4. Adds so created individuals to the offspring collection, which will become the next generation.;
        :param parents:
        :return:
        """
        offspring = []
        number_of_parents = len(parents)
        for index in range(int(number_of_parents)):
            parent_1, parent_2 = self._pick_random_parents(parents, number_of_parents)
            child_1, child_2 = self.single_point_crossover.crossover(parent_1, parent_2)
            child_1_mutated = self.mutator.mutate(child_1)
            child_2_mutated = self.mutator.mutate(child_2)
            offspring.extend((child_1_mutated, child_2_mutated))
        return offspring

    def _pick_random_parents(self, parents, number_of_parents: int):
        parent_1 = parents[random.randint(0, number_of_parents - 1)]
        parent_2 = parents[random.randint(0, number_of_parents - 1)]
        return parent_1, parent_2


class Environment:
    def __init__(self, population_size: int, parent_selector: ParentSelector, population_factory: PopulatoinFactory,
                 breeder: Breeder):
        self.population_factory = population_factory
        self.population = self.population_factory.with_random_individuals(size=population_size)
        self.parent_selector = parent_selector
        self.breeder = breeder

    def update(self):
        parents = self.parent_selector.select_parents(self.population)
        next_generation = self.breeder.produce_offspring(parents)
        self.population = self.population_factory.with_individuals(random.sample(next_generation, len(parents)))
        # self.population = self.population_factory.with_individuals(self.population.get_the_fittest(len(parents)))

    def get_the_fittest(self, n: int):
        return self.population.get_the_fittest(n)


if __name__ == '__main__':
    TOTAL_GENERATIONS = 10000
    POPULATION_SIZE = 100
    GENOTYPE_LENGTH = 6

    current_generation = 1

    x = np.random.random((GENOTYPE_LENGTH,))
    bias = np.random.random((1,))
    goal = 0
    print("X: {}\nbias: {}".format(x, bias))

    genotype_decoder = GenotypeDecoder()
    fitness_evaluator = FitnessEvaluator(genotype_decoder)
    individual_factory = IndividualFactory(GENOTYPE_LENGTH, fitness_evaluator, x, bias, goal)
    population_factory = PopulatoinFactory(individual_factory)
    single_point_crossover = SinglePointCrossover(individual_factory)
    mutator = Mutator(individual_factory)
    breeder = Breeder(single_point_crossover, mutator)
    parent_selector = ParentSelector()
    environment = Environment(POPULATION_SIZE, parent_selector, population_factory, breeder)

    highest_fitness_list = []
    while current_generation <= TOTAL_GENERATIONS:
        fittest = environment.get_the_fittest(1)[0]
        highest_fitness_list.append(fittest.fitness)
        if fittest.fitness < 0.001:
            print("Winner, winner, chicken dinner! We got there")
            break
        environment.update()
        current_generation += 1
        print(current_generation, fittest.fitness)

    print("Stopped at generation " + str(current_generation - 1) + ". The fittest individual: ")
    print(fittest)

    generations = range(1, len(highest_fitness_list) + 1)
    plt.plot(generations, highest_fitness_list)
    plt.title("Fittest individual in each generation")
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.show()
