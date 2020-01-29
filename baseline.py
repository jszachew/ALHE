import numpy as np
import pandas as pd
import copy
import constants as con
import time
import matplotlib.pyplot as plt

np.random.seed(42)
import random


def generate_random_number():
    return np.random.random_sample()


def get_gift_weight(gift_name):
    if gift_name.lower() == 'horse':
        return max(0, np.random.normal(5, 2, 1)[0])
    elif gift_name.lower() == 'ball':
        return max(0, 1 + np.random.normal(1, 0.3, 1)[0])
    elif gift_name.lower() == 'bike':
        return max(0, np.random.normal(20, 10, 1)[0])
    elif gift_name.lower() == 'train':
        return max(0, np.random.normal(10, 5, 1)[0])
    elif gift_name.lower() == 'coal':
        return 47 * np.random.beta(0.5, 0.5, 1)[0]
    elif gift_name.lower() == 'book':
        return np.random.chisquare(2, 1)[0]
    elif gift_name.lower() == 'doll':
        return np.random.gamma(5, 1, 1)[0]
    elif gift_name.lower() == 'blocks':
        return np.random.triangular(5, 10, 20, 1)[0]
    elif gift_name.lower() == 'gloves':
        return 3.0 + np.random.rand(1)[0] if np.random.rand(1) < 0.3 else np.random.rand(1)[0]
    return 1.0


def get_gift_weight_helper(gift_name_raw):
    index = gift_name_raw.find('_')
    if index != -1:
        gift_weight = get_gift_weight(gift_name_raw[:index].strip().lower())
        return gift_weight
    return 3.14


def extract_gift_name(gift_name, separator='_'):
    index_of_separator = gift_name.find(separator)
    if index_of_separator != -1:
        return gift_name[:index_of_separator].strip().lower()
    return gift_name


def get_gift_dicts(filename='gifts.csv'):
    gift_frame = pd.read_csv(filename)
    gift_idx_to_name_dict = [None]*7166
    gift_idx_to_weight_dict = [None]*7166
    for i, row in enumerate(gift_frame.values):
        gift_idx_to_name_dict[i] = row[0]
        gift_idx_to_weight_dict[i] = get_gift_weight_helper(row[0])

    return gift_idx_to_name_dict, gift_idx_to_weight_dict


def check_if_min_3_gifts_in_every_bag(individual):
    used_gifts = np.sum(individual, axis=1)
    for i in used_gifts:
        if i < 3:
            return False
    return True


def check_if_every_gift_in_one_or_none_bag(individual):
    used_gifts = np.sum(individual, axis=0)
    if np.any(used_gifts > 1):
        return False
    return True


def chcek_if_all_bags_not_overloaded(individual, weight):
    for i in range(0, 1000):
        w = bag_sum(individual, weight, i)
        if w > 50.0:
            return False
    return True


def only_mutate(population, weight_dict, mutation_probability, max_tries, list_of_selected):
    new_population = list()

    list_of_individuals = list(range(0, len(list_of_selected)))
    number_of_ind_to_mutate = int(len(population)*mutation_probability)
    list_to_mutate = list_of_individuals[:number_of_ind_to_mutate]
    list_to_pass = list_of_individuals[number_of_ind_to_mutate:]

    while len(list_to_pass) > 0:
        new_population.append(copy.deepcopy(population[list_of_selected[list_to_pass.pop()]]))

    while len(list_to_mutate) > 0:
        to_mutate_idx = list_to_mutate.pop()
        individual = copy.deepcopy(population[list_of_selected[to_mutate_idx]])
        row_to_mutate = random.randint(0, 999)
        available_gifts = list()
        used_gifts = np.sum(individual, axis=0)
        for i, val in enumerate(used_gifts):
            if not val:
                available_gifts.append(i)

        random.shuffle(available_gifts)
        gift_to_add = available_gifts.pop()

        try_counter = max_tries
        while bag_sum(individual, weight_dict, row_to_mutate) + weight_dict[gift_to_add] > 50.0 \
                and try_counter > 0 and len(available_gifts) > 0:
            gift_to_add = available_gifts.pop()
            try_counter = try_counter - 1
        if try_counter != 0 and available_gifts != 0:
            individual[row_to_mutate][gift_to_add] = 1.0
        new_population.append(individual)
    return new_population


def bag_sum(individual, weight_dict, bag_no):
    return np.sum(weight_dict*individual[bag_no])


def count_rate(individual, weight_dict):
    used_gifts = np.sum(individual, axis=0)
    return np.sum(weight_dict*used_gifts)


def my_generate_initial_population(weight_dict, initial_population_size):
    list_of_initial_population = list()

    for i in range(0, initial_population_size):
        # shape: (1000, 7166)
        individual = np.zeros((1000, 7166))
        presents_to_start = list(range(0, 7166))
        random.shuffle(presents_to_start)
        random.shuffle(presents_to_start)
        random.shuffle(presents_to_start)
        random.shuffle(presents_to_start)
        random.shuffle(presents_to_start)
        random.shuffle(presents_to_start)
        random.shuffle(presents_to_start)
        tries = 100
        for bag in range(0, 1000):
            gift_1 = presents_to_start[-1]
            gift_2 = presents_to_start[-2]
            gift_3 = presents_to_start[-3]
            while weight_dict[gift_1] + weight_dict[gift_2] + weight_dict[gift_3] > 50 and tries > 0:
                random.shuffle(presents_to_start)
                gift_1 = presents_to_start[-1]
                gift_2 = presents_to_start[-2]
                gift_3 = presents_to_start[-3]
                tries = tries - 1
            individual[bag][gift_1] = 1
            individual[bag][gift_2] = 1
            individual[bag][gift_3] = 1
            presents_to_start.pop()
            presents_to_start.pop()
            presents_to_start.pop()
        list_of_initial_population.append(individual)
    return list_of_initial_population


def selection_for_mutation(population, weight, elite_count, initial_population_size):
    population_weights = list()
    random_list = list(range(0, 1000))
    random.shuffle(random_list)
    random.shuffle(random_list)
    random.shuffle(random_list)
    for ind in population:
        w = count_rate(ind, weight)
        population_weights.append(w)
    population_sum = np.sum(population_weights)
    res = sorted(range(len(population_weights)), reverse=True, key=lambda sub: population_weights[sub])
    proportions_indication_list = list(range(0, 1000))
    end_idx = 0
    proportion_sum = 0
    for j in res:
        start_idx = end_idx
        proportion_sum = proportion_sum + population_weights[res[j]] * len(random_list) / population_sum
        end_idx = min(int(round(proportion_sum)), 1000)
        for k in range(start_idx, end_idx):
            proportions_indication_list[k] = res[j]
    list_to_cross_over = list()
    for i in range(2*(initial_population_size-elite_count)):
        list_to_cross_over.append(proportions_indication_list[random_list.pop()])
    return list_to_cross_over


def create_plot(history, plot_name):
    plt.plot(history)
    plt.title('model accuracy')
    plt.ylabel('rate')
    plt.xlabel('epoch')
    plt.savefig(
        f"{plot_name}.png",
        dpi=700, frameon='false', bbox_inches='tight', )
    plt.close("all")


def print_bags_to_file(individual, names, weight, filename):
    for k in range(len(individual)):
        print(f"\tbag no. {k}, weight: {bag_sum(individual, weight, k)}", file=filename)
        for i, val in enumerate(individual[k]):
            if val == 1:
                print(f"\t\t{names[i]}, {weight[i]}", file=filename)


def get_elitists(population, elite_count, weight):
    elitists = list()
    if elite_count == 0:
        return elitists
    sort = sorted(population, key=lambda g: count_rate(g, weight))
    for i in range(elite_count):
        elitists.append(sort[-(i+1)])

    return elitists


def algorithm(initial_population_size, elite_count, mutation_probability, max_tries, plot_name,
              elite_reduction=False, reduction_iteration=1000, reduction_size=1):
    name, weight = get_gift_dicts()
    start = time.process_time()
    start1 = time.time()
    population = my_generate_initial_population(weight, initial_population_size)
    iteration = 0
    const_counter = 0
    list_of_results = list()
    limit = con.max_bag_capacity * con.max_number_of_bags
    max_weight = 0
    sample = open(f"{plot_name}.txt", 'w')
    print(f"Start of mutation algorithm with max {con.max_iterations} iteration, "
          f"initial population size: {initial_population_size}")
    print(f"Mutation probability: {mutation_probability}")
    print(f"Start of algorithm with max {con.max_iterations} iteration, "
          f"initial population size: {initial_population_size}", file=sample)
    print(f"Mutation probability: {mutation_probability}",
          file=sample)
    while max_weight < limit and iteration < con.max_iterations and const_counter < con.max_iteration_with_same_max:
        for c in range(len(population)):
            check_if_every_gift_in_one_or_none_bag(population[0])
        if elite_reduction:
            if iteration % reduction_iteration == 0 and iteration != 0:
                if reduction_size > 0:
                    elite_count = max(elite_count - reduction_size, 0)
                else:
                    elite_count = min(elite_count - reduction_size, len(population)-2)
        elite = get_elitists(population, elite_count, weight)
        selected_list = selection_for_mutation(population, weight, elite_count, initial_population_size)
        population = only_mutate(population, weight, mutation_probability, max_tries, selected_list)
        list_of_rates = list()
        population.extend(elite)
        for g in population:
            w = count_rate(g, weight)
            list_of_rates.append(w)
        list_of_results.append(max(list_of_rates))
        if max_weight != max(list_of_rates):
            max_weight = max(list_of_rates)
            const_counter = 0
        else:
            const_counter = const_counter + 1
        if iteration % 250 == 0:
            print(f"Iteration: {iteration}, Maximum weight: {max(list_of_rates)}")
        iteration = iteration + 1
    end = time.process_time()
    end1 = time.time()
    print(f"Elapsed time: {end1 - start1}s")
    print(f"Process time: {end - start}s")
    print(f"Elapsed time: {end1 - start1}s", file=sample)
    print(f"Process time: {end - start}s", file=sample)
    print(f"END: Maximum weight: {max(list_of_rates)}")
    for i in range(len(list_of_results)):
        print(f"Iteration: {i}, best in iteration: {list_of_results[i]}", file=sample)
    print(f"PRESENTS IN BAGS", file=sample)
    best = get_elitists(population, 1, weight)
    if not check_if_every_gift_in_one_or_none_bag(best[0]):
        print(f"WARNING! Best individual has at least one present which is in more than one bag")
    else:
        print_bags_to_file(best[0], name, weight, sample)
    create_plot(list_of_results, f"{plot_name}")
    sample.close()


def main():
    for k in range(10):
        timestamp1 = time.time()
        plot_name = f"elite_0-{timestamp1}"
        algorithm(con.initial_population_size, 0, con.mutation_probability, con.max_tries, plot_name)
        timestamp2 = time.time()
        plot_name = f"elite_1-{timestamp2}"
        algorithm(con.initial_population_size, 1, con.mutation_probability, con.max_tries, plot_name)
        timestamp3 = time.time()
        plot_name = f"elite_5-{timestamp3}"
        algorithm(con.initial_population_size, 5, con.mutation_probability, con.max_tries, plot_name)
        timestamp4 = time.time()
        plot_name = f"elite_decrease{timestamp4}"
        algorithm(con.initial_population_size, 5, con.mutation_probability, con.max_tries, plot_name,
                  True, 1000, 1)
        timestamp5 = time.time()
        plot_name = f"elite_growth{timestamp5}"
        algorithm(con.initial_population_size, 0, con.mutation_probability, con.max_tries, plot_name,
                  True, 1000, -1)


if __name__ == "__main__":
    main()
