import string

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


def chcek_if_all_bags_not_overloaded(individual, weight):
    for i in range(0, 1000):
        w = bag_sum(individual, weight, i)
        if w > 50.0:
            return False
    return True


def cross_over(population, cross_over_probability, list_of_individuals_to_cross_over, initial_population_size,
               elite_count, weights, max_tries):
    new_population = list()
    list_of_split_rate = list(range(0, 7166))
    random.shuffle(list_of_split_rate)
    random.shuffle(list_of_split_rate)
    random.shuffle(list_of_split_rate)
    split_rate = list_of_split_rate.pop()

    number_of_ind_to_cross = (int(len(list_of_individuals_to_cross_over)*cross_over_probability/2))*2
    number_of_kids_from_crossing = int(number_of_ind_to_cross/2)
    list_to_cross_over = list_of_individuals_to_cross_over[:number_of_ind_to_cross]
    list_to_pass_over = list_of_individuals_to_cross_over[number_of_ind_to_cross:]
    list_to_pass_over = list_to_pass_over[:max(initial_population_size - number_of_kids_from_crossing - elite_count, 0)]
    while len(list_to_pass_over) > 0:
        pop = list_to_pass_over.pop()
        #print(f"len population: {len(population)} | last list_to_pass_over: {pop}")
        new_population.append(population[pop])

    first_parent_idx = list_to_cross_over.pop()
    second_parent_idx = list_to_cross_over.pop()
    while len(list_to_cross_over) > 1:
        #print(f"first_parent_idx: {first_parent_idx} | second_parent_idx:{second_parent_idx}")
        ind_1 = population[first_parent_idx]
        ind_2 = population[second_parent_idx]
        ind_1_back = ind_1[:, split_rate:]
        ind_1_front = ind_1[:, :split_rate]
        ind_2_back = ind_2[:, split_rate:]
        ind_2_front = ind_2[:, :split_rate]
        new_1 = np.concatenate((ind_2_front, ind_1_back), axis=1)
        new_2 = np.concatenate((ind_1_front, ind_2_back), axis=1)
        tries = max_tries
        new_1_ok = False
        new_2_ok = False
        if check_if_min_3_gifts_in_every_bag(new_1) and chcek_if_all_bags_not_overloaded(new_1, weights):
            print("new_1_ok!")
            new_1_ok = True
        if check_if_min_3_gifts_in_every_bag(new_2) and chcek_if_all_bags_not_overloaded(new_2, weights):
            print("new_2_ok!")
            new_2_ok = True
        while tries > 0 and len(list_of_split_rate) > 0 and not new_1_ok and not new_2_ok:
            split_rate = list_of_split_rate.pop()
            ind_1_back = ind_1[:, split_rate:]
            ind_1_front = ind_1[:, :split_rate]
            ind_2_back = ind_2[:, split_rate:]
            ind_2_front = ind_2[:, :split_rate]
            new_1 = np.concatenate((ind_2_front, ind_1_back), axis=1)
            new_2 = np.concatenate((ind_1_front, ind_2_back), axis=1)
            if check_if_min_3_gifts_in_every_bag(new_1) and chcek_if_all_bags_not_overloaded(new_1, weights):
                print("new_1_ok!")
                new_1_ok = True
            if check_if_min_3_gifts_in_every_bag(new_2) and chcek_if_all_bags_not_overloaded(new_2, weights):
                print("new_2_ok!")
                new_2_ok = True
            tries = tries - 1
        if new_1_ok and new_2_ok:
            print("both new ok")
            if count_rate(new_1, weights) > count_rate(new_2, weights):
                new_population.append(new_1)
            else:
                new_population.append(new_2)
        elif new_1_ok:
            print("new 1 ok")
            new_population.append(new_1)
        elif new_2_ok:
            print("new 2 ok")
            new_population.append(new_2)
        else:
            if generate_random_number() < 0.5:
                print("parent 1 appended")
                new_population.append(population[first_parent_idx])
            else:
                print("parent 2 appended")
                new_population.append(population[second_parent_idx])
        print(f"crossing - tries left: {tries}")
        first_parent_idx = list_to_cross_over.pop()
        second_parent_idx = list_to_cross_over.pop()
    return new_population


def mutate(population, weight_dict, mutation_probability, max_tries):
    list_of_individuals_to_mutate = list(range(0, len(population)))
    number_of_ind_to_mutate = int(len(population)*mutation_probability)

    random.shuffle(list_of_individuals_to_mutate)
    random.shuffle(list_of_individuals_to_mutate)
    random.shuffle(list_of_individuals_to_mutate)

    list_of_individuals_to_mutate = list_of_individuals_to_mutate[:number_of_ind_to_mutate]
    to_mutate_idx = list_of_individuals_to_mutate.pop()

    while len(list_of_individuals_to_mutate) > 0:
        individual = population[to_mutate_idx]
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
       # population[to_mutate_idx] = individual
        to_mutate_idx = list_of_individuals_to_mutate.pop()


def mutate_with_best_approx_potential(population, weight_dict, mutation_probability, max_tries):
    number_of_ind_to_mutate = int(len(population) * mutation_probability)

    population_potentials = list()
    for ind in population:
        w = count_rate(ind, weight_dict)
        population_potentials.append(w)
    res = sorted(range(len(population_potentials)),
                 key=lambda sub: population_potentials[sub])[:number_of_ind_to_mutate]
    for r in range(len(res)):
        individual = population[r]
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


def bag_sum(individual, weight_dict, bag_no):
    return np.sum(weight_dict*individual[bag_no])


def count_rate(individual, weight_dict):
    used_gifts = np.sum(individual, axis=0)
    return np.sum(weight_dict*used_gifts);


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
            gift_4 = presents_to_start[-4]
            while weight_dict[gift_1] + weight_dict[gift_2] + \
                    weight_dict[gift_3] + weight_dict[gift_4] > 50 and tries > 0:
                random.shuffle(presents_to_start)
                gift_1 = presents_to_start[-1]
                gift_2 = presents_to_start[-2]
                gift_3 = presents_to_start[-3]
                gift_4 = presents_to_start[-4]
                tries = tries - 1
            individual[bag][gift_1] = 1
            individual[bag][gift_2] = 1
            individual[bag][gift_3] = 1
            individual[bag][gift_4] = 1
            presents_to_start.pop()
            presents_to_start.pop()
            presents_to_start.pop()
            presents_to_start.pop()
        list_of_initial_population.append(individual)
    return list_of_initial_population


def selection(population, weight, elite_count, initial_population_size):
    population_weights = list()
    random_list = list(range(0, 1000))
    random.shuffle(random_list)
    random.shuffle(random_list)
    random.shuffle(random_list)
    for ind in population:
        w = count_rate(ind, weight)
        population_weights.append(w)
    population_sum = np.sum(population_weights)
    res = sorted(range(len(population_weights)),reverse=True, key=lambda sub: population_weights[sub])
    proportions_indication_list = list(range(0,1000))
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


def algorithm(initial_population_size, elite_count, mutation_probability, cross_over_probability, max_tries, plot_name):
    name, weight = get_gift_dicts()
    start = time.process_time()
    start1 = time.time()
    population = my_generate_initial_population(weight, initial_population_size)
    iteration = 0
    const_counter = 0
    list_of_results = list()
    limit = con.max_bag_capacity * con.max_number_of_bags
    max_weight = 0
    sample = open(f"random-{plot_name}.txt", 'w')
    print(f"Start of algorithm with max {con.max_iterations} iteration, "
          f"initial population size: {initial_population_size}")
    print(f"Mutation probability: {mutation_probability}, Crossing-over probability: {cross_over_probability}")
    print(f"Start of algorithm with max {con.max_iterations} iteration, "
          f"initial population size: {initial_population_size}", file=sample)
    print(f"Mutation probability: {mutation_probability}, Crossing-over probability: {cross_over_probability}",
          file=sample)
    while max_weight < limit and iteration < con.max_iterations and const_counter < con.max_iteration_with_same_max:
        list_to_cross_over = selection(population, weight, elite_count, initial_population_size)
        population = cross_over(population, cross_over_probability, list_to_cross_over, initial_population_size,
                                elite_count, weight, max_tries)

        mutate(population, weight, mutation_probability, max_tries)
        list_of_rates = list()
        for g in population:
            w = count_rate(g, weight)
            list_of_rates.append(w)
        list_of_results.append(max(list_of_rates))
        if max_weight != max(list_of_rates):
            max_weight = max(list_of_rates)
            const_counter = 0
        else:
            const_counter = const_counter + 1
        if iteration % 5 == 0:
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
    print_bags_to_file(population[0], name, weight, sample)
    create_plot(list_of_results, f"random-{plot_name}")
    sample.close()


def main():
    # for k in range(10):
    #     for l in range(len(con.mutation_probability_list)):
    #         for m in range(len(con.cross_over_probability_list)):
    #             for o in range(len(con.initial_population_size)):
    #                 for p in range(len(con.max_tries)):
    #                     timestamp1 = time.time()
    #                     plot_name = f"Comparision-Run{timestamp1}-p{con.initial_population_size[o]}-" \
    #                                 f"i{con.max_iterations}-m{con.mutation_probability_list[l]}-" \
    #                                 f"c{con.cross_over_probability_list[m]}-t{con.max_tries[p]}"
    #                     algorithm_with_potential(0.7, 0.7, plot_name)
    #                     timestamp2 = time.time()
    #                     plot_name = f"Comparision-Run{timestamp2}-p{con.initial_population_size[o]}-" \
    #                                 f"i{con.max_iterations}-m{con.mutation_probability_list[l]}-" \
    #                                 f"c{con.cross_over_probability_list[m]}-t{con.max_tries[p]}"
    #                     algorithm(0.7, 0.7, plot_name)
    timestamp1 = time.time()
    plot_name = f"Comparision-Run{timestamp1}"
    algorithm(20, 0, 0.7, 0.7, 20, plot_name)


if __name__ == "__main__":
    main()
