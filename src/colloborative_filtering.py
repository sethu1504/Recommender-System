import numpy as np
import math
import matplotlib.pyplot as plt
import hashlib


# Function to get Signature matrix
def get_signature_matrix(no_of_permutations, total_matrix):
    signature_matrix = []
    no_of_objects = len(total_matrix[0])
    for _ in range(no_of_permutations):
        signature_row = []  # Each row in the signature matrix
        perm_list = np.random.permutation(len(total_matrix))
        for object_index in range(no_of_objects):
            row_with_first_one = -1
            for row_index in perm_list:
                if total_matrix[row_index][object_index] == 1:
                    row_with_first_one = row_index
                    break
            signature_row.append(row_with_first_one)
        signature_matrix.append(signature_row)
    return signature_matrix


# Function to get pairs of r and b
def get_r_b_tuples(no_of_permutations):
    r_b_tuples = []
    for r in range(no_of_permutations):
        for b in range(no_of_permutations):
            if r * b == no_of_permutations:
                r_b_tuples.append((r, b))
    return r_b_tuples


# Function to return a dictionary containing the similarity index of each pair in the passed matrix
def give_similarity_dictionary(matrix):
    no_objects = len(matrix[0])
    sims_dictionary = dict()
    for i in range(no_objects):
        for j in range(i + 1, no_objects):
            tuple_key = (i, j)
            set1 = np.asarray(matrix[:, [i]].flatten()).astype(np.bool)
            set2 = np.asarray(matrix[:, [j]].flatten()).astype(np.bool)
            intersection = np.logical_and(set1, set2)
            union = np.logical_or(set1, set2)
            sim = intersection.sum() / float(union.sum())
            sims_dictionary[tuple_key] = sim
    return sims_dictionary


# Function to give similarity dictionary over signature matrix
def give_sim_signature_dictionary(matrix):
    no_objects = len(matrix[0])
    no_items = len(matrix)
    sims_dictionary = dict()
    for i in range(no_objects):
        for j in range(i + 1, no_objects):
            tuple_key = (i, j)
            set1 = matrix[:, [i]].flatten()
            set2 = matrix[:, [j]].flatten()
            no_of_similar = np.count_nonzero(set1 == set2)
            sim = no_of_similar / float(no_items)
            sims_dictionary[tuple_key] = sim
    return sims_dictionary


# Function to generate candidate pairs in the given matrix
def get_candidate_pairs(bands, rows, matrix, buckets=10000):
    candidates = set()  # Set contains the candidate pairs
    objects = len(matrix[0])
    start_row_index = 0
    end_row_index = rows - 1
    for band_number in range(bands):
        sim_dict = dict()
        for i in range(objects):
            band_set1 = matrix[start_row_index:end_row_index + 1, [i]].flatten()
            hash_value = int(hashlib.sha256((str(band_set1) + str(band_number)).encode('utf-8')).hexdigest(), 16) % buckets
            if hash_value in sim_dict:
                l = sim_dict[hash_value]
                l.append(i)
                sim_dict[hash_value] = l
            else:
                sim_dict[hash_value] = list([i])
        for hash_key in sim_dict.keys():
            candidate_list = sim_dict[hash_key]
            for i_index in range(len(candidate_list)):
                for j_index in range(i_index + 1, len(candidate_list)):
                    candidates.add((candidate_list[i_index], candidate_list[j_index]))
        start_row_index += rows
        end_row_index += rows
    return candidates


# Function get best pair of r and b
def get_best_r_b(r_b_tuples, similarity_sample):
    candidate_prob = []
    max_spike = 0.0
    best_r_b = ()
    for r, b in r_b_tuples:
        probabilities = []
        for similarity in similarity_sample:
            probability = 1 - math.pow((1 - math.pow(similarity, r)), b)
            probabilities.append(probability)
        spike = probabilities[4] - probabilities[2]
        if spike > max_spike:
            max_spike = spike
            candidate_prob = probabilities
            best_r_b = (r, b)
    return best_r_b, candidate_prob


def do_user_filtering(signature_matrix, total_similarities, r_b_tuples):
    # Choose the best r and b values by calculate the max slope between 0.2 and 0.4 similarity values
    similarity_sample = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    best_r_b, candidate_prob = get_best_r_b(r_b_tuples, similarity_sample)

    print('For threshold 30% BEST r = ' + str(best_r_b[0]) + ' and b = ' + str(best_r_b[1]))

    # Plot the S curve for the best b and r while t = 0.3
    plt.figure(1)
    plt.plot(similarity_sample, candidate_prob)
    plt.title('The S-Curve for Permutations = ' + str(permutation))
    plt.xlabel('Jaccard Similarity')
    plt.ylabel('Candidate Probabilities')

    signature_matrix_np = np.array(signature_matrix)

    # Calculate the Jaccard similarities of all pairs in the signature matrix
    jaccard_sims_signature = give_sim_signature_dictionary(signature_matrix_np)

    # Calculate FP and FN with respect to Signature Matrix
    b_list = []
    fp_list = []
    fn_list = []
    fp_tot_list = []
    fn_tot_list = []
    for r, b in r_b_tuples:
        b_list.append(b)
        print('r = ' + str(r) + ' b = ' + str(b))
        candidate_pairs = get_candidate_pairs(b, r, signature_matrix_np)

        print('Number of candidate pairs = ' + str(len(candidate_pairs)))

        # Calculate FP and FN in the candidate pairs
        fp = 0
        fn = 0
        for pair in list(candidate_pairs):
            if jaccard_sims_signature[pair] < 0.3:
                fp += 1
                candidate_pairs.remove(pair)  # Remove false positives
        fp_list.append(fp)

        for key in jaccard_sims_signature.keys():
            if jaccard_sims_signature[key] > 0.3 and key not in candidate_pairs:
                fn += 1
        fn_list.append(fn)

        print('FP = ' + str(fp) + ' FN = ' + str(fn))

        print('After removing FP No of Candidate Pairs = ' + str(len(candidate_pairs)))

        fp_tot = 0
        fn_tot = 0
        for pair in candidate_pairs:
            if total_similarities[pair] < 0.3:
                fp_tot += 1
        fp_tot_list.append(fp_tot)

        for key in total_similarities.keys():
            if total_similarities[key] > 0.3 and key not in candidate_pairs:
                fn_tot += 1
        fn_tot_list.append(fn_tot)

        print('Comparing original similarities and the current list of candidate pairs FP = ' + str(fp_tot) + \
              ' FN = ' + str(fn_tot))

    plt.figure(2)
    plt.plot(b_list, fp_list, color='red', label='FP')
    plt.plot(b_list, fn_list, color='blue', label='FN')
    plt.legend()
    plt.title('Signature Matrix - b vs FP FN for Permutations = ' + str(permutation))
    plt.xlabel('b')
    plt.ylabel('FP, FN')

    plt.figure(3)
    plt.plot(b_list, fp_tot_list, color='red', label='FP')
    plt.plot(b_list, fn_tot_list, color='blue', label='FN')
    plt.legend()
    plt.title('Original Matrix - b vs FP FN for Permutations = ' + str(permutation))
    plt.xlabel('b')
    plt.ylabel('FP, FN')


if __name__ == "__main__":
    data_file_loc = '../datasets/data.txt'  # Location of data file

    tot_matrix = []  # Matrix to hold the entire data (Item x User)
    sig_matrix = []  # Matrix to hold the min-hash signatures

    data_file = open(data_file_loc, 'r')

    # Read the text file into the matrix
    for line in data_file:
        if len(line) > 0:
            tot_matrix.append(list(map(int, line.split(','))))

    # Calculate Jaccard similarities of the total matrix
    tot_similarities = give_similarity_dictionary(np.array(tot_matrix))

    permutations_list = [100, 500]

    for permutation in permutations_list:
        print('PERMUTATIONS = ' + str(permutation))
        sig_matrix = get_signature_matrix(permutation, tot_matrix)
        r_b = get_r_b_tuples(permutation)
        do_user_filtering(sig_matrix, tot_similarities, r_b)
        plt.show()
