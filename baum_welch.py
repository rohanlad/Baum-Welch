import copy
from decimal import *
import sys
import random
random.seed(42)
getcontext().prec = 9

# ************************************************
# ********** INSTRUCTIONS ON HOW TO RUN **********
# ************************************************


# Open a terminal window and navigate to the directory in which this file is stored
# Enter the following command and press enter

# python3 baum_welch.py [alphabet] [number of states] [number of iterations] [observation 1] [observation 2] [observation 3] ... [observation n]

# [alphabet] should be a sequence of unique characters
# [number of states] should be an integer
# [number of iterations] should be an integer
# [observation x] should be a sequence of characters where each character is from the alphabet

# EXAMPLE COMMAND: python3 baum_welch.py abc 3 25 abbca bca aacbcba abacaba cbcbca accba aaa

# In the example above, the alphabet is 'abc', there are 3 states, the number of iterations is 25, and the observation sequences
# are [abbca, bca, aacbcba, abacaba, cbcbca, accba, aaa]

# The algorithm will then run. Status information is outputted to the terminal, and when the algorithm has completed, the final output is 
# outputted to the terminal too. This will consist of the transition probability matrix, the emission probability matrix and the initial
# probability matrix. 

# The transition probability matrix is a 2d list of size [number of states]*[number of states], where index [i][j] denotes the probability of
# transitioning from state i to j

# The emission probability matrix is a list of dictionaries, where index [i] denotes the emission probabilites for state i. The dictionary keys
# are the letters from the alphabet and their associated values denote the probability of that letter.

# The initial probability matrix is a list of size [number of states], where index[i] denotes the probability for state i.

# Note that the algorithm will successfully produce the correct output for large input sizes (e.g observation sequences of length 10,000) but
# it will take a while to run. The algorithm prints to terminal on every iteration so you can use this to track the progress.

# ************************************************
# ********** INSTRUCTIONS ON HOW TO RUN **********
# ************************************************



# THE COMMENTED OUT CODE BELOW WAS USED AS PART OF TESTING AND VALIDATION ONLY. IT HAS THEREFORE BEEN COMMENTED OUT.
'''
from hmmlearn import hmm
import numpy as np
np.random.seed(42)

# A function that can generate sequences of a desired length from a given alphabet 
def generate_random_sequences(alphabet, sequence_length, desired_num_sequences):
    list_of_sequences = []
    for _ in range(0, desired_num_sequences):
        seq = ''
        for _ in range(0, sequence_length):
            seq = seq + random.choice(alphabet)
        list_of_sequences.append(seq)
    return list_of_sequences

# This function converts observation sequences into a format that is compatible with the hmm module
def string_to_validate(str):
    final = []
    for chr in str:
        final.append([int(chr)])
    return final

def validate_func(alphabet, n, num_iterations, S, A, pi):
    # alphabet = sys.argv[1]
    # n = int(sys.argv[2])
    # S = sys.argv[4:]
    # num_iterations = int(sys.argv[3])
    # A = []
    # for _ in range(n):
    #     trans_set = [random.random() for i in range(0,n)]
    #     s = sum(trans_set)
    #     trans_set = [ i/s for i in trans_set ]  
    #     A.append(trans_set)
    # B = []
    # emission_dict = {}
    # for letter in alphabet:
    #     emission_dict[letter] = 1/len(alphabet)
    # for _ in range(n):
    #     B.append(copy.deepcopy(emission_dict))
    # pi = [random.random() for i in range(0,n)]
    # s = sum(pi)
    # pi = [ i/s for i in pi ]

    # The python HMM package, used to cross validate the ouputs from our implementation
    model = hmm.MultinomialHMM(n_components=n, n_iter=num_iterations, init_params="", params="te", tol=0, implementation='scaling')
    model.startprob_ = np.array(pi)
    model.transmat_ = np.array(A)
    model.emissionprob_ = np.array([[1/len(alphabet) for _ in range(len(alphabet))] for _ in range(n)])

    to_conc = []
    lengths = []
    for s in S:
        stri = string_to_validate(s)
        to_conc.append(stri)
        lengths.append(len(stri))
    X = np.concatenate(to_conc)
    model.fit(X, lengths)
    print(f'hmmlearn A \n{model.transmat_}')
    print(f'hmmlearn B \n{model.emissionprob_}')
    print(f'hmmlearn pi \n{model.startprob_}\n')

'''





# A helper function used to convert a float to a Decimal
def float_to_decimal(num):
    return Decimal(str(num))



# The wrapper function for the algorithm. It is this function which is called on line 291 which kickstarts the algorithm.
def EM_wrapper(alphabet, n, S, num_iterations):
    print('-------------------------------------------------\n')
    print('Your inputs are as follows:\n')
    print('Alphabet: ' + alphabet + '\n')
    print('Number of States: ' + str(n) + '\n')
    print('Observations: ' + str(S) + '\n')
    print('Number of iterations: ' + str(num_iterations) + '\n')
    print('-------------------------------------------------\n')

    A = []
    for _ in range(n):
        trans_set = [random.random() for i in range(0,n)]
        s = sum(trans_set)
        trans_set = [ i/s for i in trans_set ]  
        A.append(trans_set)
    B = []
    emission_dict = {}
    for letter in alphabet:
        emission_dict[letter] = 1/len(alphabet)
    for _ in range(n):
        B.append(copy.deepcopy(emission_dict))

    pi = [random.random() for i in range(0,n)]
    s = sum(pi)
    pi = [ i/s for i in pi ]

    # this function call was used for validation purposes and has therefore been commented out
    '''
    validate_func(alphabet, n, num_iterations, S, A, pi)
    '''

    for it in range(0, num_iterations):
        print('Starting iteration ' + str(it+1) + '/' + str(num_iterations) + ' ...')
        A, B, pi = EM_main(A, B, pi, S, alphabet)
    print('-------------------------------------------------\n')
    print('FINAL OUTPUT\n')
    print('Transition Probability Matrix: ' + str(A) + '\n')
    print('Emission Probability Matrix: ' + str(B) + '\n')
    print('Initial Probability Matrix: ' + str(pi) + '\n')
    return



# The bulk of the algorithm logic is contained in this function. This function is iteratively 
# called by EM_wrapper()
def EM_main(A, B, pi, S, alphabet):

    num_states = len(A)
    pi_bar_overall = [0]*num_states
    A_bar_numerator = [[0 for _ in range(num_states)]
                       for _ in range(num_states)]
    A_bar_denominator = [0]*num_states
    B_bar_numerator = []
    emission_dict_zero = {}
    for letter in alphabet:
        emission_dict_zero[letter] = 0
    for _ in range(num_states):
        B_bar_numerator.append(copy.deepcopy(emission_dict_zero))
    B_bar_denominator = [0]*num_states
    for O in S:

        num_timesteps = len(O)

        alpha = [[0 for _ in range(num_timesteps)] for _ in range(num_states)]
        for i in range(0, num_states):
            alpha[i][0] = float_to_decimal(pi[i])*float_to_decimal(B[i][O[0]])
        for t in range(1, num_timesteps):
            for j in range(0, num_states):
                sum = 0
                for i in range(0, num_states):
                    sum += float_to_decimal(alpha[i]
                                            [t-1])*float_to_decimal(A[i][j])
                alpha[j][t] = float_to_decimal(
                    sum)*float_to_decimal(B[j][O[t]])

        scaling_parameter = [0]*num_timesteps
        for t in range(0, num_timesteps):
            sum = 0
            for i in range(0, num_states):
                sum += alpha[i][t]
            scaling_parameter[t] = 1/sum

        alpha_hat = [[0 for _ in range(num_timesteps)]
                     for _ in range(num_states)]
        for t in range(0, num_timesteps):
            sum = 0
            for i in range(0, num_states):
                sum += alpha[i][t]
            for i in range(0, num_states):
                alpha_hat[i][t] = alpha[i][t] / sum

        beta = [[0 for _ in range(num_timesteps)] for _ in range(num_states)]
        for i in range(0, num_states):
            beta[i][num_timesteps-1] = 1
        for t in range(num_timesteps-2, -1, -1):
            for i in range(0, num_states):
                sum = 0
                for j in range(0, num_states):
                    sum += float_to_decimal(A[i][j])*float_to_decimal(
                        B[j][O[t+1]])*float_to_decimal(beta[j][t+1])
                beta[i][t] = float_to_decimal(sum)

        beta_hat = [[0 for _ in range(num_timesteps)]
                    for _ in range(num_states)]

        for t in range(num_timesteps-1, -1, -1):
            for i in range(0, num_states):
                beta_hat[i][t] = float_to_decimal(
                    scaling_parameter[t])*float_to_decimal(beta[i][t])

        gamma = [[0 for _ in range(num_timesteps)] for _ in range(num_states)]
        for t in range(0, num_timesteps):
            for i in range(0, num_states):
                sum = 0
                for j in range(0, num_states):
                    sum += alpha_hat[j][t]*beta_hat[j][t]
                gamma[i][t] = alpha_hat[i][t] * \
                    beta_hat[i][t]/sum 

        epsilon = [[[0 for _ in range(num_states)] for _ in range(
            num_states)] for _ in range(num_timesteps)]
        for t in range(0, num_timesteps-1):
            for i_ in range(0, num_states):
                for j_ in range(0, num_states):
                    outer_sum = 0
                    for i in range(0, num_states):
                        inner_sum = 0
                        for j in range(0, num_states):
                            inner_sum += float_to_decimal(alpha_hat[i][t]) * float_to_decimal(
                                A[i][j]) * float_to_decimal(B[j][O[t+1]]) * float_to_decimal(beta_hat[j][t+1])
                        outer_sum += inner_sum
                    epsilon[t][i_][j_] = float_to_decimal(alpha_hat[i_][t]) * float_to_decimal(A[i_][j_]) * float_to_decimal(
                        B[j_][O[t+1]]) * float_to_decimal(beta_hat[j_][t+1]) / float_to_decimal(outer_sum)

        for j in range(0, num_states):
            for k in B[j]:
                for t in range(0, num_timesteps):
                    if O[t] == k:
                        B_bar_numerator[j][k] += gamma[j][t]
            for t in range(0, num_timesteps):
                B_bar_denominator[j] += gamma[j][t]

        for i in range(0, len(gamma)):
            pi_bar_overall[i] += gamma[i][0]

        for i in range(0, num_states):
            for j in range(0, num_states):
                for t in range(0, num_timesteps-1):
                    A_bar_numerator[i][j] += epsilon[t][i][j]
            for t in range(0, num_timesteps-1):
                A_bar_denominator[i] += gamma[i][t]

    pi_bar_overall = list(map(lambda i: i/len(S), pi_bar_overall))
    A_bar = [[0 for _ in range(num_states)] for _ in range(num_states)]
    B_bar = copy.deepcopy(B)
    for i in range(0, num_states):
        for j in range(0, num_states):
            A_bar[i][j] = A_bar_numerator[i][j] / A_bar_denominator[i]
        for k in B[j]:
            B_bar[i][k] = B_bar_numerator[i][k] / B_bar_denominator[i]

    return A_bar, B_bar, pi_bar_overall



# Here we call the wrapper function and pass in the command line arguments to kickstart the algorithm   
EM_wrapper(sys.argv[1], int(sys.argv[2]), sys.argv[4:], int(sys.argv[3]))



# Code which was used for testing/validation and has therefore been commented out
'''
EM_wrapper(sys.argv[1], int(sys.argv[2]), generate_random_sequences('01234', 10000, 5), int(sys.argv[3]))
'''
