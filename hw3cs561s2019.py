# CSCI 561
# HW3
# Oliver Day

import copy
import sys
import time

# Class Definitions ==================================================================


class MarkovDecisionProcess:

    def __init__(self, n_size, wall_states_unadj, terminal_states_unadj, prob, reward_non_term, gamma, actions):
        self.grid_size = n_size
        self.prob = prob
        self.reward_non_term = reward_non_term
        self.gamma = gamma
        self.actions = actions

        self.wall_states = self.adjust_given_wall_coords(wall_states_unadj)
        self.terminal_state_to_reward_dict = self.adjust_given_terminal_coords_to_reward(terminal_states_unadj)
        self.terminal_states = self.terminal_state_to_reward_dict.keys()
        self.states = self.generate_state_set()
        self.reward_model = self.generate_reward_model()
        self.transition_model = self.generate_transition_model()


    def adjust_given_wall_coords(self, wall_states_unadj):
        wall_states_set = set()
        for coords in wall_states_unadj:
            wall_states_set.add((coords[0]-1, coords[1]-1))

        return wall_states_set

    def adjust_given_terminal_coords_to_reward(self, terminal_states_unadj):
        terminal_to_reward_mapping = {}
        for [row, column, reward] in terminal_states_unadj:
            terminal_to_reward_mapping[(row-1, column-1)] = reward

        return terminal_to_reward_mapping

    def generate_state_set(self):
        state_set = set()
        for row in range(0, self.grid_size):
            for col in range(0, self.grid_size):
                if (row, col) not in self.wall_states:
                    state_set.add((row, col))

        return state_set

    def generate_reward_model(self):
        reward_model = {}
        for state in self.states:
            if state in self.terminal_states:
                reward_model[state] = self.terminal_state_to_reward_dict[state]
            else:
                reward_model[state] = self.reward_non_term

        return reward_model

    def generate_transition_model(self):
        transition_model = {}
        for state in self.states:
            transition_model[state] = {}
            if state in self.terminal_states:
                action = (0, 0)
                transition_model[state][action] = [(0.0, state)]
            else:
                for action in self.actions:
                    transition_model[state][action] = self.calculate_trans_probs_for_action_pairs(state, action)

        return transition_model

    def get_state_from_action(self, curr_state, agent_direction):
        resulting_state = self.vector_addition(curr_state, agent_direction)
        if resulting_state in self.states:
            return resulting_state
        else:
            return curr_state

    def calculate_trans_probs_for_action_pairs(self, state, action):
        prob_of_correct_move = self.prob
        prob_of_wrong_move = 0.5 * (1 - prob_of_correct_move)
        return [(prob_of_correct_move, self.get_state_from_action(state, action)),
                (prob_of_wrong_move, self.get_state_from_action(state, self.turn_clockwise(action))),
                (prob_of_wrong_move, self.get_state_from_action(state, self.turn_counter_clockwise(action)))]

    def get_available_actions(self, state):
        if state not in self.terminal_states:
            return self.actions
        else:
            return [(0, 0)]

    def turn_clockwise(self, action):
        # Up
        if action == (-1, 0):
            return (-1, 1)
        # Down
        elif action == (1, 0):
            return (1, -1)
        # Left
        elif action == (0, -1):
            return (-1, -1)
        # Right
        else:
            return (1, 1)

    def turn_counter_clockwise(self, action):
        # Up
        if action == (-1, 0):
            return (-1, -1)
        # Down
        elif action == (1, 0):
            return (1, 1)
        # Left
        elif action == (0, -1):
            return (1, -1)
        # Right
        else:
            return (-1, 1)

    def vector_addition(self, vector_a, vector_b):
        return tuple(map(sum, zip(vector_a, vector_b)))

    def expected_utility_for_action(self, state, action, utility):
        return sum([prob_of_reaching_state_prime * utility[state_prime]
                   for (prob_of_reaching_state_prime, state_prime) in self.transition_model[state][action]])

    def value_iteration(self, error_bound=0.1):
        utility_prime = dict([(state, 0) for state in self.states])
        policy = dict()
        while True:
            utility = copy.deepcopy(utility_prime)
            delta = 0.0
            for state in self.states:
                policy[state] = ()
                max_calculated_utility = float("-inf")
                for action in self.get_available_actions(state):
                    calculated_utility = self.reward_model[state] + self.gamma*(self.expected_utility_for_action(state, action, utility))
                    if calculated_utility > max_calculated_utility:
                        policy[state] = action
                        max_calculated_utility = calculated_utility
                utility_prime[state] = max_calculated_utility
                delta = max(delta, abs(utility_prime[state] - utility[state]))

            if delta < error_bound * (1 - self.gamma)/self.gamma:
                return policy


# Function Definitions ===========================================================


# file_path = "./inputs/input0.txt"
# file_path = "./input.txt"
# file_path = "./inputs/inputHW.txt"
def get_mdp_info_from_file():
    file_path = "./inputs/input4.txt"
    with open(file_path, 'r') as file_pointer:
        grid_size = int(str.strip(file_pointer.readline()))

        # Split row and column
        n_walls = int(str.strip(file_pointer.readline()))
        wall_states = []
        for i in range(n_walls):
            wall_coords_str = (file_pointer.readline()).split(',')
            wall_states.append([int(wall_coords_str[0]),
                                int(wall_coords_str[1])])

        # Split line by row, column, and reward_term
        n_terminals = int(file_pointer.readline())
        terminal_states = []
        for i in range(n_terminals):
            terminal_coords_w_reward_str = (file_pointer.readline()).split(',')
            terminal_states.append([int(terminal_coords_w_reward_str[0]),
                                    int(terminal_coords_w_reward_str[1]),
                                    int(terminal_coords_w_reward_str[2])])

        prob_correct_move = float(str.strip(file_pointer.readline()))
        reward_non_term = float(str.strip(file_pointer.readline()))
        gamma = float(str.strip(file_pointer.readline()))

    return grid_size, wall_states, terminal_states, prob_correct_move, reward_non_term, gamma


def print_matrix(matrix):
    for row in matrix:
        print row


def print_dict(mapping):
    for key in mapping.keys():
        print key
        print mapping[key]


def output_solution_to_file(optimal_policy, action_to_char, grid_size, wall_states):

    file_path = "./output.txt"
    with open(file_path, 'w') as file_pointer:
        for row in range(0, grid_size):
            for column in range(0, grid_size):

                if column != (grid_size - 1):
                    if [row+1, column+1] not in wall_states:
                        file_pointer.write(str(action_to_char[optimal_policy[(row, column)]]) + ',')
                    else:
                        file_pointer.write('N,')
                else:
                    if [row+1, column+1] not in wall_states:
                        file_pointer.write(str(action_to_char[optimal_policy[(row, column)]]) + '\n')
                    else:
                        file_pointer.write('N\n')

# main ===========================================================================


def main():
    # start_time = time.time()

    grid_size, wall_states_unadj, terminal_states_unadj, prob_correct_move, reward_non_term, gamma = get_mdp_info_from_file()

    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    action_to_char = {(-1, 0): 'U',
                      (1, 0): 'D',
                      (0, -1): 'L',
                      (0, 1): 'R',
                      (0, 0): 'E'}

    mdp = MarkovDecisionProcess(grid_size, wall_states_unadj, terminal_states_unadj,
                                prob_correct_move, reward_non_term, gamma, actions)
    optimal_policy = mdp.value_iteration()
    output_solution_to_file(optimal_policy, action_to_char, grid_size, wall_states_unadj)

    # end_time = time.time() - start_time
    # print "\nTotal run-time: " + str(end_time)

# ===============================================================================


if __name__ == "__main__":
    main()
