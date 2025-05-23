import numpy as np
import mdptoolbox
import mdptoolbox.example
from scipy.sparse import lil_matrix

pH = [0.24852, 0.12858, 0.11527, 0.07902, 0.03438, 0.03000, 0.02101, 0.05289]
pA = 0.29033
print(np.sum(pH)+pA)
gamma = 0
N_pools = len(pH)
maxForkLen = 8
max_bribe = 2
states = []
N_pool_blocks = np.zeros(N_pools)
eps = 0
non_active, active = 0, 1
honest_available = False

adopt, override, wait = 0, 1, 2
match = np.zeros(max_bribe + 1, dtype=int)
match[0] = 3
for i in range(len(match)-1):
    match[i+1] = match[i] + 1
choices = 3 + len(match)
def state_generator():
    def helper(combination, current_sum):
        if len(combination) == N_pools:
            if current_sum <= maxForkLen:
                for a in range(maxForkLen + 1):
                    match_options = [0, 1] if a >= current_sum > 0 else [0]
                    for match in match_options:
                        latest_options = [0, 2] if match == 1 else [0, 1]
                        for latest in latest_options:
                            max_combination_value = max(combination) if combination else 0
                            max_bribe_value = min(max_bribe, max_combination_value)
                            bribe_options = range(max_bribe_value + 1) if match == 1 else [0]
                            for bribe in bribe_options:
                                state = [list(combination), a, latest, match, bribe]
                                states.append(state)

            return
        max_value = maxForkLen - current_sum

        for i in range(max_value + 1):
            new_combination = combination + [i]
            new_sum = current_sum + i
            helper(new_combination, new_sum)

    helper([], 0)

    return states

def stnum2st(index):
    if 0 <= index < len(states):
        return states[index]
    else:
        print(f"Index {index} is out of range.")
        return None

def st2stnum(combination, a, latest, match, bribe):
    state = (tuple(combination), a, latest, match, bribe)
    return states_dict.get(state, -1)

def MDP_matrices():
    global states_dict
    states = state_generator()
    states_dict = {tuple((tuple(state[0]),) + tuple(state[1:])): index for index, state in enumerate(states)}
    numOfStates = len(states)
    print(f"Total number of valid states: {numOfStates}")
    P = [lil_matrix((numOfStates, numOfStates), dtype=np.float64) for _ in range(choices)]
    Rs = [lil_matrix((numOfStates, numOfStates), dtype=np.float64) for _ in range(choices)]
    Diff = [lil_matrix((numOfStates, numOfStates), dtype=np.float64) for _ in range(choices)]

    for i in range(numOfStates):
        if i % 2000 == 0:
            print(f"processing state: {i}")

        [combination, a, latest, Match, bribe] = stnum2st(i)
        h = np.sum(combination)

        # adopt
        for j in range(N_pools):
            new_combination = np.zeros(N_pools, dtype=int)
            new_combination[j] = 1
            P[adopt][i, st2stnum(new_combination, 0, 1, 0, 0)] = pH[j]
            Diff[adopt][i, st2stnum(new_combination, 0, 1, 0, 0)] = h

        new_combination = np.zeros(N_pools, dtype=int)
        P[adopt][i, st2stnum(new_combination, 1, 0, 0, 0)] = pA
        Diff[adopt][i, st2stnum(new_combination, 1, 0, 0, 0)] = h

        # Define override
        if a > h:
            for j in range(N_pools):
                new_combination = np.zeros(N_pools, dtype=int)
                new_combination[j] = 1
                P[override][i, st2stnum(new_combination, a-h-1, 1, 0, 0)] = pH[j]
                Rs[override][i, st2stnum(new_combination, a-h-1, 1, 0, 0)] = h + 1
                Diff[override][i, st2stnum(new_combination, a-h-1, 1, 0, 0)] = h + 1

            new_combination = np.zeros(N_pools, dtype=int)
            P[override][i, st2stnum(new_combination, a - h, 0, 0, 0)] = pA
            Rs[override][i, st2stnum(new_combination, a - h, 0, 0, 0)] = h + 1
            Diff[override][i, st2stnum(new_combination, a - h, 0, 0, 0)] = h + 1

        else:  # Just for completeness
            P[override][i, 0] = 1
            Rs[override][i, 0] = - 10000
            Diff[override][i, 0] = 10000

        # Define wait
        if Match == non_active and a + 1 <= maxForkLen and h + 1 <= maxForkLen:
            for j in range(N_pools):
                new_combination = combination.copy()
                new_combination[j] += 1
                P[wait][i, st2stnum(new_combination, a, 1, 0, 0)] = pH[j]
            new_combination = combination.copy()
            P[wait][i, st2stnum(new_combination, a+1, 0, 0, 0)] = pA

        elif Match == active and a > h > 0 and a + 1 <= maxForkLen and h + 1 <= maxForkLen:
            # honest pool
            if honest_available:
                if latest == 2:
                    new_combination = combination.copy()
                    new_combination[0] += 1
                    P[wait][i, st2stnum(new_combination, a, 1, 0, 0)] = (1 - gamma) * pH[0]

                    new_combination = np.zeros(N_pools, dtype=int)
                    new_combination[0] += 1
                    P[wait][i, st2stnum(new_combination, a - h, 1, 0, 0)] = gamma * pH[0]
                    Rs[wait][i, st2stnum(new_combination, a - h, 1, 0, 0)] = h
                    Diff[wait][i, st2stnum(new_combination, a - h, 1, 0, 0)] = h
                else:
                    new_combination = combination.copy()
                    new_combination[0] += 1
                    P[wait][i, st2stnum(new_combination, a, 1, 0, 0)] = pH[0]

            # petty-compliant pool
            min_index = 1 if honest_available else 0
            for j in range(min_index, N_pools):
                if bribe >= combination[j]:
                    new_combination = np.zeros(N_pools, dtype=int)
                    new_combination[j] += 1
                    P[wait][i, st2stnum(new_combination, a - h, 1, 0, 0)] = pH[j]
                    Rs[wait][i, st2stnum(new_combination, a - h, 1, 0, 0)] = h - (bribe + eps)
                    Diff[wait][i, st2stnum(new_combination, a - h, 1, 0, 0)] = h
                else:
                    new_combination = combination.copy()
                    new_combination[j] += 1
                    P[wait][i, st2stnum(new_combination, a, 1, 0, 0)] = pH[j]

            # adversary
            new_combination = combination.copy()
            P[wait][i, st2stnum(new_combination, a + 1, latest, 1, bribe)] = pA


        else:
            P[wait][i, 0] = 1
            Rs[wait][i, 0] = - 10000
            Diff[wait][i, 0] = 10000


        # Define match
        for k in range(len(match)):
            if Match == non_active and a >= h > 0 and a + 1 <= maxForkLen and h + 1 <= maxForkLen and k <= max(combination):
                # honest pool
                if honest_available:
                    if latest == 1:
                        new_combination = combination.copy()
                        new_combination[0] += 1
                        P[match[k]][i, st2stnum(new_combination, a, 1, 0, 0)] = (1-gamma) * pH[0]

                        new_combination = np.zeros(N_pools, dtype=int)
                        new_combination[0] += 1
                        P[match[k]][i, st2stnum(new_combination, a - h, 1, 0, 0)] = gamma * pH[0]
                        Rs[match[k]][i, st2stnum(new_combination, a - h, 1, 0, 0)] = h
                        Diff[match[k]][i, st2stnum(new_combination, a - h, 1, 0, 0)] = h
                    else:
                        new_combination = combination.copy()
                        new_combination[0] += 1
                        P[match[k]][i, st2stnum(new_combination, a, 1, 0, 0)] = pH[0]

                # petty-compliant pool
                min_index = 1 if honest_available else 0
                for j in range(min_index, N_pools):
                    if k >= combination[j]:
                        new_combination = np.zeros(N_pools, dtype=int)
                        new_combination[j] += 1
                        P[match[k]][i, st2stnum(new_combination, a - h, 1, 0, 0)] = pH[j]
                        Rs[match[k]][i, st2stnum(new_combination, a - h, 1, 0, 0)] = h - (k + eps)
                        Diff[match[k]][i, st2stnum(new_combination, a - h, 1, 0, 0)] = h
                    else:
                        new_combination = combination.copy()
                        new_combination[j] += 1
                        P[match[k]][i, st2stnum(new_combination, a, 1, 0, 0)] = pH[j]

                # adversary
                new_combination = combination.copy()
                if latest == 1 and honest_available:
                    P[match[k]][i, st2stnum(new_combination, a+1, 2, 1, k)] = pA
                else:
                    P[match[k]][i, st2stnum(new_combination, a + 1, 0, 1, k)] = pA
            else:
                P[match[k]][i, 0] = 1
                Rs[match[k]][i, 0] = - 10000
                Diff[match[k]][i, 0] = 10000

    P = [matrix.tocsr(copy=True) for matrix in P]
    Rs = [matrix.tocsr(copy=True) for matrix in Rs]
    Diff = [matrix.tocsr(copy=True) for matrix in Diff]
    epsilon = 0.00001
    low_rou = 0
    high_rou = 1
    while high_rou - low_rou > epsilon / 8:
        rou = (high_rou + low_rou) / 2
        Wrou = [Rs[i] - rou * Diff[i] for i in range(choices)]
        mdp = mdptoolbox.mdp.RelativeValueIteration(P, Wrou, epsilon / 8)
        mdp.run()
        reward = mdp.average_reward
        # print('reward', reward)
        if reward > 0:
            low_rou = rou
        else:
            high_rou = rou
    reward_share = rou
    return reward_share

reward_share = MDP_matrices()
print(reward_share)
