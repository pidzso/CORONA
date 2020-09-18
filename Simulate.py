import quantecon as qe
import numpy as np
import matplotlib.pyplot as plt
from ProbabilityMX import mx
from Parameters import N, init, steps
from Parameters import infected_with_symptoms
from Parameters import infected_without_symptoms
from Parameters import recovered_with_immunity


# simulate step days within the SAIRD model with N people starting from ini
def simulate(NN, ini, step):
    # tracking the change in time
    track_S = []
    track_A = []
    track_I = []
    track_R = []
    track_D = []
    track_m = [[], []]

    # current cardinality
    S = int(NN * ini[0])
    A = int(NN * ini[1])
    I = int(NN * ini[2])
    R = int(NN * ini[3])
    D = int(NN * ini[4])

    # time spent within each state
    time_A = np.random.poisson(infected_without_symptoms, A)
    time_I = np.random.poisson(infected_with_symptoms, I)
    time_R = np.random.poisson(recovered_with_immunity, R)

    # state changers in the next step
    moves_A = np.sum((time_A == 0))
    moves_I = np.sum((time_I == 0))
    moves_R = np.sum((time_R == 0))

    for i in range(step):

        # record mx changes
        m = mx(S, A, I, R, D)
        track_m[0].append(m[0][0])
        track_m[1].append(m[0][1])

        mc = qe.MarkovChain(m, state_values=('S', 'A', 'I', 'R', 'D'))

        # record the current values for all the states
        track_S.append(S)
        track_A.append(A)
        track_I.append(I)
        track_R.append(R)
        track_D.append(D)

        # simulate one step for each state as many times as many people are in them
        # (ts_length=2 because it counts the starting state as 1)
        x, y = np.unique(mc.simulate(ts_length=2, init='S', num_reps=S, random_state=None), return_counts=True)
        s = dict(zip(x, y))
        x, y = np.unique(mc.simulate(ts_length=2, init='A', num_reps=moves_A, random_state=None), return_counts=True)
        a = dict(zip(x, y))
        x, y = np.unique(mc.simulate(ts_length=2, init='I', num_reps=moves_I, random_state=None), return_counts=True)
        i = dict(zip(x, y))
        x, y = np.unique(mc.simulate(ts_length=2, init='R', num_reps=moves_R, random_state=None), return_counts=True)
        r = dict(zip(x, y))

        # calculate newcomers the time they spent in states
        poi_A = np.random.poisson(infected_without_symptoms, s.get('A', 0))
        poi_I = np.random.poisson(infected_with_symptoms, a.get('I', 0))
        poi_R = np.random.poisson(recovered_with_immunity, a.get('R', 0) + i.get('R', 0))
        # update time remaining in each state and add newcomers
        time_A = np.append(np.add(time_A[time_A != 0], -1), poi_A)
        time_I = np.append(np.add(time_I[time_I != 0], -1), poi_I)
        time_R = np.append(np.add(time_R[time_R != 0], -1), poi_R)

        # state changers in the next step
        moves_A = np.sum((time_A == 0))
        moves_I = np.sum((time_I == 0))
        moves_R = np.sum((time_R == 0))

        # update the state counts
        # (need to remove the previous state counts as the new count does include them)
        S = s.get('S', 0) + r.get('S', 0) - S
        A = A - moves_A + s.get('A', 0)
        I = I - moves_I + a.get('I', 0)
        R = R - moves_R + a.get('R', 0) + i.get('R', 0)
        D = D + i.get('D', 0)

    # plot mx changes during step days
    plt.style.use('fivethirtyeight')
    plt.plot(range(step), track_m[0], label='SS')
    plt.plot(range(step), track_m[1], label='SA')

    # plotting the changes of states during step days
    #plt.plot(range(step), track_S, label='S')
    #plt.plot(range(step), track_A, label='A')
    #plt.plot(range(step), track_I, label='I')
    #plt.plot(range(step), track_R, label='R')
    #plt.plot(range(step), track_D, label='D')

    plt.legend()
    return plt.show()


simulate(N, init, steps)

#simulate(N, [0.99, 0.01, 0.00, 0.00, 0.00], steps)
#simulate(N, [0.90, 0.10, 0.00, 0.00, 0.00], steps)
#simulate(N, [0.80, 0.18, 0.02, 0.00, 0.00], steps)
#simulate(N, [0.80, 0.10, 0.10, 0.00, 0.00], steps)

