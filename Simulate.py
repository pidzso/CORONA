import quantecon as qe
import numpy as np
import matplotlib.pyplot as plt
from ProbabilityMX import mx
from Parameters import N, init, steps


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
        x, y = np.unique(mc.simulate(ts_length=2, init='A', num_reps=A, random_state=None), return_counts=True)
        a = dict(zip(x, y))
        x, y = np.unique(mc.simulate(ts_length=2, init='I', num_reps=I, random_state=None), return_counts=True)
        i = dict(zip(x, y))
        x, y = np.unique(mc.simulate(ts_length=2, init='R', num_reps=R, random_state=None), return_counts=True)
        r = dict(zip(x, y))
        x, y = np.unique(mc.simulate(ts_length=2, init='D', num_reps=D, random_state=None), return_counts=True)
        d = dict(zip(x, y))

        # update the state counts
        # (need to remove the previous state counts as the new count does include them)
        S = s.get('S', 0) + r.get('S', 0) - S
        A = a.get('A', 0) + s.get('A', 0) - A
        I = i.get('I', 0) + a.get('I', 0) - I
        R = r.get('R', 0) + a.get('R', 0) + i.get('R', 0) - R
        D = d.get('D', 0) + i.get('D', 0) - D

    # plot mx changes during step days
    plt.style.use('fivethirtyeight')
#    plt.plot(range(step), track_m[0], label='SS')
#    plt.plot(range(step), track_m[1], label='SA')

    # plotting the changes of states during step days
    plt.plot(range(step), track_S, label='S')
    plt.plot(range(step), track_A, label='A')
    plt.plot(range(step), track_I, label='I')
    plt.plot(range(step), track_R, label='R')
    plt.plot(range(step), track_D, label='D')

    plt.legend()
    return plt.show()


simulate(N, init, steps)

#simulate(N, [0.99, 0.01, 0.00, 0.00, 0.00], steps)
#simulate(N, [0.90, 0.10, 0.00, 0.00, 0.00], steps)
#simulate(N, [0.80, 0.18, 0.02, 0.00, 0.00], steps)
#simulate(N, [0.80, 0.10, 0.10, 0.00, 0.00], steps)

