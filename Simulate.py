import quantecon as qe
import numpy as np
import matplotlib.pyplot as plt


# probability matrix of transition
def mx(S, A, R, dst_r, msk_r, msk_e, sym_r, mor_r):

    # distancing only matters for S, while mask works both for S and A
    spread = (1 - dst_r) * (np.power((1 - msk_r), 2) + np.power(msk_r * (1 - msk_e), 2))

    SA = spread * A / (S + A + R)  # chance of not getting infected
    SS = 1 - SA                    # chance of getting infected
    AI = 1 / (sym_r + 1)           # chance of developing symptoms
    AR = sym_r / (sym_r + 1)       # chance of not developing symptoms
    ID = mor_r                     # chance of dieing
    IR = 1 - mor_r                 # chance of recovering

    return [[SS, SA, 0., 0., 0.],
            [0., 0., AI, AR, 0.],
            [0., 0., 0., IR, ID],
            [1,  0., 0., 0., 0.],
            [0., 0., 0., 0., 1.]]


# simulate step days within the SAIRD model with N people starting from ini
def simulate(ini=[0.9, 0.1, 0.0, 0.0, 0.0], NN=1000000, step=200, sym_r=4., mor_r=0.02,
             dst_r=0., msk_r=0., msk_e=1., inf_wo_s=7., inf_w_s=14., rec=56.):

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
    time_A = np.random.poisson(inf_wo_s, A)
    time_I = np.random.poisson(inf_w_s, I)
    time_R = np.random.poisson(rec, R)

    # state changers in the next step
    moves_A = np.sum((time_A == 0))
    moves_I = np.sum((time_I == 0))
    moves_R = np.sum((time_R == 0))

    for i in range(step):

        # record mx changes
        m = mx(S, A, R, dst_r, msk_r, msk_e, sym_r, mor_r)
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
        poi_A = np.random.poisson(inf_wo_s, s.get('A', 0))
        poi_I = np.random.poisson(inf_w_s, a.get('I', 0))
        poi_R = np.random.poisson(rec, a.get('R', 0) + i.get('R', 0))
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
    '''
    # plot mx changes during step days
    plt.style.use('fivethirtyeight')
    #plt.plot(range(step), track_m[0], label='SS')
    #plt.plot(range(step), track_m[1], label='SA')

    string = 'S' + str(init[0]) + '_A' + str(init[1]) + '_D' + str(dst_r) + \
             '_MR' + str(msk_r) + '_ME' + str(msk_e)

    # plotting the changes of states during step days
    plt.plot(range(step), track_I, label='I')
    plt.plot(range(step), track_D, label='D')

    plt.legend()
    plt.savefig(string + '_ID.png')

    plt.plot(range(step), track_S, label='S')
    plt.plot(range(step), track_A, label='A')
    plt.plot(range(step), track_R, label='R')

    plt.legend()
    plt.savefig(string + '_SAIRD.png')

    plt.show()
    '''
    return [track_m, track_S, track_A, track_I, track_R, track_D]


[base_m, base_S, base_A, base_I, base_R, base_D] = simulate(
                                                   ini=[0.9, 0.1, 0.0, 0.0, 0.0],  # initial distribution
                                                   NN=1000000,   # population size
                                                   step=200,     # simulation steps
                                                   sym_r=4,      # ratio between asymptomatic and symptoms ppl
                                                   mor_r=0.02,   # probability to die from COVID when symptotic
                                                   dst_r=0,      # ratio of ppl staying home
                                                   msk_r=0,      # ratio of ppl wearing masks
                                                   msk_e=1,      # efficiency of masks in stopping the spreading
                                                   inf_wo_s=7,   # expected number of days in state A
                                                   inf_w_s=14,   # expected number of days in state I
                                                   rec=56)       # expected number of days in state R


[dist_m,  dist_S,  dist_A,  dist_I,  dist_R,  dist_D]  = simulate(dst_r=0.5)
[msk_m,   msk_S,   msk_A,   msk_I,   msk_R,   msk_D]   = simulate(msk_r=0.5)
[dima_m,  dima_S,  dima_A,  dima_I,  dima_R,  dima_D]  = simulate(dst_r=0.5, msk_r=0.5)
[mske_m,  mske_S,  mske_A,  mske_I,  mske_R,  mske_D]  = simulate(msk_r=0.5, msk_e=0.5)
[dimae_m, dimae_S, dimae_A, dimae_I, dimae_R, dimae_D] = simulate(dst_r=0.5, msk_r=0.5, msk_e=0.5)

with open('result.txt', 'w') as f:
    print(base_m,  '\n', base_S,  '\n', base_A,  '\n', base_I,  '\n', base_R,  '\n', base_D, '\n',
          dist_m,  '\n', dist_S,  '\n', dist_A,  '\n', dist_I,  '\n', dist_R,  '\n', dist_D, '\n',
          msk_m,   '\n', msk_S,   '\n', msk_A,   '\n', msk_I,   '\n', msk_R,   '\n', msk_D,  '\n',
          mske_m,  '\n', mske_S,  '\n', mske_A,  '\n', mske_I,  '\n', mske_R,  '\n', mske_D, '\n',
          dima_m,  '\n', dima_S,  '\n', dima_A,  '\n', dima_I,  '\n', dima_R,  '\n', dima_D, '\n',
          dimae_m, '\n', dimae_S, '\n', dimae_A, '\n', dimae_I, '\n', dimae_R, '\n', dimae_D, '\n', file=f)

#plt.style.use('fivethirtyeight')
#plt.plot(range(len(base_m[0])), base_m[0], label='')
#plt.plot(range(len(dist_m[0])), dist_m[0], label='')
#plt.legend()
#plt.show()
