import quantecon as qe
import numpy as np
import matplotlib.pyplot as plt


class COVID:

    def __init__(self, ini=[0.9, 0.1, 0.0, 0.0, 0.0],  # initial distribution
                       NN=1000000,   # population size
                       step=200,     # simulation steps
                       sym_r=4,      # ratio between asymptomatic and symptoms ppl
                       mor_r=0.02,   # probability to die from COVID when symptotic
                       cost_i=10.,   # cost of having symptoms
                       cost_d=100.,  # cost of dieing
                       cost_dst=2.,  # cost of staying home
                       cost_msk=1.,  # cost of wearing a mask
                       msk_e=1,      # efficiency of masks in stopping the spreading
                       inf_wo_s=7,   # expected number of days in state A
                       inf_w_s=14,   # expected number of days in state I
                       rec=56):       # expected number of days in state R):
        self.ini      = ini
        self.NN       = NN
        self.step     = step
        self.sym_r    = sym_r
        self.mor_r    = mor_r
        self.cost_i   = cost_i
        self.cost_d   = cost_d
        self.cost_msk = cost_msk
        self.cost_dst = cost_dst
        self.msk_e    = msk_e
        self.inf_wo_s = inf_wo_s
        self.inf_w_s  = inf_w_s
        self.rec      = rec

    # probability matrix of transition
    def mx(self, S, A, R, dst_r, msk_r):

        # distancing only matters for S, while mask works both for S and A
        spread = (1 - dst_r) * (np.power((1 - msk_r), 2) + np.power(msk_r * (1 - self.msk_e), 2))

        SA = spread * A / (S + A + R)       # chance of getting infected
        SS = 1 - SA                         # chance of not getting infected
        AI = 1 / (self.sym_r + 1)           # chance of developing symptoms
        AR = self.sym_r / (self.sym_r + 1)  # chance of not developing symptoms
        ID = self.mor_r                     # chance of dieing when symptomatic
        IR = 1 - self.mor_r                 # chance of recovering from symptoms

        return [[SS, SA, 0., 0., 0.],
                [0., 0., AI, AR, 0.],
                [0., 0., 0., IR, ID],
                [1,  0., 0., 0., 0.],
                [0., 0., 0., 0., 1.]]

    # determine distancing and mask rates as individuals
    def play_alone(self, S, A, I, R, D):
        # ToDo
        dsk_r = 0.
        msk_r = 0.
        return dsk_r, msk_r

    # determine distancing and mask rates as government
    def play_together(self, S, A, I, R, D):
        # ToDo
        dsk_r = 0.
        msk_r = 0.
        return dsk_r, msk_r

    # simulate the process how the SAIRD states develop taking into account the ppl decision stepwise
    def simulate(self):

        # tracking the change in time
        track_S = []
        track_A = []
        track_I = []
        track_R = []
        track_D = []
        track_SA = []
        track_dst = []
        track_msk = []

        # starting cardinality
        S = int(self.NN * self.ini[0])
        A = int(self.NN * self.ini[1])
        I = int(self.NN * self.ini[2])
        R = int(self.NN * self.ini[3])
        D = int(self.NN * self.ini[4])

        # time spent within A, I, R
        time_A = np.random.poisson(self.inf_wo_s, A)
        time_I = np.random.poisson(self.inf_w_s, I)
        time_R = np.random.poisson(self.rec, R)

        # count the state changers in the first step
        moves_A = np.sum((time_A == 0))
        moves_I = np.sum((time_I == 0))
        moves_R = np.sum((time_R == 0))

        for i in range(self.step):

            # play the game in current state to determine distancing and mask rates
            dst_r, msk_r = self.play_alone(S, A, I, R, D)
            #dst_r, msk_r = play_together(S, A, I, R, D)

            # update the mx and the mc
            m = self.mx(S, A, R, dst_r, msk_r)
            mc = qe.MarkovChain(m, state_values=('S', 'A', 'I', 'R', 'D'))

            # record the current values
            track_S.append(S)
            track_A.append(A)
            track_I.append(I)
            track_R.append(R)
            track_D.append(D)
            track_SA.append(m[0][1])
            track_dst.append(dst_r)
            track_msk.append(msk_r)

            # simulate one step for each state as many times as many people are in them
            x, y = np.unique(mc.simulate(ts_length=2, init='S', num_reps=S, random_state=None), return_counts=True)
            s = dict(zip(x, y))
            x, y = np.unique(mc.simulate(ts_length=2, init='A', num_reps=moves_A, random_state=None), return_counts=True)
            a = dict(zip(x, y))
            x, y = np.unique(mc.simulate(ts_length=2, init='I', num_reps=moves_I, random_state=None), return_counts=True)
            i = dict(zip(x, y))
            x, y = np.unique(mc.simulate(ts_length=2, init='R', num_reps=moves_R, random_state=None), return_counts=True)
            r = dict(zip(x, y))

            # calculate for newcomers the time they spent in A, I, R
            poi_A = np.random.poisson(self.inf_wo_s, s.get('A', 0))
            poi_I = np.random.poisson(self.inf_w_s, a.get('I', 0))
            poi_R = np.random.poisson(self.rec, a.get('R', 0) + i.get('R', 0))

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

        return [track_dst, track_msk, track_SA, track_S, track_A, track_I, track_R, track_D]

    # plot the changes of states, distancing and mask habits and infection porobabilities
    def create_plot(self, track_dst, track_msk, track_SA, track_S, track_A, track_I, track_R, track_D):

        string = 'S' + str(self.init[0]) + '_A' + str(self.init[1]) + '_ME' + str(self.msk_e)

        # plot mx changes during step days
        plt.style.use('fivethirtyeight')
        plt.plot(range(self.step), track_SA, label='SA')
        plt.legend()
        plt.savefig(string + '_probability.png')
        plt.close()

        # plot distancing & mask habits during step days
        plt.style.use('fivethirtyeight')
        plt.plot(range(self.step), track_dst, label='Distancing')
        plt.plot(range(self.step), track_msk, label='Mask')
        plt.legend()
        plt.savefig(string + '_dist+mask.png')
        plt.close()

        # plotting the changes of I, D during step days
        plt.style.use('fivethirtyeight')
        plt.plot(range(self.step), track_I, label='I')
        plt.plot(range(self.step), track_D, label='D')
        plt.legend()
        plt.savefig(string + '_ID.png')
        plt.close()

        # plotting the changes of S, A, R during step days
        plt.style.use('fivethirtyeight')
        plt.plot(range(self.step), track_S, label='S')
        plt.plot(range(self.step), track_A, label='A')
        plt.plot(range(self.step), track_R, label='R')
        plt.legend()
        plt.savefig(string + '_SAIRD.png')
        plt.close()

covid = COVID()
[dst, msk, prob, S, A, I, R, D] = covid.simulate()

with open('result.txt', 'w') as f:
    print(dst, '\n', msk, '\n', prob, '\n', S, '\n', A, '\n', I, '\n', R, '\n', D, '\n', file=f)

plt.style.use('fivethirtyeight')
plt.plot(range(len(prob)), prob, label='Probability of Infection')
plt.legend()
plt.show()
