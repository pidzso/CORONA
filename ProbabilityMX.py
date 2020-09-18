from Parameters import mortality_rate
from Parameters import infected_with_symptoms
from Parameters import infected_without_symptoms
from Parameters import symptom_ratio
from Parameters import recovered_with_immunity
#from Parameters import stay_home
#from Parameters import mask_wearing_ratio
#from Parameters import mask_eff


def infection_rate(S, A, I, R, D):  # infection rate
    return A / (S + A + R)


def SS(S, A, I, R, D):  # based on the expected number of days within this state
    return 1 - infection_rate(S, A, I, R, D)


def AA(S, A, I, R, D):  # based on the expected number of days within this state
    return 1 - 1 / infected_without_symptoms


def II(S, A, I, R, D):  # based on the expected number of days within this state
    return 1 - 1 / infected_with_symptoms


def RR(S, A, I, R, D):  # based on the expected number of days within this state
    return 1 - 1 / recovered_with_immunity


def DD(S, A, I, R, D):  # NOONE COMES BACK FROM THE DEAD
    return 1


def AI(S, A, I, R, D):  # developing symptoms
    return (1 - AA(S, A, I, R, D)) / (symptom_ratio + 1)


def AR(S, A, I, R, D):  # recovering without symptoms
    return (1 - AA(S, A, I, R, D)) / (symptom_ratio + 1) * symptom_ratio


def ID(S, A, I, R, D):  # dieing
    return mortality_rate * (1 - II(S, A, I, R, D))


def IR(S, A, I, R, D):  # recovering
    return 1 - II(S, A, I, R, D) - ID(S, A, I, R, D)


def RS(S, A, I, R, D):  # loosing immunity
    return 1 - RR(S, A, I, R, D)


def SA(S, A, I, R, D):  # getting infected
    return infection_rate(S, A, I, R, D)


def mx(S, A, I, R, D):  # probability matrix of transition
    return [[SS(S, A, I, R, D), SA(S, A, I, R, D), 0.,                0.,                0.],
            [0.,                AA(S, A, I, R, D), AI(S, A, I, R, D), AR(S, A, I, R, D), 0.],
            [0.,                0.,                II(S, A, I, R, D), IR(S, A, I, R, D), ID(S, A, I, R, D)],
            [RS(S, A, I, R, D), 0.,                0.,                RR(S, A, I, R, D), 0.],
            [0.,                0.,                0.,                0.,                DD(S, A, I, R, D)]]
