# Simulation Parameters
steps = 100                 # simulation steps
init = [0.9, 0.1, 0, 0, 0]  # initial distribution
N = 1000000                 # population size

# Model Parameters
infected_without_symptoms = 7       # expected number of days
infected_with_symptoms    = 2 * 7   # expected number of days
recovered_with_immunity   = 8 * 7   # expected number of days
symptom_ratio             = 10      # ratio between asymptomatic and symptoms people
mortality_rate            = 0.02    # probability to die from COVID
#staying_home             = 0.1     # portion of population staying home
#mask_wearing_ratio       = 0.5     # portion of population wearing mask
#mask_eff                 = 1       # efficiency of the mask in stopping COVID