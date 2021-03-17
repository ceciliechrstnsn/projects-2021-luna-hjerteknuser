############################### multidimensional solver #######################################
from scipy import optimize
import numpy as np

def u_func(x, mp):
    '''
    Calculates utility for a chosen (consumption, housing) bundle

    Args:
    x (array)           : Array of length 2, first element is housing and second element is consumption 
    mp (dictionary)     : should contain the key phi and its respective value

    Returns:
    (float)             : utility of selected bundle 

    '''
    h = x[0]
    c = x[1]
    return c**(1-mp['phi'])*h**mp['phi']

def expenditure(x,mp):
    '''
    Calculates expenditure of a chosen (consumption, housing) bundle

    Args:
    x (array)           : Array of length 2, first element is housing and second element is consumption 
    mp (dictionary)     : Should contain the keys: r, tau_g, epsilon, tau_p, p_bar and their respective values

    Returns:
    (float)             : Expenditure of selected bundle 

    '''
    h = x[0]
    c = x[1]
    h_cost = mp['r']*h+mp['tau_g']*h*mp['epsilon']+mp['tau_p']*max(h*mp['epsilon']-mp['p_bar'],0)
    E = c+h_cost
    return E


def optimizer(mp, printres = False):
    '''
    Uses a multi-dimensional constrained solver, that is the minimize function from the scipy.optimize package (method SLSQP), to calculate optimal allocation of ressources, i.e. optimal choice of consumption and housing quality.

    Args:
    mp (dictionary)     : should contain keys: phi, r, tau_g, tau_p, epsilon, p_bar, m and their respective values
    printres (boolean)  : OPTIONAL, is False by default - if True, the result is printed

    Returns:
    3 (float) parameters: first parameter is choice of housing quality, second is choice of consumption and lastly is the utility given choice of spending 

    '''
    initial_guess = ([0.5, 20]) # some guess
    constraints = ({'type': 'ineq',
                    'fun': lambda x: mp['m']-expenditure(x,mp)})
    sol = optimize.minimize(lambda x: -u_func(x, mp), initial_guess,
                            method = 'SLSQP',  constraints=constraints)
    h=sol.x[0]
    c=sol.x[1]
    e=expenditure(sol.x,mp)
    u=u_func(sol.x,mp)

    if printres == True:
        emDKK = mp['m']-e
        print(f'For house quality={h:.2f} and consumption={c:.2f} with total expenditure={e:.2f}              \nLeaves us with utility of {u:.2f} and excess {emDKK:.2f}mDKK')
    return h, c, u


################################## scalar solver #######################################

def u_func_1d(c,h, mp):
    '''
    Calculates utility for a chosen (consumption, housing) bundle

    Args:
    c (float)           : choice of consumption 
    h (float)           : choice of housing quality
    mp (dictionary)     : should contain the key phi and its respective value

    Returns:
    (float)             : utility of selected bundle 

    '''
    return c**(1-mp['phi'])*h**mp['phi']

def expenditure_1d(c,h,mp):
    '''
    Calculates expenditure of a chosen (consumption, housing) bundle

    Args:
    c (float)           : choice of consumption 
    h (float)           : choice of housing quality 
    mp (dictionary)     : Should contain the keys: r, tau_g, epsilon, tau_p, p_bar and their respective values

    Returns:
    (float)             : Expenditure of selected bundle 

    '''
    h_cost = mp['r']*h+mp['tau_g']*h*mp['epsilon']+mp['tau_p']*max(h*mp['epsilon']-mp['p_bar'],0)
    E = c+h_cost
    return E


def objective_function(h,mp):
    '''
    Creates an objective function based on housing, where consumption is implicitly given by the model constraints. That is, the (negative) utility is calculated based on a choice of housing and model parameters.

    Args:
    h (float)           : choice of housing quality 
    mp (dictionary)     : Should contain the keys: m, r, tau_g, epsilon, tau_p, p_bar and their respective values

    Returns:
    (float)             : negative utility for a given level of housing and the model parameters

    '''
    h_tax =  mp['tau_g']*h*mp['epsilon']+mp['tau_p']*max(h*mp['epsilon']-mp['p_bar'],0)
    h_loanc = mp['r']*h
    c = mp['m'] - h_tax - h_loanc
    return -u_func_1d(c,h,mp)


def scalar_optimizer(mp, printres = False):
    '''
    Uses the non-bounded minimize_scalar function from the scipy.optimize package, exploiting the monotonicity of the problem, to calculate optimal allocation of ressources, i.e. optimal choice of consumption and housing quality.

    Args:
    mp (dictionary)     : should contain keys: phi, r, tau_g, tau_p, epsilon, p_bar, m and their respective values
    printres (boolean)  : OPTIONAL, is False by default - if True, the result is printed

    Returns:
    3 (float) parameters: first parameter is choice of housing quality, second is choice of consumption and lastly is the utility given choice of spending

    '''
    sol = optimize.minimize_scalar(objective_function, args = (mp), bounds = None)
    h = sol.x
    c = mp['m'] - mp['r']*h-mp['tau_g']*h*mp['epsilon']-mp['tau_p']*max(h*mp['epsilon']-mp['p_bar'],0)
    u = u_func_1d(c,h,mp)
    e = expenditure_1d(c,h,mp)

    if printres == True:
        emDKK = mp['m']-e
        print(f'For house quality={h:.2f} and consumption={c:.2f} with total expenditure={e:.2f}\nLeaves us with utility of {u:.2f} and excess {emDKK:.2f}mDKK')
    return h, c, u


############################### Calculating tax revenue for Q3 ####################################
def taxrev(mp, cash, N, printres = False):
    '''
    Calculates the average tax burden given model parameters (mp), a distribution of cash-on-hand (cash) and a population size of N. This is done by calculating the optimal choice of spending and the implied tax payments associated with this choice.

    Args:
    mp (dictionary)     : should contain keys: phi, r, tau_g, tau_p, epsilon, p_bar, m and their respective values.
    cash (array)        : array of length equal to population size. Contains cash-on-hand for each individual in population, i.e. the distribution of cash-on-hand.
    N (integer)         : denotes the size of population.
    printres (boolean)  : OPTIONAL, is False by default - if True, the result is printed.

    Returns:
     (float) parameters : average tax rate given model parameters, distribution of cash-on-hand in the population.
     (array)            : Array of length equal to population size. Contains optimal choice of housing quality for each individual in population.

    '''
    # Initializing h-star storage and total tax payments
    h_stars1 = np.empty(N)
    total_tax = 0
    
    
    # Looping over different levels of income to get choice of spending
    for i, v in enumerate(cash):
    # Storing the original m-value
        if i == 0:
            m_save = mp['m']
    
        # updating dictionary with new m-value (level of income)
        mp['m'] = v
    
        # Solving with updated dictionary
        result_vector = scalar_optimizer(mp)
    
        # loading choice of housing quality
        h_stars1[i] = result_vector[0]
        # Calculating tax revenue for choice of housing quality
        tax = (mp['tau_g']*h_stars1[i]*mp['epsilon']+mp['tau_p']*max(h_stars1[i]*mp['epsilon']-mp['p_bar'],0))
        total_tax += tax
        
        # Restoring original m in dictionary
        if i == len(cash)-1:
            mp['m'] = m_save
    
    # Calculating average tax burden for each household
    average_t = total_tax/N

    # If-block that prints the result
    if printres == True:
        avg_m = np.mean(cash)
        inc_pct = average_t/avg_m*100
        print(f'Average tax burden for the population is {average_t:.3f} mDKK, corresponding to {inc_pct:.2f} pct. of average cash-on-hand')

    return average_t, h_stars1





############################### Calculating tax revenue for Q5 ####################################
def tax_objective_function(tau_g,mp,tax_target,cash,N):
    '''
    Calculates the difference between average tax burden and tax target for model parameters (mp), a distribution of cash-on-hand (cash) and a population size of N. This is done by calculating the optimal choice of spending and the implied tax payments associated with this choice.

    Args:
    tau_g (float)       : the flat housing tax
    mp (dictionary)     : should contain keys: phi, r, tau_g, tau_p, epsilon, p_bar, m and their respective values.
    cash (array)        : array of length equal to population size. Contains cash-on-hand for each individual in population, i.e. the distribution of cash-on-hand.
    N (integer)         : denotes the size of population.

    Returns:
     (float) parameters : average tax rate given model parameters, distribution of cash-on-hand in the population.

    '''
    mp['tau_g'] = tau_g
    current_tax = taxrev(mp, cash, N)[0]
    tax_difference = abs(tax_target - current_tax)
    return tax_difference