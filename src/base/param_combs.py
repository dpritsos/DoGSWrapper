"""
    This Module contains a recursive function for returning all possible combinations form a
    list of lists of all possible values of some parameters (or set of values). In addition, 
    it contains an Iterator for returning these combinations in a ordered dictionary in the 
    same order as the order given as an Initialization/Input to the iterator. 

    Details can be found in the descriptions of the iterator's class and the recursive function 
    respectively. 

"""


import collections as coll
import numpy as np


def BuildCombinations(param_vals_ll):
    """Building all possible combinations given some lists of values in a list.

    Given several lists of values in a python list, this function will return all
    possible combination of the values of these lists. The algorithm producing the 
    combinations is a recursion over the input lists. The output is a list of lists 
    of the combination. 

    Input:
        param_vals_ll: A list of lists. Each of the lists contains the different 
            possible values of a parameter (or any set of values).

    Output: 
        param_combs: A list of all possible combination produced my the recursion.

    NOTE: The output preserves inputs order.

    """

    if len(param_vals_ll) == 0:
        #When recursion reaches the end condition return an empty list-list.
        return [[]]

    else:
        #For each list of values in the input list create a list of combinations.
        for i, param_vals_l in enumerate(param_vals_ll):
            param_combs = list()
            
            #For each value of the list of values extend the param_combs list created.
            for param_val in param_vals_l:
                
                #Giving the rest of the list of parameter values list to this function itself.
                pvals_l_residual = BuildCombinations(param_vals_ll[i+1:])

                #Extending the list of the current parameter values with the subsequent parameter values lists.
                param_combs.extend( [ [param_val] + pval for pval in pvals_l_residual ] )
            
            #Returning the list of combinations.
            return param_combs



class ParamGridIter(object):
    """Iterator returning the ordered-dictionaries of parameter value combinations derived given
    a dictionary of parameters(dictionary keys) together with their possible values(dictionary
    values). 

    Given a dictionary(or ordered-dictionary) where keys are the parameter names and values are the lists 
    of parameter possible values this class defines an iterator which outputs a combination of values. 
    The output is a dictionary with parameter names as keys and a values, of the possible ones, as a value 
    of the dictionary. The ending condition is the end of the list of possible combinations.

    Initialization/Input:
        parmas_vals_d: The dictionary of parameters contains the lists of all possible values for each parameter.
        mod: Defines the output mode of the iterator. 
            Valid options are: 
                'dict': for pyton (collections) ordered dictionary. (default)
                'list' : for python list (ordered as input dictionary).
                'array': for numpy.array (ordered as input dictionary).
                'path' : for URL like path (ordered as input dictionary). Compatible to the pyTables path tree.

    Iteration Output:
        params_comb_ordered: A possible combinations of parameter values. The output format depends on 'mod' value.
            In the case of ordered dictionary keys are the names of the parameters.

    """

    def __init__(self, params_vals_d, mod='dict'):

        #Getting the list of paramter names.
        self.params_lst = params_vals_d.keys()

        #Building the list of all possible combinations lists.
        self.params_vals_combs = BuildCombinations( params_vals_d.values() )

        #Getting the output mode. Checking for validity fist.
        if mod in ['dict', 'list', 'array', 'path']:
            self.mod = mod
        else:
            raise Exception("Invalid value for 'mod' initialization argument: Valid values are 'dict', 'lst', 'array', 'path'.")

        #Initialising the counter of all possible combination. 
        self.combs_cnt = -1


    def __iter__(self):

        return self


    def next(self):

        self.combs_cnt += 1
        
        #Ending condition when the length of combination list is equal to combination counter 
        #which is the selection index, too.
        if self.combs_cnt == len(self.params_vals_combs):
            raise StopIteration
        
        #Prepare output depending on the 'mod' value
        if self.mod == 'dict':

            #Convert the combination values list into combination ordered dictionay. 
            params_comb_ordered = coll.OrderedDict( zip(self.params_lst, self.params_vals_combs[self.combs_cnt]) )

        elif self.mod == 'list':

            #Let the combination as it is, i.e. a python list.
            params_comb_ordered = self.params_vals_combs[self.combs_cnt]

        elif self.mod == 'array':

            #Convert the combination values list into numpy.array.
            params_comb_ordered = np.array( self.params_vals_combs[self.combs_cnt] )            

        elif self.mod == 'path':

            #Convert the combination values list into URL like path.
            params_path = '/'.join( [ key+str(value).replace('.','') for key, value in zip(self.params_lst, self.params_vals_combs[self.combs_cnt]) ] )
            params_comb_ordered = '/' + params_path

        return params_comb_ordered

