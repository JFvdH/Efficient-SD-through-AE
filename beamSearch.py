"""
The following code was adapted from W. Duivesteijn, T.C. van Dijk. (2021)
    Exceptional Gestalt Mining: Combining Magic Cards to Make Complex Coalitions Thrive. 
    In: Proceedings of the 8th Workshop on Machine Learning and Data Mining for Sports Analytics.
    Available from http://wwwis.win.tue.nl/~wouter/Publ/J05-EMM_DMKD.pdf
"""

# Package imports
import heapq
import numpy as np

# Classes
class BoundedPriorityQueue:
    """
    Used to store the <q> most promising subgroups
    Ensures uniqness
    Keeps a maximum size (throws away value with least quality)
    """

    def __init__(self, bound): 
        # Initializes empty queue with maximum length of <bound>
        self.values = []
        self.bound = bound
        self.entry_count = 0

    def add(self, element, quality, **adds): 
        # Adds <element> to the bounded priority queue if it is of sufficient quality
        new_entry = (quality, self.entry_count, element, adds)
        if (len(self.values) >= self.bound):
            heapq.heappushpop(self.values, new_entry)
        else:
            heapq.heappush(self.values, new_entry)

        self.entry_count += 1

    def get_values(self):
        # Returns elements in bounded priority queue in sorted order
        for (q, _, e, x) in sorted(self.values, reverse=True):
            yield (q, e, x)

    def show_contents(self):  
        # Prints contents of the bounded priority queue (used for debugging)
        print("show_contents")
        for (q, entry_count, e) in self.values:
            print(q, entry_count, e)

class Queue:
    """
    Used to store candidate solutions
    Ensures uniqness
    """

    def __init__(self): # Initializes empty queue
        self.items = []

    def is_empty(self): # Returns True if queue is empty, False otherwise
        return self.items == []

    def enqueue(self, item): # Adds <item> to queue if it is not already present
        if item not in self.items:
            self.items.insert(0, item)

    def dequeue(self): # Pulls one item from the queue
        return self.items.pop()

    def size(self): # Returns the number of items in the queue
        return len(self.items)

    def get_values(self): # Returns the queue (as a list)
        return self.items

    def add_all(self, iterable): # Adds all items in <iterable> to the queue, given they are not already present
        for item in iterable:
            self.enqueue(item)

    def clear(self): # Removes all items from the queue
        self.items.clear()
        
# Functions
def refine(desc, more):
    # Creates a copy of the seed <desc> and adds it to the new selector <more>
    # Used to prevent pointer issues with selectors
    copy = desc[:]
    copy.append(more)
    return copy

def as_string(desc):
    # Adds ' and ' to <desc> such that selectors are properly separated when the refine function is used
    return ' and '.join(desc)

def eta(seed, df, features, n_chunks = 5):
    # Returns a generator which includes all possible refinements of <seed> for the given <features> on dataset <df>
    # n_chunks refers to the number of possible splits we consider for numerical features
    
    print("eta ", seed)
    if seed != []:              #we only specify more on the elements that are still in the subset
        d_str = as_string(seed)
        ind = df.eval(d_str)
        df_sub = df.loc[ind, ]
    else:
        df_sub = df
    for f in features:
        if (df_sub[f].dtype == 'float64') or (df_sub[f].dtype == 'float32'): #get quantiles here instead of intervals for the case that data are very skewed
            column_data = df_sub[f]
            dat = np.sort(column_data)
            dat = dat[np.logical_not(np.isnan(dat))]
            for i in range(1,n_chunks+1): #determine the number of chunks you want to divide your data in
                x = np.percentile(dat,100/i) #
                candidate = "{} <= {}".format(f, x)
                if not candidate in seed: # if not already there
                    yield refine(seed, candidate)
                candidate = "{} > {}".format(f, x)
                if not candidate in seed: # if not already there
                    yield refine(seed, candidate)
        elif (df_sub[f].dtype == 'object'):
            column_data = df_sub[f]
            uniq = column_data.dropna().unique()
            for i in uniq:
                candidate = "{} == '{}'".format(f, i)
                if not candidate in seed: # if not already there
                    yield refine(seed, candidate)
                candidate = "{} != '{}'".format(f, i)
                if not candidate in seed: # if not already there
                    yield refine(seed, candidate)
        elif (df_sub[f].dtype == 'int64'):
            column_data = df_sub[f]
            dat = np.sort(column_data)
            dat = dat[np.logical_not(np.isnan(dat))]
            for i in range(1,n_chunks+1): #determine the number of chunks you want to divide your data in
                x = np.percentile(dat,100/i) #
                candidate = "{} <= {}".format(f, x)
                if not candidate in seed: # if not already there
                    yield refine(seed, candidate)
                candidate = "{} > {}".format(f, x)
                if not candidate in seed: # if not already there
                    yield refine(seed, candidate)
        elif (df_sub[f].dtype == 'bool'):
            uniq = column_data.dropna().unique()
            for i in uniq:
                candidate = "{} == '{}'".format(f, i)
                if not candidate in seed: # if not already there
                    yield refine(seed, candidate)
                candidate = "{} != '{}'".format(f, i)
                if not candidate in seed: # if not already there
                    yield refine(seed, candidate)
        else:
            assert False
            
def satisfies_all(desc, df, threshold=0.02):
    # Function used to check if subgroup with pattern <desc> is sufficiently big relative to its dataset <df>
    # A subgroup is sufficiently big if the proportion of data included in it exceeds <threshold>   
    d_str = as_string(desc)
    ind = df.eval(d_str)
    return sum(ind) >= len(df) * 0.02 

def eval_quality(desc, df, target):
    # Function used to calculate the solution's WRAcc
    sub_group = df[df.eval(as_string(desc))] 
    prop_p_sg = len(sub_group[sub_group[target]==1])/len(sub_group)
    prop_p_df = len(df[df[target]==1])/len(df)
    wracc = ((len(sub_group)/len(df))**1) * (prop_p_sg - prop_p_df) #for WRAcc a=1
    return wracc

def EMM(w, d, q, catch_all_description, df, features, target, n_chunks=5, ensure_diversity = False):
    """
    w - width of beam, i.e. the max number of results in the beam
    d - num levels, i.e. how many attributes are considered
    q - max results, i.e. max number of results output by the algorithm
    eta - a function that receives a description and returns all possible refinements
    satisfies_all - a function that receives a description and verifies wheather it satisfies some requirements as needed
    eval_quality - returns a quality for a given description. This should be comparable to qualities of other descriptions
    catch_all_description - the equivalent of True, or all, as that the whole dataset shall match
    df - dataframe of mined dataset
    features - features in scope
    target - column name of target attribute in df
    """
    
    # Initialize variables
    resultSet = BoundedPriorityQueue(q) # Set of results, can contain results from multiple levels
    candidateQueue = Queue() # Set of candidate solutions to consider adding to the ResultSet
    candidateQueue.enqueue(catch_all_description) # Set of results on a particular level
    error = 0.00001 # Allowed error margin (due to floating point error) when comparing the quality of solutions

    # Perform BeamSearch for <d> levels
    for level in range(d):
        print("level : ", level)
        
        # Initialize this level's beam
        beam = BoundedPriorityQueue(w)

        # Go over all rules generated on previous level, or 'empty' rule if level = 0 
        for seed in candidateQueue.get_values():
            print("    seed : ", seed)
            
            # Start by evaluating the quality of the seed
            if seed != []:
                seed_quality = eval_quality(seed, df, target)
            else:
                seed_quality = 99

            # For all refinements created by eta function on descriptions (i.e features), which can be different types of columns
            # eta(seed) reads the dataset given certain seed (i.e. already created rules) and looks at new descriptions
            for desc in eta(seed, df, features, n_chunks):

                # Check if the subgroup contains at least x% of data, proceed if yes
                if satisfies_all(desc, df):

                    # Calculate the new solution's quality
                    quality = eval_quality(desc, df, target)
                    
                    # Ensure diversity by forcing difference in quality when compared to its seed
                    # if <ensure_diversity> is set to True. Principle is based on:
                    # Van Leeuwen, M., & Knobbe, A. (2012), Diverse subgroup set discovery.
                    # Data Mining and Knowledge Discovery, 25(2), 208-242.
                    if ensure_diversity:
                        if quality < (seed_quality * 1-error) or quality > (seed_quality * 1+error) :
                            resultSet.add(desc, quality)
                            beam.add(desc, quality)
                    else:
                        resultSet.add(desc, quality)
                        beam.add(desc, quality)

        # When all candidates for a search level have been explored, 
        # the contents of the beam are moved into candidateQueue, to generate next level candidates
        candidateQueue = Queue()
        candidateQueue.add_all(desc for (_, desc, _) in beam.get_values())
        
    # Return the <resultSet> once the BeamSearch algorithm has completed
    return resultSet