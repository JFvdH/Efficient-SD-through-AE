# Package imports
import copy
import itertools
import pysubgroup as ps
from pysubgroup import *

# Overwrite of the pysubgroup add_if_required function to ensure diversity by forcing difference 
# in quality when compared to its seed. Principle is based on: Van Leeuwen, M., & Knobbe, A. (2012), 
# Diverse subgroup set discovery. Data Mining and Knowledge Discovery, 25(2), 208-242.
def add_if_required(result, sg, quality, task, check_for_duplicates=True, statistics=None):
    if quality > task.min_quality:
        if not ps.constraints_satisfied(task.constraints, sg, statistics, task.data):
            return
        if check_for_duplicates:
            sg_set = set(str(sg).split())
            if 'AND' in sg_set:
                sg_set.remove('AND')
            subsets = list(map(set,itertools.combinations(sg_set, 1)))
            for exist_quality, exist_sg, _ in result:
                if (quality > exist_quality - 0.000001) and (quality < exist_quality + 0.000001):
                    exist_sg_set = set(str(exist_sg).split())
                    if 'AND' in exist_sg_set:
                        exist_sg_set.remove('AND')
                    if any(subset.issubset(exist_sg_set) for subset in subsets):
                        return
        if len(result) < task.result_set_size:
            heappush(result, (quality, sg, statistics))
        elif quality > result[0][0]:
            heappop(result)
            heappush(result, (quality, sg, statistics))
            
# Overwrite of the pysubgroup BestFirstSearch class to use the overwritten pysubgroup add_if_required function 
class adjusted_BestFirstSearch(BestFirstSearch):
    def execute(self, task):
        result = []
        queue = [(float("-inf"), ps.Conjunction([]))]
        operator = ps.StaticSpecializationOperator(task.search_space)
        task.qf.calculate_constant_statistics(task.data, task.target)
        while queue:
            q, old_description = heappop(queue)
            q = -q
            if not q > ps.minimum_required_quality(result, task):
                break
            for candidate_description in operator.refinements(old_description):
                sg = candidate_description
                statistics = task.qf.calculate_statistics(sg, task.target, task.data)
                add_if_required(result, sg, task.qf.evaluate(sg, task.target, task.data, statistics), task, statistics=statistics)
                if len(candidate_description) < task.depth:
                    optimistic_estimate = task.qf.optimistic_estimate(sg, task.target, task.data, statistics)

                    # compute refinements and fill the queue
                    if optimistic_estimate >= ps.minimum_required_quality(result, task):
                        if ps.constraints_satisfied(task.constraints_monotone, candidate_description, statistics, task.data):
                            heappush(queue, (-optimistic_estimate, candidate_description))

        result.sort(key=lambda x: x[0], reverse=True)
        return ps.SubgroupDiscoveryResult(result, task)
    
# Overwrite of the pysubgroup DFS class to use the overwritten pysubgroup add_if_required function 
class adjusted_DFS(SimpleDFS):
    def search_internal(self, task, prefix, modification_set, result, use_optimistic_estimates):
        sg = ps.Conjunction(copy.copy(prefix))
        statistics = task.qf.calculate_statistics(sg, task.target, task.data)
        if use_optimistic_estimates and len(prefix) < task.depth and isinstance(task.qf, ps.BoundedInterestingnessMeasure):
            optimistic_estimate = task.qf.optimistic_estimate(sg, task.target, task.data, statistics)
            if not optimistic_estimate > ps.minimum_required_quality(result, task):
                return result
        quality = task.qf.evaluate(sg, task.target, task.data, statistics)
        add_if_required(result, sg, quality, task, check_for_duplicates=True, statistics=statistics)
        if not ps.constraints_satisfied(task.constraints_monotone, sg, statistics=statistics, data=task.data):
            return
        
        if len(prefix) < task.depth:
            new_modification_set = copy.copy(modification_set)
            for sel in modification_set:
                prefix.append(sel)
                new_modification_set.pop(0)
                self.search_internal(task, prefix, new_modification_set, result, use_optimistic_estimates)
                # remove the sel again
                prefix.pop(-1)
        return result

# Overwrite of the pysubgroup Apriori class to use the overwritten pysubgroup add_if_required function     
class adjusted_Apriori(Apriori):
    def get_next_level_candidates(self, task, result, next_level_candidates):
        promising_candidates = []
        optimistic_estimate_function = getattr(task.qf, self.optimistic_estimate_name)
        for sg in next_level_candidates:
            statistics = task.qf.calculate_statistics(sg, task.target, task.data)
            add_if_required(result, sg, task.qf.evaluate(sg, statistics, task.target, task.data), task, statistics=statistics)
            optimistic_estimate = optimistic_estimate_function(sg, task.target, task.data, statistics)

            if optimistic_estimate >= ps.minimum_required_quality(result, task):
                if ps.constraints_hold(task.constraints_monotone, sg, statistics, task.data):
                    promising_candidates.append((optimistic_estimate, sg.selectors))
        min_quality = ps.minimum_required_quality(result, task)
        promising_candidates = [selectors for estimate, selectors in promising_candidates if estimate > min_quality]
        return promising_candidates

    def get_next_level_candidates_vectorized(self, task, result, next_level_candidates):
        promising_candidates = []
        statistics = []
        optimistic_estimate_function = getattr(task.qf, self.optimistic_estimate_name)
        for sg in next_level_candidates:
            statistics.append(task.qf.calculate_statistics(sg, task.target, task.data))
        tpl_class = statistics[0].__class__
        vec_statistics = tpl_class._make(np.array(tpl) for tpl in zip(*statistics))
        qualities = task.qf.evaluate(None, task.target, task.data, vec_statistics)
        optimistic_estimates = optimistic_estimate_function(None, None, None, vec_statistics)

        for sg, quality, stats in zip(next_level_candidates, qualities, statistics):
            add_if_required(result, sg, quality, task, statistics=stats)

        min_quality = ps.minimum_required_quality(result, task)
        for sg, optimistic_estimate in zip(next_level_candidates, optimistic_estimates):
            if optimistic_estimate >= min_quality:
                promising_candidates.append(sg.selectors)
        return promising_candidates

# Function used to run the adjusted BestFirstSearch algorithm
def adjustedBestFirstSearch(data):
    target = ps.BinaryTarget('target', True)
    searchspace = ps.create_selectors(data, ignore=['target'])
    task = ps.SubgroupDiscoveryTask(
        data,
        target,
        searchspace,
        result_set_size = 100,
        depth = 2,
        qf = ps.WRAccQF()
    )
    result = adjusted_BestFirstSearch().execute(task)
    #result = ps.BestFirstSearch().execute(task)
    return result.to_dataframe()

# Function used to run the adjusted DFS algorithm
def adjustedDFS(data):
    target = ps.BinaryTarget('target', True)
    searchspace = ps.create_selectors(data, ignore=['target'])
    task = ps.SubgroupDiscoveryTask(
        data,
        target,
        searchspace,
        result_set_size = 100,
        depth = 2,
        qf = ps.WRAccQF()
    )
    result = adjusted_DFS().execute(task)
    #result = ps.DFS().execute(task)
    return result.to_dataframe()

# Function used to run the adjusted Apriori algorithm
def adjustedApriori(data):
    target = ps.BinaryTarget('target', True)
    searchspace = ps.create_selectors(data, ignore=['target'])
    task = ps.SubgroupDiscoveryTask(
        data,
        target,
        searchspace,
        result_set_size = 100,
        depth = 2,
        qf = ps.WRAccQF()
    )
    result = adjusted_Apriori().execute(task)
    #result = ps.Apriori().execute(task)
    return result.to_dataframe()
