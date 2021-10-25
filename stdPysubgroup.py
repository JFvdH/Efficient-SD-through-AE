# Package imports
import pysubgroup as ps

# Function used to execute the BestFirstSearch algorithm from the pysubgroup package
def bestFirstSearch(data):
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
    result = ps.BestFirstSearch(beam_width=100).execute(task)
    return result.to_dataframe()

# Function used to execute the DFS algorithm from the pysubgroup package
def DFS(data):
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
    result = ps.SimpleDFS().execute(task)
    return result.to_dataframe()

# Function used to execute the Apriori algorithm from the pysubgroup package
def apriori(data):
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
    result = ps.Apriori().execute(task)
    return result.to_dataframe()

