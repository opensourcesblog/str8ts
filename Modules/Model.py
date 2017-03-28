import networkx as nx
from timeit import default_timer as timer
import numpy as np
from .Error import InfeasibleError
from sklearn.utils.extmath import cartesian
import sys

def get_true_keys_in_dict(d):
    """
     get all keys where the value is true
     return list
    """
    keys = []
    for key in d:
        if d[key]:
            keys.append(key)
    return keys

class Model:
    def __init__(self):
        # holds a list of indices where the function should be called
        self.subscribe_list_on = []
        # holds a list of all check constraint functions
        self.subscribe_list_func = []
        self.nof_calls = 0


    def subscribe(self,on,func,*args):
        # add a list of indices to the subscribe list
        self.subscribe_list_on.append(on['idx'])
        # as well as the function call
        self.subscribe_list_func.append((func,args))

    def solved(self,idx):
        """
            Check whether it is solved
            only the idx values have to be solved
        """

        self.nof_found = - self.nof_start_values
        self.nof_missing = 0
        ret = True
        for r in range(len(self.search_space)):
            for c in range(len(self.search_space[r])):
                if 'value' not in self.search_space[r][c] and idx[r][c]:
                    ret = False
                    self.nof_missing += 1
                elif 'value' in self.search_space[r][c]:
                    self.nof_found += 1
        return ret

    def get_backtrack_candidates(self):
        nof_values = sys.maxsize
        best_id = [0,0]
        for r in range(len(self.search_space)):
            for c in range(len(self.search_space[r])):
                if 'values' in self.search_space[r][c] and self.backtracking_space[r][c] == False:
                    l_nof_values = len(self.search_space[r][c]['values'])
                    if l_nof_values < nof_values:
                        nof_values = l_nof_values
                        best_id = [r,c]
        cands = self.search_space[best_id[0]][best_id[1]]['values']
        self.backtracking_space[best_id[0]][best_id[1]] = True
        return {'row':best_id[0],'col':best_id[1],'cands':cands}


    def backtrack(self,solved_idx,counter_bt=0):
        counter_bt += 1
        # save first search_space
        copy_search_space = np.copy(self.search_space)
        # get candidate list
        candidate_list = [self.get_backtrack_candidates()]
        # while candidates in list
        while candidate_list:
            candidate_entry = candidate_list.pop(0)
            cands = candidate_entry['cands']
            row, col = candidate_entry['row'],candidate_entry['col']
            for cand in cands:
                self.search_space = np.copy(copy_search_space)
                self.search_space[row][col] = {'value':cand}
                try:
                    idx = np.full(self.search_space.shape, False, dtype=bool)
                    idx[row,col] = True
                    self.fire(idx)
                    if not self.solved(solved_idx):
                        feasible, counter = self.backtrack(solved_idx,counter_bt)
                        if feasible:
                            return True,counter
                    else:
                        return True,counter_bt
                except InfeasibleError as e:
                    continue
        return False, counter_bt

    def solve(self,idx,no_val=0):
        try:
            self.fire(np.full(self.search_space.shape, True, dtype=bool))
            if not self.solved(idx):
                print("Need to backtrack")
                print("found %d values" % self.nof_found)
                print("%d values are missing" % self.nof_missing)
                feasible,counter = self.backtrack(idx)
                print("Number of backtracks", counter)
                if not feasible:
                    raise InfeasibleError("Infeasible checked backtracking")

        except InfeasibleError as e:
            print(e)
            exit(2)

    def fire(self,idx):
        """
            Check all subscribed indices and check which functions should be
            called
        """
        i = 0
        self.changed = np.full(self.changed.shape, False, dtype=bool)
        for lidx in self.subscribe_list_on:
            if np.any(np.logical_and(lidx,idx)):
                func = self.subscribe_list_func[i][0]
                args = self.subscribe_list_func[i][1]
                try:
                    func(*args)
                except InfeasibleError as e:
                    raise e
                self.nof_calls += 1

            i += 1

        # if something changed because of the initial fire
        # => fire again the changed indices
        if np.any(self.changed):
            try:
                self.fire(self.changed)
            except InfeasibleError as e:
                raise e

    def get_linked_idx(self,linked_to,constraint_type):
        on = self.subscribe_list_on[linked_to]
        nof_true_values = np.sum(on)
        links = []
        for si in range(len(self.subscribe_list_on)):
            args = self.subscribe_list_func[si][1]
            func = args[1]
            if func == constraint_type:
                sum_of_true = np.sum(np.logical_and(self.subscribe_list_on[si],on))
                if sum_of_true == nof_true_values or sum_of_true == np.sum(self.subscribe_list_on[si]):
                    links.append(si)
        return links

    def link(self):
        """
            Links different constraint types together
        """
        for si in range(len(self.subscribe_list_on)):
            on = self.subscribe_list_on[si]
            args = self.subscribe_list_func[si][1]
            func = args[1]
            # if a part is a straight and part of alldifferent
            # get all other straights and link them together
            # => the straight constraints considers what to do with it
            if func == "straight":
                temp_links = self.get_linked_idx(si,"alldifferent")
                links = []
                for t in temp_links:
                    links.extend(self.get_linked_idx(t,"straight"))
                real_links = []
                for l in links:
                    if not np.any(np.logical_and(self.subscribe_list_on[si],self.subscribe_list_on[l])):
                        real_links.append(l)
                self.subscribe_list_func[si][1][0]['links'] = real_links

    def build_search_space(self,grid,values,no_val=0):
        """
            Build the search space
            Each field which has a fixed value is set as 'value'
            and the other fields are assigned with 'values': values
        """
        self.search_space = np.empty(grid.shape,dtype=dict)
        self.backtracking_space = np.full(grid.shape, False, dtype=bool)
        self.changed = np.full(grid.shape, False, dtype=bool)
        sorted_values = sorted(values)
        self.possible_range = values
        self.highest_value = sorted_values[-1]
        self.lowest_value = sorted_values[0]

        self.nof_start_values = 0
        no_val_idx = np.where(grid == no_val)
        no_val_idx_invert = np.where(grid != no_val)
        self.search_space[no_val_idx] = {'values':values[:]}
        for idx in np.transpose(no_val_idx_invert):
            t_idx = tuple(idx)
            self.nof_start_values += 1
            self.search_space[t_idx] = {'value':grid[t_idx]}

    def get_all_reachable_edges(self,G,n,used_edges):
        """
            Get all reachable edges using a start node n
            get only edges which aren't in used_edges
        """

        reachable_edges = []

        queue = [n]
        while len(queue):
            n = queue.pop()
            for s in G.successors(n):
                e = (n,s)
                if e in used_edges:
                    continue
                used_edges[e] = 1
                reachable_edges.append(e)
                queue.append(s)

        return reachable_edges,used_edges

    def alldifferent(self,ss_idx,values):
        G = nx.MultiDiGraph()

        already_know = {}
        for i in range(len(values)):
            if 'values' in values[i]:
                for j in values[i]['values']:
                    G.add_edge('x_'+str(i),j)
            else:
                G.add_edge('x_'+str(i),values[i]['value'])
                already_know[i] = 1

        # get the maximum matching of this graph
        matching = nx.bipartite.maximum_matching(G)

        n_matching = []
        GM = nx.DiGraph()
        possible = np.empty((len(values)),dtype=dict)
        for k in matching:
            if str(k)[:2] == 'x_':
                n_matching.append({k:matching[k]})
                GM.add_edge(k,matching[k])
                possible[int(k[2:])] = {'values':set([matching[k]])}

        if len(n_matching) < len(values):
            raise InfeasibleError("Matching smaller than values")

        # every edge which is not part of the matching is added (reversed)
        for e in G.edges():
            if not GM.has_edge(e[0],e[1]):
                GM.add_edge(e[1],e[0])

        # Use lemmata of Berge
        # find even alternating path
        # find free vertex
        used_edges = {}
        for n in GM.nodes():
            if str(n)[:2] != "x_" and len(GM.predecessors(n)) == 0:
                edges,used_edges = self.get_all_reachable_edges(GM,n,used_edges)

                for e in edges:
                    if str(e[0])[:2] != 'x_':
                        e = (int(e[1][2:]),e[0])
                    else:
                        e = (int(e[0][2:]),e[1])
                    if 'values' not in possible[e[0]]:
                        possible[e[0]] = {'values': set()}
                    possible[e[0]]['values'].add(e[1])

        # find cycles
        scc = nx.strongly_connected_component_subgraphs(GM)
        for scci in scc:
            for e in scci.edges():
                if str(e[0])[:2] != 'x_':
                    e = (int(e[1][2:]),e[0])
                else:
                    e = (int(e[0][2:]),e[1])
                if 'values' not in possible[e[0]]:
                    possible[e[0]] = {'values': set()}
                possible[e[0]]['values'].add(e[1])

        new_possible = []
        new_knowledge = [False]*len(values)
        i = 0
        for p in possible:
            l = list(p['values'])
            if len(l) == 1:
                new_possible.append({'value':l[0]})
                if i not in already_know:
                    new_knowledge[i] = True
            else:
                new_possible.append({'values':l[:]})
                if len(l)<len(values[i]['values']):
                    new_knowledge[i] = True
            i += 1

        return new_knowledge, new_possible

    def array_to_obj(self,array,obj_struct):
        obj = {}
        for x in array:
            obj[x] = obj_struct
        return obj


    def get_straights(self,arr_of_values,len_straight=False,possible=False,used=False,
                    used_pattern=False,all_straights=False,def_used=False):
        """
            get all straights for an array of values
            eg. [[4], [1, 2, 3, 5, 7, 8, 9], [1, 2, 5, 6, 8, 9], [1, 3, 5, 6, 8, 9], [2, 3, 5, 6, 7, 8, 9]]
            return all_straights, new_values and definitely used values
            all_straights: [[4, 1, 2, 3, 5], [4, 1, 2, 5, 3], ... [4, 8, 5, 6, 7], [4, 8, 6, 5, 7]]
            new_values: [{4}, {1, 2, 3, 5, 7, 8}, {8, 1, 2, 5, 6}, {8, 1, 3, 5, 6}, {2, 3, 5, 6, 7, 8}]
            def_used: {1: False, 2: False, 3: False, 4: True, 5: True, 6: False, 7: False, 8: False, 9: False}
        """
        if not used:
            used = self.array_to_obj(self.possible_range,False)
        if not used_pattern:
            used_pattern = []
        if not all_straights:
            all_straights = []
        if not len_straight:
            len_straight = len(arr_of_values)
        if not possible:
            possible = [set() for x in range(len(arr_of_values))]
        if not def_used:
            # assume every digit has to be used
            def_used = self.array_to_obj(self.possible_range,True)

        # take the first entry and iterate over the possible values
        # for each build a new straight (recursive)
        for value in arr_of_values[0]:
            # only use this value if it's not already part of the straight
            if not used[value]:
                copy_used = dict(used)
                copy_used[value] = True

                # set all values which aren't possible anymore to used
                # for example if the straight has a length of 2 and one value is 2
                # only 1 and 3 are possible but not 2 and not 4-...
                for i in range(self.lowest_value,max(self.lowest_value,value-len_straight+1)):
                    copy_used[i] = True
                for i in range(min(self.highest_value+1,value+len_straight),self.highest_value+1):
                    copy_used[i] = True

                copy_used_pattern = used_pattern[:]
                copy_used_pattern.append(value)
                # if there are 2 or more values left
                if len(arr_of_values) > 1:
                    all_straights,possible,def_used = self.get_straights(arr_of_values[1:],len_straight,possible,
                                                            copy_used,copy_used_pattern,all_straights,def_used)
                else:
                    # add the used pattern to all the straights
                    all_straights.append(copy_used_pattern)
                    # check if there are values which are not used in the straight
                    not_used = [x for x in self.possible_range if x not in copy_used_pattern]
                    for nu in not_used:
                        def_used[nu] = False
                    xi = 0
                    # every digit which is used is possible for the defined position
                    for x in copy_used_pattern:
                        possible[xi].add(x)
                        xi += 1
        return all_straights, possible, def_used

    def list2values_structure(self,list_of_values):
        values_struct = []
        for l in list_of_values:
            if len(l) == 1:
                values_struct.append({'value':l[0]})
            else:
                values_struct.append({'values':l})
        return values_struct

    def straight(self, ss_idx, values, opts):
        """
            the straight constraint
            ss_idx: the indices where there has to be a straight
            values: the actual values on these indices
            opts: all options
        """
        fixed_values = []

        fixed_min_val = self.highest_value
        fixed_max_val = self.lowest_value
        len_values = len(values)

        # build and array of already fixed values
        # and get the highest and lowest fixed value
        for i in range(len(values)):
            if 'value' in values[i]:
                fixed_values.append(values[i]['value'])
                if values[i]['value'] > fixed_max_val:
                    fixed_max_val = values[i]['value']
                if values[i]['value'] < fixed_min_val:
                    fixed_min_val = values[i]['value']

        min_val = self.lowest_value
        max_val = self.highest_value

        for i in range(len(values)):
            if 'values' in values[i]:
                c_min = min(values[i]['values'])
                c_max = max(values[i]['values'])
            else:
                c_min = values[i]['value']
                c_max = values[i]['value']
            if c_max+len_values-1 < max_val:
                max_val = c_max+len_values-1
            if c_min-len_values+1 > min_val:
                min_val = c_min-len_values+1


        lowest_possible = min_val
        highest_possible = max_val

        if len(fixed_values):
            # if there is a fixed value we can determine the lowest possible and highest posssible
            # by the length of the straight
            # example: min: 1 max: 3 length: 5 => values 1-5
            # max-length+1 until min+length-1  or lowest_value until highest_value as bounding
            fixed_lowest_possible  = max(fixed_max_val-len_values+1,self.lowest_value)
            lowest_possible = max(lowest_possible,fixed_lowest_possible)
            fixed_highest_possible = min(fixed_min_val+len_values-1,self.highest_value)
            highest_possible = min(highest_possible,fixed_highest_possible)

        # remove impossible
        # example: {1,3,4,6,7,8,9} and {1,3,4,6,7,8} there 1 isn't possible
        # generate arrays:

        new_possible = [False]*len_values
        new_knowledge = [False]*len_values

        arr_of_values = []
        i = 0
        for e in values:
            if 'value' in e:
                arr_of_values.append(np.array([e['value']]))
            else:
                c_new_values = [x for x in e['values'] if  lowest_possible <= x <= highest_possible]
                arr_of_values.append(np.array(c_new_values))
                if len(c_new_values) < len(e['values']):
                    new_knowledge[i] = True
            i += 1

        estimated_nof_straights = np.prod(np.array([len(x) for x in arr_of_values]))

        if estimated_nof_straights > 10000:
            return new_knowledge, self.list2values_structure(arr_of_values)



        if 'links' in opts and len(opts['links']):
            straights, new_values, def_used = self.get_straights(arr_of_values)

            if len(straights) == 0:
                raise InfeasibleError("No straight possible")

            true_keys = get_true_keys_in_dict(def_used)
            if len(true_keys):
                for l in opts['links']:
                    self.check_constraint({'idx':self.subscribe_list_on[l],'against':set(true_keys)},"notInSet")
        else:
            return new_knowledge, self.list2values_structure(arr_of_values)


        # build up the new possible values and the new knowledge we have
        i = 0
        for entry in new_values:
            entry = list(entry)
            if len(entry) > 1:
                if len(entry) < len(values[i]['values']):
                    new_knowledge[i] = True
                if len(entry) == 1:
                    new_possible[i] = {'value':entry[0]}
                else:
                    new_possible[i] = {'values':entry}
            else:
                if 'value' not in values[i]:
                    new_knowledge[i] = True
                new_possible[i] = {'value':entry[0]}


            i += 1

        # update changed and search space
        old_changed = self.changed.copy()
        self.changed[ss_idx] = new_knowledge
        self.changed = np.logical_or(self.changed,old_changed)

        self.search_space[ss_idx] = new_possible

        return new_knowledge, new_possible

    def not_in_set(self,ss_idx,values,against):
        """
            not_in set constraint
            the values shouldn't exist in against
        """
        new_possible = []
        new_knowledge = [False]*len(values)
        i = 0
        for entry in values:
            if 'values' in entry:
                nof_pos = len(entry['values'])
                new_p = set(entry['values'])-against

                if len(new_p) == 1:
                    new_knowledge[i] = True
                    new_possible.append({'value':list(new_p)[0]})
                    continue
                if len(new_p) < nof_pos:
                    new_knowledge[i] = True
                new_possible.append({'values':list(new_p)})
            else:
                new_possible.append({'value':entry['value']})
            i += 1

        # update changed and search space
        old_changed = self.changed.copy()
        self.changed[ss_idx] = new_knowledge
        self.changed = np.logical_or(self.changed,old_changed)

        self.search_space[ss_idx] = new_possible

        return new_knowledge, new_possible



    def check_constraint(self,opts,operator):
            ss_idx = opts['idx']
            values = self.search_space[ss_idx]

            breaked = False
            for v in values:
                if 'values' in v:
                    breaked = True
                    break

            if not breaked:
                return

            try:
                if operator == "notInSet":
                    new_knowledge, new_possible = self.not_in_set(ss_idx,values,opts['against'])

                elif operator == "alldifferent":
                    new_knowledge, new_possible = self.alldifferent(ss_idx,values)

                elif operator == "straight":
                    new_knowledge, new_possible = self.straight(ss_idx,values,opts)

            except InfeasibleError as e:
                raise e

            # update changed and search space
            old_changed = self.changed.copy()
            self.changed[ss_idx] = new_knowledge
            self.changed = np.logical_or(self.changed,old_changed)

            self.search_space[ss_idx] = new_possible

            # some constraints actually need a different constraint:
            try:
                # if we have a straight constraint we also have an alldifferent constraint
                if operator == "straight":
                    self.check_constraint(opts,"alldifferent")

            except InfeasibleError as e:
                raise e


    def get_solution(self):
        grid = [[0]*9 for i in range(9)]
        for r in range(len(self.search_space)):
            for c in range(len(self.search_space[r])):
                if 'value' in self.search_space[r][c]:
                    grid[r][c] = self.search_space[r][c]['value']
        return grid

    def print_search_space(self):
        for r in range(len(self.search_space)):
            row = []
            for c in range(len(self.search_space[r])):
                if 'value' in self.search_space[r][c]:
                    row.append(self.search_space[r][c]['value'])
                else:
                    row.append(self.search_space[r][c]['values'])
            print(row)


