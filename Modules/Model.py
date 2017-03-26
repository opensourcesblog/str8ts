import networkx as nx
from timeit import default_timer as timer
import numpy as np
from .Error import InfeasibleError
from sklearn.utils.extmath import cartesian
import sys

def get_true_keys_in_dict(d):
    keys = []
    for key in d:
        if d[key]:
            keys.append(key)
    return keys

class Model:
    def __init__(self):
        self.subscribe_list_on = []
        self.subscribe_list_func = []
        self.nof_calls = 0


    def subscribe(self,on,func,*args):
        self.subscribe_list_on.append(on['idx'])
        self.subscribe_list_func.append((func,args))

    def solved(self,idx,no_val):
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


    def backtrack(self,solved_idx,no_val,counter_bt=0):
        counter_bt += 1
        # save first search_space
        copy_search_space = np.copy(self.search_space)
        # get candidate list
        candidate_list = [self.get_backtrack_candidates()]
        # while candidates in list
        while candidate_list:
            candidate_enrty = candidate_list.pop(0)
            cands = candidate_enrty['cands']
            row, col = candidate_enrty['row'],candidate_enrty['col']
            for cand in cands:
                self.search_space = np.copy(copy_search_space)
                self.search_space[row][col] = {'value':cand}
                try:
                    idx = np.full(self.search_space.shape, False, dtype=bool)
                    idx[row,col] = True
                    self.fire(idx)
                    if not self.solved(solved_idx,no_val):
                        feasible, counter = self.backtrack(solved_idx,no_val,counter_bt)
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
            if not self.solved(idx,no_val):
                print("Need to backtrack")
                print("found %d values" % self.nof_found)
                print("%d values are missing" % self.nof_missing)
                feasible,counter = self.backtrack(idx,no_val)
                print("Number of backtracks", counter)
                if not feasible:
                    raise InfeasibleError("Infeasible checked backtracking")

        except InfeasibleError as e:
            print(e)
            exit(2)

    def fire(self,idx):
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
        for si in range(len(self.subscribe_list_on)):
            on = self.subscribe_list_on[si]
            args = self.subscribe_list_func[si][1]
            func = args[1]
            if func == "street":
                temp_links = self.get_linked_idx(si,"alldifferent")
                links = []
                for t in temp_links:
                    links.extend(self.get_linked_idx(t,"street"))
                real_links = []
                for l in links:
                    if not np.any(np.logical_and(self.subscribe_list_on[si],self.subscribe_list_on[l])):
                        real_links.append(l)
                self.subscribe_list_func[si][1][0]['links'] = real_links

    def build_search_space(self,grid,values,no_val=0):
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
        # build a graph with connects the variables with the possible values
        G = nx.MultiDiGraph()
#         print('values', values)

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



    def matrix_of_values(self,values):
        matrix = np.zeros((len(values),len(self.possible_range)))
        index = {}
        ip = 0
        for p in self.possible_range:
            index[p] = ip
            ip += 1

        i = 0
        for entry in values:
            if 'value' in entry:
                matrix[i][index[entry['value']]] = 1
            else:
                for v in entry['values']:
                    matrix[i][index[v]] = 1
            i += 1
        print(matrix)
        return matrix

    def get_streets(self,arr_of_values,len_street=False,possible=False,used=False,
                    used_pattern=False,all_streets=False,def_used=False):
        if not used:
            used = self.array_to_obj(self.possible_range,False)
        if not used_pattern:
            used_pattern = []
        if not all_streets:
            all_streets = []
        if not len_street:
            len_street = len(arr_of_values)
        if not possible:
            possible = [set() for x in range(len(arr_of_values))]
        if not def_used:
            def_used = self.array_to_obj(self.possible_range,True)


        for value in arr_of_values[0]:
            if not used[value]:
                copy_used = dict(used)
                copy_used[value] = True

                for i in range(self.lowest_value,max(self.lowest_value,value-len_street+1)):
                    copy_used[i] = True
                for i in range(min(self.highest_value+1,value+len_street),self.highest_value+1):
                    copy_used[i] = True

                copy_used_pattern = used_pattern[:]
                copy_used_pattern.append(value)
                if len(arr_of_values) > 1:
                    all_streets,possible,def_used = self.get_streets(arr_of_values[1:],len_street,possible,
                                                            copy_used,copy_used_pattern,all_streets,def_used)
                else:
                    all_streets.append(copy_used_pattern)
                    not_used = [x for x in self.possible_range if x not in copy_used_pattern]
                    for nu in not_used:
                        def_used[nu] = False
                    xi = 0
                    for x in copy_used_pattern:
                        possible[xi].add(x)
                        xi += 1
        return all_streets, possible, def_used


    def street(self, ss_idx, values, opts):
        fixed_values = []

        min_val = self.highest_value
        max_val = self.lowest_value
        len_values = len(values)
        found = False

        for i in range(len(values)):
            if 'value' in values[i]:
                found = True
                fixed_values.append(values[i]['value'])
                if values[i]['value'] > max_val:
                    max_val = values[i]['value']
                if values[i]['value'] < min_val:
                    min_val = values[i]['value']

        if not found:
            min_val = self.lowest_value
            max_val = self.highest_value

            for i in range(len(values)):
                c_min = min(values[i]['values'])
                c_max = max(values[i]['values'])
                if c_max+len_values-1 < max_val:
                    max_val = c_max+len_values-1
                if c_min-len_values+1 > min_val:
                    min_val = c_min-len_values+1

            low_possible = min_val
            highest_possible = max_val
        else:
            # example: min: 1 max: 3 length: 5 => values 1-5
            # max-length+1 until min+length-1  or lowest_value until highest_value as bounding
            low_possible = max(max_val-len_values+1,self.lowest_value)
            highest_possible = min(min_val+len_values-1,self.highest_value)

        # remove unpossible
        # example: {1,3,4,6,7,8,9} and {1,3,4,6,7,8} there 1 isn't possible
        # generate arrays:
        arr_of_values = []
        for e in values:
            if 'value' in e:
                arr_of_values.append(np.array([e['value']]))
            else:
                arr_of_values.append(np.array(e['values']))

        streets, new_values, def_used = self.get_streets(arr_of_values)
        feasible_check = [x for x in new_values if len(x) > 0]
        if len(feasible_check) == 0:
            raise InfeasibleError("No street possible")

        if 'links' in opts and len(opts['links']):
            true_keys = get_true_keys_in_dict(def_used)
            if len(true_keys):
                for l in opts['links']:
                    self.check_constraint({'idx':self.subscribe_list_on[l],'against':set(true_keys)},"notInSet")

        possible = range(low_possible,highest_possible+1)

        new_possible = [False]*len_values
        new_knowledge = [False]*len_values
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

                elif operator == "street":
                    new_knowledge, new_possible = self.street(ss_idx,values,opts)

            except InfeasibleError as e:
                raise e

            # update changed and search space
            old_changed = self.changed.copy()
            self.changed[ss_idx] = new_knowledge
            self.changed = np.logical_or(self.changed,old_changed)

            self.search_space[ss_idx] = new_possible

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


