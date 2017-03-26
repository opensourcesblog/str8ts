import networkx as nx
from timeit import default_timer as timer
import numpy as np
from Modules.Model import Model
from Modules.Error import InfeasibleError
import sys

def print_str8ts(grid,grid_c):
    for r in range(len(grid)):
        row = ""
        for c in range(len(grid[r])):
            if grid_c[r][c]:
                if grid[r][c]:
                    row += " |"+str(grid[r][c])+"|"
                else:
                    row += " |=|"
            else:
                row += "  "+str(grid[r][c])+" "
        print(row)

def print_search_space(grid_c,search_space):
    shape = search_space.shape

    matrix = np.empty((shape[0]*4,shape[1]*4),dtype=str)
    empty = [' ',' ',' ',' ']
    full = ['+','+','+',' ']
    for r in range(len(search_space)):
        for c in range(len(search_space[r])):
            mr = r*4
            mc = c*4
            if 'value' in search_space[r][c]:
                if grid_c[r][c]:
                    matrix[mr:mr+4,mc:mc+4] = np.array([full,['+',str(search_space[r][c]['value']),'+',' '],full,empty])
                else:
                    matrix[mr:mr+4,mc:mc+4] = np.array([empty,[' ',str(search_space[r][c]['value']),' ',' '],empty,empty])
            elif grid_c[r][c]:
                matrix[mr:mr+4,mc:mc+4] = np.array([full,full,full,empty])
            else:
                s_matrix = np.array(search_space[r][c]['values']+[' ']*(9-len(search_space[r][c]['values']))).reshape((3,3))
                b = np.full(tuple(s+1 for s in s_matrix.shape), ' ', dtype=str)
                b[tuple(slice(0,-1) for s in s_matrix.shape)] = s_matrix
                matrix[mr:mr+4,mc:mc+4] = b

    for r in range(len(matrix)):
        print("".join(matrix[r]))

def parse_input(filename):
    grid = []
    grid_c = []
    with open('./data/'+filename) as f:
        for line in f:
            split_line = line.split(' ')
            grid_line = []
            grid_c_line = []
            for part in split_line:
                grid_line.append(int(part[0]))
                grid_c_line.append(1 if part[1] == "!" else 0)
            grid.append(grid_line)
            grid_c.append(grid_c_line)

    return grid, grid_c

def main():
    grid, grid_c = parse_input(sys.argv[1])

    grid = np.array(grid)
    grid_c = np.array(grid_c,dtype=bool)

    print("Start with %d digits" % np.count_nonzero(grid))
    start = timer()


    model = Model()
    model.build_search_space(grid,[1,2,3,4,5,6,7,8,9],0)

    # per row
    for r in range(len(grid)):
        idx = np.full(grid.shape, False, dtype=bool)
        idx[r,:] = True
        idx[r,np.where((grid_c[r,:]==1) & (grid[r,:]==0))] = False
        model.subscribe({'idx':idx},model.check_constraint,{'idx':idx},"alldifferent")


    # per col
    for c in range(len(grid[0])):
        idx = np.full(grid.shape, False, dtype=bool)
        idx[:,c] = True
        idx[np.where((grid_c[:,c]==1) & (grid[:,c]==0)),c] = False
        model.subscribe({'idx':idx},model.check_constraint,{'idx':idx},"alldifferent")



    # find all streets
    # per row
    for r in range(len(grid_c)):
        streets = []
        c_street = []
        for c in range(len(grid_c[r])):
            if grid_c[r][c]:
                if len(c_street):
                    streets.append(c_street)
                    c_street = []
            else:
                c_street.append(c)
        if len(c_street):
            streets.append(c_street)

        for street in streets:
            idx = np.full(grid.shape, False, dtype=bool)
            idx[r,street] = True
            model.subscribe({'idx':idx},model.check_constraint,{'idx':idx},"street")

    # per col
    for c in range(len(grid_c[0])):
        streets = []
        c_street = []
        for r in range(len(grid_c)):
            if grid_c[r][c]:
                if len(c_street):
                    streets.append(c_street)
                    c_street = []
            else:
                c_street.append(r)
        if len(c_street):
            streets.append(c_street)

        for street in streets:
            idx = np.full(grid.shape, False, dtype=bool)
            idx[street,c] = True
            model.subscribe({'idx':idx},model.check_constraint,{'idx':idx},"street")

    model.link()
    model.solve(np.invert(grid_c),0)
    solution = model.get_solution()

    print_str8ts(solution,grid_c)
    print("finished in ",timer()-start)
    print("nof function calls", model.nof_calls)

if __name__ == '__main__':
    main()


