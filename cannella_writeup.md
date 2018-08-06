# Motion Planning Project
### Ron Cannella 

## Introduction
This project is for basic path planning around set obstalces in the given area.

### Pipeline
Main program is the motion_planning application. It handles the drone states and waypoint programming. Planning utils is used for the graph creation, available actions, and pruning the path using the Bresenham algorithm.

### Comparison of motion_planning vs backyard flyer
Main difference in motion_planning vs the backyard_flyer is there is a PLANNING state included. Once the drone is armed, a 'plan_path' routine is called to do the path planning and then set the waypoints for the flight


### TODO Read lat/lon
Load the colliders csv and just read the first line. Split on comma, saving them to a variable, and then set the home position with the coordinates that are parsed out of hte temp variable.

```python
    # TODO: read lat0, lon0 from colliders into numpy floating point values
    file = "colliders.csv"
    with open(file) as f:
        start_lat_lon_raw = f.readline().strip()
    lltemp = start_lat_lon_raw.split(',')
    self.lat0 = np.float(lltemp[0][5:])
    self.lon0 = np.float(lltemp[1][5:])
```

### Set  Home position
Use the set_home_position function to set the home point
```python
    # TODO: set home position to (lon0, lat0, 0)
    self.set_home_position(self.lon0, self.lat0, 0)
```

### Convert global_position to local position
Using the global_to_local function get the local position by passing to it the current GPS position and home position.
```python
    # TODO: convert to current local position using global_to_local()
    print("Local Position from Global Position: {}".format(global_to_local(self.global_position, self.global_home)))
```


### Set start to current position
Determined grid position from global position and offsets, then convert to local
```python
    # TODO: convert start position to current position rather than map center
    # Get current Gloabl position relative to home in grid coords
    self.north, self.east = global_to_local(self.global_position, self.global_home)[:2]
    # Add grid offsets
    start_ne = [int(self.north + -north_offset), int(self.east+ -east_offset)]
    print("Start NE: {}".format(start_ne))
```


### Set goal by lat/lon
Determine a goal with lat/lon and set a variable for it. Determine goal in grid using the global_to_local and grid offsets
```python
    goal_ll = [ -122.3984095, 37.7962997, 4.99]
    goal_local = global_to_local(goal_ll,self.global_home)
    grid_goal = (int(goal_local[0]- north_offset),int(goal_local[1] - east_offset))
```

### Path Planning. 
The diagonal grid directions were added to the Action class. And then valid_actions was updated for the diagonals.
```python
# Assume all actions cost the same.
class Action(Enum):
    """
    An action is represented by a 3 element tuple.

    The first 2 values are the delta of the action relative
    to the current grid position. The third and final value
    is the cost of performing the action.
    """
    NORTHWEST = (-1, -1, np.sqrt(2))
    SOUTHWEST = (1, -1, np.sqrt(2))
    NORTHEAST = (-1, 1, np.sqrt(2))
    SOUTHEAST = (1, 1, np.sqrt(2))
    WEST = (0, -1, 1)
    EAST = (0, 1, 1)
    NORTH = (-1, 0, 1)
    SOUTH = (1, 0, 1)

    @property
    def cost(self):
        return self.value[2]

    @property
    def delta(self):
        return (self.value[0], self.value[1])


def valid_actions(grid, current_node):
    """
    Returns a list of valid actions given a grid and current node.
    """
    valid_actions = list(Action)
    n, m = grid.shape[0] - 1, grid.shape[1] - 1
    x, y = current_node

    # print("X: {} Y: {}".format(x,y))
    if x - 1 < 0 or grid[x - 1, y] == 1:
        valid_actions.remove(Action.NORTH)
        # print("Removed NORTH")
    if x + 1 > n or grid[x + 1, y] == 1:
        valid_actions.remove(Action.SOUTH)
        # print("Removed SOUTH")
    if y - 1 < 0 or grid[x, y - 1] == 1:
        valid_actions.remove(Action.WEST)
        # print("Removed WEST")
    if y + 1 > m or grid[x, y + 1] == 1:
        valid_actions.remove(Action.EAST)
        # print("Removed EAST")
    if x - 1 < 0 and y + 1 > m or grid[x - 1, y + 1] == 1:
        valid_actions.remove(Action.NORTHEAST)
        # print("Removed NORTHEAST")
    if x - 1 < 0 and y - 1 < 0 or grid[x - 1,y - 1] == 1:
        valid_actions.remove(Action.NORTHWEST)
        # print("Removed NORTHWEST")
    if x + 1 > n and y + 1 > m:
        valid_actions.remove(Action.SOUTHEAST)
        # print("Removed SOUTHEAST")
    try:
        if grid[x + 1,y + 1] == 1:
            valid_actions.remove(Action.SOUTHEAST)
    except:
        valid_actions.remove(Action.SOUTHEAST)
        print("SE clash")
    if x + 1 > n and y -1 < 0 or grid[x + 1,y - 1] == 1:
        valid_actions.remove(Action.SOUTHWEST)
        # print("Removed SOUTHWEST")
    return valid_actions
```


A* Implementation - For path finding I used the A* for graphs. I switched to graphs because the grid paths always stuck close to the buildings.

```python
def a_star_ng(graph, heuristic, start, goal):
    """Modified A* to work with NetworkX graphs."""
    path = []
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False
    while not queue.empty():
        item = queue.get()
        current_cost = item[0]
        current_node = item[1]

        if current_node == goal:
            print('Found a path.')
            found = True
            break
        else:
            for next_node in graph[current_node]:
                cost = graph.edges[current_node, next_node]['weight']
                new_cost = current_cost + cost + heuristic(next_node, goal)

                if next_node not in visited:
                    visited.add(next_node)
                    queue.put((new_cost, next_node))

                    branch[next_node] = (new_cost, current_node)

    path = []
    path_cost = 0
    if found:
        # retrace steps
        path = []
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    # Wanted the sets to have ints, but had floats, so casted them to int
    path = [(int(p[0]), int(p[1])) for p in path]
    return path[::-1], path_cost
```


### Prune path
The path is finally pruned using Bresenhams algorithm.
```python

# this is from Mike Hahn's example
def bres_prune(grid, path): 
    """
    Use the Bresenham module to trim uneeded waypoints from path
    """
    pruned_path = [p for p in path]

    i = 0
    while i < len(pruned_path) - 2:
        p1 = pruned_path[i]
        p2 = pruned_path[i + 1]
        p3 = pruned_path[i + 2]
        #print("Points: P1: {} P2: {} P3: {}".format(p1, p2, p3))
        # if the line between p1 and p3 doesn't hit an obstacle
        # remove the 2nd point.
        # The 3rd point now becomes the 2nd point
        # and the check is redone with a new third point
        # on the next iteration.
        if  all( (grid[pp] == 0) for pp in bresenham(int(p1[0]), int(p1[1]), int(p3[0]), int(p3[1]))):
            # Something subtle here but we can mutate
            # `pruned_path` freely because the length
            # of the list is checked on every iteration.
            pruned_path.remove(p2)
        else:
            i += 1
    return pruned_path
```

#### Conclusion

I am getting a better understanding of the coordinate systems and the relation of the grid to GPS position. I tried the graph based search, but couldnt get it to work yet. One issue I kept having is the Breseham would return points that where outside the grid.  I did validate the points that were given. But the Bresenham algorith worked great for path pruning. I would like to improve this with the graph based search and 3D paths.