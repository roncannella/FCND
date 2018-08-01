# Motion Planning Project
### Ron Cannella 

## Introduction
This project is for basic path planning around set obstalces in the given area.

### Pipeline
Main program is the motion_planning application. It handles the drone states and waypoint programming. Planning utils is used for the grid creation, available actions, and pruning the path.

### Comparison of motion_planning vs backyard flyer
Main difference in motion_planning vs the backyard_flyer is there is a PLANNING state included. Once the drone is armed, a 'plan_path' routine is called to do the path planning and then set the waypoints for hte flight


### TODO Read lat/lon
Load the colliders csv and just read the first line. Split on comma and then set the home position with the coordinates that are parsed out.

```python
    file = "colliders.csv"
    with open(file) as f:
        start_lat_lon_raw = f.readline().strip()
    lltemp = start_lat_lon_raw.split(',')
```

### Set  Home posistion
Use the set_home_position function to set the home point
```python
	self.set_home_position(np.float(lltemp[1][5:]),np.float(lltemp[0][5:]),0) 
```

### Convert global_position to local position
Using the global_to_local function get the local position by passing to it the current GPS position and home position.


```python
    curr_local = global_to_local(self.global_position,self.global_home)
```

### Set start to current position
Determined grid position from local position and offsets
```python
	grid_start = (int(self.local_position[0]- north_offset), int(self.local_position[1] - east_offset))
```


### Set goal by lat/lon
Determine a goal with lat/lon and set a variable for it. Determine goal in grid using the global_to_local and grid offsets
```python
        goal_ll = [ -122.3984095, 37.7962997, 4.99]
        goal_local = global_to_local(goal_ll,self.global_home)
        grid_goal = (int(goal_local[0]- north_offset),int(goal_local[1] - east_offset))
```

### Path planning. The diagonal grid directions were added to the Action class. And then valid_actions was updated for the diagonals
```python
class Action(Enum):
    """
    An action is represented by a 3 element tuple.

    The first 2 values are the delta of the action relative
    to the current grid position. The third and final value
    is the cost of performing the action.
    """
    NORTHWEST = (-1, -1, np.sqrt(2))
    SOUTHWEST = (1, 1, np.sqrt(2))
    NORTHEAST = (-1, 1, np.sqrt(2))
    SOUTHEAST = (1, -1, np.sqrt(2))
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

    # check if the node is off the grid or
    # it's an obstacle

    if x - 1 < 0 or grid[x - 1, y] == 1:
        valid_actions.remove(Action.NORTH)
    if x + 1 > n or grid[x + 1, y] == 1:
        valid_actions.remove(Action.SOUTH)
    if y - 1 < 0 or grid[x, y - 1] == 1:
        valid_actions.remove(Action.WEST)
    if y + 1 > m or grid[x, y + 1] == 1:
        valid_actions.remove(Action.EAST)
    if x - 1 < 0 or y + 1 > m or grid[x - 1, y + 1] == 1:
        valid_actions.remove(Action.NORTHEAST)
    if x - 1 < 0 or y - 1 < 0 or grid[x - 1,y - 1] == 1:
        valid_actions.remove(Action.NORTHWEST)
    if x + 1 > n or y -1 < 0 or grid[x + 1,y - 1] == 1:
        valid_actions.remove(Action.SOUTHEAST)
    if x + 1 > n or y + 1 > m or grid[x + 1,y + 1] == 1:
        valid_actions.remove(Action.SOUTHWEST)
        
    return valid_actions
```

### Prune path
Path is pruned using a collineratity algorithm in the planning_utils code
```python

def collinear(points,eps=1e-2):
    matrix = np.vstack((points[0],points[1],points[2]))
    determinate = np.linalg.det(matrix)
    return determinate < eps

def prune(path):
    pruned = [p for p in path]
    i = 0
    while i < len(pruned)-2:
        if collinear([point(pruned[i]),point(pruned[i+1]),point(pruned[i+2])]):
            pruned.remove(pruned[i+1])
        else:
            i+=1
    print("Path Length:{}\nPruned Path:{}".format(len(path),len(pruned)))
    return pruned

```
#### Conclusion

I am getting a better understanding of the coordinate systems and the relation of the grid to GPS position. I tried the graph based search, but couldnt get it to work yet. One issue I kept having is the breseham would return points that where outside the grid. I did validate the points that were given. I would like to improve this with the graph based search and 3D paths.