import argparse
import time
import msgpack
from enum import Enum, auto
import numpy as np
import networkx as nx

from planning_utils import a_star_ng, heuristic, create_grid_and_edges, bres_prune
from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local
import matplotlib.pyplot as plt


class States(Enum):
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()


class MotionPlanning(Drone):

    def __init__(self, connection):
        super().__init__(connection)

        self.target_position = np.array([0.0, 0.0, 0.0])
        self.waypoints = []
        self.in_mission = True
        self.check_state = {}

        # initial state
        self.flight_state = States.MANUAL

        # register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def local_position_callback(self):
        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < 1.0:
                if len(self.waypoints) > 0:
                    self.waypoint_transition()
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()

    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            if self.global_position[2] - self.global_home[2] < 0.1:
                if abs(self.local_position[2]) < 0.01:
                    self.disarming_transition()

    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if self.armed:
                    self.plan_path()
            elif self.flight_state == States.PLANNING:
                self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()

    def arming_transition(self):
        self.flight_state = States.ARMING
        print("arming transition")
        self.arm()
        self.take_control()

    def takeoff_transition(self):
        self.flight_state = States.TAKEOFF
        print("takeoff transition")
        self.takeoff(self.target_position[2])

    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        print("waypoint transition")
        self.target_position = self.waypoints.pop(0)
        print('target position', self.target_position)
        self.cmd_position(self.target_position[0], self.target_position[1],
                            self.target_position[2], self.target_position[3])

    def landing_transition(self):
        self.flight_state = States.LANDING
        print("landing transition")
        self.land()

    def disarming_transition(self):
        self.flight_state = States.DISARMING
        print("disarm transition")
        self.disarm()
        self.release_control()


    def manual_transition(self):
        self.flight_state = States.MANUAL
        print("manual transition")
        self.stop()
        self.in_mission = False
        plt.show()

    def send_waypoints(self):
        print("Sending waypoints to simulator ...")

        data = msgpack.dumps(self.waypoints)
        self.connection._master.write(data)

    def plan_path(self):
        self.flight_state = States.PLANNING
        print("Searching for a path ...")
        TARGET_ALTITUDE = 5
        SAFETY_DISTANCE = 3

        self.target_position[2] = TARGET_ALTITUDE

        # TODO: read lat0, lon0 from colliders into floating point values
        file = "colliders.csv"
        with open(file) as f:
            start_lat_lon_raw = f.readline().strip()
        lltemp = start_lat_lon_raw.split(',')
        self.lat0 = np.float(lltemp[0][5:])
        self.lon0 = np.float(lltemp[1][5:])

        # TODO: set home position to (lon0, lat0, 0)
        self.set_home_position(self.lon0, self.lat0, 0)

        # TODO: retrieve current global position
        print("Current Global Position: {} ".format(self.global_position))

        # TODO: convert to current local position using global_to_local()
        print("Local Position from Global Position: {}".format(global_to_local(self.global_position, self.global_home)))

        # Read in obstacle map
        data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)

        # Define a grid for a particular altitude and safety margin around obstacles
        grid, edges, north_offset, east_offset = create_grid_and_edges(data, TARGET_ALTITUDE, SAFETY_DISTANCE)
        print("North Offset: {} East Offset: {}".format(north_offset, east_offset))

        # TODO: convert start position to current position rather than map center
        # Get current Gloabl position relative to home in grid coords
        self.north, self.east = global_to_local(self.global_position, self.global_home)[:2]
        # Add grid offsets
        start_ne = [int(self.north + -north_offset), int(self.east+ -east_offset)]
        print("Start NE: {}".format(start_ne))

        # TODO: adapt to set goal as latitude / longitude position and convert
        # Specify the coords in Long/Lat
        global_goal = [-122.399189, 37.796400, 0]
        # Get the local grid from global position coords
        goal_gl = global_to_local(global_goal, self.global_home)[:2]
        # Add the grid offset
        goal_ne = [int(goal_gl[0] + -north_offset), int(goal_gl[1] + -east_offset)]

        #goal_ne = (-north_offset + 400, -east_offset - 100)
        print("Goal NE from Lat/Lon: {}".format(goal_ne))

        G = nx.Graph()
        for e in edges:
            p1 = e[0]
            p2 = e[1]
            dist = np.linalg.norm(np.array(p2) - np.array(p1))
            G.add_edge(p1, p2, weight=dist)

        start_ne_g = closest_point(G, start_ne)

        goal_ne_g = closest_point(G, goal_ne)

        path, cost = a_star_ng(G, heuristic, start_ne_g, goal_ne_g)

        # TODO: prune path to minimize number of waypoints
        path = bres_prune(grid, path)
        print("Path Length: {}".format(len(path)))

        # Convert path to waypoints
        waypoints = [[int(p[0] + north_offset), int(p[1] + east_offset), TARGET_ALTITUDE, 0] for p in path]

        # Set self.waypoints
        self.waypoints = waypoints

        # TODO: send waypoints to sim (this is just for visualization of waypoints)
        self.send_waypoints()


def closest_point(graph, current_point):
    """
    Compute the closest point in the `graph`
    to the `current_point`.
    """
    closest_point = None
    dist = 100000
    for p in graph.nodes:
        d = np.linalg.norm(np.array(p) - np.array(current_point))
        if d < dist:
            closest_point = p
            dist = d
    return closest_point

    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        self.connection.start()

        # Only required if they do threaded
        # while self.in_mission:
        #    pass

        self.stop_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    args = parser.parse_args()

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=60)
    drone = MotionPlanning(conn)
    time.sleep(1)

    drone.start()
