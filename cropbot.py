import itertools
import numpy as np
import networkx as nx

from collections import defaultdict


class CropBot:
    """
    A class representing the CropBot.
    """
    def __init__(self, width, height, weights):
        self.width = width
        self.height = height
        self.weights = np.array(weights) / sum(weights)  # Make weights sum to one.
        self.state = (0, 0)
        self.unchecked = np.zeros((width, height))
        self.intervention_values = np.zeros((width, height))
        self.history = np.zeros((width, height, 2))  # History of how many times visited with number of interventions.
        self.distances = self.calc_distances(width, height)
        self.directions = {(0, 1): "forward", (0, -1): "backward", (-1, 0): "left", (1, 0): "right"}
        self.graph = nx.grid_2d_graph(width, height).to_directed()
        self.edge_values = defaultdict(int)
        self.vertex_values = defaultdict(int)

    @staticmethod
    def calc_distance(state1, state2):
        """Compute the Manhattan distance between two states.

        Args:
            state1 (Tuple[int]): The first state.
            state2 (Tuple[int]): The second state.

        Returns:
            int: The Manhattan distance between the two.
        """
        return abs(state1[0] - state2[0]) + abs(state1[1] - state2[1])

    def calc_distances(self, width, height):
        """Compute the distances between every pair of nodes.

        Args:
            width (int): The width of the environment.
            height (int): The height of the environment.

        Returns:
            ndarray: An array of distances from each state to all others.
        """
        distances = np.zeros((width, height, width, height))
        for state1 in itertools.product(range(width), range(height)):
            for state2 in itertools.product(range(width), range(height)):
                distance = self.calc_distance(state1, state2)
                w1, h1 = state1
                w2, h2 = state2
                distances[w1, h1, w2, h2] = distance
        return distances

    def get_weight_for_edge(self, start_vertex, end_vertex, attributes):
        """Get the weight for a given edge.

        This is used in computing the optimal path between the current state and the end point.

        Args:
            start_vertex (Tuple[int]): The start vertex of the edge.
            end_vertex (Tuple[int]): The end vertex of the edge.
            attributes (Dict): A dictionary of attributes for an edge.

        Returns:
            float: The weight for a given edge.
        """
        return self.edge_values[(start_vertex, end_vertex)]

    def update_vertex_values(self):
        """Update the values for the vertices in the graph.

        This computes the expected value of visiting each vertex by using the weighting of the different objectives.
        """
        for vertex in self.graph.nodes:
            unchecked = self.weights[0] * self.unchecked[vertex]
            distance = self.weights[1] * self.distances[self.state + vertex]
            intervention_value = self.weights[2] * self.intervention_values[vertex]
            vertex_value = unchecked + distance + intervention_value
            self.vertex_values[vertex] = vertex_value

    def update_edge_weights(self):
        """Update the weights for all edges.

        This scales the value of the end vertex such that best edges have lowest weight. We can later use these weights
        in a path finding algorithm to compute the optimal trajectory.
        """
        max_weight = max(self.vertex_values.values()) + 1

        for edge in self.graph.edges:
            self.edge_values[edge] = max_weight - self.vertex_values[edge[1]]

    def get_plan(self, target):
        """Compute the optimal trajectory from the current state to a given target.

        Args:
            target (Tuple[int]): The required end state.

        Returns:
            List[Tuple[int]]: A list of nodes to go to.
        """
        return nx.astar_path(self.graph, self.state, target, self.get_weight_for_edge)

    def update_area(self, state):
        min_x = max(0, state[0] - 1)
        max_x = min(self.width, state[0] + 1)
        min_y = max(0, state[1] - 1)
        max_y = min(self.height, state[1] + 1)

        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                print(x, y)

    def get_direction_from_next(self, next_state):
        """Determine the direction to move from the next state.

        Args:
            next_state (Tuple[int]): The next state.

        Returns:
            str: The direction to move into.
        """
        move_tpl = tuple(np.array(next_state) - np.array(self.state))
        return self.directions[move_tpl]

    def command_driver(self, direction):
        response = 0
        return response

    def execute_plan(self, plan):
        """Execute the current monitoring plan.

        Args:
            plan (List[Tuple[int]]): A list of nodes to go to.
        """
        for next_state in plan:
            direction = self.get_direction_from_next(next_state)
            response = self.command_driver(direction)
            if response == 'success':
                self.state = next_state
            elif response in ['fail', 'stop']:
                return False
        return True

    def next_endpoint(self):
        """Compute the next endpoint.

        Returns:
            Tuple[int]: The optimal end vertex.
        """
        self.update_vertex_values()
        return max(self.vertex_values, key=self.vertex_values.get)

    def monitor(self):
        keep_monitoring = True
        while keep_monitoring:
            endpoint = self.next_endpoint()
            self.update_edge_weights()
            plan = self.get_plan(endpoint)
            keep_monitoring = self.execute_plan(plan)
