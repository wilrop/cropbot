import itertools
import serial
import json
import time
import multiprocessing
import copy

import numpy as np
import pandas as pd
import networkx as nx

from collections import defaultdict


class CropBot:
    """
    A class representing the CropBot.
    """
    def __init__(self, width, height, weights):
        self.width = width
        self.height = height
        self.weights = self.normalise_weights(weights)
        self.state = (0, 0)
        self.unchecked = np.zeros((width, height))
        self.current_risk = np.zeros((width, height))
        self.history = np.zeros((width, height, 2))  # History of how many times visited with number of interventions.
        self.distances = self.calc_distances(width, height)
        self.directions = {(0, 0): "stay", (0, 1): "forward", (0, -1): "backward", (-1, 0): "left", (1, 0): "right"}
        self.graph = nx.grid_2d_graph(width, height).to_directed()
        self.edge_values = defaultdict(int)
        self.vertex_values = defaultdict(int)
        self.driver = serial.Serial('/dev/ttys002', timeout=3)
        self.log_world_state()

    @staticmethod
    def normalise_weights(weights):
        """Normalise weights to sum to one.

        Args:
            weights (List[float]): A list of weights over the objectives.

        Returns:
            ndarray: An array of weights summing to one.
        """
        normalised_weights = np.array(weights) / sum(weights)
        return normalised_weights

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
        max_distance = width + height
        distances = np.zeros((width, height, width, height))
        for state1 in itertools.product(range(width), range(height)):
            for state2 in itertools.product(range(width), range(height)):
                distance = self.calc_distance(state1, state2)
                w1, h1 = state1
                w2, h2 = state2
                distances[w1, h1, w2, h2] = max_distance - distance  # Flip distances so lower is better.
        return distances

    def get_weight_for_edge(self, start_vertex, end_vertex):
        """Get the weight for a given edge.

        This is used in computing the optimal path between the current state and the end point.

        Args:
            start_vertex (Tuple[int]): The start vertex of the edge.
            end_vertex (Tuple[int]): The end vertex of the edge.

        Returns:
            float: The weight for a given edge.
        """
        return self.edge_values[(start_vertex, end_vertex)]

    @staticmethod
    def normalise_scalar(x, min_x, max_x):
        return x if x == 0 else (x - min_x)/(max_x - min_x)

    def update_vertex_values(self):
        """Update the values for the vertices in the graph.

        This computes the expected value of visiting each vertex by using the weighting of the different objectives.
        """
        min_unchecked = np.min(self.unchecked)
        max_unchecked = np.max(self.unchecked)

        min_distance = np.min(self.distances)
        max_distance = np.max(self.distances)

        intervention_values = []
        for vertex in self.graph.nodes:
            if self.history[vertex][0] == 0:
                intervention_freq = 0
            else:
                intervention_freq = self.history[vertex][0] / self.history[vertex][1]
            intervention_value = self.current_risk[vertex] * (1 + intervention_freq)
            intervention_values.append(intervention_value)

        min_intervention_value = min(intervention_values)
        max_intervention_value = max(intervention_values)

        for vertex, intervention_value in zip(self.graph.nodes, intervention_values):
            unchecked_value = self.weights[0] * self.normalise_scalar(self.unchecked[vertex], min_unchecked, max_unchecked)
            distance_value = self.weights[1] * self.normalise_scalar(self.distances[self.state + vertex], min_distance, max_distance)
            intervention_value = self.weights[2] * self.normalise_scalar(intervention_value, min_intervention_value, max_intervention_value)
            vertex_value = unchecked_value + distance_value + intervention_value
            self.vertex_values[vertex] = vertex_value

    def update_edge_weights(self):
        """Update the weights for all edges.

        This scales the value of the end vertex such that best edges have lowest weight. We can later use these weights
        in a path finding algorithm to compute the optimal trajectory.
        """
        max_weight = max(self.vertex_values.values()) + 1

        for edge in self.graph.edges:
            self.edge_values[edge] = max_weight - self.vertex_values[edge[1]]

    def update_unchecked(self):
        """Increment the counter for unchecked cells."""
        self.unchecked += 1
        self.unchecked[self.state] = 0

    def update_area(self):
        """Update the values of an area around an intervention."""
        min_x = max(0, self.state[0] - 1)
        max_x = min(self.width - 1, self.state[0] + 1)
        min_y = max(0, self.state[1] - 1)
        max_y = min(self.height - 1, self.state[1] + 1)

        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                self.current_risk[x, y] += 1

    def classify_color(self, color):
        if color == 'red':
            return 'diseased'
        elif color == 'black':
            return 'pests'
        elif color == 'yellow':
            return 'drought'
        elif color in ['blue', 'cyan']:
            return'flooding'
        else:
            return 'normal'

    def update_interventions(self, res_dict):
        """Update the value of a cell when an intervention should take place.

        Note: Handle the interventions differently

        Args:
            res_dict (Dict): The response from the driver with their observation.
        """
        color = res_dict['response']
        classification = self.classify_color(color)
        if classification == 'diseased':
            self.update_area()
        elif classification == 'pests':
            self.update_area()
        elif classification == 'flooding':
            self.update_area()
        elif classification == 'drought':
            self.update_area()
        else:
            self.current_risk[self.state] = 0

    def get_plan(self, target):
        """Compute the optimal trajectory from the current state to a given target.

        Args:
            target (Tuple[int]): The required end state.

        Returns:
            List[Tuple[int]]: A list of nodes to go to.
        """
        return nx.astar_path(self.graph, self.state, target, self.get_weight_for_edge)[1:]

    def get_direction_from_next(self, next_state):
        """Determine the direction to move from the next state.

        Args:
            next_state (Tuple[int]): The next state.

        Returns:
            str: The direction to move into.
        """
        move_tpl = tuple(np.array(next_state) - np.array(self.state))
        return self.directions[move_tpl]

    def read_buffer(self, return_dict):
        driver = serial.Serial('/dev/ttys002')
        return_dict['response'] = driver.readline()

    def command_driver(self, direction):
        """Command the driver to move in a specific direction.

        Args:
            direction (str): The direction to move into.

        Returns:
            Dict: A response dictionary.
        """
        formatted_direction = direction + '\n'
        byte_command = formatted_direction.encode("utf-8")
        self.driver.write(byte_command)
        self.driver.flush()
        print('The command: ' + direction)
        start = time.time()
        while 1:
            if self.driver.in_waiting:
                manager = multiprocessing.Manager()
                return_dict = manager.dict()
                p = multiprocessing.Process(target=self.read_buffer, name="Reading", args=(return_dict,))
                p.start()
                p.join(5)

                if p.is_alive():
                    print("Failed to respond, retrying...")
                    p.terminate()
                    p.join()
                    return self.command_driver(direction)
                else:
                    byte_response = return_dict['response']

                res_str = byte_response.decode("utf-8")

                if not res_str.endswith('\n'):
                    print("Failed to respond, retrying...")
                    return self.command_driver(direction)
                elif '\r' in res_str:
                    res_str = res_str.split('\r')[-1]

                res_str = res_str.strip('\n')
                print("The response: " + res_str)
                try:
                    if res_str.startswith('ERROR'):
                        print("Encountered an error, retrying...")
                        return self.command_driver(direction)
                    else:
                        res_dict = json.loads(res_str)
                        break
                except Exception as e:
                    raise Exception(e)
            else:
                if time.time() - start > 3:
                    print("Failed to respond, retrying...")
                    return self.command_driver(direction)
        return res_dict

    def execute_plan(self, plan):
        """Execute the current monitoring plan.

        Args:
            plan (List[Tuple[int]]): A list of nodes to go to.
        """
        for next_state in plan:
            direction = self.get_direction_from_next(next_state)
            res_dict = self.command_driver(direction)
            if res_dict['response'] in ['fail', 'stop']:
                return False
            else:
                self.state = next_state
                self.update_unchecked()
                self.update_interventions(res_dict)
                self.log_world_state()
        return True

    def next_endpoint(self):
        """Compute the next endpoint.

        Returns:
            Tuple[int]: The optimal end vertex.
        """
        self.update_vertex_values()
        vertex_values = copy.deepcopy(self.vertex_values)
        del vertex_values[self.state]
        return max(vertex_values, key=vertex_values.get)

    def log_world_state(self):
        world_dict = {'X': [], 'Y': [], 'Unvisited': [], 'Risk': [], 'Value': [], 'Robot': []}
        for vertex in self.graph.nodes:
            world_dict['X'].append(vertex[0])
            world_dict['Y'].append(vertex[1])
            world_dict['Unvisited'].append(max(0.001, self.unchecked[vertex]))
            if self.history[vertex][0] == 0:
                intervention_freq = 0
            else:
                intervention_freq = self.history[vertex][0] / self.history[vertex][1]
            intervention_value = self.current_risk[vertex] * (1 + intervention_freq)
            world_dict['Risk'].append(intervention_value)
            world_dict['Value'].append(self.vertex_values[vertex])
            world_dict['Robot'].append(self.state == vertex)
        df = pd.DataFrame.from_dict(world_dict)
        df.to_csv('world_state.csv', index=False)

    def read_weights(self):
        try:
            weight_df = pd.read_csv('weights.csv')
            weights = weight_df.iloc[0].tolist()
            self.weights = self.normalise_weights(weights)
        except Exception:
            pass

    def monitor(self):
        keep_monitoring = True
        while keep_monitoring:
            self.read_weights()
            endpoint = self.next_endpoint()
            self.update_edge_weights()
            plan = self.get_plan(endpoint)
            keep_monitoring = self.execute_plan(plan)
