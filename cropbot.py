import copy
import itertools
import json
import multiprocessing
from collections import defaultdict

import networkx as nx
import numpy as np
import serial


class CropBot:
    """
    A class representing the CropBot.
    """

    def __init__(self, width, height, weights, start=(0, 0), prior_sick=0.1, prior_conditional_sick=0.5):
        self.width = width
        self.height = height
        self.weights = self.normalise_weights(weights)
        self.state = start
        self.plan = []

        self.serial_port = '/dev/ttys002'
        self.driver = serial.Serial(self.serial_port)
        self.directions = {(0, 0): 'stay', (0, 1): 'forward', (0, -1): 'backward', (-1, 0): 'left', (1, 0): 'right'}
        self.driver_delay = 15

        self.unchecked = np.zeros((width, height))
        self.inverse_distances = self.calc_inverse_distances(width, height)
        self.current_risk = np.zeros((width, height))
        self.risk_model = np.zeros((width, height, 2))
        self.prior_sick = prior_sick
        self.prior_conditional_sick = prior_conditional_sick

        self.graph = nx.grid_2d_graph(width, height).to_directed()
        self.edge_values = defaultdict(int)
        self.vertex_values = defaultdict(int)
        self.risk_values = defaultdict(int)

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

    def calc_inverse_distances(self, width, height):
        """Compute the inverse distances between every pair of nodes.

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
                distances[w1, h1, w2, h2] = max_distance - distance
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
        return x if x == 0 else (x - min_x) / (max_x - min_x)

    def compute_intervention_freq(self, vertex):
        """Compute the intervention frequency for a given vertex.

        Args:
            vertex (Tuple[int]): The coordinates of the vertex.

        Returns:
            float: The intervention frequency.
        """
        if self.risk_model[vertex][0] == 0:
            intervention_freq = 0
        else:
            intervention_freq = self.risk_model[vertex][0] / self.risk_model[vertex][1]
        return intervention_freq

    def update_vertex_values(self):
        """Update the values for the vertices in the graph.

        This computes the expected value of visiting each vertex by using the weighting of the different objectives.

        Note:
            Higher vertex values are better.
        """
        min_unchecked = np.min(self.unchecked)
        max_unchecked = np.max(self.unchecked)

        min_inv_dist = np.min(self.inverse_distances)
        max_inv_dist = np.max(self.inverse_distances)

        risk_values = []
        for vertex in self.graph.nodes:
            intervention_freq = self.compute_intervention_freq(vertex)
            risk_value = self.current_risk[vertex] * (1 + intervention_freq)
            risk_values.append(risk_value)

        min_intervention_value = min(risk_values)
        max_intervention_value = max(risk_values)

        for vertex, risk_value in zip(self.graph.nodes, risk_values):
            unchecked_value = self.normalise_scalar(self.unchecked[vertex], min_unchecked, max_unchecked)
            distance_value = self.normalise_scalar(self.inverse_distances[self.state + vertex], min_inv_dist,
                                                   max_inv_dist)
            risk_value = self.normalise_scalar(risk_value, min_intervention_value, max_intervention_value)
            vertex_value = self.weights[0] * unchecked_value + self.weights[1] * distance_value + self.weights[
                2] * risk_value
            self.risk_values[vertex] = risk_value
            self.vertex_values[vertex] = vertex_value

    def update_edge_weights(self):
        """Update the weights for all edges.

        The value of the end vertex is inverted so that the best edges have the lowest weight. These weights are later
        used in a path finding algorithm to compute the optimal trajectory.

        Note:
            Lower edge values are better such that they can be used in the shortest path finding algorithm.
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

    @staticmethod
    def classify_color(color):
        """Classify an observed color by the driver.

        Args:
            color (str): The observed color.

        Returns:
            str: A classification of the possible risk.
        """
        if color == 'red':
            return 'diseased'
        elif color == 'yellow':
            return 'drought'
        elif color in ['blue', 'cyan']:
            return 'flooding'
        else:
            return 'normal'

    def update_interventions(self, res_dict):
        """Update the value of a cell when an intervention should take place.

        Note: Handle the interventions differently

        Args:
            res_dict (Dict): The response from the driver with their observation.
        """
        color = res_dict['color']
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
        """
        self.plan = nx.astar_path(self.graph, self.state, target, self.get_weight_for_edge)[1:]

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
        """Read the buffer in a separate process.

        Args:
            return_dict (Dict): A shared dictionary between processes to record read messages.
        """
        driver = serial.Serial(self.serial_port)
        return_dict['response'] = driver.readline()

    @staticmethod
    def encode_command(command):
        """Encode a command for the driver.

        Args:
            command (str): The command to send to the driver.

        Returns:
            bytes: An encoded command.
        """
        return command.encode('utf-8')

    @staticmethod
    def decode_response(byte_response):
        """Decode a response from the driver.

        Args:
            byte_response (bytes): A response in bytes.

        Returns:
            str: A str containing the byte response.
        """
        return byte_response.decode('utf-8')

    @staticmethod
    def format_response(res_str):
        """Format the response from the driver.

        Args:
            res_str (str): A driver response as a string.

        Returns:
                str: A response string stripped of useless info and trailing symbols.
        """
        if not res_str.endswith('\n'):
            return False

        elif '\r' in res_str:
            res_str = res_str.split('\r')[-1]

        return res_str.strip('\n')

    @staticmethod
    def format_command(command):
        """Format a command for the driver.

        Args:
            command (str): A command for the driver.

        Returns:
            str: A command with a trailing newline added.
        """
        return command + '\n'

    def command_driver(self, direction):
        """Command the driver to move in a specific direction.

        Args:
            direction (str): The direction to move into.

        Returns:
            Dict: A response dictionary.
        """
        formatted_direction = self.format_command(direction)
        encoded_direction = self.encode_command(formatted_direction)
        self.driver.write(encoded_direction)
        self.driver.flush()
        print('The command: ' + direction)

        while 1:
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            p = multiprocessing.Process(target=self.read_buffer, name='Reading', args=(return_dict,))
            p.start()
            p.join(self.driver_delay)

            if p.is_alive():
                print('Failed to respond, retrying...')
                p.terminate()
                p.join()
                return self.command_driver(direction)
            else:
                byte_response = return_dict['response']

            res_str = self.decode_response(byte_response)
            response = self.format_response(res_str)

            if not response:
                return self.command_driver(direction)

            print('The response: ' + res_str)
            try:
                if res_str.startswith('ERROR'):
                    print('Encountered an error, retrying...')
                    return self.command_driver(direction)
                else:
                    res_dict = json.loads(res_str)
                    break
            except Exception as e:
                raise Exception(e)
        return res_dict

    def execute_plan(self):
        """Execute the current monitoring plan."""
        while self.plan:
            next_state = self.plan.pop(0)
            direction = self.get_direction_from_next(next_state)
            res_dict = self.command_driver(direction)

            if res_dict['success']:
                self.state = next_state
                self.update_unchecked()
                self.update_interventions(res_dict)
                self.log_world_state()
            else:
                return False
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
        """Log the world state for visualisation by the dashboard."""
        world_dict = defaultdict(list)

        for vertex in self.graph.nodes:
            world_dict['X'].append(vertex[0])
            world_dict['Y'].append(vertex[1])
            world_dict['Unvisited'].append(max(0.1, self.unchecked[vertex]))
            world_dict['Risk'].append(self.risk_values[vertex])
            world_dict['Value'].append(self.vertex_values[vertex])

        world_dict['Robot'] = self.state
        world_dict['plan'] = self.plan

        with open(f'world_state.json', 'w') as f:
            json.dump(world_dict, f)

    def read_weights(self):
        """Read new weights from a file.

        Raises:
            FileNotFoundError: When the file does not exist.
        """
        try:
            with open('weights.json') as f:
                weights_dict = json.load(f)
            self.weights = self.normalise_weights(weights_dict['weights'])
        except FileNotFoundError:
            pass

    def monitor(self):
        """Continuously monitor the environment."""
        keep_monitoring = True

        while keep_monitoring:
            self.read_weights()
            endpoint = self.next_endpoint()
            self.update_edge_weights()
            self.get_plan(endpoint)
            keep_monitoring = self.execute_plan()
