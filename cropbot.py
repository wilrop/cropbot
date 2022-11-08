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

    def __init__(self, width, height, weights, start=(0, 0), prior_risk=0.1, prior_conditional_risk=0.5,
                 empirical_threshold=10, field=None):
        self.field = field
        self.width = width
        self.height = height
        self.weights = self.normalise_weights(weights)
        self.state = start
        self.prev = None

        self.serial_port = '/dev/ttys013'
        self.driver = serial.Serial(self.serial_port)
        self.directions = {(0, 0): 'stay', (0, 1): 'forward', (0, -1): 'backward', (-1, 0): 'left', (1, 0): 'right'}
        self.driver_delay = 10

        self.graph = nx.grid_2d_graph(width, height).to_directed()
        self.edge_values = defaultdict(int)
        self.vertex_values = defaultdict(int)
        self.risk_values = defaultdict(int)

        self.unchecked = np.zeros((width, height))
        self.inverse_distances = self.calc_inverse_distances(width, height)
        self.active_interventions = np.zeros((width, height))
        self.intervention_freqs = np.zeros((width, height, 2))
        self.conditional_freqs = self.setup_conditional_freqs()
        self.empirical_threshold = empirical_threshold
        self.prior_risk = prior_risk
        self.prior_conditional_risk = prior_conditional_risk

        self.plan = self.get_plan(self.state)
        self.next_plan = []

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

    def setup_conditional_freqs(self):
        """Build the dictionary of conditional frequencies.

        Returns:
            Dict: A dictionary of conditional intervention frequencies of vertices and neighbours of them.
        """
        conditional_freqs = {}
        for vertex in self.graph:
            conditional_freqs[vertex] = {}
            for neighbour in self.graph.neighbors(vertex):
                conditional_freqs[vertex][neighbour] = np.zeros(2)
        return conditional_freqs

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
        """Normalise a scalar in a range.

        Args:
            x (float): The value to normalise.
            min_x (float): The minimum value of x.
            max_x (float): The maximum value of x.

        Returns:
            float: The normalised value.
        """
        return x if x == 0 else (x - min_x) / (max_x - min_x)

    def get_active_neighbours(self, vertex):
        """Get the neighbours who have active interventions.

        Args:
            vertex (Tuple[int]): The coordinates of the vertex.

        Returns:
            List[Tuple[int]]: A tuple of active neighbours.
        """
        active_neighbours = []

        for neighbour in self.graph.neighbors(vertex):
            if self.active_interventions[neighbour]:
                active_neighbours.append(neighbour)
        return active_neighbours

    def compute_prob(self, interventions, total, conditional=False):
        """Compute the empirical probability for an intervention over a total if the total exceeds the threshold.

        Args:
            interventions (int): The number of interventions.
            total (int): The total number of samples.

        Returns:
            float: A probability estimate.
        """
        if total >= self.empirical_threshold:
            risk_prob = interventions / total
        else:
            if conditional:
                risk_prob = self.prior_conditional_risk
            else:
                risk_prob = self.prior_risk
        return risk_prob

    def compute_risk_prob(self, vertex):
        """Compute the current risk estimate for a given vertex.

        Args:
            vertex (Tuple[int]): A given vertex.

        Returns:
            float: A risk estimate for this vertex as a probability.
        """
        interventions, total = self.intervention_freqs[vertex]
        risk_prob = self.compute_prob(interventions, total)
        active_neighbours = self.get_active_neighbours(vertex)

        if active_neighbours:
            neighbours_probs = []
            conditional_neighbours_probs = []

            for neighbour in active_neighbours:
                interventions, total = self.intervention_freqs[neighbour]
                conditional_interventions, conditional_total = self.conditional_freqs[neighbour][vertex]
                neighbours_probs.append(self.compute_prob(interventions, total))
                conditional_neighbours_probs.append(self.compute_prob(conditional_interventions, conditional_total, conditional=True))

            neighbours_risk_prob = np.prod(neighbours_probs)
            conditional_risk_prob = np.prod(conditional_neighbours_probs)
            conditional_risk = conditional_risk_prob * risk_prob / neighbours_risk_prob
            conditional_risk = min(conditional_risk, 0.95)
            return conditional_risk
        else:
            return risk_prob

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

        for vertex in self.graph.nodes:
            unchecked_value = self.normalise_scalar(self.unchecked[vertex], min_unchecked, max_unchecked)
            distance_value = self.normalise_scalar(self.inverse_distances[self.state + vertex], min_inv_dist,
                                                   max_inv_dist)
            risk_value = self.risk_values[vertex]  # Is already a probability.
            vertex_value = self.weights[0] * unchecked_value + self.weights[1] * distance_value + self.weights[
                2] * risk_value
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

    def classify_color(self, color):
        """Classify an observed color by the driver.

        Args:
            color (str): The observed color.

        Returns:
            str: A classification of the possible risk.
        """
        if self.field is not None:
            color = self.field[self.state]

        if color == 'red':
            return 'diseased'
        elif color == 'yellow':
            return 'drought'
        elif color in ['blue', 'cyan']:
            return 'flooding'
        else:
            return 'normal'

    def update_model(self, res_dict):
        """Update the statistical model of the environment.

        Args:
            res_dict (Dict): A dictionary containing the result of the driver.
        """
        color = res_dict['color']
        classification = self.classify_color(color)
        intervention_necessary = classification in ['diseased', 'pests', 'flooding', 'drought']

        if self.active_interventions[self.prev]:
            self.conditional_freqs[self.state][self.prev][0] += intervention_necessary
            self.conditional_freqs[self.state][self.prev][1] += 1

        if intervention_necessary:
            self.active_interventions[self.state] = 1
            self.intervention_freqs[self.state][0] += 1
            self.conditional_freqs[self.prev][self.state][0] += self.active_interventions[self.prev]
            self.conditional_freqs[self.prev][self.state][1] += 1
        else:
            self.active_interventions[self.state] = 0

        self.intervention_freqs[self.state][1] += 1

    def get_plan(self, start):
        """Compute the optimal trajectory from the current state to a given target.

        Args:
            start (Tuple[int]): The starting state for the plan.
        """
        self.update_vertex_values()
        vertex_values = copy.deepcopy(self.vertex_values)
        del vertex_values[start]
        target = max(vertex_values, key=vertex_values.get)
        return nx.astar_path(self.graph, start, target, self.get_weight_for_edge)[1:]

    def get_direction_from_next(self, state, next_state):
        """Determine the direction to move from the next state.

        Args:
            state (Tuple[int]): The start state.
            next_state (Tuple[int]): The next state.

        Returns:
            str: The direction to move into.
        """
        move_tpl = tuple(np.array(next_state) - np.array(state))
        return self.directions[move_tpl]

    def read_buffer(self, return_dict):
        """Read the buffer in a separate process.

        Args:
            return_dict (Dict): A shared dictionary between processes to record read messages.
        """
        driver = serial.Serial(self.serial_port)
        return_dict['response'] = driver.readline()

    @staticmethod
    def encode_command(direction, next_direction):
        """Encode a command for the driver.

        Args:
            direction (str): The direction to move into.
            next_direction (str): The next direction to move into.

        Returns:
            bytes: An encoded command.
        """
        command_dict = {'direction': direction, 'next': next_direction}
        command_str = json.dumps(command_dict) + '\n'
        return command_str.encode('utf-8')

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

    def command_driver(self, direction, next_direction):
        """Command the driver to move in a specific direction.

        Args:
            direction (str): The direction to move into.
            next_direction (str): The next direction to move into.

        Returns:
            Dict: A response dictionary.
        """
        encoded_direction = self.encode_command(direction, next_direction)
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
                return self.command_driver(direction, next_direction)
            else:
                byte_response = return_dict['response']

            res_str = self.decode_response(byte_response)
            response = self.format_response(res_str)

            if not response:
                return self.command_driver(direction, next_direction)

            print('The response: ' + response)
            try:
                if response.startswith('ERROR'):
                    print('Encountered an error, retrying...')
                    return self.command_driver(direction, next_direction)
                else:
                    res_dict = json.loads(response)
                    break
            except Exception as e:
                raise Exception(e)
        return res_dict

    def execute_plan(self):
        """Execute the current monitoring plan."""
        while self.plan:
            self.log_world_state()
            next_state = self.plan.pop(0)
            if self.plan:
                next_next_state = self.plan[0]
            else:
                self.next_plan = self.get_plan(next_state)
                next_next_state = self.next_plan[0]

            direction = self.get_direction_from_next(self.state, next_state)
            next_direction = self.get_direction_from_next(next_state, next_next_state)
            res_dict = self.command_driver(direction, next_direction)

            if res_dict['success']:
                self.prev = self.state
                self.state = next_state
                self.update_model(res_dict)
                self.update_unchecked()
                self.risk_values = {vertex: self.compute_risk_prob(vertex) for vertex in self.graph}
            else:
                return False
        return True

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
            self.update_edge_weights()
            keep_monitoring = self.execute_plan()
            self.plan = self.next_plan
