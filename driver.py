import json
import math
import time

import hub
from mindstorms import ColorSensor


class Driver:
    def __init__(self, color_checks=20):
        self.color_sensor = ColorSensor('D')

        self.motor_left_turn = hub.port.C.motor
        self.motor_right_turn = hub.port.F.motor
        self.motor_left_drive = hub.port.A.motor
        self.motor_right_drive = hub.port.E.motor

        self.turn_pair = self.motor_left_turn.pair(self.motor_right_turn)
        self.drive_pair = self.motor_left_drive.pair(self.motor_right_drive)

        self.drive_sleep = 2.75
        self.flip_sleep = 1.5
        self.turn_sleep = 2.5
        self.color_sleep = 0.01

        self.drive_vertical_degrees = 215
        self.drive_horizontal_degrees = 215
        self.drive_adapt_degrees = 80
        self.flip_degrees = 108
        self.left_turn_degrees = 105
        self.right_turn_degrees = 106

        self.move_speed = 20
        self.flip_speed = 10
        self.turn_speed = -10
        self.reset_speed = 10

        self.turned = False
        self.front_straight = True
        self.orientation = 'forward'
        self.horizontal = {'left', 'right'}
        self.vertical = {'forward', 'backward'}
        self.swap_direction = {'left': 'right', 'right': 'left', 'forward': 'backward', 'backward': 'forward'}
        self.color_checks = color_checks
        self.commands = {'stop', 'scan', 'forward', 'backward', 'left', 'right'}
        self.bt = hub.BT_VCP()

    def decode_command(self, byte_command):
        """Decode an incoming command from CropBot.

        Args:
            byte_command (bytes): The CropBot command as a bytestring.

        Returns:
            str: The direction to drive into.
        """
        command = byte_command.decode('utf-8').strip('\n')

        try:
            command_dict = json.loads(command)
            direction = command_dict['direction']
            next_direction = command_dict['next']
        except json.decoder.JSONDecodeError:
            raise json.decoder.JSONDecodeError

        if direction in self.commands and next_direction in self.commands:
            return direction, next_direction
        else:
            raise ValueError

    @staticmethod
    def format_response(success, color):
        """Format a response for the CropBot.

        Args:
            success (bool): Whether the operation finished successfully.
            color (str): The observed color.

        Returns:
            Dict: The response as a dictionary.
        """
        response = {'success': success, 'color': color}
        return response

    @staticmethod
    def encode_response(response):
        """Encode a response dictionary.

        Args:
            response (Dict): A dictionary with the driver's response.

        Returns:
            bytes: The response dictionary encoded as bytes with a trailing endline.
        """
        response_str = json.dumps(response) + '\n'
        byte_response = response_str.encode('utf-8')
        return byte_response

    def reset_motor_angles(self):
        """Compensate the offset of a motor angle every time it wants to drive."""
        offset_left = self.motor_left_turn.get()[2]
        offset_right = self.motor_right_turn.get()[2]

        self.motor_left_turn.run_for_degrees(offset_left, speed=-math.copysign(self.reset_speed, offset_left))
        self.motor_right_turn.run_for_degrees(offset_right, speed=-math.copysign(self.reset_speed, offset_right))

    def flip_turn_wheels(self):
        """Flip the turn wheels to make turning or driving possible."""
        self.turn_pair.run_for_degrees(self.flip_degrees, speed_0=self.flip_speed, speed_1=-self.flip_speed)
        self.flip_speed *= -1
        time.sleep(self.flip_sleep)

    def turn_driver(self, direction):
        """Turn the driver in a new direction.

        Args:
            direction (str): The direction the driver wants to go to next.

        Returns:
            str: The new direction the driver should move to.
        """
        if direction == 'left':
            speed = self.turn_speed
            turn_degrees = self.left_turn_degrees
        elif direction == 'right':
            speed = -self.turn_speed
            turn_degrees = self.right_turn_degrees
        else:
            raise ValueError

        self.drive_pair.run_for_degrees(turn_degrees, speed_0=speed, speed_1=speed)
        time.sleep(self.turn_sleep)

    def turn(self, direction, new_orientation):
        """Turn the driver in a specific direction.

        Args:
            direction (str): The direction to turn into.
            new_orientation (str): The orientation after turning.

        Returns:
            str: The new direction that the robot should drive into to drive into the originally requested direction.
        """
        self.flip_turn_wheels()
        self.turn_driver(direction)
        self.flip_turn_wheels()
        self.orientation = new_orientation
        self.turned = True

    def needs_turn(self, direction):
        """Check whether a direction will need a turn.

        Args:
            direction (str): The direction to turn into.

        Returns:
            bool: Whether this direction requires a turn or not.
        """
        if self.orientation in self.vertical and direction in self.horizontal:
            return True
        elif self.orientation in self.horizontal and direction in self.vertical:
            return True
        else:
            return False

    def drive(self, direction, next_direction):
        """Drive into a specified direction.

        Args:
            direction (str): The direction to drive to.
            next_direction (str): The next direction to drive to.
        """
        if self.orientation in self.horizontal:  # Todo: Add adapt degrees.
            degrees = self.drive_horizontal_degrees
        else:
            degrees = self.drive_vertical_degrees

        if self.turned:
            self.turned = False
            if direction == 'forward':
                degrees -= self.drive_adapt_degrees
            else:
                degrees += self.drive_adapt_degrees

        if self.needs_turn(next_direction):
            degrees += self.drive_adapt_degrees

        if direction == 'forward':
            speed_0 = self.move_speed
            speed_1 = -self.move_speed
        else:
            speed_0 = -self.move_speed
            speed_1 = self.move_speed

        self.reset_motor_angles()
        self.drive_pair.run_for_degrees(degrees, speed_0=speed_0, speed_1=speed_1)
        time.sleep(self.drive_sleep)

    def move(self, direction, next_direction):
        """Move the driver to a new location.

        Args:
            direction (str): The direction to move to.
            next_direction (str): The next direction to move to.

        Returns:
            bool: Always assumes the move succeeded.
        """
        if direction in self.horizontal and self.orientation in self.horizontal:  # Direction & orientation horizontal.
            if self.orientation == direction:
                new_direction = 'forward'
            else:
                new_direction = 'backward'
        elif direction in self.horizontal:  # Direction horizontal and front facing.
            self.turn(direction, direction)
            new_direction = 'forward'
        elif self.orientation in self.vertical:  # If direction and orientation are forward.
            new_direction = direction
        elif self.orientation in self.horizontal:  # If orientation is horizontal, turn in the opposite direction.
            turn_direction = self.swap_direction[self.orientation]
            self.turn(turn_direction, 'forward')
            new_direction = direction
        else:
            raise ValueError

        self.drive(new_direction, next_direction)
        return True

    def detect_color(self):
        """Detect the color currently underneath the sensor.

        Returns:
            str: The most frequently observed color from a number of samples.
        """
        colors = []
        for i in range(self.color_checks):
            colors.append(self.color_sensor.get_color())
            time.sleep(self.color_sleep)
        most_frequent_color = max(set(colors), key=colors.count)
        return most_frequent_color

    def execute_command(self, command):
        """Execute an incoming command.

        Args:
            command (bytes): A command from CropBot.

        Returns:
            bytes: A response with the driving success and observed color.
        """
        direction, next_direction = self.decode_command(command)
        success = self.move(direction, next_direction)
        color = self.detect_color()
        response_dict = self.format_response(success, color)
        byte_response = self.encode_response(response_dict)
        return byte_response

    def listen(self):
        """Listen to incoming commands from the CropBot."""
        while True:
            if self.bt.isconnected():
                hub.display.show(hub.Image.HAPPY)

                if self.bt.any():
                    hub.display.show(hub.Image.YES)
                    try:
                        command = self.bt.readline()
                        response = self.execute_command(command)
                    except AttributeError as e:
                        hub.display.show(hub.Image.NO)
                        response = 'ERROR: ' + str(e) + '\n'
                        response = response.encode('utf-8')
                    self.bt.write(response)
            else:
                hub.display.show(hub.Image.SAD)


def check_color():
    driver = Driver()
    while 1:
        color = driver.detect_color()
        print(color)
        time.sleep(1)


def turn_demo():
    driver = Driver()
    driver.turn('right', 'right')
    driver.turn('left', 'forward')


def drive_demo(direction):
    driver = Driver()
    driver.move(direction, 'stay')


def square_demo(squares=1):
    driver = Driver()
    commands = ['forward', 'right', 'backward', 'left', 'stay']
    for _ in range(squares):
        for command, next_command in zip(commands, commands[1:]):
            driver.move(command, next_command)


def listen():
    driver = Driver()
    driver.listen()


# check_color()
# turn_demo()
# drive_demo('right')
# square_demo(2)
listen()
