import json
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

        self.drive_sleep = 3
        self.flip_sleep = 1.5
        self.turn_sleep = 2
        self.color_sleep = 0.01

        self.drive_vertical_degrees = 420
        self.drive_horizontal_degrees = 375
        self.drive_adapt_degrees = 50
        self.flip_degrees = 100
        self.turn_degrees = 114

        self.move_speed = 20
        self.flip_speed = 10
        self.turn_speed = -20

        self.front_straight = True
        self.orientation = 'forward'
        self.horizontal = ['left', 'right']
        self.vertical = ['forward', 'backward']
        self.color_checks = color_checks
        self.commands = {'stop', 'scan', 'forward', 'backward', 'left', 'right'}
        self.bt = hub.BT_VCP()

    @staticmethod
    def decode_command(byte_command):
        """Decode an incoming command from CropBot.

        Args:
            byte_command (bytes): The CropBot command as a bytestring.

        Returns:
            str: The direction to drive into.
        """
        command = byte_command.decode('utf-8').strip('\n')
        return command

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
            speed = -self.turn_speed
            new_direction = 'forward'
        elif direction == 'right':
            speed = -self.turn_speed
            new_direction = 'forward'
        elif self.orientation == 'left':
            speed = self.turn_speed
            new_direction = direction
        else:
            speed = -self.turn_speed
            new_direction = direction

        self.drive_pair.run_for_degrees(self.turn_degrees, speed_0=speed, speed_1=speed)
        time.sleep(self.turn_sleep)
        return new_direction

    def turn(self, direction):
        self.flip_turn_wheels()
        new_direction = self.turn_driver(direction)
        self.flip_turn_wheels()
        self.orientation = direction
        return new_direction

    def drive(self, direction):
        """Drive into a specified direction.

        Args:
            direction (str): The direction to drive to.
        """
        if self.orientation in self.horizontal:  # Todo: Add adapt degrees.
            degrees = self.drive_horizontal_degrees
        else:
            degrees = self.drive_vertical_degrees

        if direction == 'forward':
            speed_0 = self.move_speed
            speed_1 = -self.move_speed
        else:
            speed_0 = -self.move_speed
            speed_1 = self.move_speed

        self.drive_pair.run_for_degrees(degrees, speed_0=speed_0, speed_1=speed_1)
        time.sleep(self.drive_sleep)

    def move(self, direction):
        """Move the driver to a new location.

        Args:
            direction (str): The direction to move to.

        Returns:
            bool: Always assumes the move succeeded.
        """
        if not all(x in self.horizontal for x in [self.orientation, direction]):
            new_direction = self.turn(direction)
        elif not all(x in self.vertical for x in [self.orientation, direction]):
            new_direction = self.turn(direction)
        else:
            new_direction = direction
        self.drive(new_direction)
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
        """Execute an incomming command.

        Args:
            command (bytes): A command from CropBot.

        Returns:
            bytes: A response with the driving success and observed color.
        """
        command = self.decode_command(command)
        success = self.move(command)
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


def drive_demo(direction):
    driver = Driver()
    driver.move(direction)


def square_demo():
    driver = Driver()
    driver.move('forward')
    driver.move('right')
    driver.move('backward')
    driver.move('left')


def check_color():
    driver = Driver()
    while 1:
        color = driver.detect_color()
        print(color)
        time.sleep(1)


def listen():
    driver = Driver()
    driver.listen()


# drive_demo('right')
# square_demo()
# check_color()
listen()
