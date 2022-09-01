import json
import time

import hub

from mindstorms import ColorSensor


class Driver:
    def __init__(self, cell_length):
        self.color_sensor = ColorSensor('D') #self.color_sensor = hub.port.D.device

        self.motor_back_move = hub.port.F.motor
        self.motor_front_move = hub.port.C.motor

        self.motor_back_turn = hub.port.E.motor
        self.motor_front_turn = hub.port.A.motor

        self.turn_pair = self.motor_front_turn.pair(self.motor_back_turn)
        self.move_pair = self.motor_back_move.pair(self.motor_front_move)

        self.cell_length = cell_length

        self.move_sleep = 2.5
        self.turn_sleep = 1.5
        self.move_speed_0 = -25
        self.move_speed_1 = - self.move_speed_0
        self.turn_speed = -10
        self.turn_degrees = 33
        self.motor_pos = 0
        self.flipped = False
        self.commands = {'stop', 'scan', 'forward', 'backward', 'left', 'right'}

        # hub.bluetooth.rfcomm_connect('F8:4D:89:81:E6:DC')
        # time.sleep(5)
        self.bt = hub.BT_VCP()

    def switch_left_right(self):
        self.turn_speed *= -1
        self.motor_pos *= -1

        self.flipped = not self.flipped

    def turn(self, direction):
        if (direction == 'left' and self.motor_pos == -1) or (direction == 'right' and self.motor_pos == 1):
            self.switch_left_right()

        if direction == 'left':
            if not self.flipped:
                self.move_speed_0 *= -1
                self.move_speed_1 *= -1
            self.motor_pos -= 1
            turn_speed = -self.turn_speed
        elif direction == 'right':
            if not self.flipped:
                self.move_speed_0 *= -1
                self.move_speed_1 *= -1
            self.motor_pos += 1
            turn_speed = self.turn_speed
        else:
            raise Exception("Not a valid direction")

        self.motor_back_turn.run_for_degrees(self.turn_degrees, speed=turn_speed)
        self.motor_front_turn.run_for_degrees(self.turn_degrees, speed=turn_speed)
        time.sleep(self.turn_sleep)

    def drive(self, direction, num_cells):
        degrees = 360 * num_cells * self.cell_length

        if direction == 'forward':
            speed_0 = self.move_speed_0
            speed_1 = self.move_speed_1
        elif direction == 'backward':
            speed_0 = -self.move_speed_0
            speed_1 = -self.move_speed_1
        else:
            raise Exception("Not a valid direction")

        self.move_pair.run_for_degrees(degrees, speed_0=speed_0, speed_1=speed_1)
        time.sleep(num_cells * self.move_sleep)

    def move(self, direction, num_cells):
        if direction in ['forward', 'backward']:
            if self.motor_pos == -1:
                self.turn('right')
            if self.motor_pos == 1:
                self.turn('left')
            self.drive(direction, num_cells)
        elif direction in ['left', 'right']:
            if self.motor_pos == 0:
                self.turn(direction)

            if direction == 'left':
                if self.motor_pos == -1:
                    self.drive('forward', num_cells)
                else:
                    self.drive('backward', num_cells)
            elif direction == 'right':
                if self.motor_pos == 1:
                    self.drive('forward', num_cells)
                else:
                    self.drive('backward', num_cells)
        elif direction == 'stay':
            time.sleep(1)
        else:
            raise Exception("Not a valid direction: " + direction)

    def classify_color(self, color):
        if color == 'red':
            return {'response': 'diseased'}
        elif color == 'black':
            return {'response': 'pests'}
        elif color == 'yellow':
            return {'response': 'drought'}
        elif color in ['blue', 'cyan']:
            return {'response': 'flooding'}
        else:
            return {'response': 'normal'}

    def execute_command(self, command):
        command = command.decode("utf-8").strip()
        self.move(command, 1)
        color = self.detect_color()
        response_dict = self.classify_color(color)
        response_str = json.dumps(response_dict) + '\n'
        byte_response = response_str.encode("utf-8")
        return byte_response

    def listen(self):
        while True:
            if self.bt.isconnected():
                hub.display.show(hub.Image.HAPPY)
                if self.bt.any():
                    try:
                        hub.display.show(hub.Image.YES)
                        command = self.bt.readline()
                        response = self.execute_command(command)
                    except Exception as e:
                        response = 'ERROR: ' + str(e) + '\n'
                        response = response.encode("utf-8")
                    self.bt.write(response)
            else:
                hub.display.show(hub.Image.SAD)

    def detect_color(self):
        return self.color_sensor.get_color()


def turn_debug(direction):
    driver = Driver(1)
    driver.turn(direction)


def small_demo():
    driver = Driver(1)
    driver.turn("left")
    driver.turn("right")
    driver.drive("forward", 1)
    driver.drive("backward", 1)


def square_demo(direction):
    driver = Driver(1)
    driver.drive("forward", 1)
    driver.turn(direction)
    driver.drive("forward", 1)
    driver.turn(direction)
    driver.drive("forward", 1)
    driver.turn(direction)
    driver.drive("forward", 1)
    driver.turn(direction)


def command_demo():
    driver = Driver(1)
    driver.move('forward', 1)
    driver.move('left', 1)
    driver.move('backward', 1)
    driver.move('right', 1)


def listen():
    driver = Driver(1)
    try:
        driver.listen()
    except Exception as e:
        error = 'Error: ' + str(e) + '\n'
        error_bytes = error.encode('utf-8')
        driver.bt.write(error_bytes)


def check_color():
    driver = Driver(1)
    while 1:
        color = driver.detect_color()
        print(color)
        time.sleep(1)


# turn_debug('left')
# square_demo("left")
# square_demo("right")
# command_demo()
listen()
