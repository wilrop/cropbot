import serial
import time
from cropbot import CropBot
import os


def main():
    bot = CropBot(5, 5, [1, 1, 1])
    print(bot.update_edge_weights())
    raise Exception
    hub = serial.Serial('/dev/ttys009')  # 006 tty.LEGOHubA8E2C19D7ABF

    for i in range(10):
        hub.write(b'hello\n')
        time.sleep(1)

main()