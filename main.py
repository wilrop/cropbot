import serial
import time
from cropbot import CropBot
import os


def main():
    bot = CropBot(4, 5, [1, 1, 1])
    bot.monitor()


if __name__ == '__main__':
    main()
