import serial
import time
from cropbot import CropBot
import os


def main():
    bot = CropBot(5, 5, [1, 1, 1])
    bot.monitor()

main()