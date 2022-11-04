from cropbot import CropBot

FIELD = {(0, 0): 'green', (0, 1): 'blue', (0, 2): 'blue', (0, 3): 'green', (0, 4): 'green', (0, 5): 'green', (0, 6): 'yellow',
         (1, 0): 'green', (1, 1): 'blue', (1, 2): 'blue', (1, 3): 'green', (1, 4): 'green', (1, 5): 'green', (1, 6): 'yellow',
         (2, 0): 'green', (2, 1): 'green', (2, 2): 'green', (2, 3): 'green', (2, 4): 'green', (2, 5): 'green', (2, 6): 'yellow',
         (3, 0): 'green', (3, 1): 'green', (3, 2): 'yellow', (3, 3): 'green', (3, 4): 'green', (3, 5): 'green', (3, 6): 'red',
         (4, 0): 'yellow', (4, 1): 'yellow', (4, 2): 'yellow', (4, 3): 'red', (4, 4): 'green', (4, 5): 'green', (4, 6): 'yellow',
         (5, 0): 'yellow', (5, 1): 'red', (5, 2): 'yellow', (5, 3): 'yellow', (5, 4): 'green', (5, 5): 'green', (5, 6): 'green',
         (6, 0): 'green', (6, 1): 'green', (6, 2): 'green', (6, 3): 'green', (6, 4): 'green', (6, 5): 'green', (6, 6): 'blue',
         (7, 0): 'green', (7, 1): 'green', (7, 2): 'green', (7, 3): 'red', (7, 4): 'green', (7, 5): 'green', (7, 6): 'blue',
         (8, 0): 'green', (8, 1): 'blue', (8, 2): 'blue', (8, 3): 'green', (8, 4): 'green', (8, 5): 'green', (8, 6): 'red',
         (9, 0): 'green', (9, 1): 'green', (9, 2): 'green', (9, 3): 'green', (9, 4): 'green', (9, 5): 'green', (9, 6): 'green'}


def main():
    bot = CropBot(10, 7, [1, 1, 1], field=FIELD)
    bot.monitor()


if __name__ == '__main__':
    main()
