#!/usr/bin/env python3.6
#
# Get and print live keyboard input from user
#

import keyboard


def do_on_press(key):
    print(key.name)


keyboard.on_press(do_on_press)


while True:
    pass
