#!/usr/bin/env python3
# ================================================
# BP Launcher — bp_launcher.py
# File: /home/ritam2002/bp_project/bp_launcher.py
# ================================================
# Runs on boot via LXDE autostart.
# Waits for BUTTON_START (GPIO17, Pin 11) to be
# pressed, then launches bp_main.py in lxterminal.
#
# AUTOSTART SETUP (run once):
#   mkdir -p /home/ritam2002/.config/lxsession/LXDE-pi
#   nano /home/ritam2002/.config/lxsession/LXDE-pi/autostart
#
#   Add this line:
#   @python3 /home/ritam2002/bp_project/bp_launcher.py
#
#   Save and reboot.
#
# WIRING:
#   BUTTON_START -> GPIO17 (Pin 11) + GND (Pin 9)
# ================================================

import os
import subprocess
import time
import RPi.GPIO as GPIO

PIN_START = 17
BP_MAIN   = "/home/ritam2002/bp_project/bp_main.py"
PYTHON    = "/usr/bin/python3"

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(PIN_START, GPIO.IN, pull_up_down=GPIO.PUD_UP)

print("BP Launcher ready.")
print(f"  Press BUTTON_START (GPIO{PIN_START}, Pin 11) to launch bp_main.py")

bp_process = None

try:
    while True:
        if GPIO.input(PIN_START) == GPIO.LOW:
            time.sleep(0.05)                          # debounce
            if GPIO.input(PIN_START) == GPIO.LOW:

                if bp_process is not None and bp_process.poll() is None:
                    print("bp_main.py already running — ignoring.")
                else:
                    print("BUTTON_START pressed — launching bp_main.py ...")

                    env = os.environ.copy()
                    env['DISPLAY']    = ':0'
                    env['XAUTHORITY'] = '/home/ritam2002/.Xauthority'

                    bp_process = subprocess.Popen(
                        ['lxterminal', '--command',
                         f'{PYTHON} {BP_MAIN}'],
                        env=env
                    )
                    print(f"  Launched (PID {bp_process.pid})")

                # Wait for button release
                while GPIO.input(PIN_START) == GPIO.LOW:
                    time.sleep(0.01)

        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nLauncher stopped.")

finally:
    GPIO.cleanup()
    if bp_process and bp_process.poll() is None:
        bp_process.terminate()
