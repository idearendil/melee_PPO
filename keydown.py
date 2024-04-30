from pynput.keyboard import Key, Controller
import time

if __name__ == "__main__":

    control = Controller()

    time.sleep(5)
    control.press(Key.tab)
    time.sleep(3600 * 24)
    control.release(Key.tab)
