from djitellopy import Tello
import time

def hover_drone():
    tello = Tello()
    tello.connect()

    tello.takeoff()
    time.sleep(5)  # Hover for 5 seconds

    tello.land()
    tello.end()

if __name__ == "__main__":
    hover_drone()
