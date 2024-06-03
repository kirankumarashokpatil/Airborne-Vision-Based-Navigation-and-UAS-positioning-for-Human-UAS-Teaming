import airsim
import cv2
import numpy as np
import orb_slam3

def main():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    vocab_file = "ORBvoc.txt"
    settings_file = "settings.yaml"

    slam = orb_slam3.ORBSLAM3(vocab_file, settings_file)

    while True:
        responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
        img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(responses[0].height, responses[0].width, 3)

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        slam.TrackMonocular(gray, airsim.to_timestamp(responses[0].time_stamp) / 1000000.0)
        cv2.imshow("Frame", img_rgb)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    slam.Shutdown()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
