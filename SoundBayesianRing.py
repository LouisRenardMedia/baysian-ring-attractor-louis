import csv
import queue
import time

from flask import Flask, Response
import cv2
import numpy as np
from collections import deque
import angle_utils
import random
import threading
import sounddevice as sd

from IMUReader import IMUReader
from robot_toy import _set_motors, stop, SPIN_SPEED, main
import Recorder
from scipy.spatial.distance import euclidean
from start_cam import UnwarpCamera
import Baysian_Ring_Attractor
from visual_compass import VisualCompass
import Circular_Kalman_Filter
from wait_for_start import wait_for_start
from ring_attractor_audio_array_DOA import update_das, get_card, audio_callback


app = Flask(__name__)
cam = UnwarpCamera()
#cam.start_stream(port=8080)

USE_SMOOTHING = True    # Toggle on or off for circle position averaging over history
MODE = "RNN"            # "CKF" for circular kalman filter and "RNN" for baysian ring attractor
ROBOT_TURN = False
REALTIMESYNC = True
STARTSYNC = False
ROBOT_CONTROL = False

log_file = open('rnn_estimates.csv', 'w', newline='')
log_writer = csv.writer(log_file)
log_writer.writerow(['timestamp', 'mu', 'kappa', 'IMU_omega', 'IMU_compass', 'opt_flow'])
if REALTIMESYNC:
    recorder = Recorder.Recorder(cam)


# Buffer to make detection stable
circle_history = deque(maxlen=5)
mu = deque(maxlen=1)
vc = VisualCompass()

N = 30                      # Neuron count
k_v = [3,0.5,3]              # certainty of angular velocity input
kappa_phi = 0.001              # Diffusion parameter (inverse so high number is low diffusion)
k_z0 = 10                    # Certainty of HD input
tau = 1
sigma_N = 0
phi_0 = 0
kappa_0 = 10
w_const = 0
w_quad = 1/2.5
stoch_corr = 0

dt=1/30  # 1/fps

imu = IMUReader(phi_0)
# Start the audio stream
fs = 16000  # sampling frequency
print(f"Sampling Frequency: {fs} Hz")
channels = 5
usb_card_index = get_card(sd.query_devices())
global q
q = queue.Queue()

stream = sd.InputStream(
    device=usb_card_index,
    samplerate=fs,
    channels=channels,
    blocksize=128,
    callback=audio_callback,
    latency="low",
)

stream.start()

def generate_frames():
    if MODE == "CKF":
        filter = Circular_Kalman_Filter.CKF(kappa_phi,dt,k_z0,k_v)
    elif MODE == "RNN":
        filter = Baysian_Ring_Attractor.BayesianRingAttractor(N, dt, tau, kappa_phi, k_v, k_z0, w_const, w_quad, kappa_0, phi_0, stoch_corr)

    if STARTSYNC:
        t_start = wait_for_start()  # blocks until signal received

    if ROBOT_TURN:
        rotation_thread = threading.Thread(target=random_rotation, args=(60,), daemon=True)
        rotation_thread.start()
    if ROBOT_CONTROL:
        control_thread = threading.Thread(target=main, args=(), daemon=True)
        control_thread.start()

    prev_angle = np.inf
    frames_since_detection = 1
    try:
        while True:
            if REALTIMESYNC:
                frame, time_stamp, frames_since_detection = recorder.get() # dy only comes in if realtimesync is on
                if frame is None:
                    continue
            else:
                success, frame = cam.read()
                if not success:
                    print("Camera read failed")
                    break
            w, dtheta = imu.get(filter.mu[-1])
            opt_flow_dis = vc.update(frame)
            dy = np.array([w * frames_since_detection, dtheta/dt, opt_flow_dis/dt]) #collect data from the IMU at a similar time as the frame
            dy = np.clip(dy, -8,8)
            output = frame.copy()

            sound_curve = update_das()
            if sound_curve is not None:
                mu, k_z = sound_to_von_mises(sound_curve)
                filter.step(dy=dy, z=mu, k_z=k_z)
            else:
                filter.step(dy=dy)

            # Log CSV file
            try:
                if REALTIMESYNC:
                    log_writer.writerow([time_stamp, filter.mu[-1], filter.kappa[-1], dy[0], dy[1], dy[2]])
                else:
                    log_writer.writerow([time.time(), filter.mu[-1], filter.kappa[-1]])
                log_file.flush()
            except Exception as e:
                print("CSV ERROR:", repr(e))


            # HD indicator — top right corner
            if len(filter.mu) > 1:
                output = draw_hd_indicator(output, filter.mu[-1], filter.kappa[-1], imu.getdistance())

            # Encode as MJPEG and yield
            ret, buffer = cv2.imencode('.jpg', output)
            frame_bytes = buffer.tobytes()
            yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
            )
    except KeyboardInterrupt:
        print("Stopping the audio stream.")
    finally:
        stream.stop()
        stream.close()

def sound_to_von_mises(sound_curve, h=0.1, s=5):
    mu = -(np.radians(np.argmax(sound_curve)-1) - np.pi)
    min_s = ItoDb(np.partition(sound_curve, s)[s-1])
    max_s = ItoDb(np.partition(sound_curve, -s)[-s])

    k_z = (h*(max_s - min_s))**2

    k_z = min(10, k_z)
    print('mu is::::',mu,k_z)
    return mu, k_z

def ItoDb(I):
    return 10*np.log10(I/10**-12)

def draw_hd_indicator(frame, mean, kappa, distance, size=80):
    """
    Draws a circular HD indicator in the top-right corner.
    - The arrow direction encodes mean (mu)
    - The arrow length + arc encodes kappa (certainty)
    """

    h, w = frame.shape[:2]
    margin = 10
    cx = w - margin - size  # centre x
    cy = margin + size  # centre y

    # --- background circle ---
    cv2.circle(frame, (cx, cy), size, (30, 30, 30), -1)  # dark fill
    cv2.circle(frame, (cx, cy), size, (180, 180, 180), 1)  # grey border

    # --- cardinal direction ticks ---
    for angle_deg in [0, 90, 180, 270]:
        a = np.radians(angle_deg)
        x_tick = int(cx + (size - 6) * np.cos(a))
        y_tick = int(cy - (size - 6) * np.sin(a))
        x_end = int(cx + size * np.cos(a))
        y_end = int(cy - size * np.sin(a))
        cv2.line(frame, (x_tick, y_tick), (x_end, y_end), (180, 180, 180), 1)

    # --- uncertainty arc ---
    # kappa_max: beyond this we consider certainty "full"
    kappa_max = 15.0
    certainty = float(np.clip(kappa / kappa_max, 0.0, 1.0))
    arc_thickness = max(2, int(6 * certainty))  # thicker arc = more certain

    # draw arc as a filled colour band — use ellipse with angle sweep
    # opencv ellipse: angles are clockwise from 3-o-clock
    # mean=0 → east in maths → we convert to opencv angle (clockwise from east)
    mean_deg_cv = -float(np.degrees(mean))  # flip y axis for screen coords
    sweep = int(certainty * 360)
    start_angle = int(mean_deg_cv - sweep / 2)
    end_angle = int(mean_deg_cv + sweep / 2)

    # colour goes green (certain) → red (uncertain)
    # BGR format
    color_certain = (0, 255, 0)  # green
    color_uncertain = (0, 0, 255)  # red
    arc_color = (
        0,
        int(255 * certainty),  # G channel
        int(255 * (1 - certainty))  # R channel
    )

    cv2.ellipse(frame, (cx, cy), (size - 4, size - 4),
                0, start_angle, end_angle, arc_color, arc_thickness)

    # --- direction arrow ---
    arrow_len = int(size * 0.7 * certainty + size * 0.2)  # longer = more certain
    ax = int(cx + arrow_len * np.cos(mean))
    ay = int(cy - arrow_len * np.sin(mean))  # flip y for screen
    cv2.arrowedLine(frame, (cx, cy), (ax, ay),
                    (255, 255, 255), 2, tipLength=0.3)

    # --- kappa text ---
    cv2.putText(frame, f'k={kappa:.1f}', (cx - 20, cy + size + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    # --- distance text ---
    text_y = cy + size + 16

    if distance is not None:
        cv2.putText(
            frame,
            f'd={distance:.2f} m',
            (cx - 110, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (100, 60, 64),
            2,
        )

    return frame

#UNUSED may be useful if integrating 2+ landmarks
def get_stable_circles(recent_circles, max_dist=20):
    all_detections = [c for frame in recent_circles for c in frame]
    if not all_detections:
        return []

    groups = []
    for c in all_detections:
        matched = False
        for group in groups:
            # compare to the first circle in the group
            if euclidean(c[:2], group[0][:2]) < max_dist:
                group.append(c)
                matched = True
                break
        if not matched:
            groups.append([c])

    # only draw groups seen in enough frames
    stable = []
    for group in groups:
        if len(group) >= 2:
            avg_x = int(np.mean([c[0] for c in group]))
            avg_y = int(np.mean([c[1] for c in group]))
            avg_r = int(np.mean([c[2] for c in group]))
            stable.append((avg_x, avg_y, avg_r))

    return stable

def get_smoothed_circle(recent_circles):
    """
    Averages x, y, radius across recent detections to reduce jitter.
    Reuses the same recent_circles list already built in generate_frames().
    """
    if not recent_circles:
        return None

    all_circles = np.concatenate(recent_circles)

    angles = np.array([angle_utils.calc_angle(c[0]) for c in all_circles])
    avg_cos = np.mean(np.cos(angles))
    avg_sin = np.mean(np.sin(angles))
    avg_angle = np.arctan2(avg_sin, avg_cos)  # back to polar

    # Convert avg_angle back to a pixel x position
    avg_x_circ = angle_utils.calc_position(avg_angle)

    avg_y      = int(np.mean(all_circles[:, 1]))
    avg_radius = int(np.mean(all_circles[:, 2]))

    return np.array([[avg_x_circ, avg_y, avg_radius]])

def random_rotation(duration_total=60):
    """
    Randomly spins the robot left/right for duration_total seconds.
    Runs in a thread so it doesn't block main logging loop.
    """
    end_time = time.time() + duration_total

    while time.time() < end_time:
        # random direction
        direction = random.choice([-1, 1])  # -1 = left, 1 = right
        speed = random.randint(1200, SPIN_SPEED)
        spin_duration = random.uniform(0.5, 2.0)

        _set_motors(-direction * speed, direction * speed)
        time.sleep(spin_duration)

        stop()
        time.sleep(random.uniform(0.1, 0.4))  # brief pause

    stop()





@app.route('/')
def index():
    return "<h1>Stable Red Circle Stream</h1><img src='/video_feed'>"

@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

