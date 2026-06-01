import time

from flask import Flask, Response
import cv2
import numpy as np
from collections import deque

from scipy.spatial.distance import euclidean

from start_cam import UnwarpCamera
import Circular_Kalman_Filter
from circularFiltering import vM_Projection
import robot_toy

app = Flask(__name__)
cam = UnwarpCamera()
#cam.start_stream(port=8080)

USE_SMOOTHING = True    # Toggle on or off for circle position averaging over history
MODE = "RNN"

mean_ = [0.]            # Innitialize position to 0
activations_ = []
kappa_ = [1.]           # Innitialize certainty to 1

# Buffer to make detection stable
circle_history = deque(maxlen=5)

N = 30                      # Neuron count
kappa_v = 5              # certainty of angular velocity input
kappa_fi = 5.0              # Diffusion parameter (inverse so high number is low diffusion)
kappa_z = 10                 # Certainty of HD input
tau = 0.5
I_ext = 0
sigma_N = 0
phi_0=0
kappa_0 = 1
w_const = 1
w_quad = 0.2
stoch_corr = 0
r = deque(maxlen=5)
z = None
dy = None
phi_0_r = None
dt=1/30             # 1/fps


#### calculated params #####
w_asym = kappa_v / (kappa_fi + kappa_v)
w_sym = 1/tau

def RNN_init():

    # vector of preferred HD
    phi = np.linspace(-np.pi, np.pi - (2 * np.pi) / N, N)

    # Set up weight matrix
    diff = phi[:, None] - phi[None, :]  # shape (N, N)
    W_sym = w_sym * (2 / N) * np.cos(diff)
    W_asym = (2 / N) * np.sin(diff) * w_asym
    W_const = 1 / N * np.ones((N, N)) * w_const

    # init
    r.append(kappa_0 * np.cos(phi - phi_0))

    return phi, W_sym, W_asym, W_const





def RNN_step(phi, dt=dt,
                   W_sym=0, W_asym=0, W_const=0, w_quad=w_quad,
                    stoch_corr=stoch_corr, dy=0, z=0, k_z=0):
    """" Runs a recurrent neural network dynamics, with parameters matched to
    approximate the circKF.

    Input:
    dt          - time step
    w_sym      - even recurrent connectivity
    w_asym       - odd recurrent connectivity
    tau         - decay time constant
    w_quad      - quadratic weight
    stoch_corr  - stochastic correction (additional decay due to Ito conversion)
    dy          - increment observation

    Output:
    mu      - mean estimate after update
    kappa   - certainty estimate after update """

    f_act = lambda x: np.maximum(0, x)

    # set up all-to-all summation
    M = np.pi / N * np.ones([N, N])

    # add Wiener process if there is neural noise
    # if sigma_N != 0:
    #     dW = np.sqrt(dt) * np.random.randn(int(T / dt), N)
    # else:
    #     dW = np.zeros((int(T / dt), N))

    # run network filter
    W = W_sym + W_asym * (dy / dt) + W_const

    r.append((r[-1]
            - stoch_corr * r[-1] * dt  # stochastic correction
            - 1 / tau * r[-1] * dt  # decay
            + np.dot(W, r[-1]) * dt  # angular velocity integration, recurrent stabilization
            - w_quad * np.dot(M, f_act(r[-1])) * r[-1] * dt  # quadratic inhibition
            + k_z * dt * np.cos(phi - z)))  # absolute heading info (external input)
            #+ sigma_N * dW[i]))

    # decode stochastic variables
    basis = np.array([np.cos(phi), np.sin(phi)])  # (2, N)

    theta = (2 / N) * (basis @ r[-1])  # (2,)
    # theta[0] = κ·cos(μ) and theta[1] = κ·sin(μ)

    mu = np.arctan2(theta[1], theta[0])
    kappa = np.linalg.norm(theta)

    return mu, kappa




def generate_frames():
    if MODE == "CKF":
        filter = Circular_Kalman_Filter.CKF(kappa_fi,dt,kappa_z,kappa_v)
    elif MODE == "RNN":
        filter = 0

    prev_angle = np.inf
    frames_since_detection = 1
    phi, W_sym, W_asym, W_const = RNN_init()

    while True:
        success, frame = cam.read()
        if not success:
            break

        output = frame.copy()

        # Convert to HSV and mask red
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Clean up noise in mask
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        # Isolate red channel for edge detection
        red_only = cv2.bitwise_and(frame, frame, mask=red_mask)
        gray = cv2.cvtColor(red_only, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 2)

        # Detect circles via Hough transform
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT,
            dp=1.5, minDist=30,
            param1=70, param2=22,
            minRadius=5, maxRadius=100
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            circle_history.append(circles[0])
        else:
            circle_history.append([])

        
        # Only draw if detected in at least 3 of the last 5 frames
        recent_circles = [c for c in circle_history if len(c) > 0]
        if len(recent_circles) >= 3:

            if USE_SMOOTHING:
                display_circles = get_smoothed_circle(recent_circles)
            else:
                display_circles = recent_circles[-1]
           
            for c in display_circles: #TODO this loop does not really make sense to have multiple circles
                center = (c[0], c[1])
                radius = c[2]
                cv2.circle(output, center, radius, (0, 255, 0), 2)   # outline
                cv2.circle(output, center, 2, (0, 0, 255), 3)        # center dot

                if MODE == "KF":
                    angle = filter.run_CircKF(prev_angle=prev_angle,frames_since_detection=frames_since_detection, c=c)
                elif MODE == "RNN":
                    angle = run_RNN(phi, W_sym,W_asym,W_const,prev_angle=prev_angle,frames_since_detection=frames_since_detection, c=c)

                frames_since_detection = 1
                prev_angle = angle

        else:
            if MODE == "KF":
                filter.run_CircKF()
            elif MODE == "RNN":
                run_RNN(phi, W_sym,W_asym,W_const)

            frames_since_detection += 1

        # HD indicator — top right corner
        if len(mean_) > 1:
            output = draw_hd_indicator(output, mean_[-1], kappa_[-1])
     ################ Robot looking for red ball #############
            # if mean_[-1]<-0.05:
            #     robot_toy._set_motors(-1600,1600)
            #     time.sleep(0.04)
            #     robot_toy.stop()
            # elif mean_[-1]>0.05:
            #     robot_toy._set_motors(1600, -1600)
            #     time.sleep(0.04)
            #     robot_toy.stop()
        ###########################################################

        # Small red mask overlay in top-left corner
        mask_resized = cv2.resize(red_mask, (320, 50))
        mask_colored = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
        output[0:50, 0:320] = mask_colored

        # Encode as MJPEG and yield
        ret, buffer = cv2.imencode('.jpg', output)
        frame_bytes = buffer.tobytes()
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )

def run_RNN(phi, W_sym, W_asym, W_const, prev_angle=None, frames_since_detection=1, c=None):
    if c is not None:
        angle = calc_angle(c[0])

        if prev_angle != np.inf:
            dy = (((prev_angle - angle) + np.pi) % (2 * np.pi) - np.pi) / frames_since_detection # Wrapped ngular displacement in last frame
            mean, kappa = RNN_step(phi, dt,W_sym, W_asym, W_const, dy=dy, z=angle, k_z=kappa_z)

        else:
            mean, kappa = RNN_step(phi, dt,W_sym, W_asym, W_const)
    else:

        mean, kappa = RNN_step(phi, dt,W_sym, W_asym, W_const)
        angle = None

    mean_.append(mean)
    kappa_.append(kappa)
    return angle





def draw_hd_indicator(frame, mean, kappa, size=80):
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

    angles = np.array([calc_angle(c[0]) for c in all_circles])
    avg_cos = np.mean(np.cos(angles))
    avg_sin = np.mean(np.sin(angles))
    avg_angle = np.arctan2(avg_sin, avg_cos)  # back to polar

    # Convert avg_angle back to a pixel x position
    avg_x_circ = int(avg_angle * 65 / 18 * (180 / np.pi) + 650)

    avg_y      = int(np.mean(all_circles[:, 1]))
    avg_radius = int(np.mean(all_circles[:, 2]))

    return np.array([[avg_x_circ, avg_y, avg_radius]])

#TODO check if pixels start at 0 or 1
def calc_angle(x):
    '''
    calibrated to a 1300 pixel wide image, returns angle in radians (-pi,pi]
    '''
    y = (float(x)-650)*18/65
    return np.radians(y)
    
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

