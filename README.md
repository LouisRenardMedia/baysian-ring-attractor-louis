# Sound Bayesian Ring attractor 

Robot heading and homing experiments using a Bayesian ring attractor. The project combines an omnidirectional camera, visual odometry, IMU readings, microphone-array direction-of-arrival estimates, and optional motor control on a Raspberry Pi robot.

## What This Project Does

The main realtime pipeline is `SoundBayesianRing.py`. It:

- reads unwrapped panoramic camera frames;
- reads IMU angular velocity and compass heading from a BNO055;
- estimates visual rotation with `VisualCompass`;
- estimates sound direction with a ReSpeaker microphone array;
- fuses those signals in `Baysian_Ring_Attractor.BayesianRingAttractor`;
- streams an MJPEG debug view with the current heading estimate;
- optionally sends the current heading estimate to the robot motor controller.



## Hardware Assumptions

Most realtime scripts are intended to run on the Raspberry Pi attached to the robot. They assume:

- Raspberry Pi with GPIO access;
- PCA9685 PWM motor driver;
- motor driver pins configured in `robot_toy.py`;
- BNO055 IMU over I2C;
- camera available through OpenCV, usually `/dev/video0`;
- ReSpeaker microphone array visible to `sounddevice`;
- Python packages for Raspberry Pi hardware, including `RPi.GPIO`, `Adafruit_PCA9685`, `board`, `busio`, and `adafruit_bno055`.


## Important Files

- `SoundBayesianRing.py` - main realtime sound/IMU/visual ring-attractor pipeline and Flask video stream.
- `Baysian_Ring_Attractor.py` - Bayesian ring attractor model used for sensor fusion.
- `Circular_Kalman_Filter.py` - circular Kalman filter implementation.
- `IMUReader.py` - threaded BNO055 reader.
- `visual_compass.py` - rotational odometry.
- `ring_attractor_audio_array_DOA.py` - ReSpeaker microphone-array DOA estimation.
- `robot_toy.py` - motor primitives, manual keyboard control, experiment path, and go-home worker thread.
- `Recorder.py` - threaded synchronized camera frame recorder.
- `Camera/start_cam.py` - camera wrapper that crops and unwraps frames.
- `Camera/unwarp_cfg.json` - current omnidirectional camera crop/unwarp configuration.
- `Simulation/Simulation.ipynb` - Simulation of the Bayesian ring attractor and the circular kalman filter.

## Running The Main Pipeline

From the project root:

```bash
python SoundBayesianRing.py
```

Then open:

```text
http://<pi-ip>:5000/
```

The stream overlays the ring-attractor heading and confidence. Logs are written to `rnn_estimates.csv`.

## Main Runtime Flags

The main feature switches are near the top of `SoundBayesianRing.py`:

```python
MODE = "RNN"                 # "RNN" or "CKF"
ROBOT_TURN = False           # random turning experiment
REALTIMESYNC = True          # use Recorder for synchronized frames
ROBOT_CONTROL = False        # old/manual robot_toy keyboard control thread
ROBOT_GO_HOME_CONTROL = False # autonomous go_home worker thread
ROBOT_PATH = False           # run predefined experimentPath()
```

Use `MODE = "RNN"` to use the bayesian ring attractor or `MODE = "CKF"` to use the circlar kalman filter. Use `ROBOT_GO_HOME_CONTROL = True` when you want `SoundBayesianRing.py` to send `filter.mu[-1]` to the motor-control worker in `robot_toy.py`. That worker waits for an input, runs `go_home(mu)`, and then waits for the next input.

## Motor Control

`robot_toy.py` defines the low-level motor commands:

- `w` - forward
- `s` - backward
- `a` - arc left
- `d` - arc right
- `q` - spin left
- `e` - spin right
- `x` - stop

Manual control:

```bash
python robot_toy.py
```

Autonomous go-home control is started from `SoundBayesianRing.py` through:

```python
start_go_home_control()
submit_go_home_target(mu)
stop_go_home_control()
```

The worker queue keeps only the most recent target, so stale motor commands do not build up if the perception loop runs faster than the robot action.

## Offline Simulation

Tune the bayesian ring attractor parameters individually or run parameter sweep in the jupyter notebook: `Simulation/Simulation.ipynb`

## Using Bayesian ring attractor

The `BayesianRingAttractor` object can be initialized with either one `k_v` value representing angular velocity certainty, or an array of values which indicates that there are several sources of odometry to be fused.

The `step()` function will run one step of the model given an angular velocity and an external input. Either of these can simply be omitted and the model run as expected when not receiving an input. A new external input certainty `k_z` can be passed which will update the certainty of the external input.

The `update_weights()` function can be called to dynamically update the weight of each odometry sensor updating `k_v` values accordingly. 