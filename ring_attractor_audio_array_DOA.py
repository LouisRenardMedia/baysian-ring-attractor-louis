#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created: 2026-1-13
Author: Alberto Doimo
email: alberto.doimo@uni-konstanz.de

Description:

DOA estimation using Delay And Sum (DAS) method
with a Uniform Circular Array (UCA) of microphones.
"""

import numpy as np
import sounddevice as sd
import scipy.signal as signal
import queue


from utilities import *

# Define DAS filter function
# Constants
global fs, channels, radius, sos

c = 343.0  # speed of sound
fs = 16000  # sampling frequency
print(f"Sampling Frequency: {fs} Hz")
channels = 5
# radius = 0.0323  # radius of the microphone array in meters
radius = 0.04625  # radius of the microphone array in meters

cutoff = 100  # high-pass filter cutoff frequency in Hz
sos = signal.butter(1, cutoff, "hp", fs=fs, output="sos")


def get_card(device_list):
    for i, each in enumerate(device_list):
        if "reSpeaker" in each["name"]:
            return i


global q
q = queue.Queue()

usb_card_index = get_card(sd.query_devices())
print(sd.query_devices())
print(f"Using device index: {usb_card_index}")


def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""

    # Fancy indexing with mapping creates a (necessary!) copy:
    q.put(indata[::1, :])


def update_das():
    """Calculates DOA using Delay And Sum (DAS) method.

    Returns
    -------
        spatial_resp:
            Spatial response of the DAS beamformer.
    """

    try:
        in_sig = q.get_nowait()
    except queue.Empty:
        return None

    # in_sig = signal.sosfiltfilt(sos, in_sig, axis=0)
    in_sig = in_sig[
        :, 1:5
    ]  # ReSpeaker 4-mic array: use channels 1 to 4 for the processing; ch0=summed signal from all mics, ch5 = speaker out signal

    theta, spatial_resp, f_spec_axis, spectrum, bands = higher_order_dmas_UCA(
        in_sig,
        fs,
        4,
        radius,
        45,
        [400, 4200],
        np.linspace(0, 360, 361),
        show=True,
    )

    return spatial_resp


def main():

    # Start the audio stream
    stream = sd.InputStream(
        device=usb_card_index,
        samplerate=fs,
        channels=channels,
        blocksize=128,
        callback=audio_callback,
        latency="low",
    )

    stream.start()
    try:
        while True:
            spatial_resp = update_das()
            print(f"Spatial Response: {spatial_resp}")
            if spatial_resp is not None:
                # Process the spatial response as needed
                pass
    except KeyboardInterrupt:
        print("Stopping the audio stream.")
    finally:
        stream.stop()
        stream.close()


if __name__ == "__main__":
    main()
