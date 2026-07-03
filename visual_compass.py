"""
Streaming visual compass (Labrosse-style panoramic rotation estimator).

Operates purely on unwarped 360 panoramas — no calibration / unwarp deps, so it
drops straight into a bigger pipeline that already produces panoramas.

Usage
-----
    from visual_compass import VisualCompass

    vc = VisualCompass(t_match=0.5)

    for panorama in stream:                 # each = unwarped 360 image
        shift = vc.update(panorama)         # yaw (deg) relative to the reference frame
        yaw   = vc.wrapped_heading           # accumulated heading wrapped to [-180, 180)
"""
import cv2
import numpy as np

class VisualCompass:
    def __init__(self, t_match=0.30):
        self.t_match = t_match          # swap keyframe when A_n < t_match (low = poor/ambiguous)
        self.W = None                   # panorama width (columns), inferred on 1st frame
        self.deg_per_col = None
        self.reset()

    # -- lifecycle ---------------------------------------------------------
    def reset(self):
        """Forget all state; the next panorama becomes the reference keyframe."""
        self.I_r = self.I_c = None      # reference and current signals
        self.w_r = 0.0                  # banked (reference) orientation, deg
        self.r_rc = 0.0                 # last rotation reference -> current, deg
        self.frame_idx = 0
        self.last_shift = 0.0

        # last-frame diagnostics (handy for visualization / debugging)
        self.heading = 0.0              # accumulated orientation (w_r + r_rc), can drift
        self.shift = 0.0                # yaw relative to the reference frame (returned)
        self.quality = 0.0              # normalized amplitude A_n (Eqs. 7-8), HIGH = good
        self.shift_cols = 0
        self.ref_id = 0
        self.last_pano = None
        self.ref_pano = None


    # -- signal / matching -------------------------------------------------
    @staticmethod
    def to_signal(panorama):
        """Collapse panorama to a 1D azimuth signal (gray, row-averaged, mean-removed)."""
        if panorama.ndim == 3:
            panorama = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
        sig = panorama.astype(np.float32).mean(axis=0)
        sig -= sig.mean()
        return sig

    def alpha_m(self, sig_ref, sig_cur):
        """Best circular shift + normalized match amplitude (paper Eqs. 7-8),
        computed fast via FFT cross-correlation.
        Returns (shift_deg, A_n, shift_cols). A_n = normalized amplitude, HIGH = good."""
        n = sig_ref.size
        R = np.fft.rfft(sig_ref)
        C = np.fft.rfft(sig_cur)
        corr = np.fft.irfft(R * np.conj(C), n=n)        # circular cross-correlation
        best = int(np.argmax(corr))                     # max corr == min SSD

        # distance curve from the correlation: d^2 = E_ref + E_cur - 2*corr
        E_ref = float(np.dot(sig_ref, sig_ref))
        E_cur = float(np.dot(sig_cur, sig_cur))
        opp = (best + n // 2) % n                        # alpha_m + w/2: stable opposite
        d_best = np.sqrt(max(E_ref + E_cur - 2.0 * corr[best], 0.0))
        d_opp = np.sqrt(max(E_ref + E_cur - 2.0 * corr[opp], 0.0))
        amp = d_opp - d_best                             # Eq. (7)

        corr_self_half = float(np.dot(sig_ref, np.roll(sig_ref, n // 2)))
        self_amp = np.sqrt(max(2.0 * E_ref - 2.0 * corr_self_half, 0.0)) + 1e-9
        A_n = amp / self_amp                             # Eq. (8)

        best_shift = best if best <= n // 2 else best - n
        return best_shift * self.deg_per_col, A_n, best_shift

    # -- main entry point --------------------------------------------------
    def update(self, panorama):
        """Feed one unwarped panorama; returns the yaw (deg) relative to the
        reference frame. The accumulated heading is available as `self.heading`."""
        if self.W is None:                      # infer geometry from the first panorama
            self.W = panorama.shape[1]
            self.deg_per_col = 2*np.pi / self.W

        sig = self.to_signal(panorama)
        prev_pano = self.last_pano              # image behind the old current signal
        self.last_pano = panorama

        if self.I_r is None:                    # first frame: set the reference
            self.I_r = self.I_c = sig
            self.ref_id = self.frame_idx
            self.ref_pano = panorama
            self.heading = 0.0
            self.shift = 0.0
            self.frame_idx += 1
            return self.shift

        I_p = self.I_c                          # previous = old current
        self.I_c = sig                          # new current
        r_rp = self.r_rc                        # rotation ref -> previous

        r_rc, quality, cols = self.alpha_m(self.I_r, self.I_c)
        r_rc = -r_rc                            # world shifts opposite to camera turn

        if quality < self.t_match:              # dip too shallow -> ref and current too different
            self.w_r += r_rp                    # bank accumulated rotation
            r_rc, _, cols = self.alpha_m(I_p, self.I_c)  # re-measure vs fresh keyframe
            r_rc = -r_rc
            self.I_r = I_p                       # previous becomes new reference
            self.ref_id = self.frame_idx - 1
            self.ref_pano = prev_pano

        self.r_rc = r_rc
        self.heading = self.w_r + r_rc          # accumulated orientation (kept, not returned)
        self.shift = r_rc                        # yaw relative to the reference frame
        self.quality = quality                   # normalized amplitude A_n (high = good)
        self.shift_cols = cols
        self.frame_idx += 1
        angular_dis = (((self.shift - self.last_shift) + np.pi) % (2 * np.pi)) - np.pi
        self.last_shift = self.shift
        return angular_dis