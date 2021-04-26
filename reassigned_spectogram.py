import numpy as np
import math
from math import exp,pi
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import spectrogram, chirp
from scipy.signal.windows import hann
from obspy import Stream, Trace, UTCDateTime



#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------
def sine_sum_signal(time, fmin, n, dt):
    """
    Continuous source-signal
    """
    n = n+1
    sig = np.sin(2*pi*fmin*(time ))
    for i in range(2,n):
        sig += np.sin(2*pi*fmin*(i*0.8)*(time))

    # sig = sig + (np.random.rand(time.size) * 2 - 1)*1 #*np.exp(-0.05*time) + (np.random.rand(time.size) * 2 - 1)*1e-2

    tr = Trace()
    tr.data = sig
    tr.stats.delta = dt
    tr.stats.starttime = 0
    tr.normalize()

    return tr


#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------
def continuous_signal(time, low, high, dt):
    """
    Continuous source-signal
    """

    sig=(np.random.rand(time.size) * 2 - 1)

    tr = Trace()
    tr.data = sig
    tr.stats.delta = dt
    tr.stats.starttime = 0
    tr.filter('bandpass', freqmin=low, freqmax=high,
              corners=4)

    tr.normalize()

    return tr


#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------
def chirp_signal(time, fmin, fmax, dt):
	"""
	Chirp signal
	"""


	w1 = chirp(time, f0=fmax, f1=fmin, t1=time[-1], method='quadratic',
	          vertex_zero=False)

	f_w1 = fmin - (fmin - fmax) * (time[-1] - time)**2 / time[-1]**2

	fmin = fmin
	fmax = fmax + 5
	w2 = chirp(time, f0=fmax, f1=fmin, t1=time[-1], method='quadratic',
	          vertex_zero=False)

	f_w2 = fmin - (fmin - fmax) * (time[-1] - time)**2 / time[-1]**2

	sig = w1 + w2[::-1]
	tr = Trace()
	tr.data = sig
	tr.stats.delta = dt
	tr.stats.starttime = 0
	tr.normalize()

	return tr


#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------

def get_stft(signal_, w_, fft_len_, overlap_):
    '''
    signal is a trace
    w_: window length
    fft_len: length of stft
    overlap_: window overlap
    '''
    if fft_len_/w_<4:
        fft_len_ = w_*4
        print('change fft length')
    dt = signal_.stats.delta
    fs = 1/dt
    window_s = int(fs*w_)
    window_s2 =window_s * 2
    n = np.arange(window_s)
    t_n = np.arange(-int(window_s/2), int(window_s/2))
    ovlp = int(overlap_*window_s)
    dn = window_s - ovlp
    h = hann(window_s)

    fft_npts_ = np.int(fft_len_/dt)

    signal = signal_.data

    t_l = len(signal)
    time = np.arange(0,t_l*dt, dt)
    pos = np.arange(0, t_l-window_s, dn)
    time_new = time[pos]
    dt_new = time_new[1]
    pos_l = len(pos)


    freq = np.fft.fftfreq(fft_npts_, dt)
    S = np.zeros([fft_npts_, pos_l], dtype = np.complex64)
    S_del = np.zeros([fft_npts_, pos_l], dtype = np.complex64)

    norm_h = np.linalg.norm(h, ord=2)

    for i,j in enumerate(pos):

        S[window_s2+n,i] = signal[j+n] * h / norm_h
        S_del[window_s2+n,i] = signal[j+n-1] * h / norm_h

    F_S = np.fft.fft(S, axis = 0)
    F_S_del = np.fft.fft(S_del, axis = 0)
    F_S_f_del = np.roll(F_S, 1, axis=0)

    return F_S, F_S_del, F_S_f_del, freq, time_new




#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------


def nelson_reassigned(signal_, w_, fft_len_, overlap_, f_min, f_max, lim_amp):


    F_S, F_S_del, F_S_f_del, freq, time_new = get_stft(signal_, w_, fft_len_, overlap_)
    fs = signal_.stats.sampling_rate
    dt_new = time_new[1]
    l_f = freq.size
    l_t = time_new.size
    Z_reassigned = np.zeros([l_f, l_t], dtype=np.complex64)


    CIF_1 = np.zeros([l_f, l_t], dtype=np.float32)
    LGD_1 = np.zeros([l_f, l_t], dtype=np.float32)

    min_freq_pos = (np.abs(freq - f_min)).argmin()
    max_freq_pos = (np.abs(freq - f_max)).argmin()

    lim = np.mean(np.abs(F_S))*lim_amp

    for i in range(0,l_t-2):
        for j in range(min_freq_pos, max_freq_pos):

            if np.abs(F_S[j,i])>=lim :

                ARG = np.angle(F_S[j,i]) - np.angle(F_S_del[j,i])
                CIF_1[j,i] = (fs/(2*pi)) * np.mod(ARG, 2*pi)

                ARG = np.angle(F_S_f_del[j,i]) - np.angle(F_S[j,i])
                LGD_1[j,i] = (-l_t/(2*pi*fs)) * np.mod(ARG, 2*pi)
                t_new = time_new[i] + LGD_1[j,i]

                pos_t = int(np.ceil(t_new/dt_new)) #(np.abs(time_new - t_new)).argmin()
                pos_f = (np.abs(freq - CIF_1[j,i])).argmin()
                Z_reassigned[pos_f, pos_t] += np.abs(F_S[j,i] )


    return F_S, Z_reassigned, CIF_1, LGD_1, freq, time_new
