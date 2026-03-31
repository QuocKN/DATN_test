import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

def get_spectrogram(
    x, 
    Fs, 
    fc,
    n_fft=2048, 
    hop_length=None,
    window='hann',
    normalize=True
):
    """
    x: IQ signal (complex)
    Fs: sample rate
    fc: center frequency
    n_fft: FFT size
    hop_length: step (default = 50% overlap)
    """

    if hop_length is None:
        hop_length = n_fft // 2  # 50% overlap

    # ===== Window =====
    if window == 'hann':
        w = np.hanning(n_fft)
    else:
        w = np.ones(n_fft)

    # ===== STFT =====
    n_frames = max(0, (len(x) - n_fft) // hop_length)
    if n_frames == 0:
        raise ValueError("Signal is too short for selected n_fft/hop_length.")

    # Pre-allocate to avoid list growth and duplicated memory copies.
    spec = np.empty((n_frames, n_fft), dtype=np.float32)
    for frame_idx in range(n_frames):
        i = frame_idx * hop_length
        segment = x[i:i+n_fft] * w

        spectrum = np.fft.fft(segment)
        spectrum = np.fft.fftshift(spectrum)

        power = np.abs(spectrum)**2
        spec[frame_idx] = power.astype(np.float32)

    # ===== Log scale (dB) =====
    spec = 10 * np.log10(spec + 1e-12)

    # ===== Normalize =====
    # if normalize:
    #     spec = (spec - np.mean(spec)) / (np.std(spec) + 1e-12)
        # --- Sửa đoạn Normalize trong hàm get_spectrogram ---
    if normalize:
        # Thay vì dùng std, hãy đưa về thang [0, 1] dựa trên min/max 
        # hoặc cắt ngưỡng để loại bỏ nhiễu nền
        spec = spec - np.max(spec) # Đưa đỉnh cao nhất về 0dB
        spec = np.clip(spec, -60, 0) # Chỉ lấy khoảng 60dB từ đỉnh trở xuống

    # ===== Trục =====
    time_axis = np.arange(spec.shape[0]) * hop_length / Fs * 1000  # ms
    freq_axis = np.linspace(fc - Fs/2, fc + Fs/2, n_fft)

    return spec, time_axis, freq_axis

def plot_spectrogram(spec, time_axis, freq_axis, save_path=None):
    plt.figure(figsize=(8, 5))

    plt.imshow(
        spec.T,
        aspect='auto',
        origin='lower',
        extent=[time_axis[0], time_axis[-1], freq_axis[0]/1e9, freq_axis[-1]/1e9]
    )

    plt.xlabel("Time (ms)")
    plt.ylabel("Frequency (GHz)")
    plt.title("Spectrogram")
    plt.colorbar(label="Normalized Power (dB)")
    plt.tight_layout()

    if save_path is not None:
        output_dir = os.path.dirname(save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        if os.path.exists(save_path):
            os.remove(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved spectrogram image to: {save_path}")

    # plt.show()

if __name__ == "__main__":
    # # đọc file (1 dòng hoặc nhiều dòng) đọc csv
    # raw = np.loadtxt("10000H_20.csv", delimiter=",")

    # # flatten về 1D
    # raw = raw.flatten()

    # # tách I và Q
    # I = raw[0::2]
    # Q = raw[1::2]

    # # tạo IQ complex
    # x = I + 1j * Q

    # # --- Đọc file .mat  ---
    # f = h5py.File("Frysky_1.mat", "r")

    # x_raw = np.array(f['uhd_samps']).squeeze()

    # # convert structured → complex
    # x = x_raw['real'] + 1j * x_raw['imag']

    # --- Đọc file .dat (float32) ---
    file_path = "RF_Processing/WIFI/MP1_FY/MA1_1010_00.dat"
    Fs = 60e6
    duration_ms = 200

    # Read only a short chunk for plotting to keep RAM usage low.
    n_complex_samples = int(Fs * (duration_ms / 1000.0))
    n_float32_values = n_complex_samples * 2  # I/Q interleaved float32
    raw = np.fromfile(file_path, dtype=np.float32, count=n_float32_values)

    if raw.size < 2:
        raise ValueError(f"File {file_path} does not contain enough samples.")

    # Ensure an even number of float32 values (I/Q interleaved)
    if raw.size % 2 != 0:
        raw = raw[:-1]

    x = raw.view(np.complex64)
    print(x.shape)
    print(np.iscomplexobj(x))  # phải True

    spec, t, f = get_spectrogram(
        x,           # IQ signal
        Fs=Fs,
        fc=2.4375e9,
        n_fft=2048
    )
    print("OK")
    # --- Trong phần main ---
    # spec, t, f = get_spectrogram(
    #     x, 
    #     Fs=100e6,
    #     fc=2.44e9,
    #     n_fft=1024,      # Giảm n_fft để tăng độ phân giải thời gian
    #     hop_length=128   # Tăng overlap (n_fft // 2) giúp ảnh "dày" hơn theo trục dọc
    # )

    plot_spectrogram(spec, t, f, save_path="RF_Processing/WIFI/MP1_FY/MA1_1010_00.png")