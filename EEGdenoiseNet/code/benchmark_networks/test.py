"""
每个指标单独验证，使用确定性合成信号，预期值可手算。

信号 shape: (n_ch, T)
"""
import numpy as np


def compute_metrics(X_in, X_out, X_gt):
    T = min(X_in.shape[1], X_out.shape[1], X_gt.shape[1])
    X_in, X_out, X_gt = X_in[:, :T], X_out[:, :T], X_gt[:, :T]
    X_artifact = X_in - X_gt
    residual   = X_out - X_gt

    def rms_per_ch(X):
        return np.sqrt(np.mean(X**2, axis=1))

    def rms_ratio_per_ch(num, den):
        r = rms_per_ch(num) / (rms_per_ch(den) + 1e-12)
        return 20 * np.log10(np.mean(r))

    snr_b = rms_ratio_per_ch(X_gt, X_artifact)
    snr_a = rms_ratio_per_ch(X_gt, residual)
    rmse  = np.mean(np.sqrt(np.mean(residual**2, axis=1)))
    sf    = rms_ratio_per_ch(X_in, X_out)
    rnr   = rms_ratio_per_ch(residual, X_artifact)
    le    = rms_ratio_per_ch(residual, X_gt)
    return snr_b, snr_a, rmse, sf, rnr, le


def check(name, got, expected, tol=1e-6):
    ok = abs(got - expected) < tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got={got:.6f}, expected={expected:.6f}")
    return ok


# ── SNR_before ────────────────────────────────────────────────────────────
def test_snr_before():
    """
    ch0: X_gt=[1,1], noise=[0.1,0.1] -> RMS_gt=1,   RMS_noise=0.1, ratio=10
    ch1: X_gt=[2,2], noise=[0.2,0.2] -> RMS_gt=2,   RMS_noise=0.2, ratio=10
    mean(ratio) = 10
    SNR_before = 20*log10(10) = 20 dB
    """
    print("\n[Test] SNR_before")
    X_gt  = np.array([[1.0, 1.0],
                       [2.0, 2.0]])
    noise = np.array([[0.1, 0.1],
                       [0.2, 0.2]])
    X_in  = X_gt + noise
    X_out = X_gt.copy()

    snr_b, *_ = compute_metrics(X_in, X_out, X_gt)
    return check("SNR_before", snr_b, 20.0, tol=0.01)


# ── SNR_after ─────────────────────────────────────────────────────────────
def test_snr_after():
    """
    ch0: X_gt=[1,1], residual=[0.1,0.1] -> RMS_gt=1,   RMS_res=0.1, ratio=10
    ch1: X_gt=[2,2], residual=[0.2,0.2] -> RMS_gt=2,   RMS_res=0.2, ratio=10
    mean(ratio) = 10
    SNR_after = 20*log10(10) = 20 dB
    """
    print("\n[Test] SNR_after")
    X_gt  = np.array([[1.0, 1.0],
                       [2.0, 2.0]])
    X_in  = np.zeros((2, 2))
    X_out = np.array([[1.1, 1.1],
                       [2.2, 2.2]])

    _, snr_a, *_ = compute_metrics(X_in, X_out, X_gt)
    return check("SNR_after", snr_a, 20.0, tol=0.01)


# ── RMSE ──────────────────────────────────────────────────────────────────
def test_rmse():
    """
    ch0: residual=[0.5,0.5] -> RMSE_ch0 = 0.5
    ch1: residual=[0.5,0.5] -> RMSE_ch1 = 0.5
    mean(0.5, 0.5) = 0.5
    """
    print("\n[Test] RMSE")
    X_gt  = np.array([[1.0, 1.0],
                       [2.0, 2.0]])
    X_in  = np.zeros((2, 2))
    X_out = np.array([[1.5, 1.5],
                       [2.5, 2.5]])

    _, _, rmse, *_ = compute_metrics(X_in, X_out, X_gt)
    return check("RMSE", rmse, 0.5, tol=1e-10)


# ── SF ────────────────────────────────────────────────────────────────────
def test_sf():
    """
    ch0: X_in=[2,2], X_out=[1,1] -> RMS_in=2, RMS_out=1, ratio=2
    ch1: X_in=[4,4], X_out=[2,2] -> RMS_in=4, RMS_out=2, ratio=2
    mean(ratio) = 2
    SF = 20*log10(2) = 6.0206 dB
    """
    print("\n[Test] SF")
    X_gt  = np.zeros((2, 2))
    X_in  = np.array([[2.0, 2.0],
                       [4.0, 4.0]])
    X_out = np.array([[1.0, 1.0],
                       [2.0, 2.0]])

    *_, sf, _, _ = compute_metrics(X_in, X_out, X_gt)
    expected = 20 * np.log10(2.0)
    return check("SF", sf, expected, tol=0.01)


# ── RNR ───────────────────────────────────────────────────────────────────
def test_rnr():
    """
    ch0: artifact=[1,1], residual=[0.1,0.1] -> RMS_res=0.1, RMS_art=1.0, ratio=0.1
    ch1: artifact=[2,2], residual=[0.2,0.2] -> RMS_res=0.2, RMS_art=2.0, ratio=0.1
    mean(ratio) = 0.1
    RNR = 20*log10(0.1) = -20 dB
    """
    print("\n[Test] RNR")
    X_gt  = np.zeros((2, 2))
    X_in  = np.array([[1.0, 1.0],
                       [2.0, 2.0]])
    X_out = np.array([[0.1, 0.1],
                       [0.2, 0.2]])

    *_, rnr, _ = compute_metrics(X_in, X_out, X_gt)
    return check("RNR", rnr, -20.0, tol=0.01)


# ── LE ────────────────────────────────────────────────────────────────────
def test_le():
    """
    ch0: X_gt=[1,1], residual=[0.1,0.1] -> RMS_res=0.1, RMS_gt=1.0, ratio=0.1
    ch1: X_gt=[2,2], residual=[0.2,0.2] -> RMS_res=0.2, RMS_gt=2.0, ratio=0.1
    mean(ratio) = 0.1
    LE = 20*log10(0.1) = -20 dB
    """
    print("\n[Test] LE")
    X_gt  = np.array([[1.0, 1.0],
                       [2.0, 2.0]])
    X_in  = np.zeros((2, 2))
    X_out = np.array([[1.1, 1.1],
                       [2.2, 2.2]])

    *_, le = compute_metrics(X_in, X_out, X_gt)
    return check("LE", le, -20.0, tol=0.01)


if __name__ == '__main__':
    tests = [
        test_snr_before,
        test_snr_after,
        test_rmse,
        test_sf,
        test_rnr,
        test_le,
    ]
    passed = sum(t() for t in tests)
    print(f"\n{'='*40}")
    print(f"Results: {passed}/{len(tests)} tests passed")
