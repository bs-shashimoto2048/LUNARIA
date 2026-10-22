"""WAV 検証 (Phase 0 / FR-009)。歌詞本文は扱わない。

使い方: python -I validate_wav.py <wav> <expected_seconds> [tolerance_seconds]
標準出力に JSON を出力し、全条件を満たせば終了コード 0。
"""
import json
import sys

SILENCE_RMS = 1e-3


def validate(path, expected_sec, tol=1.5):
    # numpy / soundfile は検証時にだけ import する（import だけで commit が約 780MB 増えるため、
    # 診断ランナー等が phase0_run を import しても軽量なままにする）
    import numpy as np
    import soundfile as sf

    info = sf.info(path)
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    frames = data.shape[0]
    finite = bool(np.isfinite(data).all())
    clean = np.nan_to_num(data)
    peak = float(np.abs(clean).max()) if frames else 0.0
    rms = float(np.sqrt(np.mean(clean**2))) if frames else 0.0
    dur = frames / sr if sr else 0.0
    checks = {
        "readable": True,
        "sample_rate_48k": sr == 48000,
        "stereo": data.shape[1] == 2,
        "frames_nonzero": frames > 0,
        "duration_in_tolerance": abs(dur - expected_sec) <= tol,
        "peak_nonzero": peak > 0.0,
        "rms_above_silence": rms > SILENCE_RMS,
        "no_nan_inf": finite,
    }
    return {
        "file": path,
        "format": info.format,
        "subtype": info.subtype,
        "sample_rate": sr,
        "channels": int(data.shape[1]),
        "frames": int(frames),
        "duration_sec": round(dur, 3),
        "expected_sec": expected_sec,
        "peak": round(peak, 5),
        "rms": round(rms, 5),
        "checks": checks,
        "ok": all(checks.values()),
    }


if __name__ == "__main__":
    tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
    try:
        r = validate(sys.argv[1], float(sys.argv[2]), tol)
    except Exception as e:  # 読み込み不能も失敗として扱う
        r = {"file": sys.argv[1], "ok": False, "error": type(e).__name__}
    print(json.dumps(r, ensure_ascii=False, indent=2))
    sys.exit(0 if r.get("ok") else 1)
