"""Transformasi ruang warna RGB -> Grayscale, HS, YCbCr, dan CMY.

Semua transformasi dihitung manual dengan NumPy (tanpa OpenCV untuk
mengubah citra). OpenCV / matplotlib hanya dipakai untuk membaca citra
dan visualisasi.
"""

import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(__file__)
IMG_PATH = os.path.join(BASE, "data", "images.jpg")
OUT_DIR = os.path.join(BASE, "output")
os.makedirs(OUT_DIR, exist_ok=True)

DPI = 110


# ---------------------------------------------------------------------------
# 1. Konversi manual (input RGB float [0..1], shape (H, W, 3))
# ---------------------------------------------------------------------------
def rgb_to_grayscale(rgb):
    """Lightness perceptif, standar ITU-R BT.601."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def rgb_to_hs(rgb):
    """Hue [0..360) derajat dan Saturation [0..1] (dari HSV)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.max(rgb, axis=-1)
    mn = np.min(rgb, axis=-1)
    delta = mx - mn

    h = np.zeros_like(mx)
    m = delta > 0
    num = np.zeros_like(delta)
    np.divide(g - b, delta, out=num, where=m)
    h = np.where(m & (mx == r), 60 * (num % 6), 0.0)
    np.divide(b - r, delta, out=num, where=m)
    h = np.where(m & (mx == g), 60 * (num + 2), h)
    np.divide(r - g, delta, out=num, where=m)
    h = np.where(m & (mx == b), 60 * (num + 4), h)
    h = (h + 360) % 360

    s = np.zeros_like(mx)
    np.divide(delta, mx, out=s, where=mx > 0)
    return h, s


def rgb_to_ycbcr(rgb):
    """Standar ITU-R BT.601, skala 8-bit (0..255)."""
    r, g, b = rgb[..., 0] * 255, rgb[..., 1] * 255, rgb[..., 2] * 255
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return y, cb, cr


def rgb_to_cmy(rgb):
    """Subtractive CMY = pelengkap (complement) RGB dalam [0..1]."""
    return 1.0 - rgb


# ---------------------------------------------------------------------------
# 2. Membaca citra
# ---------------------------------------------------------------------------
rgb_src = mpimg.imread(IMG_PATH)
if rgb_src.max() > 1.0:
    rgb_src = rgb_src / 255.0

print(f"Citra sumber : {IMG_PATH}")
print(f"Dimensi      : {rgb_src.shape[1]} x {rgb_src.shape[0]} px")
print(f"Dependency   : numpy {np.__version__}, matplotlib {plt.matplotlib.__version__}")
print("Semua konversi ditulis manual (tanpa cv2.cvtColor).", end="\n\n")


# ---------------------------------------------------------------------------
# 3. Visualisasi umum: gambar pratinjau + per-kanal, lalu simpan PNG
# ---------------------------------------------------------------------------
def plot_channels(fig_title, images, labels, cmaps, save_name):
    n = len(images)
    fig, axes = plt.subplots(1, n + 1, figsize=(3.2 * (n + 1), 3.4))
    fig.suptitle(fig_title, fontsize=13, fontweight="bold")

    axes[0].imshow(rgb_src)
    axes[0].set_title("RGB (asli)")
    axes[0].axis("off")

    for ax, img, label, cmap in zip(axes[1:], images, labels, cmaps):
        ax.imshow(img, cmap=cmap)
        ax.set_title(label)
        ax.axis("off")

    fig.tight_layout()
    fp = os.path.join(OUT_DIR, save_name)
    fig.savefig(fp, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {save_name}")


# --- a. RGB -> Grayscale ----------------------------------------------------
gray = rgb_to_grayscale(rgb_src)
plot_channels(
    "RGB -> Grayscale (0.299R + 0.587G + 0.114B)",
    [gray],
    ["Grayscale"],
    ["gray"],
    "grayscale.png",
)

# --- b. RGB -> HS -----------------------------------------------------------
h, s = rgb_to_hs(rgb_src)
plot_channels(
    "RGB -> HS (Hue & Saturation)",
    [h / 360.0, s],
    ["Hue (H) 0-360°", "Saturation (S) 0-1"],
    ["hsv", "gray"],
    "hs_channels.png",
)

# --- c. RGB -> YCbCr --------------------------------------------------------
y, cb, cr = rgb_to_ycbcr(rgb_src)
plot_channels(
    "RGB -> YCbCr (ITU-R BT.601, 8-bit)",
    [y / 255.0, cb / 255.0, cr / 255.0],
    ["Y (Luma)", "Cb (Chroma Blue)", "Cr (Chroma Red)"],
    ["gray", "gray", "gray"],
    "ycbcr_channels.png",
)

# --- d. RGB -> CMY ----------------------------------------------------------
cmy = rgb_to_cmy(rgb_src)
plot_channels(
    "RGB -> CMY (Subtractive)",
    [cmy[..., 0], cmy[..., 1], cmy[..., 2]],
    ["Cyan (C = 1-R)", "Magenta (M = 1-G)", "Yellow (Y = 1-B)"],
    ["Blues", "Purples", "YlOrBr"],
    "cmy_channels.png",
)

# --- e. Montase gabungan ----------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle("Ringkasan Transformasi Ruang Warna", fontsize=14, fontweight="bold")

grid = [
    (rgb_src, "RGB (asli)", None),
    (gray, "Grayscale", "gray"),
    (np.dstack([h / 360.0, s, np.ones_like(s)]), "HS (H & S)", "hsv"),
    (np.dstack([y, cb, cr]) / 255.0, "YCbCr (Y, Cb, Cr)", None),
    (cmy, "CMY (C, M, Y)", None),
]
for ax, (img, title, cmap) in zip(axes.ravel(), grid):
    ax.imshow(img, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")

axes[1, 2].axis("off")

fig.tight_layout(rect=(0, 0, 1, 0.95))
montage = os.path.join(OUT_DIR, "ringkasan_transformasi.png")
fig.savefig(montage, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("[saved] ringkasan_transformasi.png")

print("Selesai. Semua hasil tersimpan di folder output/.")