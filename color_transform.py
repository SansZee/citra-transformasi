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


def invert_grayscale(gray):
    return 1.0 - gray


def logarithmic_transform(gray):
    c = 1.0 / np.log(2.0)
    return np.clip(c * np.log1p(gray), 0.0, 1.0)


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


def _sample_inverse_affine(image, matrix, fill_value=1.0):
    height, width = image.shape[:2]
    grid_y, grid_x = np.indices((height, width))
    source_x = matrix[0, 0] * grid_x + matrix[0, 1] * grid_y + matrix[0, 2]
    source_y = matrix[1, 0] * grid_x + matrix[1, 1] * grid_y + matrix[1, 2]

    valid = (
        (source_x >= 0)
        & (source_x <= width - 1)
        & (source_y >= 0)
        & (source_y <= height - 1)
    )
    x0 = np.floor(source_x).astype(np.intp)
    y0 = np.floor(source_y).astype(np.intp)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    x0_valid = np.clip(x0[valid], 0, width - 1)
    x1_valid = np.clip(x1[valid], 0, width - 1)
    y0_valid = np.clip(y0[valid], 0, height - 1)
    y1_valid = np.clip(y1[valid], 0, height - 1)
    x_weight = (source_x - x0)[valid]
    y_weight = (source_y - y0)[valid]
    channel_shape = (1,) * (image.ndim - 2)
    x_weight = x_weight.reshape(x_weight.shape + channel_shape)
    y_weight = y_weight.reshape(y_weight.shape + channel_shape)

    top_left = image[y0_valid, x0_valid]
    top_right = image[y0_valid, x1_valid]
    bottom_left = image[y1_valid, x0_valid]
    bottom_right = image[y1_valid, x1_valid]
    top = top_left * (1.0 - x_weight) + top_right * x_weight
    bottom = bottom_left * (1.0 - x_weight) + bottom_right * x_weight

    output = np.full(
        image.shape,
        fill_value,
        dtype=np.result_type(image.dtype, np.float64),
    )
    output[valid] = top * (1.0 - y_weight) + bottom * y_weight
    return output


def scale_image(image, scale_x=0.5, scale_y=0.5, fill_value=1.0):
    if scale_x <= 0 or scale_y <= 0:
        raise ValueError("Faktor skala harus lebih besar dari nol.")

    height, width = image.shape[:2]
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    matrix = np.array(
        [
            [1.0 / scale_x, 0.0, center_x * (1.0 - 1.0 / scale_x)],
            [0.0, 1.0 / scale_y, center_y * (1.0 - 1.0 / scale_y)],
        ]
    )
    return _sample_inverse_affine(image, matrix, fill_value)


def translate_image(image, dx=50, dy=30, fill_value=1.0):
    matrix = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy]])
    return _sample_inverse_affine(image, matrix, fill_value)


def rotate_image(image, angle=30.0, center=None, fill_value=1.0):
    height, width = image.shape[:2]
    if center is None:
        center = ((width - 1) / 2.0, (height - 1) / 2.0)

    center_x, center_y = center
    theta = np.deg2rad(angle)
    cosine = np.cos(theta)
    sine = np.sin(theta)
    matrix = np.array(
        [
            [
                cosine,
                -sine,
                center_x - cosine * center_x + sine * center_y,
            ],
            [
                sine,
                cosine,
                center_y - sine * center_x - cosine * center_y,
            ],
        ]
    )
    return _sample_inverse_affine(image, matrix, fill_value)


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

inverse_gray = invert_grayscale(gray)
log_gray = logarithmic_transform(gray)
plot_channels(
    "Transformasi Citra Tingkat Keabuan",
    [gray, inverse_gray, log_gray],
    ["Grayscale", "Invers (1-r)", "Log (c=1/ln(2))"],
    ["gray", "gray", "gray"],
    "grayscale_transformations.png",
)
plot_channels(
    "2a. Invers Citra (Negative)",
    [gray, inverse_gray],
    ["Grayscale", "Hasil Invers (1-r)"],
    ["gray", "gray"],
    "2a_invers_citra.png",
)
plot_channels(
    "2b. Log Transform (s = c log(1+r))",
    [gray, log_gray],
    ["Grayscale", "Hasil Log"],
    ["gray", "gray"],
    "2b_log_transform.png",
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

scaled = scale_image(rgb_src, 0.5, 0.5)
translated = translate_image(rgb_src, 50, 30)
rotated = rotate_image(rgb_src, 30.0)
plot_channels(
    "Transformasi Geometri",
    [scaled, translated, rotated],
    ["Skala 0,5x", "Translasi (50, 30) px", "Rotasi 30° CCW"],
    [None, None, None],
    "geometric_transformations.png",
)
plot_channels(
    "3a. Skala",
    [scaled],
    ["Skala 0,5x"],
    [None],
    "3a_skala.png",
)
plot_channels(
    "3b. Translasi",
    [translated],
    ["Translasi (50, 30) px"],
    [None],
    "3b_translasi.png",
)
plot_channels(
    "3c. Rotasi",
    [rotated],
    ["Rotasi 30° CCW"],
    [None],
    "3c_rotasi.png",
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