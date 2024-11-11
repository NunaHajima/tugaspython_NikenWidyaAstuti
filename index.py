import numpy as np
import cv2
from matplotlib import pyplot as plt

#Menampilkan Gambar
"""
photo = cv2.imread("allofus.jpg")
cv2.imshow('allofus_image', photo)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

#Menyimpan gambar
"""
photo2 = cv2.imread("C:/Users/Niken/.vscode/Niken Widya Astuti_23170028_UTS/allofus.jpg", 0)
cv2.imshow('save_allofus', photo2)
key = cv2.waitKey(0)
if key == ord('s'):
    cv2.imwrite("C:/Users/Niken/.vscode/Niken Widya Astuti_23170028_UTS/allofus_duplikat.jpg", photo2)
    print("Gambar berhasil disimpan sebagai allofus_duplikat.jpg")
cv2.destroyAllWindows()
"""

#Mengubah gambar berwarna ke gambar keabuan dan gambar binary
"""
photo3 = cv2.imread("allofus.jpg")
cv2.imshow("Original", photo3)
key = cv2.waitKey(0)
# Mengonversi gambar ke grayscale
gray = cv2.cvtColor(photo3, cv2.COLOR_BGR2GRAY)
cv2.imshow("Photo_AbuAbu", gray)
key = cv2.waitKey(0)
if key == ord('s'):
    cv2.imwrite("C:/Users/Niken/.vscode/Niken Widya Astuti_23170028_UTS/allofus_gray.jpg", gray)
    print("Gambar grayscale berhasil disimpan sebagai allofus_gray.jpg")
# Mendeteksi tepi pada gambar grayscale
edged = cv2.Canny(gray, 30, 150)
cv2.imshow("Edged", edged)
key = cv2.waitKey(0)
if key == ord('s'):
    cv2.imwrite("C:/Users/Niken/.vscode/Niken Widya Astuti_23170028_UTS/allofus_edged.jpg", edged)
    print("Gambar edged berhasil disimpan sebagai allofus_edged.jpg")
# Menerapkan threshold binary inverse
thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]
cv2.imshow("Thresh", thresh)
key = cv2.waitKey(0)
if key == ord('s'):
    cv2.imwrite("C:/Users/Niken/.vscode/Niken Widya Astuti_23170028_UTS/allofus_thresh.jpg", thresh)
    print("Gambar threshold berhasil disimpan sebagai allofus_thresh.jpg")
cv2.destroyAllWindows()
"""

#Hitung histogram dari gambar
"""
photo4 = cv2.imread("allofus.jpg")
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    hist = cv2.calcHist([photo4], [i], None, [400], [0, 400])
    plt.plot(hist, color=col)
    plt.xlim([0, 400])
# Menyimpan gambar histogram ke file
plt.savefig("C:/Users/Niken/.vscode/Niken Widya Astuti_23170028_UTS/allofus_histogram.jpg")
print("Histogram berhasil disimpan sebagai allofus_histogram.jpg")
plt.show()
"""

#Hitung normalisasi histogram dari gambar
"""
photo5 = cv2.imread("allofus.jpg")
print("Data gambar sebelum di normalisasi:\n", photo5)
photo5_normalized = cv2.normalize(photo5, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
cv2.imshow('Gambar di Normalisasi', photo5_normalized)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Data gambar setelah dinormalisasi:\n", photo5_normalized)
"""

#Hitung equalization histogram dari gambar
"""
image_path = "allofus.jpg"
image = cv2.imread(image_path, 0)
hist, bins = np.histogram(image.flatten(), 256, [0, 256])
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(image, cmap="gray", vmin=0, vmax=255)
axs[0].axis("off")
axs[1].plot(cdf_normalized, color="black", linestyle="--", linewidth=1)
axs[1].hist(image.flatten(), 256, [0, 256], color="r", alpha=0.5)
axs[1].set_xlim([0, 256])
axs[1].legend(("CDF", "Histogram"), loc="upper left")
plt.savefig("C:/Users/Niken/.vscode/Niken Widya Astuti_23170028_UTS/allofus_histogram_equalized.jpg")
print("Histogram berhasil disimpan sebagai allofus_histogram_equalized.jpg")
plt.show()
"""