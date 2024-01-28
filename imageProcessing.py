import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görüntüyü oku ve keskinleştir
img = cv2.imread('hucre.png', cv2.IMREAD_GRAYSCALE)
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
img_sharp = cv2.filter2D(img, -1, kernel)

# Görüntüyü ikili görüntüye dönüştür
# Otsu eşikleme yöntemi kullan
_, img_bin = cv2.threshold(img_sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Morfolojik erozyon uygula
kernel = np.ones((3,3), np.uint8)
img_erode = cv2.erode(img_bin, kernel, iterations = 1)

# Bağlantılı bileşen etiketleme algoritması uygula
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_erode, 8, cv2.CV_32S)

# Bileşenleri farklı renklerle göster
colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
colors[0] = [0, 0, 0] # Arka plan siyah olsun
img_color = colors[labels]

for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    x = labels == i
    x = x.astype(np.uint8)
    _, contours, _ = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    moments = cv2.moments(contours[0])
    orientation = 0.5 * np.arctan2(2 * moments['mu11'], moments['mu20'] - moments['mu02'])
    perimeter = cv2.arcLength(contours[0], True)
    circularity = 4 * np.pi * area / (perimeter ** 2)
    edge = cv2.Canny(x, 100, 200)
    perimeter = np.sum(edge) / 255
    compactness = area / perimeter
    # Sonuçları yazdır
    print(f'Bileşen {i}:')
    print(f'Alan: {area}')
    print(f'Yön: {orientation}')
    print(f'Dairesellik: {circularity}')
    print(f'Çevre: {perimeter}')
    print(f'Kompaktlık: {compactness}')
    print()

# Görüntüleri göster
plt.figure(figsize=(15,15))
plt.subplot(221)
plt.imshow(img, cmap='gray')
plt.title('Orijinal Görüntü')
plt.subplot(222)
plt.imshow(img_sharp, cmap='gray')
plt.title('Keskinleştirilmiş Görüntü')
plt.subplot(223)
plt.imshow(img_erode, cmap='gray')
plt.title('Erozyon Uygulanmış Görüntü')
plt.subplot(224)
plt.imshow(img_color)
plt.title('Bileşenlerin Renklendirilmiş Görüntüsü')
plt.show()