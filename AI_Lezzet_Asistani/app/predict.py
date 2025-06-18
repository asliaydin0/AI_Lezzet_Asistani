import numpy as np
from model.ai_model import SimpleNN

# Eğitim verileri
X = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1]
])
y = np.array([[1], [0], [0], [1]])

# Modeli oluştur ve eğit
model = SimpleNN(input_size=3)
model.train(X, y)

# Yeni bir yemek: Baharatlı, Tatlı değil, Sebzeli değil
yeni_yemek = np.array([[1, 0, 0]])
tahmin = model.predict(yeni_yemek)

print("Yeni yemek için beğenme ihtimali:", round(float(tahmin), 2))
