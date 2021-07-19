import numpy as np
from sklearn.linear_model import LinearRegression

np.set_printoptions(precision = 1)

n = 50
# assume these are the original features
x1 = np.random.randint(size = n, low = 50, high = 100)
x2 = np.random.randint(size = n, low = 2, high = 20)
x3 = np.random.randint(size = n, low = 5, high = 50)

# and this is the model with some gaussian noise
y = 12 + x1 + 4 * x2 - 5 * x3 \
    + 3 * x1**2 \
    - 2 * np.sqrt(x2) \
    + 4 * np.log(x1) \
    + 2 * x1 * x3 \
    + np.random.normal(size = n, loc = 0, scale = 0.05)

# first do regression as such
X = np.column_stack((x1, x2, x3))

raw = LinearRegression()
raw.fit(X, y)
print(f'A very bad model: {raw.intercept_:.1f}', np.round(raw.coef_, 1))

# TRANSFORMATIONS
# add squares of raw features
tf1 = np.power(x1, 2)
tf2 = np.power(x2, 2)
tf3 = np.power(x3, 2)
X = np.column_stack((X, tf1, tf2, tf3))
# add roots
tf4 = np.sqrt(x1)
tf5 = np.sqrt(x2)
tf6 = np.sqrt(x3)
X = np.column_stack((X, tf4, tf5, tf6))
# add logarithms
tf7 = np.log(x1)
tf8 = np.log(x2)
tf9 = np.log(x3)
X = np.column_stack((X, tf4, tf5, tf6))
# add products
tf10 = np.multiply(x1, x2)
tf11 = np.multiply(x2, x3)
tf12 = np.multiply(x1, x3)
X = np.column_stack((X, tf10, tf11, tf12))

model = LinearRegression()
model.fit(X, y)
for (desired, obtained) in zip([12, 1, 4, -5, 3, 0, 0, 0, -2, 0, 4, 0, 0, 0, 0, 2],
                                   [round(model.intercept_)] + list([int(i) for i in np.round(model.coef_)])):
    print(desired, obtained, 'ok' if desired == obtained else 'mismatch')
