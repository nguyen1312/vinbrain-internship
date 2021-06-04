from lib import *

def predict(X):
    X_p = np.copy(X)
    preds = (X_p > 0.5).astype('uint8')
    return preds