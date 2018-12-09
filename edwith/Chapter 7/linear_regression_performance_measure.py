# Linear Regression - Performance measure

# 핵심 키워드
#
# Mean Absolute Error(MAE)
# Root Mean Squared Error(RMSE)
# R squared
# training & test set
# scikit-learn

# Mean Absolute Error(MAE)
from sklearn.metrics import median_absolute_error

y_true = [3,-0.5,2,7]
y_pred = [2.5,0.0,2,8]

median_absolute_error(y_true,y_pred)

# Root Mean Squared Error(RMSE)
from sklearn.metrics import mean_squared_error

mean_squared_error(y_true,y_pred)

# R squared
from sklearn.metrics import r2_score

r2_score(y_true,y_pred)

# training & test set
import numpy as np
from sklearn.model_selection import train_test_split

X, y = np.arange(10).reshape((5,2)), range(5)
X
y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=42)
X_train, X_test, y_train, y_test
