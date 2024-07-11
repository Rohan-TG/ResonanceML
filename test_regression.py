import xgboost as xg
import pandas as pd
import resml_functions
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('ENDFBVIII_MT102_XS_with_QZA.csv')

al = resml_functions.range_setter(df=df, la=38, ua=42)

tempsmall = df[df.A > 86]
tempsmall2 = tempsmall[tempsmall.A < 93]
tempsmall2.index = range(len(tempsmall2))

energy_grid, zrxs = resml_functions.General_plotter(df=tempsmall2, nuclides=[[40,90]])

print('Data loaded. Forming matrices...')

validation_nuclides = [[40,90]]

X_train, y_train = resml_functions.train_matrix(df=tempsmall2, val_nuclides=validation_nuclides, LA=87, UA=92)

X_test, y_test = resml_functions.test_matrix(df=tempsmall2, val_nuclides=validation_nuclides)

print('Matrices formed. Training...')

model = xg.XGBRegressor(n_estimators= 500,
						max_depth=6,
						learning_rate=0.1,
						max_leaves=0)

model.fit(X_train, y_train, verbose=True, eval_set=[(X_test, y_test)])

print('Training complete. Evaluating...')

predictions = model.predict(X_test)

logp = [np.log(p) for p in predictions]
loge = [np.log(e) for e in energy_grid]
logxs = [np.log(x) for x in zrxs]

plt.figure()
plt.plot(loge, logxs, label = 'ENDF/B-VIII')
plt.plot(loge, logp, label = 'Predictions')
plt.legend()
plt.grid()
plt.xlabel('Log energy')
plt.ylabel('Log XS')
plt.show()