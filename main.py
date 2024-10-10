from src.data_loader import load_data
from src.model import train_model, print_linear_model
from src.metrics import compute_metrics
from src.visualization import plot_predictions

data = load_data('data/stock_delta.csv')

train_data = data.iloc[:36]
val_data = data.iloc[36:40]

X_train = train_data[['Time from Start']].values
y_train = train_data['Price'].values

X_val = val_data[['Time from Start']].values
y_val = val_data['Price'].values

model = train_model(X_train, y_train)

feature_names = ['Time from Start']
print_linear_model(model, feature_names)

y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)

plot_predictions(train_data, val_data, y_train_pred, y_val_pred)

mse, rmse, mad, mape = compute_metrics(y_val, y_val_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'Mean Absolute Deviation (MAD): {mad}')
print(f'Mean Absolute Percentage Error (MAPE): {mape}%')

X_all = data[['Time from Start']].values
y_all = data['Price'].values
model.fit(X_all, y_all)

next_quarter = [[41]]
future_price = model.predict(next_quarter)
print(f'Predicted stock price for the next quarter: {future_price[0]}')
