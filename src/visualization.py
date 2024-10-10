import matplotlib.pyplot as plt


def plot_predictions(train_data, val_data, y_train_pred, y_val_pred):
    plt.figure(figsize=(10, 6))

    plt.plot(train_data['Time from Start'], train_data['Price'],
             label='Actual Train Price', color='blue', marker='o')

    plt.plot(train_data['Time from Start'], y_train_pred,
             label='Predicted Train Price', color='blue', linestyle='--', marker='x')

    plt.plot(val_data['Time from Start'], val_data['Price'],
             label='Actual Validation Price', color='green', marker='o')

    plt.plot(val_data['Time from Start'], y_val_pred,
             label='Predicted Validation Price', color='green', linestyle='--', marker='x')

    plt.title('Actual vs. Predicted Stock Prices')
    plt.xlabel('Time from Start (Quarter)')
    plt.ylabel('Stock Price')

    plt.grid(True)
    plt.legend()

    plt.savefig('output/predictions_plot.png')

