from sklearn.linear_model import LinearRegression

def train_model(x_train, y_train):
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model


def print_linear_model(model, feature_names):
    equation = f"Price = {model.intercept_:.2f}"
    for coef, feature in zip(model.coef_, feature_names):
        equation += f" + {coef:.2f} * {feature}"

    print("Linear Model Equation:")
    print(equation)
