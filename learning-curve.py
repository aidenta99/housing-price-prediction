from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from modelling import chosen_model, final_features
import matplotlib.pyplot as plt

data_dir = Path("data/")
img_dir = Path("../img")

# Load data
all_data = pd.read_csv(data_dir / "housing-data.csv", index_col="Order")

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=10
    )
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], np.log(y_train[:m]))
        y_train_predict = np.exp(model.predict(X_train[:m]))
        y_val_predict = np.exp(model.predict(X_val))
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=1, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=1, label="val")
    plt.legend(loc="upper right", fontsize=14)
    plt.xlabel("Training set size", fontsize=14)
    plt.ylabel("RMSE", fontsize=14)

plot_learning_curves(chosen_model, final_features.drop(columns="SalePrice"), final_features["SalePrice"])
plt.axis([0, 200, 0, 100000])
plt.show()
