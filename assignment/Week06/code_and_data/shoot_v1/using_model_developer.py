from joblib import load

from plotting import data_visualization

# local model load as variables
model_MLPr_i3_h = load("_train_model/MLPr_i3_h.joblib")
model_MLPr_i3_r = load("_train_model/MLPr_i3_r.joblib")
model_RFr_h = load("_train_model/RFr_h.joblib")
model_RFr_r = load("_train_model/RFr_r.joblib")


def generate_data():
    """Generate a set of predicting data boundary"""
    velocity_init = 0
    velocity_end = 200
    theta_init = 0
    theta_end = 180
    sample_amount = 1000
    return (velocity_init, velocity_end, sample_amount), (theta_init, theta_end, sample_amount)


def main():
    # Start plotting
    data_visualization(generate_data(), (model_RFr_h, model_RFr_r))


if __name__ == "__main__":
    main()
