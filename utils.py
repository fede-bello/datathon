import unicodedata
from sklearn.metrics import mean_absolute_error



def detect_separator(filename):
    with open(filename, "r") as file:
        first_line = file.readline()
        return ";" if first_line.count(";") > first_line.count(",") else ","


def convert_to_int_else_object(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return value


def remove_accents(input_str):
    if isinstance(input_str, str):
        nfkd_form = unicodedata.normalize("NFKD", input_str)
        return nfkd_form.encode("ASCII", "ignore").decode("ASCII")
    else:
        return input_str
    
def evaluate_model(regressor, X_val, y_val):
    y_pred = regressor.predict(X_val)
    return mean_absolute_error(y_pred, y_val)
