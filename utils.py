import unicodedata


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
