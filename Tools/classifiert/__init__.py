CLASSIFIERS = {}

def register_classifier(name, data_format, runner):

    global CLASSIFIERS
    assert name not in CLASSIFIERS
    CLASSIFIERS[name] = {"data_format": data_format, "runner": runner}
