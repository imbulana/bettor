import pickle as pkl

def load_model(model_path):
    """
    Load the pre-trained model from the specified path.
    """

    with open(model_path, 'rb') as f:
        model = pkl.load(f)
    return model
