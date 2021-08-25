from cnn import Turtlebot_CNN
from lstm import Turtlebot_LSTM
from read_data import make_dataset,split_data
import tensorflow as tf
from inception import Inception
def predict(path_model,data):#Data must be a list of two arrays : the first one, with dimensions (Time,13) for the first one and (Time,360) for the second one.
    model=tf.keras.models.load_model(path_model)
    labels=['NoNoise','OdomNoise','ScanNoise']
    pred=model.predict(data)
    return labels[np.argmax(pred)]

