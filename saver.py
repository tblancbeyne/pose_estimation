from keras.models import load_model

import pickle as pkl
import numpy as np
import glob
import time

def poses():
    # Load model
    estimator = load_model('network.hd5')

    real = pkl.load(open("one_minute_synthese.pkl", "rb"))
    real = (2 * real - np.amax(real)) / np.amax(real)

    # Predict outputs
    poses = estimator.predict(real)

    # Save generated data
    pkl.dump(poses, open('one_minute_poses.pkl', 'wb'))

if __name__ == '__main__':
    poses()
