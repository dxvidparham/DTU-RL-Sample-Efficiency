"""
This files is to evaluate the hyperparameter testing. We saved the iterations into *.model files in the results folder.
They include the model, as well as the performance of each of them.

"""
import logging
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level='INFO',
                    handlers=[logging.StreamHandler()]
                    )
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

import logging
import pickle

filename = "hp_result_04_12_2020-02_54_51.model"

with open(f"results/{filename}", "rb") as f:
    eval = pickle.load(f)

logging.info(eval)


if __name__ == '__main__':
    pass