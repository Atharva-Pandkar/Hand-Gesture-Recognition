import logging

from ASL import t

logging.basicConfig(format="[LINE:%(lineno)d] %(levelname)-8s [%(asctime)s]  %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    t.train_model()
    t.display()
