from vis.visualization import visualize_cam, overlay
from vis.utils import utils
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

# Find the index of the to be visualized layer above
layer_index = utils.find_layer_idx(model, 'visualized_layer')