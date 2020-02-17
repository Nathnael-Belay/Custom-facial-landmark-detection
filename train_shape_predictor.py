import multiprocessing
import argparse
import dlib

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training", required=True,
	help="path to input training XML file")
ap.add_argument("-m", "--model", required=True,
	help="path serialized dlib shape predictor model")
args = vars(ap.parse_args())

# grab the default options for dlib's shape predictor
print("[INFO] setting shape predictor options...")
options = dlib.shape_predictor_training_options()

# depth of the regression tree -- there will be a total of 
# 2^tree_depth leaves in the tree
options.tree_depth = 2

# regularization parameter
options.nu = 0.25

# the number of cascades used to train the shape predictor 
options.cascade_depth = 12

# number of pixel used to generate features for the random trees at each cascade
options.feature_pool_size = 500

# selects best features at each cascade when training
options.num_test_splits = 20

# controls the amount of jitter
options.oversampling_amount = 20

# amount of translation jitter to apply
options.oversampling_translation_jitter = 0

# tell the dlib shape predictor to be verbose and print out status
# messages the model trains
options.be_verbose = True

# number of threads/CPU cores to be used when training 
options.num_threads = multiprocessing.cpu_count()

print("[INFO] shape predictor options:")
print(options)

# train the shape predictor
print("[INFO] training shape predictor...")
dlib.train_shape_predictor(args["training"], args["model"], options)