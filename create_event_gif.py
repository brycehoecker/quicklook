import imageio
import os, sys
from tqdm import tqdm

try:
    runID = sys.argv[1]
    eventID = sys.argv[2]
except:
    print('Must provide runID and eventID as arguments.')
    sys.exit()

images = []

#homedir = os.environ['HOME']
#datadir = "/data/user/bmode/analysis_output/camera_movies"
datadir = "/data/analysis_output/camera_movies/run{}_ev{}_movie".format(runID, eventID)

for time in tqdm(range(128)):
    images.append(imageio.imread('{}/BaseSubWaveform_run{}_event{}_time{}.png'.format(datadir, runID, eventID, time)))

imageio.mimsave('{}/run{}_ev{}.gif'.format(datadir , runID, eventID), images)

imageio.mimsave('{}/run{}_ev{}_centered.gif'.format(datadir, runID, eventID), images[40:91])
