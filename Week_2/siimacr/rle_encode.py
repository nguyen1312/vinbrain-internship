from lib import *

# encode mask of XRays Image into rle 
def run_length_encode(component):
    # convert into vector
    component = component.T.flatten()

    start = np.where(component[1:] > component[:-1])[0] + 1
    end = np.where(component[:-1] > component[1:])[0] + 1
    length = end - start
    rle = []
    for i in range(len(length)):
        if i == 0:
            rle.extend([start[0], length[0]])
        else:
            rle.extend([start[i] - end[i-1], length[i]])
    rle = ' '.join([str(r) for r in rle])
    return rle

