from example_sim import get4d
import numpy as np
import matplotlib.pyplot as plt

def slidingwindow(slc1,slc2):
    pixelcoh=np.zeros(slc1.shape)

    for i in range(0,slc1.shape[1]-6,1):

        for j in range(0,slc1.shape[0]-6,1):
            z1 = slc1[i:i+6,j:j+6]
            z2 = slc2[i:i+6,j:j+6]
            delta = np.abs(np.sum(z1 * np.conj(z2)) / np.sqrt(np.sum(np.abs(z1) ** 2.) * np.sum(np.abs(z2) ** 2.)))
            #print('delta',delta)
            pixelcoh[i+3,j+3] = delta
    pixelcoh[0:3,0:pixelcoh.shape[1]]=0
    pixelcoh[(pixelcoh.shape[0]-3):pixelcoh.shape[0],0:pixelcoh.shape[1]]=0
    pixelcoh[0:pixelcoh.shape[0],0:3]=0
    pixelcoh[0:pixelcoh.shape[0],(pixelcoh.shape[1]-3):pixelcoh.shape[1]]=0
    return pixelcoh

if __name__ == "__main__":
    IFG_PATH = "../sim_data/S1-Flow-FS-Test/ifg_fr/1slc1_1slc2.noisy"
    COH_PATH = "../sim_data/S1-Flow-FS-Test/ifg_fr/1slc1_1slc2.filt.coh"
    SLC1_PATH = "../sim_data/S1-Flow-FS-Test/ifg_fr/1slc1.rslc"
    SLC2_PATH = "../sim_data/S1-Flow-FS-Test/ifg_fr/1slc2.rslc"
    WIDTH = 300
    amp_slc1, amp_slc2, real_ifg_phase, imag_ifg_phase, slc1, slc2, coh_3vg = get4d(IFG_PATH,COH_PATH,SLC1_PATH,SLC2_PATH,WIDTH)
    pixelcoh = slidingwindow(slc1,slc2)
    #print ('coh_3vg',coh_3vg)
    #print('pixelcoh',pixelcoh)
    '''
    fig = plt.figure()
    plt.imshow(pixelcoh, cmap='gray')
    fig.savefig('basecoh2.jpg')
    '''
    MSE = np.sum((pixelcoh-coh_3vg)**2)/slc1.shape[0]/slc1.shape[1]
    print('MSE',MSE)

