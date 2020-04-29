import numpy as np
from SLIC_fast import *
from example_sim import *
import pickle
from statistics import mean
from utils import readFloatComplex, readShortComplex, readFloat
import cv2
from basemodel import *

WIDTH = 300
m = 1

#Select different

def iteratefile(i):

    print('i=', i)
    IFG_PATH = "../sim_data/S1-Flow-FS-Test/ifg_fr/{}slc1_{}slc2.noisy".format(i, i)
    COH_PATH = "../sim_data/S1-Flow-FS-Test/ifg_fr/{}slc1_{}slc2.filt.coh".format(i, i)
    SLC1_PATH = "../sim_data/S1-Flow-FS-Test/ifg_fr/{}slc1.rslc".format(i, i)
    SLC2_PATH = "../sim_data/S1-Flow-FS-Test/ifg_fr/{}slc2.rslc".format(i, i)

    return IFG_PATH, COH_PATH, SLC1_PATH, SLC2_PATH

def save_pixelcoh(IFG_PATH, COH_PATH, SLC1_PATH, SLC2_PATH):
    allpixelcoh = []
    for K in range(1000,4000,1000):

        print('K=', K)

        amp_slc1, amp_slc2, real_ifg_phase, imag_ifg_phase, slc1, slc2, coh_3vg = get4d(IFG_PATH, COH_PATH, SLC1_PATH, SLC2_PATH)
        #print('slc1', slc1)
        slic = SLIC(IFG_PATH, COH_PATH, SLC1_PATH, SLC2_PATH, K, m)

        slic.generate_pixels()
        slic.create_connectivity()
        centers = slic.calculate_centers()
        pixelcoh, patchcoh = slic.calc_pixelcoh(centers)
        '''
        cv2.imshow('',pixelcoh)
        cv2.waitKey(0)
        '''
        allpixelcoh.append(pixelcoh)

    return allpixelcoh

def cal_mse(Test_coh, coh_3vg):
    MSE = np.sum((Test_coh - coh_3vg) ** 2) / WIDTH / WIDTH
    return MSE

if __name__ == "__main__":
    MSE_avg=[]
    MSE_min=[]
    MSE_max=[]
    MSE_bse=[]
    MSEall=[]
    MSE_1000=[]
    MSE_2000=[]
    MSE_3000=[]
    WIDTH = 300

    for i in range(20):

        IFG_PATH, COH_PATH, SLC1_PATH, SLC2_PATH = iteratefile(i)
        #print('slc1_path',SLC1_PATH)


        #allpixelcoh = save_pixelcoh(IFG_PATH, COH_PATH, SLC1_PATH, SLC2_PATH)

        with open('SPpixelcoh_1000_3000_slc{}.pkl'.format(i), 'rb') as f:
            allpixelcoh = pickle.load(f)

        #amp_slc1, amp_slc2, real_ifg_phase, imag_ifg_phase, slc1, slc2, coh_3vg = get4d(IFG_PATH, COH_PATH, SLC1_PATH, SLC2_PATH)

        '''
        basepixelcoh = slidingwindow(slc1, slc2)
        cv2.imshow('base',basepixelcoh)
        cv2.waitKey(0)
        cv2.imwrite('basecoh_slc{}.png'.format(i),basepixelcoh*255)
        MSE_temp0 = cal_mse(basepixelcoh, coh_3vg)
        MSE_bse.append(MSE_temp0)
        
        '''

        cv2.imshow('1000',allpixelcoh[0])
        cv2.waitKey(0)
        #cv2.imwrite('1000SPcoh_slc{}.png'.format(i),allpixelcoh[0]*255)
        cv2.imshow('2000',allpixelcoh[1])
        cv2.waitKey(0)
        #cv2.imwrite('2000SPcoh_slc{}.png'.format(i), allpixelcoh[1]*255)
        cv2.imshow('3000',allpixelcoh[2])
        cv2.waitKey(0)
        #cv2.imwrite('2000SPcoh_slc{}.png'.format(i), allpixelcoh[2]*255)

        coh_3vg = readFloat(COH_PATH, WIDTH)
        MSE_temp4 = cal_mse(allpixelcoh[0], coh_3vg)
        MSE_1000.append(MSE_temp4)
        print('MSE k=1000 slc={}'.format(i),MSE_temp4)
        MSE_temp5 = cal_mse(allpixelcoh[1], coh_3vg)
        MSE_2000.append(MSE_temp5)
        print('MSE k=2000 slc={}'.format(i), MSE_temp5)
        MSE_temp6 = cal_mse(allpixelcoh[2], coh_3vg)
        MSE_3000.append(MSE_temp6)
        print('MSE k=3000 slc={}'.format(i), MSE_temp6)

        avg_pixelcoh = (allpixelcoh[0]+allpixelcoh[1]+allpixelcoh[2])/3.0
        #print('avg pixelcoh',avg_pixelcoh)
        min_pixelcoh = np.amin(allpixelcoh,axis=0)
        #print('min pixelcoh',min_pixelcoh)
        max_pixelcoh = np.maximum.reduce(allpixelcoh)
        #print('max pixelcoh',max_pixelcoh)

        cv2.imshow('avg', avg_pixelcoh)
        cv2.waitKey(0)
        #cv2.imwrite('avgcoh_slc{}.png'.format(i), avg_pixelcoh * 255)

        cv2.imshow('min', min_pixelcoh)
        cv2.waitKey(0)
        #cv2.imwrite('mincoh_slc{}.png'.format(i), min_pixelcoh * 255)

        cv2.imshow('max', max_pixelcoh)
        cv2.waitKey(0)
        #cv2.imwrite('maxcoh_slc{}.png'.format(i), max_pixelcoh * 255)


        MSE_temp1 = cal_mse(avg_pixelcoh, coh_3vg)
        MSE_avg.append(MSE_temp1)
        print('MSE avg slc={}'.format(i), MSE_temp1)
        MSE_temp2 = cal_mse(min_pixelcoh,coh_3vg)
        MSE_min.append(MSE_temp2)
        print('MSE min slc={}'.format(i), MSE_temp2)
        MSE_temp3 = cal_mse(max_pixelcoh,coh_3vg)
        MSE_max.append(MSE_temp3)
        print('MSE max slc={}'.format(i), MSE_temp3)

    #MSEall = [MSE_1000, MSE_2000, MSE_3000, MSE_avg, MSE_min, MSE_max]
    '''
    with open('SPMSE_all.pkl','wb') as f:
        pickle.dump(MSEall,f)
    '''

    print('\n MSE K=1000 for all slc', mean(MSE_1000))
    print('MSE K=2000 for all slc', mean(MSE_2000))
    print('MSE K=3000 for all slc', mean(MSE_3000))

    print('MSE_avg for all slc',mean(MSE_avg))
    print('MSE_min for all slc', mean(MSE_min))
    print('MSE_max for all slc', mean(MSE_max))

    '''
    with open('MSEbse.pkl', 'wb') as f:
        pickle.dump(MSE_bse, f)

    with open('MSEbse.pkl','rb') as f:
        MSEbase = pickle.load(f)
    print(MSE_bse)
    '''