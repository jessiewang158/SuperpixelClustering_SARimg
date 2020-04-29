#The code was modified to take 4D SAR data input based on following code:
# Aleena Watson
# Final Project - Computer Vision Simon Niklaus
# Winter 2018 - PSU
#————————————————
#版权声明：本文为CSDN博主「请叫我西木同学」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
#原文链接：https://blog.csdn.net/qq965194745/java/article/details/104243782

import numpy
import sys
import cv2
import tqdm
from skimage import io
from example_sim import get4d
import matplotlib.pyplot as plt
import math
import cmath
from utils import readFloatComplex, readShortComplex, readFloat

class SLIC:
    def __init__(self, IFG_PATH, COH_PATH, SLC1_PATH, SLC2_PATH, K, m):
        # global variables

        self.amp_slc1, self.amp_slc2, self.real_ifg_phase, self.imag_ifg_phase, self.slc1, self.slc2, self.coh_3vg = get4d(IFG_PATH, COH_PATH, SLC1_PATH, SLC2_PATH)

        #img = numpy.abs(self.amp_slc1) ** 0.3

        #K_num = int(sys.argv[1])
        self.K = K
        self.step = int((self.slc1.shape[0]*self.slc1.shape[1] / int(self.K)) ** 0.5)
        #SLIC_m = int(sys.argv[2])
        self.SLIC_m = m
        self.SLIC_ITERATIONS = 10
        self.SLIC_height, self.SLIC_width = self.slc1.shape
        self.SLIC_4Dimg = numpy.empty((self.slc1.shape[0], self.slc1.shape[1], 4))

        for h in range(self.slc1.shape[0]):
            for w in range(self.slc1.shape[1]):
                self.SLIC_4Dimg[h][w][0] = self.amp_slc1[h][w]
                self.SLIC_4Dimg[h][w][1] = self.amp_slc2[h][w]
                self.SLIC_4Dimg[h][w][2] = self.real_ifg_phase[h][w]
                self.SLIC_4Dimg[h][w][3] = self.imag_ifg_phase[h][w]

        self.SLIC_distances = 1 * numpy.ones(self.SLIC_4Dimg.shape[:2])
        self.SLIC_clusters = -1 * self.SLIC_distances
        # SLIC_center_counts = numpy.zeros(len(calculate_centers(step, SLIC_width, SLIC_height)))
        self.SLIC_centers = numpy.array(self.calculate_centers())

    def generate_pixels(self):
        #print('SLIC_center',SLIC_centers)
        indnp = numpy.mgrid[0:self.SLIC_height, 0:self.SLIC_width].swapaxes(0, 2).swapaxes(0, 1)
        for i in tqdm.tqdm(range(self.SLIC_ITERATIONS)):
            self.SLIC_distances = 1 * numpy.ones(self.SLIC_4Dimg.shape[:2])
            for j in range(self.SLIC_centers.shape[0]):
                x_low, x_high = int(self.SLIC_centers[j][4] - self.step), int(self.SLIC_centers[j][4] + self.step)
                y_low, y_high = int(self.SLIC_centers[j][5] - self.step), int(self.SLIC_centers[j][5] + self.step)

                if x_low <= 0:
                    x_low = 0

                if x_high > self.SLIC_width:
                    x_high = self.SLIC_width

                if y_low <= 0:
                    y_low = 0

                if y_high > self.SLIC_height:
                    y_high = self.SLIC_height


                cropimg = self.SLIC_4Dimg[y_low: y_high, x_low: x_high]
                var_diff = cropimg - self.SLIC_4Dimg[int(self.SLIC_centers[j][5]), int(self.SLIC_centers[j][4])]
                var_distance = numpy.sqrt(numpy.sum(numpy.square(var_diff), axis=2))

                yy, xx = numpy.ogrid[y_low: y_high, x_low: x_high]
                pixdist = ((yy - self.SLIC_centers[j][5]) ** 2 + (xx - self.SLIC_centers[j][4]) ** 2) ** 0.5

                # SLIC_m is "m" in the paper, (m/S)*dxy
                dist = ((var_distance / self.SLIC_m) ** 2 + (pixdist / self.step) ** 2) ** 0.5

                distance_crop = self.SLIC_distances[y_low: y_high, x_low: x_high]
                idx = dist < distance_crop
                distance_crop[idx] = dist[idx]
                self.SLIC_distances[y_low: y_high, x_low: x_high] = distance_crop
                self.SLIC_clusters[y_low: y_high, x_low: x_high][idx] = j

            for k in range(len(self.SLIC_centers)):
                idx = (self.SLIC_clusters == k)
                varnp = self.SLIC_4Dimg[idx]
                distnp = indnp[idx]
                self.SLIC_centers[k][0:4] = numpy.sum(varnp, axis=0)
                sumy, sumx = numpy.sum(distnp, axis=0)
                self.SLIC_centers[k][4:] = sumx, sumy
                self.SLIC_centers[k] /= numpy.sum(idx)


    '''
    def cal_exp_z():
        exp_z2 = numpy.zeros((300,300),dtype=complex)
        exp_z1 = numpy.zeros((300,300),dtype=complex)
        for k in range(len(SLIC_centers)):
            idx = (SLIC_clusters ==k)
            compnum1 = slc1[idx]
            compnum2 = slc2[idx]
            norm_z1 = numpy.sum(compnum1,axis=0)
            norm_z2 = numpy.sum(compnum2,axis=0)
            norm_z1 /=numpy.sum(idx)
            norm_z2 /=numpy.sum(idx)
            exp_z1[idx] = norm_z1
            exp_z2[idx] = norm_z2

        return exp_z1, exp_z2
    '''

    def create_connectivity(self):
        label = 0
        adj_label = 0
        lims = int(self.SLIC_width * self.SLIC_height / self.SLIC_centers.shape[0])

        new_clusters = -1 * numpy.ones(self.SLIC_4Dimg.shape[:2]).astype(numpy.int64)
        elements = []
        for i in range(self.SLIC_width):
            for j in range(self.SLIC_height):
                if new_clusters[j, i] == -1:
                    elements = []
                    elements.append((j, i))
                    for dx, dy in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                        x = elements[0][1] + dx
                        y = elements[0][0] + dy
                        if (x >= 0 and x < self.SLIC_width and
                                y >= 0 and y < self.SLIC_height and
                                new_clusters[y, x] >= 0):
                            adj_label = new_clusters[y, x]

                count = 1
                counter = 0
                while counter < count:
                    for dx, dy in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                        x = elements[counter][1] + dx
                        y = elements[counter][0] + dy

                        if (x >= 0 and x < self.SLIC_width and y >= 0 and y < self.SLIC_height):
                            if new_clusters[y, x] == -1 and self.SLIC_clusters[j, i] == self.SLIC_clusters[y, x]:
                                elements.append((y, x))
                                new_clusters[y, x] = label
                                count += 1

                    counter += 1

                if (count <= lims >> 2):
                    for counter in range(count):
                        new_clusters[elements[counter]] = adj_label

                    label -= 1

                label += 1


        SLIC_new_clusters = new_clusters


    def display_contours(self,color=1000):
        is_taken = numpy.zeros(self.SLIC_4Dimg.shape[:2], numpy.bool)
        contours = []
        contourimg = self.coh_3vg
        for i in range(self.SLIC_width):
            for j in range(self.SLIC_height):
                nr_p = 0
                for dx, dy in [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]:
                    x = i + dx
                    y = j + dy
                    if x >= 0 and x < self.SLIC_width and y >= 0 and y < self.SLIC_height:
                        if is_taken[y, x] == False and self.SLIC_clusters[j, i] != self.SLIC_clusters[y, x]:
                            nr_p += 1

                if nr_p >= 2:
                    is_taken[j, i] = True
                    contours.append([j, i])

        #print(contours[0])
        for i in range(len(contours)):
            contourimg[contours[i][0], contours[i][1]] = color

        #io.imsave("SLIC_contours.jpg",contourimg)
        return contourimg

    # end
    def display_center(self):

        center_img = numpy.zeros([self.SLIC_height,self.SLIC_width,4]).astype(numpy.float64)
        for i in range(self.SLIC_width):
            for j in range(self.SLIC_height):
                k = int(self.SLIC_clusters[j, i])
                center_img[j,i] = self.SLIC_centers[k][0:4]

        return center_img

    def find_local_minimum(self,center):
        min_grad = 1
        loc_min = center
        for i in range(center[0] - 1, center[0] + 2):
            for j in range(center[1] - 1, center[1] + 2):
                c1 = self.SLIC_4Dimg[j + 1, i]
                c2 = self.SLIC_4Dimg[j, i + 1]
                c3 = self.SLIC_4Dimg[j, i]
                if numpy.sum(((c1 - c3) ** 2) ** 0.5 + ((c2 - c3) ** 2) ** 0.5) < min_grad:
                    min_grad = numpy.sum(abs(c1 - c3) + abs(c2 - c3))
                    loc_min = [i, j]

        return loc_min

    def calculate_centers(self):

        centers = []
        for i in range(self.step, self.SLIC_width - int(self.step / 2), self.step):
            for j in range(self.step, self.SLIC_height - int(self.step / 2), self.step):
                nc = self.find_local_minimum(center=(i, j))
                fourvar = self.SLIC_4Dimg[nc[1], nc[0]]
                center = [fourvar[0], fourvar[1], fourvar[2], fourvar[3], nc[0], nc[1]]
                centers.append(center)

        return centers

    def calc_pixelcoh(self,centers):
        patchcoh = []
        pixelcoh = numpy.zeros(self.slc1.shape)
        for k in range(len(centers)):
            idx = (self.SLIC_clusters==k)
            z1 = self.slc1[idx]
            z2 = self.slc2[idx]
            delta = numpy.abs(numpy.sum(z1 * numpy.conj(z2)) / numpy.sqrt(numpy.sum(numpy.abs(z1) ** 2.) * numpy.sum(numpy.abs(z2) ** 2.)))
            patchcoh.append(delta)
            pixelcoh[idx]=delta
        #print('patchcoh shape',len(patchcoh))
        return pixelcoh, patchcoh

    '''
    def patchcoh(slc1,slc2):
        coh = []
        #centers = calculate_centers()
    
        for i in range(len(centers)):
    
            x,y = centers[i][4:]
            
            
            pho = (slc1[x,y]**2*slc2[x,y]**2)/(cmath.sqrt(slc1[x,y]**4*slc2[x,y]**4))
            if pho > 1/2:
                CE = cmath.sqrt(2*pho -1)
    
            else:
                CE = 0
            coh.append(CE)
        return coh
    
    def patchcoh_to_pixelcoh():
    
    
        pixelcoh = numpy.zeros((300,300),dtype=complex)
      
        for k in range(len(coh)):
            idx = (SLIC_clusters==k)
            pixelcoh[idx]=coh[k]
        return pixelcoh
    # end
    '''

    # main
if __name__ == "__main__":
    IFG_PATH = "../sim_data/S1-Flow-FS-Test/ifg_fr/1slc1_1slc2.noisy"
    COH_PATH = "../sim_data/S1-Flow-FS-Test/ifg_fr/1slc1_1slc2.filt.coh"
    SLC1_PATH = "../sim_data/S1-Flow-FS-Test/ifg_fr/1slc1.rslc"
    SLC2_PATH = "../sim_data/S1-Flow-FS-Test/ifg_fr/1slc2.rslc"
    WIDTH = 300
    slic = SLIC(IFG_PATH, COH_PATH, SLC1_PATH, SLC2_PATH, 1000, 1)
    slic.generate_pixels()
    #print('\n SLIC_clusters',list(SLIC_clusters))
    slic.create_connectivity()

    centers=slic.calculate_centers()
    print('centers',len(centers))
    img_contours = slic.display_contours()
    img_center = slic.display_center()
    #exp_z1, exp_z2=cal_exp_z()
    #coh = patchcoh(slc1,slc2)
    #pixelcoh = patchcoh_to_pixelcoh()
    pixelcoh, patchcoh = slic.calc_pixelcoh(centers)
    #print('patchcoh',patchcoh)
    #print('pixelcoh',pixelcoh)
    coh_3vg = readFloat(COH_PATH, WIDTH)
    print('coh_3vg',coh_3vg)

    print('MSE',numpy.sum((pixelcoh-coh_3vg)**2)/WIDTH/WIDTH)


    #fig = plt.figure()
    #plt.imshow(img,cmap='gray')
    #plt.imshow(pixelcoh,cmap='gray')
    #plt.imshow(img_contours,cmap='gray')
    #fig.show()
    #fig.savefig('SLICimg{}_{}.jpg'.format(SLIC_ITERATIONS,int(sys.argv[1])))
    #fig.savefig('SP_coh{}_{}.jpg'.format(SLIC_ITERATIONS, int(sys.argv[1])))


    #print('img',img.shape,'\n','img_center','\n',img_center[50][20:30],'\n', 'contours','\n',img_contours,'\n','center','\n',centers)
