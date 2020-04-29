#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# example.py
# Copyright (c) 2020 Alvin(Xinyao) Sun <xinyao1@ualberta.ca>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from utils import readFloatComplex, readShortComplex, readFloat
import matplotlib.pyplot as plt
import numpy as np
import argparse


def get4d(IFG_PATH,COH_PATH,SLC1_PATH,SLC2_PATH):
    WIDTH = 300
    # Load binary interfergram data;
    ifg = readFloatComplex(IFG_PATH, WIDTH)
    #print('ifg',ifg)
    coh_3vg = readFloat(COH_PATH, WIDTH)
    #print('coh_3vg',coh_3vg)
    slc1 = readFloatComplex(SLC1_PATH, WIDTH)
    #print(slc1,'slc1')
    slc2 = readFloatComplex(SLC2_PATH, WIDTH)
    #print('slc',slc1)

    # 4D representation

    # Dim-0
    amp_slc1 = np.abs(slc1)
    # Dim-1
    amp_slc2 = np.abs(slc2)

    # Phase of Ifg
    phase_ifg = np.angle(ifg)
    # Force amp to one
    phase_bar_ifg = 1 * np.exp(1j * phase_ifg)

    # Dim-2
    real_ifg_phase = np.real(phase_bar_ifg)
    # Dim-3
    imag_ifg_phase = np.imag(phase_bar_ifg)
    return amp_slc1, amp_slc2, real_ifg_phase, imag_ifg_phase, slc1, slc2, coh_3vg

if __name__ == "__main__":
    IFG_PATH = "../sim_data/S1-Flow-FS-Test/ifg_fr/1slc1_1slc2.noisy"
    COH_PATH = "../sim_data/S1-Flow-FS-Test/ifg_fr/1slc1_1slc2.filt.coh"
    SLC1_PATH = "../sim_data/S1-Flow-FS-Test/ifg_fr/1slc1.rslc"
    SLC2_PATH = "../sim_data/S1-Flow-FS-Test/ifg_fr/1slc2.rslc"
    amp_slc1, amp_slc2, real_ifg_phase, imag_ifg_phase, slc1, slc2, coh_3vg = get4d(IFG_PATH,COH_PATH,SLC1_PATH,SLC2_PATH,WIDTH)
    '''
    print('amp_slc1',amp_slc1.shape)
    print('amp_slc2', amp_slc2.shape)
    print('real_ifg_phase', real_ifg_phase.shape)
    print('imag_ifg_phase', imag_ifg_phase.shape)
    '''