# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 15:22:37 2020

Declares what is made available to the final user, so that they can do

import microstructure_fingerprinting as mf
mfu = mf.mf_utils
mcf = mf.mcf
[...]
MF_model = mf.MFModel(dictionary_file)
mfu.


@author: rensonnetg
"""

# Everything made visible to the final user
from .mf import MFModel, cleanup_2fascicles
import microstructure_fingerprinting.mcf
import microstructure_fingerprinting.mf_utils
