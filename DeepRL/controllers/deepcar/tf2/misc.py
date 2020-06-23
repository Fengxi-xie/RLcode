#   Other miscellaneous funtions
#   Vision = 1.0
import tensorflow as tf
import numpy as np

def update_tar_var(target,source,deg =1.,name = 'update_variables',use_lock = False):
    '''Network parameter set update'''
    if not isinstance(deg,float):
        raise TypeError("deg must be float! ")
    if deg<0 or deg >1:
        raise ValueError("Invalid parameter deg")

    same_shape = all(trg.get_shape() == src.get_shape()
                     for trg, src in zip(target, source))
    if not same_shape:
        raise ValueError("target and source are not the shape!")

    if deg !=1:
        new_source = deg * source + (1.0 - deg) * source
        [target_var.assign(source_var,use_lock) for target_var, source_var
                in zip(target, new_source)]
    else :
        [target_var.assign(source_var,use_lock) for target_var, source_var
                in zip(target, source)]



def learing_rate_adjust(init_rate,end_rate,ep_max,now_ep,piece = 300):
    '''Change learning rate in steps'''
    if init_rate <=end_rate:
        raise ValueError("learning target rate is wrong!")

    if now_ep >= ep_max:  now_ep = ep_max

    rate_piece = (init_rate - end_rate) / piece    #change the size of the learning rate

    piece_ = (ep_max / piece)   #How often does it change
    
    c_= int(now_ep / piece_)

    now_lrn_rate = init_rate - c_ * rate_piece

    if now_lrn_rate <  0:
        print('There is an error in learing_rate_adjust about now_lrn_rate!')

    if now_lrn_rate <= end_rate: now_lrn_rate = end_rate

    return np.round(now_lrn_rate,6)






