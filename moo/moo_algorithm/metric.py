from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
import numpy as np
def cal_hv(front, ref_point):
    #front = np.array([[0.1, 0.5], [0.3, 0.7], [0.2, 0.4], [0.4, 0.8], [0.9, 0.9]])
    #ref_point = np.array([1, 1])
    ind = HV(ref_point=ref_point)
    return ind(front)


def cal_igd(front, optimal_front):
    #front = np.array([[0.1, 0.5], [0.3, 0.7], [0.2, 0.4], [0.4, 0.8], [0.9, 0.9]])
    #optimal_front = np.array([[0.1, 0.5], [0.3, 0.7], [0.2, 0.4], [0.4, 0.8], [0.9, 0.9]])
    ind = IGD(optimal_front)
    return ind(front)
            
def cal_hv_front(indi_list, ref_point):
    #print("cal hv font")
    front = []
    for indi in indi_list:
        front.append(indi.objectives)
    return cal_hv(np.array(front), ref_point)