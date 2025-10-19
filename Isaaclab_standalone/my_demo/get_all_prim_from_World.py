
import omni.isaac.core.utils.prims as prims_utils
from omni.isaac.core import World
import numpy as np


def get_sub_prim(prim_parents_list):
    prim_list=[]
    for prim_parent in prim_parents_list:
        prim_children = prims_utils.get_prim_children(prim_parent)
        if prim_children:
            prim_list.extend(prim_children)
    return prim_list

def get_all_prim_from_World():

    all_prim_list=[]

    prim_world = prims_utils.get_prim_at_path("/World")
    all_prim_list.append(prim_world)

    prim_parents_list=[prim_world]

    while prim_parents_list:
        sub_prime = get_sub_prim(prim_parents_list)
        print(sub_prime)
        all_prim_list.extend(sub_prime)
        prim_parents_list = sub_prime

    #print(all_prim_list)
    for prim in all_prim_list: # print name of all prime
        print(prims_utils.get_prim_path(prim))
    
    return all_prim_list