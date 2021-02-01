import os
import numpy as np
import math
from Bio import SeqIO
from scipy import stats

def get_fna_sequence(filename,index):
    '''
    index start from 1
    '''
    i=1
    for seq_record in SeqIO.parse(filename, "fasta"):
        if i==index:
            print(seq_record.description)
            print('len',len(seq_record))
            return str(seq_record.seq)
        i+=1


def split_sequence(sequence,split_every=100,is_slide=False,drop_last=True):
    seq_len=len(sequence)
    seq_list=[]
    list_len=seq_len/split_every
    if drop_last:
        list_len=math.floor(list_len)
    else:
        list_len=math.ceil(list_len)

    for i in range(list_len):
        seq_list.append(sequence[i*split_every:(i+1)*split_every])
    return seq_list

def count_gc(seq_list):
    count_list=np.zeros((len(seq_list),))
    for i,seq in enumerate(seq_list):
        count=0
        for element in seq:
            if element == 'g' or element == 'G' or element == 'c' or element == 'C':
                count+=1
        count_list[i]=count/len(seq)

    return count_list

def seqence2emitted_state(seq,num_state=5,win_size=100):
    '''
    return state index
    '''
    seq_list=split_sequence(seq,drop_last=False,split_every=win_size)
    count_list=count_gc(seq_list)

    per_div=np.arange(1,num_state)*100/num_state
    div_point=np.percentile(count_list,per_div,interpolation='lower')
    
    # another binning scheme
    # max,min=np.max(count_list),np.min(count_list)
    # div_point=min+range(1,num_state)*(max-min)/num_state

    print(div_point)

    state=np.digitize(count_list,div_point,right=True)

    return state
