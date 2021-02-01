import numpy as np
import matplotlib.pyplot as plt

from HMM import HMM
from util import *

if __name__ == "__main__":
    # params write
    params={
        'hidden_state':np.array([0,1]),
        'emission_state':np.array([1,2,3,4,5]),
        'init_prob':np.array([0.5,0.5]),
        'trans_prob':np.array([[0.8,0.2],[0.1,0.9]]),
        'emission_prob_type':'matrix',
        'emission_matrix':np.array([[0.2,0.5,0.2,0.1,0],[0,0.1,0.4,0.4,0.1]])
    }
    hmm=HMM(init_params=params)
    hmm.write_params_to_file('params')

    # params read
    hmm=HMM(folder_name='params')

    # generate
    hidden_state_chain,emitted_state_chain=hmm.generate(115)
    hidden_state_chain_,emitted_state_chain_=hidden_state_chain,emitted_state_chain
    hidden_state_chain,emitted_state_chain=hmm.index2state(hidden_state_chain,'hidden_state'),hmm.index2state(emitted_state_chain,'emission_state')
    
    # plot the generated result
    fig, ax = plt.subplots()
    x=np.arange(115)
    ax.plot(x,emitted_state_chain,label='emitted state')
    ax.plot(x,hidden_state_chain,label='hidden state')
    ax.set_xlabel('position index')
    ax.set_ylabel('state')
    ax.legend()
    fig.savefig('result/q1.pdf')

    fig, ax = plt.subplots()
    x=np.arange(115)
    ax.plot(x,emitted_state_chain,'.',label='emitted state')
    ax.plot(x,hidden_state_chain,'.',label='hidden state')
    ax.set_xlabel('position index')
    ax.set_ylabel('state')
    ax.legend()
    fig.savefig('result/q1_.pdf')

    # save chain
    np.savetxt('result/hidden_state_chain',hidden_state_chain,fmt='%d')
    np.savetxt('result/emitted_state_chain',emitted_state_chain,fmt='%d')

    # load chain
    hidden_state_chain=np.loadtxt('result/hidden_state_chain',dtype=int)
    emitted_state_chain=np.loadtxt('result/emitted_state_chain',dtype=int)

    hidden_state_chain,emitted_state_chain=hmm.state2index(hidden_state_chain,'hidden_state'),hmm.state2index(emitted_state_chain,'emission_state')
    assert (hidden_state_chain==hidden_state_chain_).all()
    assert (emitted_state_chain==emitted_state_chain_).all()

    # forward

    print('--naive forward--')
    likelihood=hmm.forward(emitted_state_chain,is_log=False)
    print('likelihood',likelihood)
    log_likelihood=hmm.forward(emitted_state_chain,is_log=True)
    print('log_likelihood',log_likelihood)

    print('--scaled forward--')
    likelihood=hmm.scaled_forward(emitted_state_chain,is_log=False)
    print('likelihood',likelihood)
    log_likelihood=hmm.scaled_forward(emitted_state_chain,is_log=True)
    print('log_likelihood',log_likelihood)

    # forward multiple times test

    print('--forward multiple times test--')
    test_time=100
    test_res=np.zeros((test_time,))
    test_res_log=np.zeros((test_time,))
    for i in range(test_time):
        hidden_state_chain,emitted_state_chain=hmm.generate(115)
        test_res[i]=hmm.scaled_forward(emitted_state_chain,is_log=False)
        test_res_log[i]=hmm.scaled_forward(emitted_state_chain,is_log=True)
    
    print('likelihood mean',np.mean(test_res))
    print('likelihood std',np.std(test_res))
    print('likelihood log mean',np.mean(test_res_log))
    print('likelihood log std',np.std(test_res_log))

    # random forward test

    print('--random sequence forward test--')
    test_time=100
    test_res=np.zeros((test_time,))
    test_res_log=np.zeros((test_time,))
    for i in range(test_time):
        emitted_state_chain=np.random.choice(np.arange(5),size=115)
        test_res[i]=hmm.scaled_forward(emitted_state_chain,is_log=False)
        test_res_log[i]=hmm.scaled_forward(emitted_state_chain,is_log=True)
    
    print('likelihood mean',np.mean(test_res))
    print('likelihood std',np.std(test_res))
    print('likelihood log mean',np.mean(test_res_log))
    print('likelihood log std',np.std(test_res_log))
    

    # get sequence

    # sequence=get_fna_sequence('sequence/GCA_003086655.1_ASM308665v1_genomic.fna',3)
    sequence=get_fna_sequence('sequence/GCF_000146045.2_R64_genomic.fna',3)

    # binning

    emitted_state_chain=seqence2emitted_state(sequence)

    # forward

    # likelihood=hmm.forward(emitted_state_chain,is_log=False)
    # print('likelihood',likelihood)
    # log_likelihood=hmm.forward(emitted_state_chain,is_log=True)
    # print('log_likelihood',log_likelihood)

    # likelihood=hmm.scaled_forward(emitted_state_chain,is_log=False)
    # print('likelihood',likelihood)
    log_likelihood=hmm.scaled_forward(emitted_state_chain,is_log=True)
    print('log_likelihood',log_likelihood)
    
    # the code above does not run because they will overflow

    # get alpha and beta

    scaled_alpha,scaled_beta,likelihood=hmm.get_scaled_alpha_beta(emitted_state_chain)

    # infer params

    # random initialize params
    print('--random initialize params--')
    init_prob=np.random.uniform(0,1,2)
    init_prob=init_prob/np.sum(init_prob)
    trans_prob=np.random.uniform(0,1,(2,2,))
    trans_prob=trans_prob/np.sum(trans_prob,axis=1,keepdims=True)
    emission_matrix=np.random.uniform(0,1,(2,5,))
    emission_matrix=emission_matrix/np.sum(emission_matrix,axis=1,keepdims=True)

    hmm.params['init_prob']=init_prob
    hmm.params['trans_prob']=trans_prob
    hmm.params['emission_matrix']=emission_matrix

    likelihood_adjusted=hmm.infer_params(emitted_state_chain)
    print('likelihood_adjusted',likelihood_adjusted)

    print('--initial params from q1--')
    hmm=HMM(folder_name='params')
    likelihood_adjusted=hmm.infer_params(emitted_state_chain)
    print('likelihood_adjusted',likelihood_adjusted)

    # plot real emitted state and generated emitted state

    hidden_state_gen,emitted_state_gen=hmm.generate(len(emitted_state_chain))

    fig, ax = plt.subplots()
    plot_len=100
    start_from=500
    x=np.arange(start_from,start_from+plot_len)
    ax.plot(x,emitted_state_gen[start_from:start_from+plot_len],label='generated emitted state')
    ax.plot(x,emitted_state_chain[start_from:start_from+plot_len],label='real emitted state')
    ax.set_xlabel('position index')
    ax.set_ylabel('state')
    ax.legend()
    fig.savefig('result/q4_emitted_state.pdf')

    # infer hidden state
    hidden_state_infer=hmm.infer_hidden_state(emitted_state_chain)

    hidden_state_infer,emitted_state_chain=hmm.index2state(hidden_state_infer,'hidden_state'),hmm.index2state(emitted_state_chain,'emission_state')

    # plot the infered hidden state and real emitted state
    
    fig, ax = plt.subplots()
    plot_len=100
    start_from=500
    x=np.arange(start_from,start_from+plot_len)
    ax.plot(x,emitted_state_chain[start_from:start_from+plot_len],label='real emitted state')
    ax.plot(x,hidden_state_infer[start_from:start_from+plot_len],label='infered hidden state')
    ax.set_xlabel('position index')
    ax.set_ylabel('state')
    ax.legend()
    fig.savefig('result/q5.pdf')

    # plot infered hidden state and generated hidden state

    fig, ax = plt.subplots()
    plot_len=100
    start_from=500
    x=np.arange(start_from,start_from+plot_len)
    ax.plot(x,hidden_state_gen[start_from:start_from+plot_len],label='generated hidden state')
    ax.plot(x,hidden_state_infer[start_from:start_from+plot_len],label='infered hidden state')
    ax.set_xlabel('position index')
    ax.set_ylabel('state')
    ax.legend()
    fig.savefig('result/q5_hidden_state.pdf')

    # try infer with another genome sequence
    
    hmm=HMM(folder_name='params')
    sequence=get_fna_sequence('sequence/GCF_000146045.2_R64_genomic.fna',9)
    emitted_state_chain=seqence2emitted_state(sequence)
    likelihood_adjusted=hmm.infer_params(emitted_state_chain)
    print('likelihood_adjusted',likelihood_adjusted)

    hidden_state_infer=hmm.infer_hidden_state(emitted_state_chain)

    hidden_state_infer,emitted_state_chain=hmm.index2state(hidden_state_infer,'hidden_state'),hmm.index2state(emitted_state_chain,'emission_state')

    fig, ax = plt.subplots()
    plot_len=100
    start_from=1000
    x=np.arange(start_from,start_from+plot_len)
    ax.plot(x,emitted_state_chain[start_from:start_from+plot_len],label='real emitted state')
    ax.plot(x,hidden_state_infer[start_from:start_from+plot_len],label='infered hidden state')
    ax.set_xlabel('position index')
    ax.set_ylabel('state')
    ax.legend()
    fig.savefig('result/q5_1.pdf')
    
    # try infer with another species
    
    hmm=HMM(folder_name='params')
    sequence=get_fna_sequence('sequence/GCA_001413975.1_ASM141397v1_genomic.fna',1)
    emitted_state_chain=seqence2emitted_state(sequence)
    likelihood_adjusted=hmm.infer_params(emitted_state_chain)
    print('likelihood_adjusted',likelihood_adjusted)

    hidden_state_infer=hmm.infer_hidden_state(emitted_state_chain)

    hidden_state_infer,emitted_state_chain=hmm.index2state(hidden_state_infer,'hidden_state'),hmm.index2state(emitted_state_chain,'emission_state')

    fig, ax = plt.subplots()
    plot_len=100
    start_from=1500
    x=np.arange(start_from,start_from+plot_len)
    ax.plot(x,emitted_state_chain[start_from:start_from+plot_len],label='real emitted state')
    ax.plot(x,hidden_state_infer[start_from:start_from+plot_len],label='infered hidden state')
    ax.set_xlabel('position index')
    ax.set_ylabel('state')
    ax.legend()
    fig.savefig('result/q5_2.pdf')

    # change emission distributions -- add state

    print('--change emission distributions -- add state--')
    emission_matrix = np.array([[0.1, 0.4, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05], [
                               0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.4, 0.1]])

    # emission_matrix=np.random.uniform(0,1,(2,10,))
    # emission_matrix=emission_matrix/np.sum(emission_matrix,axis=1,keepdims=True)
    # uncomment above code to enable random initialize

    params={
        'hidden_state':np.array([0,1]),
        'emission_state':np.array([1,2,3,4,5,6,7,8,9,10]),
        'init_prob':np.array([0.5,0.5]),
        'trans_prob':np.array([[0.8,0.2],[0.1,0.9]]),
        'emission_prob_type':'matrix',
        'emission_matrix':emission_matrix
    }
    hmm=HMM(init_params=params)

    sequence=get_fna_sequence('sequence/GCF_000146045.2_R64_genomic.fna',3)
    emitted_state_chain=seqence2emitted_state(sequence,num_state=10,win_size=100)

    likelihood_adjusted=hmm.infer_params(emitted_state_chain)
    print('likelihood_adjusted',likelihood_adjusted)

    hidden_state_infer=hmm.infer_hidden_state(emitted_state_chain)

    hidden_state_infer,emitted_state_chain=hmm.index2state(hidden_state_infer,'hidden_state'),hmm.index2state(emitted_state_chain,'emission_state')

    fig, ax = plt.subplots()
    plot_len=100
    start_from=100
    x=np.arange(start_from,start_from+plot_len)
    ax.plot(x,emitted_state_chain[start_from:start_from+plot_len],label='real emitted state')
    ax.plot(x,hidden_state_infer[start_from:start_from+plot_len],label='infered hidden state')
    ax.set_xlabel('position index')
    ax.set_ylabel('state')
    ax.legend()
    fig.savefig('result/q6.pdf')
    
    fig, ax = plt.subplots()
    x=np.arange(hmm.params['emission_matrix'].shape[1])
    ax.plot(x,hmm.params['emission_matrix'][0],label='emission_matrix first row')
    ax.plot(x,hmm.params['emission_matrix'][1],label='emission_matrix second row')
    ax.set_xlabel('emitted state')
    ax.set_ylabel('probability')
    ax.legend()
    fig.savefig('result/q6_params.pdf')
    
    # change emission distributions -- linear distribution

    print('--change emission distributions -- linear distribution--')
    def update_emission_matrix_with_mean(mean,index_array):
        b=(19/3-mean)*3/55
        if b<0:
            b=0
        if b>0.2:
            b=0.2
        res=(1-10*b)/45*index_array+b
        return res
        
    emission_matrix_0=np.arange(0,10).reshape((1,-1))
    emission_matrix_0=emission_matrix_0*0.02+0.01
    emission_matrix_1=np.arange(0,10).reshape((1,-1))
    emission_matrix_1=emission_matrix_1*-0.02+0.19

    # emission_matrix_0=np.arange(0,10).reshape((1,-1))
    # emission_matrix_0=emission_matrix_0/90+0.05
    # emission_matrix_1=np.arange(0,10).reshape((1,-1))
    # emission_matrix_1=emission_matrix_1/-90+0.15
    # uncomment code above to do another initiaisation

    emission_matrix=np.concatenate((emission_matrix_0,emission_matrix_1),axis=0)

    params={
        'hidden_state':np.array([0,1]),
        'emission_state':np.array([1,2,3,4,5,6,7,8,9,10]),
        'init_prob':np.array([0.5,0.5]),
        'trans_prob':np.array([[0.8,0.2],[0.1,0.9]]),
        'emission_prob_type':'matrix',
        'emission_matrix':emission_matrix    
    }
    hmm=HMM(init_params=params)

    sequence=get_fna_sequence('sequence/GCF_000146045.2_R64_genomic.fna',3)
    emitted_state_chain=seqence2emitted_state(sequence,num_state=10,win_size=100)

    likelihood_adjusted=hmm.infer_params(emitted_state_chain,{'need':['mean'],'update_fun':update_emission_matrix_with_mean})
    print('likelihood_adjusted',likelihood_adjusted)

    hidden_state_infer=hmm.infer_hidden_state(emitted_state_chain)

    hidden_state_infer,emitted_state_chain=hmm.index2state(hidden_state_infer,'hidden_state'),hmm.index2state(emitted_state_chain,'emission_state')

    fig, ax = plt.subplots()
    plot_len=100
    start_from=100
    x=np.arange(start_from,start_from+plot_len)
    ax.plot(x,emitted_state_chain[start_from:start_from+plot_len],label='real emitted state')
    ax.plot(x,hidden_state_infer[start_from:start_from+plot_len],label='infered hidden state')
    ax.set_xlabel('position index')
    ax.set_ylabel('state')
    ax.legend()
    fig.savefig('result/q6_2.pdf')

    fig, ax = plt.subplots()
    x=np.arange(hmm.params['emission_matrix'].shape[1])
    ax.plot(x,hmm.params['emission_matrix'][0],label='emission_matrix first row')
    ax.plot(x,hmm.params['emission_matrix'][1],label='emission_matrix second row')
    ax.set_xlabel('emitted state')
    ax.set_ylabel('probability')
    ax.legend()
    fig.savefig('result/q6_params_2.pdf')
    
    print('end')
