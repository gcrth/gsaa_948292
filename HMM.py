from itertools import chain
import numpy as np
import os
import pickle

class HMM:
    '''
    params content
    
    params['hidden_state'] np.array
    params['emission_state'] np.array
    params['init_prob'] np.array
    params['trans_prob'] np.array
    params['emission_prob_type'] matrix or other
    params['emission_matrix'] np.array
    '''
    def __init__(self,init_params=None,folder_name=None):
        if init_params is not None:
            self.params=init_params
        elif folder_name is not None:
            self.read_params_from_file(folder_name)
        else:
            self.params=None
    
    def read_params_from_file(self,folder_name):        
        assert os.path.exists(folder_name)
        self.params={}
        with open(os.path.join(folder_name,'hidden_state'),'r') as f:
            lines=f.readlines()
            assert len(lines)==2
            line=lines[0].strip('\n').split(' ')
            state_num=int(line[0])
            state_type=line[1]
            line=lines[1].strip(' \n').split(' ')
            if state_type =='int':
                self.params['hidden_state']=[int(element) for element in line]
            else:
                self.params['hidden_state']=line
            self.params['hidden_state']=np.array(self.params['hidden_state'])

        if os.path.exists(os.path.join(folder_name,'emission_state')):
            with open(os.path.join(folder_name,'emission_state'),'r') as f:
                lines=f.readlines()
                assert len(lines)==2
                line=lines[0].strip('\n').split(' ')
                state_num=int(line[0])
                state_type=line[1]
                line=lines[1].strip(' \n').split(' ')
                if state_type =='int':
                    self.params['emission_state']=[int(element) for element in line]
                else:
                    self.params['emission_state']=line
                self.params['emission_state']=np.array(self.params['emission_state'])

        with open(os.path.join(folder_name,'init_prob'),'r') as f:
            self.params['init_prob']=np.loadtxt(f)
        
        with open(os.path.join(folder_name,'trans_prob'),'r') as f:
            self.params['trans_prob']=np.loadtxt(f)
        
        if os.path.exists(os.path.join(folder_name,'emission_matrix')):
            self.params['emission_prob_type'] = 'matrix'
            with open(os.path.join(folder_name,'emission_matrix'),'r') as f:
                self.params['emission_matrix']=np.loadtxt(f)
        else:
            raise NotImplementedError

    def write_params_to_file(self,folder_name):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        
        with open(os.path.join(folder_name,'hidden_state'),'w') as f:
            if np.issubdtype(np.dtype(self.params['hidden_state'][0]), np.integer):
                state_type = 'int'
            else:
                state_type = 'str'
            f.write(str(len(self.params['hidden_state']))+' '+state_type+'\n')
            for hidden_state in self.params['hidden_state']:
                f.write(str(hidden_state)+' ')
            f.write('\n')
        
        if self.params['emission_state'] is not None:
            with open(os.path.join(folder_name,'emission_state'),'w') as f:
                if np.issubdtype(np.dtype(self.params['emission_state'][0]), np.integer):
                    state_type = 'int'
                else:
                    state_type = 'str'
                f.write(str(len(self.params['emission_state']))+' '+state_type+'\n')
                for emission_state in self.params['emission_state']:
                    f.write(str(emission_state)+' ')
                f.write('\n')
        
        with open(os.path.join(folder_name,'init_prob'),'w') as f:
            np.savetxt(f,self.params['init_prob'])
        
        with open(os.path.join(folder_name,'trans_prob'),'w') as f:
            np.savetxt(f,self.params['trans_prob'])
        
        if self.params['emission_prob_type'] == 'matrix':
            with open(os.path.join(folder_name,'emission_matrix'),'w') as f:
                np.savetxt(f,self.params['emission_matrix'])
            with open(os.path.join(folder_name,'meta'),'w') as f:
                f.write('emission_prob_type matrix\n')
        else:
            raise NotImplementedError

    def generate(self,gen_len):
        '''
        output state index
        '''
        hidden_state_chain=[]
        emitted_state_chain=[]
        hidden_state_chain.append(np.random.choice(np.arange(len(self.params['hidden_state'])),
                                p=self.params['init_prob']))
        if self.params['emission_prob_type']=='matrix':
            emitted_state_chain.append(np.random.choice(np.arange(len(self.params['emission_state'])),
                                    p=self.params['emission_matrix'][hidden_state_chain[-1]]))
        else:
            raise NotImplementedError

        for i in range(1,gen_len):
            hidden_state_chain.append(np.random.choice(np.arange(len(self.params['hidden_state'])),
                                    p=self.params['trans_prob'][hidden_state_chain[-1]]))
            if self.params['emission_prob_type']=='matrix':
                emitted_state_chain.append(np.random.choice(np.arange(len(self.params['emission_state'])),
                                        p=self.params['emission_matrix'][hidden_state_chain[-1]]))
            else:
                raise NotImplementedError

        return (np.array(hidden_state_chain),np.array(emitted_state_chain))

    def index2state(self,chain,type):
        chain=np.array(chain)
        if type=='hidden_state':
            return self.params['hidden_state'][chain]
        elif type=='emission_state':
            return self.params['emission_state'][chain]
        else:
            raise ValueError

    def state2index(self,chain,type):
        chain=np.array(chain)

        if type=='hidden_state':
            state=self.params['hidden_state']
        elif type=='emission_state':
            state=self.params['emission_state']
        else:
            raise ValueError

        chain_out=[]
        for i in range(chain.shape[0]):
            chain_out.append(np.where(chain[i]==state)[0][0])

        chain_out=np.array(chain_out).reshape((-1))
        
        return chain_out
    
    def forward(self,emitted_state_chain,is_log=False):
        '''
        emitted_state_chain should be index
        '''
        if self.params['emission_prob_type'] != 'matrix':
            raise NotImplementedError
        
        chain_length=len(emitted_state_chain)
        alpha_old=np.zeros((len(self.params['hidden_state']),))
        alpha_cur=np.zeros((len(self.params['hidden_state']),))
        for i in range(len(self.params['hidden_state'])):
            alpha_old[i]=self.params['emission_matrix'][i,emitted_state_chain[0]]*self.params['init_prob'][i]
        for step in range(1,chain_length):
            for i in range(len(self.params['hidden_state'])):
                prob_sum=0.0
                for j in range(len(self.params['hidden_state'])):
                    prob_sum+=alpha_old[j]*self.params['trans_prob'][j,i]
                alpha_cur[i]=self.params['emission_matrix'][i,emitted_state_chain[step]]*prob_sum
            alpha_old=alpha_cur.copy()
        likelihood=np.sum(alpha_cur)
        if is_log:
            likelihood=np.log(likelihood)
        return likelihood

    def scaled_forward(self,emitted_state_chain,is_log=True,is_for_backward=False):
        '''
        emitted_state_chain should be index
        '''
        if self.params['emission_prob_type'] != 'matrix':
            raise NotImplementedError

        chain_length=len(emitted_state_chain)

        c=np.zeros((chain_length,))
        scaled_alpha_old=np.zeros((len(self.params['hidden_state']),))
        scaled_alpha_cur=np.zeros((len(self.params['hidden_state']),))

        if is_for_backward:
            scaled_alpha=np.zeros((chain_length,len(self.params['hidden_state']),))
        for i in range(len(self.params['hidden_state'])):
            scaled_alpha_old[i]=self.params['emission_matrix'][i,emitted_state_chain[0]]*self.params['init_prob'][i]
        c[0]=np.sum(scaled_alpha_old)
        scaled_alpha_old=scaled_alpha_old/c[0]
        if is_for_backward:
            scaled_alpha[0]=scaled_alpha_old
        for step in range(1,chain_length):
            for i in range(len(self.params['hidden_state'])):
                scaled_alpha_cur[i]=self.params['emission_matrix'][i,emitted_state_chain[step]]*np.sum(scaled_alpha_old*self.params['trans_prob'][:,i])
            c[step]=np.sum(scaled_alpha_cur)
            scaled_alpha_cur=scaled_alpha_cur/c[step]
            scaled_alpha_old=scaled_alpha_cur.copy()
            if is_for_backward:
                scaled_alpha[step]=scaled_alpha_old
        
        
        likelihood=np.sum(np.log(c))
        if not is_log:
            likelihood=np.exp(likelihood)

        if is_for_backward:
            return scaled_alpha,c,likelihood
        else:
            return likelihood

    def get_scaled_alpha_beta(self,emitted_state_chain,is_log=True):
        '''
        emitted_state_chain should be index
        '''
        if self.params['emission_prob_type'] != 'matrix':
            raise NotImplementedError

        chain_length=len(emitted_state_chain)

        scaled_alpha,c,likelihood=self.scaled_forward(emitted_state_chain,is_log=is_log,is_for_backward=True)
        scaled_beta=np.zeros((chain_length,len(self.params['hidden_state']),))

        scaled_beta[-1]=1/c[-1]
        for step in range(chain_length-2,-1,-1):
            for i in range(len(self.params['hidden_state'])):
                scaled_beta[step,i]=np.sum(self.params['emission_matrix'][:,emitted_state_chain[step]]*scaled_beta[step+1]*self.params['trans_prob'][i])/c[step]

        return scaled_alpha,scaled_beta,likelihood

    def infer_params(self,emitted_state_chain,update_option=None):
        '''
        emitted_state_chain should be index
        '''
        if self.params['emission_prob_type'] != 'matrix':
            raise NotImplementedError

        chain_length=len(emitted_state_chain)
        likelihood_old=0.0

        # 2000 upper bound
        for itr in range(2000):

            scaled_alpha,scaled_beta,likelihood=self.get_scaled_alpha_beta(emitted_state_chain)
            if itr !=0:
                delta=likelihood-likelihood_old
                # delta=np.abs(delta)
                if delta<0.01:
                    return likelihood
                print(itr,': ','likelihood ',likelihood,' delta ',delta)
            else:
                print(itr,': ','likelihood ',likelihood)
            likelihood_old=likelihood

            expect=np.zeros((len(self.params['hidden_state']),len(self.params['hidden_state']),))
            for i in range(len(self.params['hidden_state'])):
                for j in range(len(self.params['hidden_state'])):
                    expect[i,j]=np.sum(scaled_alpha[:-1,i]*self.params['trans_prob'][i,j]*self.params['emission_matrix'][j,emitted_state_chain[1:]]*scaled_beta[1:,j])
            
            # update            
            init_prob=np.zeros_like(self.params['init_prob'])
            emission_matrix=np.zeros_like(self.params['emission_matrix'])

            trans_prob=expect/np.sum(expect,axis=1,keepdims=True)
            # check axis
            for i in range(len(self.params['hidden_state'])):
                init_prob[i]=np.sum(scaled_alpha[0,i]*self.params['trans_prob'][i]*self.params['emission_matrix'][:,emitted_state_chain[1]]*scaled_beta[1])
                
            if update_option is None:
                for i in range(len(self.params['hidden_state'])):
                    for vk in range(len(self.params['emission_state'])):
                        sum_value=0.0
                        for j in range(len(self.params['hidden_state'])):
                            for n in range(chain_length-1):
                                if emitted_state_chain[n]==vk:
                                    sum_value+=scaled_alpha[n,i]*self.params['trans_prob'][i,j]*self.params['emission_matrix'][j,emitted_state_chain[n+1]]*scaled_beta[n+1,j]
                        emission_matrix[i,vk]=sum_value
                emission_matrix=emission_matrix/np.sum(expect,axis=1,keepdims=True)
            else:
                for need in update_option['need']:
                    if need == 'mean':
                        mean=np.zeros((len(self.params['hidden_state']),))
                        for i in range(len(self.params['hidden_state'])):
                            sum_value=0.0
                            for j in range(len(self.params['hidden_state'])):
                                for n in range(chain_length-1):
                                    sum_value+=emitted_state_chain[n]*scaled_alpha[n,i]*self.params['trans_prob'][i,j]*self.params['emission_matrix'][j,emitted_state_chain[n+1]]*scaled_beta[n+1,j]
                            mean[i]=sum_value
                        mean=mean/np.sum(expect,axis=1)
                    else:
                        raise NotImplementedError
                if 'mean' in update_option['need'] and len(update_option['need'])==1:
                    for i in range(len(self.params['hidden_state'])):
                        update_index=np.arange(0,len(self.params['emission_state'])).reshape(1,-1)
                        update_res=update_option['update_fun'](mean[i],update_index)
                        emission_matrix[i]=update_res
                else:
                    raise NotImplementedError

            # trans_prob_old=self.params['trans_prob']
            # init_prob_old=self.params['init_prob']
            # emission_matrix_old=self.params['emission_matrix']
            self.params['trans_prob']=trans_prob
            self.params['init_prob']=init_prob
            self.params['emission_matrix']=emission_matrix

        return likelihood

    def infer_hidden_state(self,emitted_state_chain):
        '''
        emitted_state_chain should be index
        '''
        if self.params['emission_prob_type'] != 'matrix':
            raise NotImplementedError

        chain_length=len(emitted_state_chain)

        phi=np.zeros((chain_length,len(self.params['hidden_state']),))
        psi=np.zeros((chain_length,len(self.params['hidden_state']),))

        phi[0]=np.log(self.params['init_prob'])+np.log(self.params['emission_matrix'][:,emitted_state_chain[0]])

        for step in range(1,chain_length):
            for i in range(len(self.params['hidden_state'])):
                # max_value should be small enough
                max_value=-9e19
                max_index=0
                for j in range(len(self.params['hidden_state'])):
                    cur=np.log(self.params['trans_prob'][j,i])+phi[step-1,j]
                    if cur>max_value:
                        max_value=cur
                        max_index=j
                phi[step,i]=np.log(self.params['emission_matrix'][i,emitted_state_chain[step]])+max_value
                psi[step,i]=max_index
        
        hidden_state=np.zeros((chain_length,),dtype=np.int)

        hidden_state[-1]=np.argmax(phi[-1,:])

        for step in range(chain_length-2,-1,-1):
            hidden_state[step]=psi[step+1,hidden_state[step+1]]
        
        return hidden_state

