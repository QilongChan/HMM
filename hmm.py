import numpy as np
import time
import matplotlib.pyplot as plt
np.set_printoptions(suppress = True, precision = 3)

class HMM:
    def __init__(self, A, B, pi):
        """
        A: Transition matrix N * N
        B: Emission matrix N * M
        pi: Initial state probability N*1
        """
        self.A = A
        self.B = B
        self.pi = pi
    
    def forward(self, obs):
        N = self.A.shape[0]
        T = len(obs)
        
        # use a matrix to store alpha_t (i) in forward algorithm
        Alpha = np.zeros((N, T))
        
        # t = 0
        Alpha[:,0] = self.pi * self.B[:,obs[0]]
        
        # t > 0
        for t in range(1,T):
            for i in range(N):
                Alpha[i,t] = np.dot(Alpha[:,t-1], self.A[:,i]) * self.B[i,obs[t]]
        
        return Alpha
        
    def backward(self, obs):
        N = self.A.shape[0]
        T = len(obs)
        
        # use a matrix to store beta_t (i) in backward algorithm
        Beta = np.zeros((N, T))
        
        # t = T - 1
        Beta[:,T - 1] = 1
        
        # t < T - 1
        for t in reversed(range(T-1)):
            for i in range(N):
                Beta[i,t] = np.dot(self.A[i,:], Beta[:,t+1] * self.B[:,obs[t+1]])
        
        return Beta
    
    def baum_welch(self, multi_obs, tol = 0.001):
        start_time = time.time()
        N = self.A.shape[0]
        K, T = multi_obs.shape
        
        Xi = np.zeros((K,T-1,N,N))
        Gamma = np.zeros((K,T,N))
        
        flag = 1
        count = 0
        
        ave_log_likelihoods = []
        ave_log_likelihoods.append(self.ave_log_likelihood(multi_obs))
        
        while flag:
            for k in range(K):
                # get current observation sequence
                obs = multi_obs[k]
                
                # get Alpha matrix using forward algorithm
                Alpha = self.forward(obs)
                
                # get Beta matrix using backward algorithm
                Beta = self.backward(obs)
                
                # get Xi
                for t in range(T-1):
                    for i in range(N):
                        # first set nominator as entry
                        Xi[k,t,i,:] = Alpha[i,t] * self.A[i,:] * self.B[:,obs[t+1]] * Beta[:,t+1]
                    # divided by denominator, the sum of the matrix at t
                    Xi[k,t,:,:] /= np.sum(Xi[k,t])
                
                # get Gamma
                for t in range(T):
                    Gamma[k,t,:] = Alpha[:,t] * Beta[:,t] / np.sum(Alpha[:,t] * Beta[:,t])
                
            
            # update a_ij
            A_new = np.sum(Xi,axis = (0,1)) / np.sum(Gamma[:,:-1],axis = (0,1))[:, None] # row-wise divide 
            
            # update b_jk
            M = self.B.shape[1]
            B_new = np.zeros((N,M))
            for m in range(M):
                numer = 0
                for k in range(K):
                    index = multi_obs[k] == m
                    numer += np.sum(Gamma[k,index], axis = 0)
                B_new[:,m] = numer / np.sum(Gamma[:,:], axis = (0,1))
            
            # update pi_i
            pi_new = np.mean(Gamma[:,0], axis = 0)
            
            self.A, self.B, self.pi = A_new, B_new, pi_new
            
            # update log likelihood
            ave_log_likelihoods.append(self.ave_log_likelihood(multi_obs))
            
            count += 1 
            # check if converge
            if (ave_log_likelihoods[-1] - ave_log_likelihoods[-2]) < tol:
                flag = 0
                print("iteration: ", count)
    
        print("training time: %0.3fs" % (time.time() - start_time))
        return ave_log_likelihoods

    
    def viterbi(self, obs):
        N = self.A.shape[0]
        T = len(obs)
        
        paths = np.zeros((N,T-1))
        Delta = np.zeros((N,T))
        
        # initialization
        Delta[:,0] = self.pi * self.B[:,obs[0]]
        
        # update forwards
        for t in range(1,T):
            for i in range(N):
                Delta[i,t] = np.max(Delta[:,t-1] * self.A[:,i]) * self.B[i,obs[t]]
                paths[i,t-1] = np.argmax(Delta[:,t-1] * self.A[:,i])
        
        opt_path = np.zeros(T)
        
        opt_path[T-1] = np.argmax(Delta[:,T-1])
        for t in reversed(range(T-1)):
            opt_path[t] = paths[int(opt_path[t+1]),t]
            
        return opt_path.astype(int), Delta
    
    def predict_next(self, obs):
        log_likelihoods = []
        N, M = self.B.shape
        for i in range(M):
            new_obs = np.append([obs],[i])
            log_likelihoods.append(self.log_prob_of_obs(new_obs))
            
        return np.argmax(log_likelihoods)
    
    def predict_next_likelihood(self, obs):
        likelihoods = []
        N, M = self.B.shape
        denom = np.exp(self.log_prob_of_obs(obs))
        for i in range(M):
            new_obs = np.append([obs],[i])
            likelihoods.append(np.exp(self.log_prob_of_obs(new_obs)) / denom)
        return likelihoods
            
    def simulate(self, T):
        # simulate observation sequences given A, B, pi
        state_seq = np.zeros(T).astype(int)
        obs_seq = np.zeros(T).astype(int)
        N, M = self.B.shape
        
        # t = 0
        state_seq[0] = np.random.choice(N, p = self.pi)
        obs_seq[0] = np.random.choice(M, p = self.B[state_seq[0]])
        
        # t > 0
        for t in range(1,T):
            state_seq[t] = np.random.choice(N, p = self.A[state_seq[t-1]])
            obs_seq[t] = np.random.choice(M, p = self.B[state_seq[t]])
        
        return state_seq, obs_seq
    
    def log_prob_of_obs(self, obs):
        # calculate the probablity of the observation sequence P(O_1:T | model)        
        # use forward algorithm to get the probability
        Alpha = self.forward(obs)
        prob = np.sum(Alpha[:,-1])
        
        return np.log(prob)
    
    def ave_log_likelihood(self, multi_obs):
        n = multi_obs.shape[0]
        log_likelihood = 0
        for i in range(n):
            log_likelihood += self.log_prob_of_obs(multi_obs[i])
            
        return log_likelihood / n
            
    def sort(self):
        # s.t. the state with the most uniform emission probability is labeled 0
        # give states with the most deterministic emission probabilities highest numbers
        var_sort = np.var(self.B, axis = 1).argsort()
        self.B = self.B[var_sort]
        self.A = self.A[var_sort][:,var_sort]
        self.pi = self.pi[var_sort]

def initialize(N,M):
    # Initialization of A, B, pi
    # randomly select from a uniform distribution s.t. each row sums up to 1
    A = np.random.rand(N,N)
    A /= np.sum(A, axis = 1)[:,None]
    
    B = np.random.rand(N,M)
    B /= np.sum(B, axis = 1)[:,None]
    
    pi = np.random.rand(N)
    pi /= np.sum(pi) 
    
    return A, B, pi

def cross_validate(multi_obs, N, M = 4, k = 5):
    n = multi_obs.shape[0]
    vali_size = int(n / k)
    log_likelihoods = []
    
    for i in range(k):
        test_index = list(set(range(i * vali_size, (i+1) * vali_size)))
        train_index = list(set(range(n)) - set(test_index))
        A, B, pi = initialize(N,M)
        model = HMM(A,B,pi)
        model.baum_welch(multi_obs[train_index])
        log_likelihoods.append(model.ave_log_likelihood(multi_obs[test_index]))
        
    return np.mean(log_likelihoods)

def pred_accuracy(model, multi_obs):
    n = multi_obs.shape[0]
    num_accu = 0
    for i in range(n):
        out = model.predict_next(multi_obs[i,:-1])
        num_accu += out == multi_obs[i,-1]
    
    return num_accu / n


TRAIN_PATH = "./train534.dat"
TEST_PATH = "./test1_534.dat"
M = 4 # number of possible outputs 

if __name__ == "__main__":
    # read training data
    train_seqs = np.genfromtxt(TRAIN_PATH,dtype = int)
    test_seqs = np.genfromtxt(TEST_PATH,dtype = int)
    
    # use cv to find optimal N
    vali_errors = []
    N_possible = list(range(5,11))
    for i in N_possible:
        vali_errors.append(cross_validate(train_seqs,i))
    
    plt.plot(N_possible,vali_errors)
    plt.xlabel("N",fontsize = 14)
    plt.ylabel("log-likelihood",fontsize = 14)
    plt.title("Avg log-likelihood vs. N on validation set",fontsize = 18)
    plt.show() # the plot shows that N = 9 is optimal
    
    # For N = 5,6,7,8,9,10, report covergence plot, avg log-likelihoods and accuracy(training and testing)
    avg_log_likelihoods = []
    models = []
    train_accus = []
    test_accus = []
    
    for N in N_possible:
        A, B, pi = initialize(N,M)
        model = HMM(A,B,pi)
        # store avg log likelihoods
        likelihood = model.baum_welch(train_seqs)
        avg_log_likelihoods.append(likelihood)
        # store model
        model.sort()
        models.append(model)
        print("A:\n", model.A)
        print("B:\n", model.B)
        print("pi:\n", model.pi)
        # training accuracy
        train_accus.append(pred_accuracy(model,train_seqs))
        # test accuracy
        test_accus.append(pred_accuracy(model,test_seqs))
    
    # plot of convergence
    for (i,N) in enumerate(N_possible):
        plt.plot(avg_log_likelihoods[i], label = "N = {}".format(N))
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Average Log-likelihoods")
    plt.title("Plot of Convergence")
    plt.savefig("convergence")
    
    # log likelihood of training set and test set
    for (i,N) in enumerate(N_possible):
        model = models[i]
        train_logl = model.ave_log_likelihood(train_seqs)
        test_logl = model.ave_log_likelihood(test_seqs)
        print("N = {}".format(N))
        print("Train: {0}    {1}".format(round(train_logl,3),round(train_logl * train_seqs.shape[0], 3)))
        print("Test: {0}    {1}".format(round(test_logl,3),round(test_logl * test_seqs.shape[0], 3)))
        
    # viterbi sequence for N = 9
    print(models[4].viterbi(train_seqs[1])[0])
    
    t0 = time.time()
    for i in range(train_seqs.shape[0]):
        models[4].viterbi(train_seqs[i])[0]
    print("Time cost %0.3fs" % (time.time() - t0))
    
    t0 = time.time()
    for i in range(test_seqs.shape[0]):
        models[4].viterbi(train_seqs[i])[0]
    print("Time cost %0.3fs" % (time.time() - t0))
    
    # predict the accuracy for training set and test set
    print("Training:\n",train_accus)
    print("Test:\n",test_accus)
    
    # final model with N = 9
    final_hmm = HMM(A = models[4].A, B = models[4].B, pi = models[4].pi)
    
    # loglik
    loglik = final_hmm.ave_log_likelihood(test_seqs) * test_seqs.shape[0]
    
    # viterbi
    viterbi = []
    for i in range(test_seqs.shape[0]):
        viterbi.append(final_hmm.viterbi(test_seqs[i])[0])
    viterbi = np.array(viterbi)
    
    # predict
    output_likelihoods = []
    for i in range(test_seqs.shape[0]):
        output_likelihoods.append(final_hmm.predict_next_likelihood(test_seqs[i]))
    output = np.array(output_likelihoods)
    
    # save results
    # A,B,pi
    np.savetxt('a.txt',models[4].A,delimiter = ',')
    np.savetxt('b.txt',models[4].B,delimiter = ',')
    np.savetxt('pi.txt',models[4].pi.reshape((1,-1)),delimiter = ',')
    
    # save "loglik.dat"
    np.savetxt('loglik.dat', np.array([loglik]))
    
    # save "viterbi.dat"
    np.savetxt('viterbi.dat', viterbi, delimiter = ',', fmt='%d')
    
    # save predict.dat
    np.savetxt("predict.dat", output, delimiter = ',')
    
    # save predictoutput.dat
    predictoutput = np.zeros(test_seqs.shape[0]).astype(int)
    for i in range(test_seqs.shape[0]):
        predictoutput[i] = final_hmm.predict_next(test_seqs[i])
    np.savetxt("predictoutput.dat", predictoutput, fmt = '%d')
    
    
