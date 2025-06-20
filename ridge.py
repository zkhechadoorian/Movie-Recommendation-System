# -*- coding: utf-8 -*-

import numpy as np
import util
import matplotlib.pyplot as plt
import pickle
from array import array

# -----
KEEP_N_ITERS = 9066 # slice width - set smaller to save RAM/time
MAX_ITERS = 2       # None -> unlimited convergence loop
# -----

class Ridge():
    """Weighted Ridge Regression"""
    def __init__(self, f=200, alpha=20, lambd=0.1, thres=0, epsilon=0.1):
        self.f = f
        self.alpha = alpha
        self.lambd = lambd
        self.thres = thres          # ← save it!
        self.epsilon = epsilon


    # Input: Rating matrix A
    # Hyperparameters: Vector Space Dimension: f ; alpha ; lambd
    def fit(self, A, max_iter=3):
        """
        Memory-safe Alternating Least Squares for implicit feedback.
        Works in < 200 MB for the full 671 × 9 066 matrix.
        """
        k, n = A.shape
        P = (A > self.thres).astype(np.float32)        # preference matrix
        C = (1 + self.alpha * A).astype(np.float32)    # confidence matrix

        self.X = np.random.rand(k, self.f).astype(np.float32)
        self.Y = np.random.rand(self.f, n).astype(np.float32)
        eye_f = np.eye(self.f, dtype=np.float32)

        for _ in range(max_iter):
            # ---------- update X ----------
            YTY = self.Y @ self.Y.T                    # (f × f)
            for u in range(k):
                cu = C[u]                              # (n,)
                A_u = YTY + (self.Y * cu) @ self.Y.T + self.lambd * eye_f
                b_u = (self.Y * cu) @ P[u]
                self.X[u] = np.linalg.solve(A_u, b_u)

            # ---------- update Y ----------
            XTX = self.X.T @ self.X                    # (f × f)
            for m in range(n):
                cm = C[:, m]                           # (k,)
                A_m = XTX + (self.X.T * cm) @ self.X + self.lambd * eye_f
                b_m = (self.X.T * cm) @ P[:, m]
                self.Y[:, m] = np.linalg.solve(A_m, b_m)
      
    # K is the number of recommended movies   
    def predict(self, u, K = 5):

        P_u_hat = np.dot(self.X[u] , self.Y)
        indices = np.argsort(P_u_hat)
        
#        Recommended movies that have not been rated yet
#        k=0
#        i = 0
#        recommended_movies = []
#        while k < K and i < len(indices) :
#            if self.P[u][indices[i]] == 0 : 
#                k += 1
#                recommended_movies.append(indices[i])
#            i += 1
        
        recommended_movies = indices[:K].tolist()
            
        return recommended_movies
            
def rank(mat1, r):
    '''
    Evaluates the quality of recommendations. Calculates expected percentile rank for recommendation.
    
    Input: 
        mat1: validation matrix
        r: ridge model instance

    Steps:
        - Computes a percentile ranking for recommendations based on their position in the sorted list
        - Aggregates rankings across all users to calculate an overall score
    '''

    k = len(mat1) #number of users
    n = len(mat1[0]) #number of movies
    sum_numerator = 0
    sum_denominator = np.sum(mat1) # sum of all movie ratings

    # iterate through users
    for u in range(k):

        # generate recommendations for user u
        recommendations = r.predict(u)
        K = len(recommendations)

        rank_u = np.zeros(n)

        # iterate through movies
        for m in range(n):

            # if this movie was a recommendation
            if m in recommendations :

                # update rank for this movie to store index in recommendadtions 
                # normalize to length of recommendations - 1
                rank_u[m] = recommendations.index(m)/(K-1)
        
        # iterate through movies
        for m in range(n): 
            # add to numerator each rating*rank from user u for movie m
            sum_numerator += mat1[u,m]*rank_u[m]
    
    return(sum_numerator / sum_denominator)
    
if __name__ == "__main__":
    
    '''Basic test of the algorithm'''

    # Load the user-movie matrix

    # A = util.load_data_matrix()
    A = util.load_data_matrix()[:,:100] # if tested on a laptop, please use the first 100 movies 

    # Train the model using Ridge object
    r = Ridge()
    r.fit(A)

    # Generate movie recommendations for user with ID 1 (movie IDs)
    recommendations = r.predict(1) # predicts the top K movies for user 1

    # Open dictionary that maps movieIDs to movie titles
    B = pickle.load( open('{}'.format('data/data_dicts.p'), 'rb'))

    # Uncomment if you want to print all movies for which this user gave 5.0 rating
    # for movie_id,rating in B['userId_rating'][2]:
    #     if rating ==5 :
    #         print(B['movieId_movieName'][movie_id] , ", rating:" , rating )
        
    l = recommendations # movie columns in matrix
    k_list =[] # movieId values

    # iterate through movie columns
    for movie_column in l :

        # iterate through movieId_movieCol dictionary
        for k, v in B['movieId_movieCol'].items():

            # if movie column in recommendations matches the col value in the dictionary
            if v == movie_column:
                # add this movieId to k_list
                k_list.append(k)

    print("Recommendations")
    for movie_id in k_list :
        print(B['movieId_movieName'][movie_id])
    

    # Hyperparameter tuning
    '''Choice of hyperparameters'''
    # A = util.load_data_matrix()
    A = util.load_data_matrix()[:,:100] # if tested on a laptop, please use the first 100 movies 
    f_range = np.arange(10,40,10)
    ranks_f = []
    alpha_range = np.arange(10) # np.arange(10, 80, 10)
    ranks_alpha = []
    lambd_range = np.logspace(-1, 1, 2)
    ranks_lambd = []
    thres_range = np.arange(0, 3.5, 1.0)
    ranks_thres = []

    k = 4
    train_mats, val_mats, masks = util.k_cross(k=k)

    '''Choice of f'''
    for f in f_range :
        print("f: ", f)
        x=[]
        for i in range(k):
            train_mat = train_mats[i]
            val_mat = val_mats[i]
            r = Ridge(f=f)
            r.fit(train_mat)
            x.append(rank(val_mat, r))
            
        ranks_f.append(np.mean(x)*100)
        
    plt.plot(f_range,ranks_f)
    plt.ylabel('expected percentile ranking (%)')
    plt.xlabel('f')
    #plt.show()
    plt.savefig('hyperparameter_f.pdf')

    '''Choice of alpha'''
    for alpha in alpha_range :
        print("alpha: ", alpha)
        x=[]
        for i in range(k):
            train_mat = train_mats[i]
            val_mat = val_mats[i]
            r = Ridge(alpha=alpha)
            r.fit(train_mat)
            x.append(rank(val_mat, r))
            
        ranks_alpha.append(np.mean(x)*100)
        
    plt.plot(alpha_range,ranks_alpha)
    plt.ylabel('expected percentile ranking (%)')
    plt.xlabel('alpha')
    #plt.show()
    plt.savefig('hyperparameter_alpha.pdf')
    
    '''Choice of lambda'''
    for lambd in lambd_range :
        print("lambda: ", lambd)
        x=[]
        for i in range(k):
            train_mat = train_mats[i]
            val_mat = val_mats[i]
            r = Ridge(lambd=lambd)
            r.fit(train_mat)
            x.append(rank(val_mat, r))
            
        ranks_lambd.append(np.mean(x)*100)
        
    plt.semilogx(lambd_range,ranks_lambd)
    plt.ylabel('expected percentile ranking (%)')
    plt.xlabel('lambda')
    #plt.show()
    plt.savefig('hyperparameter_lambda.pdf')

    '''Choice of threshold'''
    for thres in thres_range :
        print("thres: ", thres)
        x=[]
        for i in range(k):
            train_mat = train_mats[i]
            val_mat = val_mats[i]
            r = Ridge(thres=thres)
            r.fit(train_mat)
            x.append(rank(val_mat, r))
            
        ranks_thres.append(np.mean(x)*100)
        
    plt.plot(thres_range,ranks_thres)
    plt.ylabel('expected percentile ranking (%)')
    plt.xlabel('threshold')
    #plt.show()
    plt.savefig('hyperparameter_threshold.pdf')
