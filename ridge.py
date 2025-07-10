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
    def __init__(self, f=250, alpha=200, lambd=0.01, thres=0, epsilon=0.1):
        self.f = f         # latent feature space dimension
        self.alpha = alpha # confidence scaling factor
        self.lambd = lambd # regularization parameter
        self.thres = thres # ← save it!
        self.epsilon = epsilon # convergence tolerance for ALS


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
      
        #print(f"Range of X: Min = {self.X.min()}, Max = {self.X.max()}")
        #print(f"Range of Y: Min = {self.Y.min()}, Max = {self.Y.max()}")

    # K is the number of recommended movies   
    def predict(self, u, K = 5):

        P_u_hat = np.dot(self.X[u] , self.Y)
        indices = np.argsort(P_u_hat)[::-1] 
        
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
    #print(f"Min value in A: {A.min()}, Max value in A: {A.max()}")

    # compute average of nonzero elements in A
    nonzero_elements = A[A != 0]
    average_nonzero = np.mean(nonzero_elements)
    #print(f"Average of nonzero elements in A: {average_nonzero}")

    # Train the model using Ridge object
    r = Ridge(thres=2.0,lambd=0.1)
    r.fit(A)

    # generate predicted ratings for all users
    predicted_ratings = np.clip( (r.X @ r.Y) * 5 , 0, 5) # scale ratings from 0-1 to 0-5
    #print(f"Min predicted rating: {predicted_ratings.min()}")
    #print(f"Max predicted rating: {predicted_ratings.max()}")

    # Plot the distribution of predicted ratings
    plt.figure(figsize=(8, 6))
    plt.hist(predicted_ratings.reshape(-1).tolist(), bins=np.arange(0, 6, 1.0), edgecolor='black')
    plt.title('Distribution of Predicted Ratings (Ridge Algorithm)')
    plt.xlabel('Predicted Rating')
    plt.ylabel('Frequency')
    plt.grid(axis='y')
    plt.savefig('predictions_ridge.png')

    # Generate movie recommendations for user with ID 1 (list of movie IDs)
    userId = 1
    recommendations = r.predict(userId) # predicts the top K movies for user with userId

    # Open dictionary that maps movieIDs to movie titles
    B = pickle.load( open('{}'.format('data/data_dicts.p'), 'rb'))
        
    l = recommendations # list of recommended movie columns in matrix
    k_list =[] # list of recommended movieId values

    # iterate through movie columns
    for movie_column in l :

        # iterate through movieId_movieCol dictionary
        for k, v in B['movieId_movieCol'].items():

            # if movie column in recommendations matches the col value in the dictionary
            if v == movie_column:
                # add this movieId to k_list
                k_list.append(k)

    print("Recommendations")
    for rec_movie_column in l:  # Iterate through recommended movie columns
        for movie_id, movie_col in B['movieId_movieCol'].items():
            if movie_col == rec_movie_column:
                pred_title = B['movieId_movieName'][movie_id]
                # Predicted rating for the movie (scaled from 0-1 to 0-5)
                pred_rating = np.clip( 5 * r.X[userId] @ r.Y[:, rec_movie_column], 0, 5 )
                # Predicted rating for the movie
                print(f"{pred_title:80} Predicted Rating: {pred_rating:.2f}")

    run_scans = False
    if (not run_scans):
        print("Ridge complete. Script is configured to skip hyperparameter turning. "
        "To turn this feature back on, set 'run_scans = True.' ")
        exit(0)

    '''
    The rest of the code is focused on hyperparameter tuning. 
    More specifically, it performs a scan over f, lambda, and threshold.'''

    # Hyperparameter tuning
    '''Choice of hyperparameters'''
    # A = util.load_data_matrix()
    A = util.load_data_matrix()[:,:100] # if tested on a laptop, please use the first 100 movies 
    f_range = np.arange(200,400,50)
    ranks_f = []
    alpha_range = np.arange(100, 300, 50)
    ranks_alpha = []
    lambd_range = np.logspace(-1, 1, 2)
    ranks_lambd = []
    thres_range = np.arange(0, 3.5, 0.5)
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
    
    plt.figure()
    plt.plot(f_range,ranks_f)
    plt.ylabel('expected percentile ranking (%)')
    plt.xlabel('f')
    plt.show()
    plt.savefig('hyperparameter_f.png')

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

    plt.figure()  
    plt.plot(alpha_range,ranks_alpha)
    plt.ylabel('expected percentile ranking (%)')
    plt.xlabel('alpha')
    plt.show()
    plt.savefig('hyperparameter_alpha.png')
    
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

    plt.figure()  
    plt.semilogx(lambd_range,ranks_lambd)
    plt.ylabel('expected percentile ranking (%)')
    plt.xlabel('lambda')
    plt.show()
    plt.savefig('hyperparameter_lambda.png')

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

    plt.figure()  
    plt.plot(thres_range,ranks_thres)
    plt.ylabel('expected percentile ranking (%)')
    plt.xlabel('threshold')
    plt.show()
    plt.savefig('hyperparameter_threshold.png')
