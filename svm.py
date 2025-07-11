# uses an SVM to predict movie ratings
import time
from sklearn import svm
import pandas as pd
import numpy as np
import util
import pickle
import matplotlib.pyplot as plt  # Add this import for plotting
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import precision_recall_curve, auc

class MovieSVM():


    def __init__(self, threshold, delta):
        self.threshold = threshold
        self.e = delta
        self.accuracy = []

    # fits the SVM according to the algorithm described
    def fit(self, A, V):
 
        # Create a negative of A 
        #N = self._buildNegative(A)
        #A = self._applyThreshold(A, self.threshold)

        T = np.copy(A)
        A = np.copy(A)
        for i in range(len(V)):
            for j in range(len(V[i])):
                if V[i, j] >= self.threshold:
                    V[i, j] = 1
                elif V[i, j] != 0:
                    V[i, j] = 0
                else:
                    V[i, j] = -1 
        for i in range(len(A)):
            for j in range(len(A[i])):
                if A[i,j] >= self.threshold:
                    A[i,j] = 1
                    T[i,j] = 1
                elif A[i,j] != 0:
                    A[i,j] = 0
                    T[i,j] = 0
                else:
                    A[i,j] = np.random.randint(0, 2, size=1)[0]
                    T[i,j] = -1
        
        print(A.shape, np.count_nonzero(A==0), np.count_nonzero(A==1)) 
        print(T.shape, np.count_nonzero(T==0), np.count_nonzero(T==1)) 
        print(V.shape, np.count_nonzero(V==0), np.count_nonzero(V==1)) 
        #return []
        
        totalValidation = np.count_nonzero(V!=-1)
        iteration = 0
            
        svms = [ svm.SVC() for i in range(len(A[0]))  ]
        #print(len(svms))
        self.accuracy = []
        acc_prev, acc_k = 0, 2*self.e
        #print(np.delete(A, 1, axis=1).shape)    
        #print()
        self.train_accuracy = []
        while acc_k - acc_prev > self.e:
            train_correct = 0
            total_train = 0
            start_time = time.time()
            iteration += 1
            for i in range(len(svms)):
                print("fit:", str(i) + "/" + str(len(svms)), end='\r')
                X = np.delete(A, i, axis=1)
                Y = A[:,i]
        
                try:
                    svms[i].fit(X, Y)
                except:
                    dummy = 0
        
                A[:, i] = svms[i].predict(X)
                for j in range(len(A[:,i])):
                    if T[j,i] != -1 and A[j,i] != T[j,i]:
                        #A[j,i] = T[j,i]     
                        total_train += 1
                    elif T[j,i] != -1:
                        train_correct += 1   
                        total_train += 1
            self.train_accuracy.append((train_correct*1.0)/total_train)
            # calculate iteration accuracy
            countMatched = 0
            # go through each column, predict that column, check matching on V
            for i in range(len(svms)):
                print("Validate:", str(i) + "/" + str(len(svms)), end='\r')
                X = np.delete(V, i, axis=1)
                #print(X.shape)
                Y = V[:,i]
                Yhat = svms[i].predict(X)
                countMatched += np.sum(Yhat==Y)
            acc_prev = acc_k
            acc_k = (countMatched*1.0)/totalValidation
            
            self.accuracy.append(acc_k)
            print("\n - Iteration:", iteration, "\n - With Accuracy:", acc_k*100, "\n Difference:", acc_k - acc_prev, "\n Time (seconds):", time.time() - start_time)
        return self.accuracy, self.train_accuracy

                
        # while acc_k - acc_k-1 > e:
            # for each column, i
                # compute svm   -- X = [A[:,i-1],A[i+1,:]], Y = A[:,i] 
                # store svm in svms[i]
                # predict values
                # replace A[:,i] = predictions
            
            # predict values for each test point
            # acc_k = accuracy
            
    def countCorrect(self,T, A):
        count = np.sum(T==A)
        return count


    #def countTrue(A, N)

    # creates a matrix of the same shape as A
    #  All positions of values in A are replaced with -1
    #  All 0 positions in A are replaced with randomly chosen 0, 1
    def _buildNegative(self, A, thresh):
        #N = np.copy(A)
        N = np.random.randint(2, size=A.shape)
        N = N - 5*A
        N[N < 0] = -1
        return N

    def _applyTreshold(A, thresh):
        A[A >= thresh] = 2*thresh
        A[A < thresh] = 0
        return A
    
# ---------------------------------------------------- #
# Save Histogram of Predicted Ratings for All Users
# ---------------------------------------------------- #
def save_predicted_ratings_histogram(predicted_ratings: np.ndarray, filename: str = "predictions_svm.png"):
    """
    Saves a histogram of predicted ratings for all users.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(predicted_ratings.flatten(), bins=20, color='skyblue', edgecolor='black')
    plt.title("Histogram of Predicted Ratings (SVM)")
    plt.xlabel("Predicted Rating")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(filename)
    print(f"Histogram saved as {filename}")
    plt.close()

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: list, filename: str = "svm_confusion_matrix.png"):
    """
    Plots and saves the confusion matrix for SVM predictions.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    plt.figure(figsize=(10, 8))  # Adjust the width and height as needed

    disp.plot(cmap='Blues', ax=plt.gca())
    plt.title("Confusion Matrix for SVM")
    plt.savefig(filename)
    print(f"Confusion Matrix saved as {filename}")
    plt.close()

if __name__== "__main__":
    NUM_MOVIES = 1000
    Data = util.load_data_matrix()
    A = Data[:400, :NUM_MOVIES]
    movieSVM = MovieSVM(1.5, .01) # sets threshold and delta. respectively
    V = Data[401:, :NUM_MOVIES]
    v_non_zero = np.count_nonzero(V)
    for i in range(len(A[0]) - 1, 0, -1):
        if np.count_nonzero(A[:,i]) == 0:
            A = np.delete(A, i, axis=1)
            V = np.delete(V, i, axis=1)
    
    # Ensure consistent dimensions for training and validation
    assert A.shape[1] == V.shape[1], "Training and validation matrices must have the same number of columns."

    accuracy, train_accuracy = movieSVM.fit(A, V)
    print("\n\n sparsity:", 1 - (v_non_zero*1.0)/(V.shape[0] * V.shape[1] * 1.0))
    print("\n\nFinished Accuracy Values:", accuracy)
    print("\n\nTraining Accuracy:", train_accuracy)

    # ---------------------------------------------------- #
    # Save Histogram of Predicted Ratings for All Users
    # ---------------------------------------------------- #
    save_predicted_ratings_histogram(A)

    # ---------------------------------------------------- #
    # Recommend Top Movies for User 1
    # ---------------------------------------------------- #
    user_id = 1  # Specify the user ID
    predicted_ratings = A[user_id]  # Get predicted ratings for user 1
    top_movie_indices = np.argsort(predicted_ratings)[-5:][::-1]  # Top 5 movies

    # Load movie dictionary
    movie_dict = pickle.load(open('data/data_dicts.p', 'rb'))

    print(f"\nTop 5 movie recommendations for User {user_id}:")
    for movie_idx in top_movie_indices:
        for movie_id, movie_col in movie_dict['movieId_movieCol'].items():
            if movie_col == movie_idx:
                movie_title = movie_dict['movieId_movieName'][movie_id]
                predicted_rating = predicted_ratings[movie_idx]
                print(f"{movie_title:80} Predicted Rating: {predicted_rating:.2f}")

    # ---------------------------------------------------- #
    # Plot Confusion Matrix
    # ---------------------------------------------------- #
    class_names = ["Below Threshold", "Above Threshold"]

    # Filter out invalid entries from y_true
    y_true = V.flatten()
    y_true = y_true[y_true != -1]  # Remove invalid entries

    # Ensure y_pred matches the length of y_true
    y_pred = A.flatten()
    y_pred = y_pred[:len(y_true)]  # Match the length of y_true

    # Convert continuous predictions to binary categories
    y_pred_binary = np.where(y_pred >= 0.5, 1, 0)  # Threshold at 0.5

    # Plot the confusion matrix
    plot_confusion_matrix(y_true, y_pred_binary, class_names)