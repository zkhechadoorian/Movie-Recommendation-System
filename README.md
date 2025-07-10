# 🎬 Movie Recommendation System: Predicting User Preferences

This project was developed as part of the 2025 CompuFlair Data Science bootcamp.

----

This project addresses the problem of accurately predicting user movie preferences in the context of sparse rating data, a common challenge in recommendation systems. The project implements and compares four collaborative filtering approaches using the MovieLens dataset: 

- Neighborhood-Based Collaborative Filtering (CF)
- Ridge Regression
- Neural Networks
- Support Vector Machines (SVM)

Each method was tailored to handle data sparsity and assessed based on prediction quality, computational efficiency, and system applicability. The models were evaluated using metrics like mean squared error, expected percentile ranking, and classification accuracy to determine their effectiveness in recommending movies.

# 📁 Project Structure

```
📁 movie-recommendation-system/
│
├── 📄 README.md                       # Project overview and setup instructions
├── requirements.txt                   # Python dependencies
│
├── 📂 data/                           # Directory for storing datasets (e.g., MovieLens)
│
├── create_data_matrix.ipynb           # Jupyter notebook for preprocessing and building data matrix
│
├── cf.py                              # Neighborhood-based Collaborative Filtering implementation
├── ridge.py                           # Ridge Regression-based Collaborative Filtering implementation
├── nn.py                              # Neural Network model definition for rating prediction
├── train_nn.py                        # Script for training the neural network
├── svm.py                             # Support Vector Machine approach for recommendations
├── util.py                            # Shared utility functions (e.g., similarity metrics, evaluation)
```
# 📊 Dataset Metadata (ml-latest-small)

## 📄 Summary

This dataset describes 5-star rating and free-text tagging activity from [MovieLens](http://movielens.org), a movie recommendation service. It contains 100,004 ratings and 1,296 tag applications across 9,125 movies. These data were collected from 671 users between January 09, 1995 and October 16, 2016. The dataset was generated on October 17, 2016.

* Each user rated at least 20 movies.
* Users were selected randomly and anonymized.
* No demographic information is included.

**Files included:**

* `links.csv`
* `movies.csv`
* `ratings.csv`
* `tags.csv`

⚠️ *Note: This is a development dataset, and may change. For publication-ready benchmarks, use official benchmark datasets.*

## 📁 File Descriptions

### 🧾 `ratings.csv`

* **Format:** `userId,movieId,rating,timestamp`
* **Scale:** 0.5 to 5.0 (in 0.5 steps)
* **Notes:** Ordered by userId, then movieId

### 🏷️ `tags.csv`

* **Format:** `userId,movieId,tag,timestamp`
* **Notes:** User-generated metadata; short phrases or words

### 🎞️ `movies.csv`

* **Format:** `movieId,title,genres`
* **Genres:** Pipe-separated, e.g., `Comedy|Drama|Romance`
* **Notes:** Titles include release year

### 🔗 `links.csv`

* **Format:** `movieId,imdbId,tmdbId`
* **Use:** Link MovieLens movies to IMDB and TMDB

## 🔧 Setup and Installation Instructions

**Important note:** please run this project on the server for the sake of version consistency.

### Step 1: Download Project Files

Run the following command in your local terminal to download the project:

```bash
git clone https://github.com/zkhechadoorian/Movie-Recommendation-System
```
### Step 2: Setup Virtual Environment

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### Step 3: Install Dependencies

Install necessary Python packages:

```bash
pip install -r requirements.txt
```

# 🚀 Step-by-Step Guide

## 🗂️ File Execution Order & Purpose

### 1. `create_data_matrix.ipynb`

**Purpose:**

* Preprocess raw MovieLens data and build user-item rating matrices. This file combines information from `movies.csv` and `ratings.csv` to create a matrix where row i corresponds to the user with userId i and column j corresponds to the movie with movieId j. Cells (i,j) store the rating from user i for movie j. Cells that store 0 indicate that no rating exists for the given user-movie combination.

Use Jupyter to manually execute cells by clicking `Run All`, or running cells one at a time.

---

### 2. `cf.py`

**Purpose:**

* Implements Neighborhood-based Collaborative Filtering using similarity metrics (e.g., cosine, Pearson). Collaborative Filtering is a technique that uses similarities between users and ratings to suggest new films to a given user. 

**How to Run:**

```bash
python cf.py
```

**How it Works:**

The `cf.py` script takes as input the data matrix created by `create_data_matrix.ipynb`, where rows represent users, columns represent movies, and cells hold numerical values to indicate user ratings for a given film. Missing ratings,  indicated by 0, are replaced with predicted values based on similarities between other users and films. After matrix completion, the script identifies top-rated films for a specific user (user-1) and prints the first five titles as their personalized movie recommendations. Each title is accompanied by its predicted rating. The output for user-1 is:

```bash
Green Mile, The (1999)                                                           Predicted Rating: 5.0
Shrek (2001)                                                                     Predicted Rating: 5.0
Léon: The Professional (a.k.a. The Professional) (Léon) (1994)                   Predicted Rating: 5.0
Independence Day (a.k.a. ID4) (1996)                                             Predicted Rating: 5.0
Fifth Element, The (1997)                                                        Predicted Rating: 5.0
```

Movie titles are printed in the same order in which they appear in the user-movie data matrix. 

Internal parameters for the CF technique can be adjusted as needed (e.g., similarity type, top-K neighbors). The CF algorithm only predicts ratings as multiples of 0.5, similar to the allowed user ratings. The distribution of predicted ratings is shown below.

![](./predictions_cf.png)

---

### 3. `ridge.py`

**Purpose:**

* Applies Ridge Regression to estimate user ratings using a linear model and regularization. The algorithm employes Alternating Least Squares (ALS) factorization, which decomposes the sparse user-movie rating matrix into separate user and movie matrices whose shapes are determined by a tunable number of latent features. The multiplication of these two matrices results in a user-movie matrix that stores predicted values. Unlike the previous algorithm, CF, the ALS factorization enables a continuous range of predictions rather than predictions that are strictly mutiples of 0.5. 

**How to Run:**

```bash
python ridge.py
```

You may need to install `scikit-learn` if not already available.

```bash
Sense and Sensibility (1995)                                                     Predicted Rating: 5.00
Clueless (1995)                                                                  Predicted Rating: 5.00
GoldenEye (1995)                                                                 Predicted Rating: 4.99
Seven (a.k.a. Se7en) (1995)                                                      Predicted Rating: 4.99
Usual Suspects, The (1995)                                                       Predicted Rating: 4.99
```

The distribution of ratings predicted by the Ridge algorithm is shown below. Noticeably, the distribution is bimodal with ratings that are either very close to 0 or very close to 5.0. This is a result of applying some threshold ratings below which existing user-movie ratings are discarded prior to ALS factorization. As such, the Ridge model focuses on predicting movie preferences, rather than realistic ratings for all movies.  

![](./predictions_ridge.png)

---

### 4. `nn.py`

**Purpose:**

* Defines the architecture for the neural network used in rating prediction.
* Used as a module, not directly executed.

**No command needed.**

---

### 5. `train_nn.py`

**Purpose:**

* Trains the neural network using user-item matrix data.
* Implements Collaborative Filtering Neural Network (CFNN) for movie recommendations.
* Utilizes the model defined in `nn.py`.

**How to Run:**

```bash
python train_nn.py
```

Ensure PyTorch is installed:

```bash
pip install torch torchvision
```

---

### 6. `svm.py`

**Purpose:**

* Uses Support Vector Machines to predict ratings.
* Ratings are classified into one of three categories: ratings above threshold (hard-coded as 3.5), ratings below threshold, and missing ratings. 
* The SVM model is trained to predict ratings in each movie column of the user-item matrix using other columns as features.
* Predictions are iteratively refined until convergence.

**How to Run:**

```bash
python svm.py
```

Install dependencies if needed:

```bash
pip install scikit-learn
```

---

### 7. `util.py`

**Purpose:**

* Contains utility functions shared across models (e.g., similarity calculation, evaluation metrics).

**No command needed.** Used as a helper module by other scripts.

---
Comparison of Results
---

```
CF Recommendations:
    Clerks (1994)
    Shallow Grave (1994)
    Highlander III: The Sorcerer (a.k.a. Highlander: The Final Dimension) (1994)
    In the Line of Fire (1993)
    Santa Clause, The (1994)
```

```
Ridge Recommendations:
    Dunston Checks In (1996)
    Sudden Death (1995)
    Don't Be a Menace to South Central While Drinking Your Juice in the Hood (1996)
    Dead Presidents (1995)
    Balto (1995)
```

```
NN Recommendations:
```

```
SVM Recommendations:
    Savage Nights (Nuits fauves, Les) (1992)
    It Takes Two (1995)
    Secret Garden, The (1993)
    Hoop Dreams (1994)
    The Glass Shield (1994)
```

## 📌 Notes

* All data files should reside in the `data/` directory.
* Execute models independently to compare results.
* Output predictions or evaluation results will typically be printed to console or saved as files (you may customize).

Feel free to adapt file parameters and code structures to fine-tune model performance.
