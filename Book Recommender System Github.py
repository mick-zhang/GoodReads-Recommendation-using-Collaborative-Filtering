#!/usr/bin/env python
# coding: utf-8

# # 1 Business Problem

# With the emergence of Machine Learning and AI, we're now able to utilize these tools and make predictions using big data at a level that we could have never previously imagined. However, despite all these tools that are available at our disposal, it is still important on how we apply the data in a business context, and how we make the most out of it.
# 
# For retailers, more specifically book retailers, we can work with big data using consumer reviews from books that were purchased, and then use those ratings to make recommendations, for similar books that interests our consumers, with the goal of recurring businesses and improved sales.
# 
# The dataset that we are using is from "[GoodReads](https://www.goodreads.com/)," which is a website that allows users to freely search their database for books and book reviews.
# <br>
# <br>
# 
# **Business Solution:**
# - We will use the user-based Collaborative Filtering model to make predictions and recommend books to consumers
# - This model will determine similar interests from other consumers, for making our predictions and recommendations
# <br>
# <br>
# 
# **Benefits:**
# - Improve marketing strategies
# - Better inventory control for each products
# - Higher sales and revenues
# <br>
# <br>
# 
# **Robustness:**
# - We are using the Euclidean Distance Model to determine the similarities between consumers, from products that each consumers previously rated
# - The Euclidean Distance Model measures the closest proximity among users, which translate to similar interests for consumer book ratings
# <br>
# <br>
# 
# **Assumptions:**
# - We're assuming that 300K+ samples in our data is sufficient to determine the accuracy of our model for making predictions and recommendations
# - We're assuming that the sample size in our data is similar to the size of a real word data
# <br>
# <br>
# 
# **Future:**
# <br>
# 
# This model is an introductory model for creating a recommendation system. In the future, I will use Affinity Analysis for recommendations on cross product selling.

# # 2 Data Overview

# In[1]:


import pandas as pd
csv = "br.csv"
df = pd.read_csv(csv, engine='python', error_bad_lines=False)
df.head(20)


# Here we see that our dataset includes numerous columns, but for the purpose of building our recommendation system, we will only use the following columns:
# - reviewerName
# - title
# - reviewerRatings
# 
# <u>Note</u> that line 312,075 has an "error bad line" which we will skip, since one row does not have significant impact on the entirety of the 300K+ sample size.

# In[2]:


# Filters the items that we need
df = df.filter(items=['reviewerName', 'title', 'reviewerRatings'])
df.head(20)


# ## 2.1 Data Cleaning

# Now that our dataset is filtered down to the columns that we will be using for our model, our next step will be data cleaning by performing the following:
# - Check for missing data
# - Remove all data that has missing values since we cannot make predictions without actual reviewerName, title or reviewerRatings
# - Remove all non-ASCII characters as we see a lot of symbols that we cannot interpret
# - Reset the index of our DataFrame after data is cleaned so that all rows are indexed sequentially 

# In[3]:


# Checks for missing value for each column
column = ['reviewerName', 'title', 'reviewerRatings']
for columns in column:
    missing = df[columns].isnull().value_counts()
    print(missing)


# In[4]:


# Drops missing values for all columns
df = df.dropna(how='any')

# Remove rows with non-ASCII characters in reviewerName and title column
df = df[~df.reviewerName.str.contains(r'[^\x00-\x7F]')]
df = df[~df.title.str.contains(r'[^\x00-\x7F]')]

# Resets the index
df = df.reset_index(drop=True)

# Check and see the cleaned data
df.head(20)


# ## 2.2 Data Analysis

# Now that our data is cleaned, we have a remaining sample size of 217K+ which is still large enough to make meaningful recommendations. We will also analyze our data briefly, before feeding the data into our model.

# In[5]:


# Check how many times each books are rated
from collections import Counter
Counter(df['title'].head(20))


# Here we see that there are books that got rated multiple times.

# In[6]:


# See how many times different reviewers rated the same title
df.groupby("reviewerName")["title"].unique().head(20)


# Here we see that there are not many books that were rated by different users.

# In[7]:


# See all the different users that rated "Anne of Avonlea" multiple times
same_names = df[df['title'] == 'Anne of Avonlea']["reviewerName"].unique()
for name in same_names:
    print(name)


# Based on our analysis so far:
# - First we checked to see if certain books were rated multiple times
# - Then we confirmed if the books were rated multiple times by different users
# - Next we took a random book called "Anne of Avonlea," which has 12 separate ratings, and checked to see how many users rated that book
# - As a result, we see that only "Maureen" rated the book "Anne of Avonlea," with an occurrence of 12 ratings
# 
# The purpose of our analysis was to confirm if there were two users who rated the same book multiple times, so that we can use theses reviews and see how close they rank amongst each other, using the Euclidean Distance Model.
# 
# Since we were not able to <u>manually/experimentally</u> find a book that were rated by the two users above, then our next step will be creating a function for finding the Euclidean Distance between "Maureen" and other users with similarly rated books. Once we have a list of these Euclidean Distances, we can then rank the similarity levels for recommendations.
# 
# <u>Note</u>: We have reduced the number of rows down to 20 rows for more web friendly displays. As a result, you may not see the "Anne of Avonlea" book in the output above. However, in order to display the full dataset, we just need to remove the `.head(20)` method in each of the three lines under "Data Cleaning" and "Data Analysis" section.

# ## 2.3 Data Preparation

# In order to fit the data into our model, we need to manipulate our `DataFrame` as following:
# 
#     {
#     reviewerName: { title : reviewerRatings }
#     }
#     
# Next, we will transform our `DataFrame` into a dictionary for model testing.
# 
# To visualize this, we will use the `sort_index()` method below.

# In[8]:


# Filters the unique reviewerName for their corresponding title and reviewRatings
df1 = df.set_index(['reviewerName', 'title']).sort_index()
df1.head(20)


# In[9]:


# Converts dataframe to dictionary
d = (df.groupby('reviewerName')['title','reviewerRatings'].apply(lambda x: dict(x.values)).to_dict())


# In[10]:


# Prints the first 20 items in our dictionary
n = 20
{key:value for key,value in list(d.items())[0:n]}


# <u>Note</u> that we may run into an error showing "IOPub data rate exceeded", this issue can be solved [here](https://stackoverflow.com/questions/43288550/iopub-data-rate-exceeded-in-jupyter-notebook-when-viewing-image).
# 
# Now that we finished cleaning, analyzing and preparing our data, we will build a function to run our model.

# # 3 Modeling

# The model we are using to determine similarities between users is the **Euclidean Distance Model**, which measures the distance between users, and takes calculates the closest distance for similarity considerations.
# 
# Here is the formula that we are going to use:
# 
#     dist((x,y),(a,b)) = √((x - a)² + (y - b)²)
#    
# First we will create a "sim_distance" function by implementing the Euclidean Distance formula between two users. We will also add the results by 1 and then divide it by 1 so the Euclidean Score is between 0 to 1.
# 
# Next, we will create a "top_matches" function, while also using the "sim_distance" function to determine the list of all other users who are similar to "Maureen". This is also where we are able to find the Euclidean Distance between "Maureen" and other users without <u>manually/experimentally</u> finding two users that rated the same book.
# 
# Finally, we will put everything together and create a "get_recommendations" function as our recommendation system. To create our recommendation system, we will use the following approach:
# - Firstly, we will calculate the similarity score between all users that are closest to "Maureen" using the "sim_distance" function, this will provide us with a similarity score for each user that is similar to "Maureen"
# - Secondly, we will take the similarity score of each user and multiply it by the book rating for the same user, on a book which "Maureen" has yet to rate
# - Now, we have a obtained a new value called rated similarity score for each user
# - Thirdly, we will add the rated similarity score for all users as our sum of rated similarity score, as well as adding the similarity score for all users as our sum of similarity score
# - Fourthly, we will divide the sum of rated similarity score by the sum of similarity score based on ratings by all other users, to determine and recommended books that "Maureen" would be interested in purchasing 
# 
# 
# We will repeat this process for each book in our "get_recommendations" function for predicting and recommending similar books that "Maureen" is interested in.

# In[11]:


# Returns a distance-based similarity score for person1 and person2
def sim_distance(prefs,person1,person2):
    si = {} 
    for item in prefs[person1]:
        if item in prefs[person2]:
            si[item]=1    
    if len(si) == 0: 
        return 0
    sum_of_squares = sum([pow(prefs[person1][item] - prefs[person2][item],2) 
                          for item in prefs[person1] if item in prefs[person2]])
    return 1/(1+sum_of_squares)

# Checks for similarity to Maureen using Euclidean distance score
def top_matches(prefs, person, n=10, similarity = sim_distance):
    scores = [(similarity(prefs,person,other), other)
            for other in prefs if other!=person]
    scores.sort()
    scores.reverse()
    return scores[0:n]

# Gets recommendation using a weighted average of all other users using Euclidean distance score
def get_recommendations(prefs, person, n=10, similarity = sim_distance):
    totals = {} 
    simSums = {}
    for other in prefs:
        if other == person:
            continue
        sim = similarity(prefs, person, other)
        if sim <= 0:
            continue
        for item in prefs[other]:
            if item not in prefs[person] or prefs[person][item] == 0:
                totals.setdefault(item,0) 
                totals[item] += prefs[other][item] * sim
                simSums.setdefault(item,0)
                simSums[item] += sim
    rankings = [(total/simSums[item], item) for item, total in totals.items()]
    rankings.sort()
    rankings.reverse()
    return rankings[0:n]


# # 4 User Similarities and Recommendations

# In[12]:


# Top 10 similar users to Maureen
top_matches(d, 'Maureen')


# In[13]:


# Top recommendation for Maureen
get_recommendations(d, 'Maureen')


# After running our model, we see the top 10 users that are similar to Maureen's preferences using the Euclidean Distance. Also, we made a list of recommended books for "Maureen" based on the similarities of all other users to "Maureen", and their respective books that they rated.

# # 5 Answering the Questions

# From the recommendations above, we see that the model was able to recommend books to a consumer that has yet to read/rate, based on similarities of other consumers and their respective book ratings.
# 
# With these recommendations, we have created a model that can make predictions and recommend books to consumers.
# 
# In addition, we also determined similar preferences of other consumers in comparison to our the original consumer, and vice versa.
# 
# Lastly, based on our model's ability make recommendations and determine interests of other consumers, we conclude that our model can effective help a retail book store improve their marketing strategies, and control inventory, which can potentially translate into higher sales and profits.

# # 6 Special Thanks

# Special thanks to Samir Madhavan for inspirations of creating this project, which we were able to apply it to real world data. Some of the earlier steps were similarly replicated from his book, Mastering Python for Data Science.
# 
# ![Book Cover](files/book_cover.jpg)
