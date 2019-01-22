
# coding: utf-8

# # Framing the Problem

# With the emergence of Machine Learning and AI, we are now able to utilize big data at a level that we could have never previously imagined. However, despite all these tools that are available at our disposal, it is still important on how we apply the data in a business context and how we make the most out of it.
# 
# For retailers, more specifically book retailers, we can work with these big data using consumer reviewers from that was purchased, and then use those ratings to make recommendations for similar books that interests our consumers for recurring businesses and improved sales.
# 
# The dataset that we are using is from **GoodReads**, which is a website that allows users to freely search thier database for books and book reviews.
# <br>
# <br>
# 
# **Business Solution:**
# - I will use an user-based Collaborative Filtering model to make predictions and recommend books to consumers
# - The model is also able to determine similar interests of other consumers that we made our predictions and recommendations for
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
# - We are using the Euclidean Distance Model to determine the similarities between consumers based on all the products that each consumers previously rated
# - The Euclidean Distance Model ensures the closest proximity among users which translate to similar interests among consumers for book ratings
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

# # Data Overview

# In[1]:


import pandas as pd
csv = "br.csv"
df = pd.read_csv(csv, engine='python', error_bad_lines=False)
# use python engine for more feature-complete
# skips error bad line
df.head(20)


# Here we see our dataset includes numerous columns, but for the purpose of building our recommendation system, we will only use the following columns:
# - reviewerName
# - title
# - reviewerRatings
# 
# <u>Note</u> that there is one line that showed "error bad line" which we will skip, since one row does not have significant impact on the entirety of the 300K+ sample size.

# In[2]:


# filters the items that we need
df = df.filter(items=['reviewerName', 'title', 'reviewerRatings'])
df.head(20)


# ### Data Cleansing

# Now that our dataset is filtered to the columns that we are interested in for our model, we will proceed to cleanse our data by performing the following:
# - Check for missing data
# - Remove all data that has missing values since we cannot make predictions without actual reviewerName, title or reviewerRatings
# - Remove all non-ASCII characters as we see a lot of symbols that we cannot interpret
# - Reset the index of our DataFrame after data cleansing so that all rows are indexed sequentially 

# In[3]:


# checks for missing value for each column
for missing in (df['reviewerName'].isnull().value_counts(),
                df['title'].isnull().value_counts(),
                df['reviewerRatings'].isnull().value_counts()):
    print(missing)


# In[4]:


# drops missing values for all columns
df = df.dropna(how='any')

# remove rows with non-ASCII characters in reviewerName and title column
df = df[~df.reviewerName.str.contains(r'[^\x00-\x7F]')]
df = df[~df.title.str.contains(r'[^\x00-\x7F]')]

# resets the index
df = df.reset_index(drop=True)

# check and see the cleansed data
df.head(20)


# ### Data Analysis

# Now that our data is cleansed, we still have a sample size of 217K+ which is large enough to make meaningful recommendations. We also want to analyze our data briefly before feeding the data into our model.

# In[5]:


# see how many times each book comes up rated (calls the collection package)
from collections import Counter
Counter(df['title'].head(20))


# Here we see that there are multiple books that got rated multiple times.

# In[6]:


# see how many times multiple reviewers rated the same title
df.groupby("reviewerName")["title"].unique().head(20)


# Here we see that there are not many books that was rated by multiple users.

# In[7]:


# see all the different users rated this one title that have multiple users
same_names = df[df['title'] == 'Anne of Avonlea']["reviewerName"].unique()
for name in same_names:
    print(name)


# Based on our analysis so far:
# - First we looked to see if certain books were rated multiple times
# - Then we want to see if these certain books that were rated multiple times have different users that rated them
# - Next we took a book called "Anne of Avonlea" which has 12 ratings, and check to see how many users rated that book
# - Lastly, our result showed that only "Maureen" rated the book "Anne of Avonlea" 12 times
# 
# The purpose of our analysis so far was to see if there were two users that rated the same book multiple times, so we can then use those two users and see how close they rank from each other using the Euclidean Distance Model.
# 
# Since we were not able to find a book that was rated by two users manually, we will create a function in our model that will find the Euclidean Distance between "Maureen" and other users, and rank their similarities level in comparison to "Maureen" based on similar books that was previously rated.
# 
# <u>Note</u>: I have reduced the number of rows of data displayed to 20 rows, since it will take a long time to scroll through the results once the project is uploaded online. As a result, you may not see the "Anne of Avonlea" book displayed in the output above. However, in order to display the full dataset, we just need to remove the .head(20) method in each of the three lines under "Data Cleansing" and "Data Analysis" section.

# ### Data Preparation

# In order to fit our DataFrame to our model, we need to structure the data in this form:
# 
#     {
#     reviewerName: { title : reviewerRatings }
#     }
#     
# This applies to all books for each user. Once we manipulated the data into our desired structure, we will then transform our DataFrame into a dictionary for our model testing.
# 
# To visualize this, below is how we want to show our data.

# In[8]:


# filters the unique reviewerName for their corresponding title and reviewRatings
df1 = df.set_index(['reviewerName', 'title']).sort_index()
df1.head(20)


# In[9]:


# converts dataframe to dictionary
d = (df.groupby('reviewerName')['title','reviewerRatings'].apply(lambda x: dict(x.values)).to_dict())
# use groupy with lambda function per reviewerName, then use to_dict to convert from DataFrame to dictionary


# In[10]:


# prints the first 20 key:value of our dictionary
n = 20
{key:value for key,value in list(d.items())[0:n]}


# <u>Note</u> that you may run into an error stating that "IOPub data rate exceeded", you can solve this issue [here](https://stackoverflow.com/questions/43288550/iopub-data-rate-exceeded-in-jupyter-notebook-when-viewing-image).
# 
# Now that we finished cleansing, analyzing and preparing our data, we will build a function to run our model.

# # Modeling

# The model we are using to determine similarities betweens users is the Euclidean Distance Model, which measures the distance between users, and takes the closest distance for similarity considerations.
# 
# Here is the formula that we are going to use:
# 
#     dist((x,y),(a,b)) = √(x - a)² + (y - b)²
#    
# First we will create a "sim_distance" function by implementing the Euclidean Distance formula between two users. We will also add the results by 1 and then divide it by 1 so the Euclidean Score is between 0 to 1.
# 
# Next, we will create a "top_matches" function while also using the "sim_distance" function to determine the list of all other users who are similar to "Maureen". This is also where we are able to find the Euclidean Distance between "Maureen" and other users without manually finding two users that rated the same book.
# 
# Finally, we will put everything together and create a "get_recommendations" function as our recommendation system. To create our recommendation system, we will use the following approach:
# - Firstly, we will calculate the similarity score between all users that are closest to "Maureen" using the "sim_distance" function, this will provide us with a similarity score for each user that is similar to "Maureen"
# - Secondly, we will take the similarity score of each user and multiply it by the book rating for the same user on a book which "Maureen" has yet to rate
# - Now, we have a obtained a new value called rated similarity score for each user
# - Thirdly, we will add the rated similarity score for all users as our sum of rated similarity score, and also add the similarity score for all users as our sum of similarity score
# - Fourthly, we will divide the sum of rated similarity score by the sum of similarity score to determine the recommended rating that "Maureen" would prefer on the specific book that was rated by all other users which "Maureen" has yet to rate
# 
# 
# We will repeat this process for each book in our "get_recommendations" function for predicting and recommending similar books that "Maureen" is interested in.

# In[11]:


# returns a distance-based similarity score for person1 and person2
def sim_distance(prefs,person1,person2):
    # get the list of shared_items
    si = {} 
    for item in prefs[person1]:
        if item in prefs[person2]:
            si[item]=1
    
    # if they have no ratings in common, return 0
    if len(si) == 0: 
        return 0
    
    # add up the squares of all the differences
    sum_of_squares = sum([pow(prefs[person1][item] - prefs[person2][item],2) 
                          for item in prefs[person1] if item in prefs[person2]])
    
    return 1/(1+sum_of_squares)


# checks for similarity to myself (Maureen) using Euclidean distance score
# n is the length of shared items si
def top_matches(prefs, person, n=10, similarity = sim_distance):
    # sets other parameter to exclude myself
    scores = [(similarity(prefs,person,other), other)
            for other in prefs if other!=person]
    
    # sort the list so the higest score appear at the top
    scores.sort()
    scores.reverse()
    return scores[0:n] # slices from first index to last index


# Gets recommendation for a person by using a weighted average of all other users
# Using Euclidean distance score
def get_recommendations(prefs, person, n=10, similarity = sim_distance):
    totals = {} 
    # get the list of each book for sum of similarity score x actual rating
    simSums = {} # get the list of each book for sum of similartiy score
    for other in prefs:
        # don't compare me to myself
        if other == person:
            continue
        sim = similarity(prefs, person, other) # then use sim
        
        # ignore scores of zero or lower
        if sim <= 0:
            continue
        for item in prefs[other]: # item in prefs from sim_distance
            
            # only score books I haven't seen yet
            if item not in prefs[person] or prefs[person][item] == 0:
                # similarity * score
                totals.setdefault(item,0) 
                totals[item] += prefs[other][item] * sim
                # sum of similarities
                simSums.setdefault(item,0)
                simSums[item] += sim
            # setdefault calls the key, and returns 0 if it does not exist
            # similar to get()
            
    # create the normalized list
    rankings = [(total/simSums[item], item) for item, total in totals.items()]
    # total(singular item, sum of (sim*actual ratings for each user)),
    # divide by simSum for each item,
    # and run total and item for each item, 
    # while adding each result in the totals list,
    # then returns a dictionary by calling totals.items()
    # items() returns a list of dictionary
    
    # return the sorted list
    rankings.sort()
    rankings.reverse()
    return rankings[0:n]


# # Similar Users and Recommendation

# In[12]:


# top 10 similar users to Maureen
top_matches(d, 'Maureen')


# In[13]:


# top recommendation for Maureen
get_recommendations(d, 'Maureen')


# After running our model, we see the top 10 users that are similar to Maureen's preferences using the Euclidean Distance. Also, we made a list of recommended books for "Maureen" based on the similarities of all other users to "Maureen", and their respective books that they rated, which "Maureen" has yet to rate them or read them.

# # Answering the Questions

# From the recommendations above, we see that the model was able to recommend books to a consumer that has yet to read or rate, based on similarities of other consumers and their respective book ratings.
# 
# With these recommendations, we have created a model that can make predictions and recommend books to consumers.
# 
# In addition, we also determined others consumers who has similar preferences to the original consumer and vice versa.
# 
# Lastly, based on our model to make recommendations and determine interests of other consumers, we conclude that our model is able to effective help a retail book store improve their marketing strategies and inventory control which can potentially translate into higher sales and profits.

# # Special Thanks

# Special thanks to Samir Madhavan for inspiration on creating this project by applying it to real world data. Some of the earlier steps were similarly replicated from his book, Mastering Python for Data Science.
# 
# ![Book Cover](files/book_cover.jpg)
