import pandas as pd
from sklearn.utils import shuffle
import os

# Getting my watchlist
pd.set_option('display.max_columns', None)

my_list = pd.read_csv("Data/Raw-Data/my_watchlist.csv", encoding='unicode_escape')

# Looking how it looks
my_list.head()
# I will keep the title and description
my_list = my_list.loc[:, ["Title", "Description", "IMDb Rating", "My Rate"]]
my_list.head()  # Sounds nice
# Now time to get the data from IMDB
top_list = pd.read_csv("Data/Raw-Data/Top 1000 IMDB movies.csv")
top_list.head()
# I will keep the title and description as well
top_list = top_list.loc[:, ["Description", "Movie Name", "Movie Rating"]]
top_list.head()
# Better to shuffle the data now
my_list = shuffle(my_list)
top_list = shuffle(top_list)
# We should look at our data
my_list.describe()
len(my_list["Title"].unique())
len(my_list["Description"].unique())

prep_data = my_list

# we will extract it to another folder to keep things organized
# curr_month = "2022-08" # Will be used to put data into sub-dir
prep_data_dir = "Data/Prep-Data/"  # +curr_month
# test_dir = "Data/Test-Data/"#+curr_month
if not os.path.exists(prep_data_dir):
    os.makedirs(prep_data_dir)

prep_data.to_csv(prep_data_dir + "/prep_data.csv",
                 index=False)  # Saving with index will cause problems with the model, it will take it as an inpur

print("finished executing script")
