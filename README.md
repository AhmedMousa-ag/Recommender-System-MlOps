# Recommender-System-MlOps
Using software practice to initiate Recommender system for what to watch next using IMDb data and my personal watchlist to predict my personal rating.

Used feature: 1- Movie description

              2- IMDb rating
              
Used pretrained tf_hub universal embedding to preprocess string feature "Description" and a Biderctional layer.

![first_recommender_plot_model](https://user-images.githubusercontent.com/59775002/190160559-2749f7d8-8cb3-428f-a4b7-f41204af2cea.png)


# How repo works
*- Runing main file will run all process "No jupyter, only .py file".

*- Utils folder where we define our pipleine components and having a few other functions to help in our journey.

*- We can explore our data inside jupyter notebook with statistics gen inside our Data folder.

*- Piple line using transform, tuning and training modules which can be find inside Modules folder.

*- Config file where we can define training hyperparameters and other required locations.

