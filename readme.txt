This is our 3rd year 2nd term project in NN, CUFE

to run the project please follow the instructions:
    1. install all dependancies in requirements.txt

    2. run the file "project.ipynb" as follows:
        2.a set the values of hyperparameters 
                data_path    : the path to the images set
                kmeans_sets  : the number of words in the "bag of words" algorithm
                number_of_estimators : number of weak esitmators used in AdaBoost / XGBoost
                train_test_ratio : the train set to test set ratio 
        2.b after running the first cell, you will need to load the images, to do this, run the second
            cell
        2.c the 3rd cell, just defines some functions, and tests thems
        2.d running the 4th cell will extract all features from images (this can be skipped if you plan to
            use pre-built models)
        2.e for the next two cells, you have option between the following:
                first cell: read the kmeans from provided models
                second cell: calculate the kmeans from loaded images (only do if you claculated features in 2.d)
        2.f the next 3 cells, you select the classifier you want to use:
                1. load classifier from file
                2. train AdaBoost from images features
                3. train XGBoost from images features
        2.g now you can run the tests and check the accuracy of the classifier
        2.h the next cell is just for outputing the kmeans and the classifier into files

    3. testing with the deployed server
        we already have deployed a testing server on http://xabdomo.pythonanywhere.com/ , you can just go and
        use the simple UI we created to test images.
    
    4. testing with local server:
        after performing items in section 2 and generating the models files, head to /server and set the models paths
        then run server.py using "python server.py", this will run a local flash server similar to the one we deployed.

    5. testing deployed server with script:
        head to /server and run "python server_post.py *image_file_path*"
        ex:
            python server_post.py my_image.jpeg
    
    6. batch testing:
        we provived a simple utility to test the generated models against large numbers of images (to calculte the 
        accuracy of a model), to use it, head to /analysis and run the following command 
        "python analysis.py path_to_folder_containing_images"
        ex:
            python analysis.py /data    where "data" is a folder that contains images to be tested
        it will generate time.txt which will contain the time taken by each image to be 
        and results.txt which will contain the classification result for each image 