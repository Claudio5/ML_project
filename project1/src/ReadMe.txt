Structure of the different files

src/

	- implementations.py : Contains the different ML functions required in the project

	- utils.py : Contains useful functions but not directly related to 
	     	implementations.py functions (i.e standardize() function)

	- cross-validation.py : Contains the cross-validation function

	- run.py: Executable that outputs our predictions in a CSV file. It is using the method that
	  		provides us the best accuracy.

	- Main.ipynb: It's our main jupyter notebook that allowed us to run the different 
			cross-validations for each method aswell as tuning the different parameters.

	- proj1_helpers.py: Helper file given by the project template

	- acc.csv: File containing the accuracy for different lambdas and different polynomial degrees
		    this file is already computed to save up time.

	- pred.csv: File containing the predictions for the testing dataset.


data/
	- train.csv
	- test.csv


To run the executable run.py you need to run it in the src folder in order 

P.S.: Some of the functions in implementations were taken for the corrections of the lab sessions
