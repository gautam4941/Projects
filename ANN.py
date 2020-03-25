import pandas as pd

data = pd.read_csv( r'C:\Users\gauta\PycharmProjects\DeepLearning\2) K-Fold Cross Validation\Churn_Modelling.csv' )
print( f"data.head()\n :- { data.head() }\n" )

print( f"data.isnull().sum():- \n{ data.isnull().sum() }\n" )

x = data.loc[ :, 'CreditScore' : 'EstimatedSalary' ]
y = data.loc[ :, 'Exited' ]
print( f"x.head() :- \n{ x.head() }\n" )
print( f"y.head() :- \n{ y.head() }\n" )

print( f"x.dtypes :- \n{ x.dtypes }\n" )

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

x['Geography'] = le.fit_transform( x['Geography'] )
x['Gender'] = le.fit_transform( x['Gender'] )

print( f"x.dtypes :- \n{ x.dtypes }\n" )
print( f"x.head() :- \n{ x.head() }\n" )
print( f"x.shape :- \n{ x.shape }\n" )

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size = 0.2 )

print( f"x_train :- \n{ x_train }\n" )
print( f"x_test :- \n{ x_test }\n" )
print( f"y_train :- \n{ y_train }\n" )
print( f"y_test :- \n{ y_test }\n" )

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_train = sc.fit_transform( x_train )
x_test = sc.fit_transform( x_test )

print( f"x_train :- \n{ x_train }\n" )
print( f"x_test :- \n{ x_test }\n" )


import matplotlib.pyplot as plt
plt.plot( y, linestyle = '', marker = '*' )
plt.ylabel( 'Y' )
plt.title( 'Y_Graph' )
plt.show()

#Tip :- If you data is linearly Seperable then, ANN is not required. If it non-linearly Seperable then,
#ANN is required

from keras.models import Sequential
from keras.layers import Dense

#Using Sequential we can initialize the ANN Network in which layer will be added.
ann = Sequential()

#Adding the 1st layer -> Input layer and linking to 1st Hidden Layer.
#we need to mention output of input layer as well as input of hidden layer.
#Now the biggest question is ,
    #No. of Input nodes for input layer = no. fo columns in x.
    #No. of Input nodes for Hidden layer = ????

#No. of Input nodes for Hidden layer = ( No. of Input Layer + No. of Output Layers ) / 2
#No. of Input nodes for Hidden layer = ( 10( x.shape ) + 1 ) / 2 = 5.5 = 6

#output_dim = 6
#init = 'uniform' # It will set weights randomly and close to 0 but not 0.
#activation = activation_function(). we need to set relu().

ann.add( Dense( output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 10 ) )

#we are not suppose to provide the input_Layer as in previous layer we have already mentioned
#Here input layer will be the output of previous hidden layer
ann.add( Dense( output_dim = 6, init = 'uniform', activation = 'relu' ) )

#Now we have 3 layer:- 1 Input layer and 2 Hidden Layer. Now we need to create last and one more layer i.e.,
#output_layer
#Here, we have to mention output_dim = 1 as at last we need answer either 0 or 1.
#Here, we have to mention activation = sigmoid as at last we need answer either 0 or 1.
#In case if we have more than 2 categories then, we need to set output_dim = possible types of output
#and activation = softmax
ann.add( Dense( output_dim = 1, init = 'uniform', activation = 'sigmoid' ) )

#Why to Compile ?
#Ans :
#optimizer is used to find optimal set of weights in neural network.
#loss function is the value of error using which ANN will understand what should be next optimum weight :
#    Bascially, loss = ( y - y_pred ) ** 2
#we are choosing binary_crossentropy as we have outcome either 0 or 1
#If we have more than 2 possible outcome then, we need to choose categorical_crossentropy
ann.compile( optimizer='adam', loss = 'binary_crossentropy', metrics = [ 'accuracy' ] )

#batch_size is 10 means after every 10 rows update the weights. No. of Bacth = 8000/10 = 800 bacthes
#nb_epoch = 100 means Once 800 bacthes or ANN passes all the rows of our train dataset then
#again it will pass the same train_dataset in same mentioned batch manner and this will be done for 100
#times.
ann.fit( x_train, y_train, batch_size = 10, nb_epoch = 100 )

#This is not the final answer. This is the prediction answer
y_pred = ann.predict( x_test )
print( f"y_pred :- \n{ y_pred }\n" )

#It will give us answer in True/False. Where True means customer will leave the bank and False means
#customer will stay with the bank.
y_pred = y_pred > 0.5

print( f"y_pred :- \n{ y_pred }\n" )

plt.plot( y_test, label = 'y_test', marker = '*', linestyle = '' )
plt.plot( y_pred, label = 'y_pred', marker = '*', linestyle = '' )
plt.ylabel( "y_test V/s y_pred" )
plt.title( 'Actual V/s Prediction Graph' )
plt.legend()
plt.show()

from sklearn.metrics import confusion_matrix
cm = confusion_matrix( y_test, y_pred )
print( cm )
