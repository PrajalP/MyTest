import numpy as np
from sklearn import linear_model, datasets, tree
import matplotlib.pyplot as plt
number_of_samples = 100
x = np.linspace(-np.pi, np.pi, number_of_samples)
y = 0.5 * x + np.sin(x)+np.random.random(x.shape)
plt.scatter(x,y,color='black')
plt.show()
random_indices = np.random.permutation(number_of_samples)
x_train = x[random_indices[:70]]
y_train = y[random_indices[:70]]
x_val = x[random_indices[70:85]]
y_val = y[random_indices[70:85]]
x_test = x[random_indices[85:]]
y_test = y[random_indices[85:]]
maximum_depth_of_tree = np.arange(10)+1
train_err_arr = []
val_err_arr = []
test_err_arr = []
for depth in maximum_depth_of_tree:
    model = tree.DecisionTreeRegressor(max_depth=depth)
    x_train_for_line_fitting = np.matrix(x_train.reshape(len(x_train),1))
    y_Train_for_line_fitting = np.matrix(y_train.reshape(len(y_train), 1))
    model.fit(x_train_for_line_fitting, y_Train_for_line_fitting)
    plt.figure()
    plt.scatter(x_train, y_train, color='black')
    plt.plot(x.reshape((len(x), 1)), model.predict(x.reshape((len(x), 1))), color='blue')
    plt.title('Line fit to training data with max_depth='+str(depth))

    plt.show()
    mean_train_error = np.mean( (y_train - model.predict(x_train.reshape(len(x_train),1)))**2 )
    mean_val_error = np.mean( (y_val - model.predict(x_val.reshape(len(x_val),1)))**2 )
    mean_test_error = np.mean( (y_test - model.predict(x_test.reshape(len(x_test),1)))**2 )
    
    train_err_arr.append(mean_train_error)
    val_err_arr.append(mean_val_error)
    test_err_arr.append(mean_test_error)

    print 'Training MSE: ', mean_train_error, '\nValidation MSE: ', mean_val_error, '\nTest MSE: ', mean_test_error
