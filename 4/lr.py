import numpy as np
import argparse

def sigmoid(x : np.ndarray):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (np.ndarray): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)


def train(
    #theta : np.ndarray, # shape (D,) where D is feature dim
    X : np.ndarray,     # shape (N, D) where N is num of examples
    y : np.ndarray,     # shape (N,)
    num_epoch : int, 
    learning_rate : float
):
    theta= np.zeros(len(X[0]))
    bias= 0 
    for i in range(num_epoch):
        for j in range(len(X)):
            gradient = X[j] * (sigmoid(theta@X[j]+bias) - y[j])
            bias-=learning_rate*(sigmoid(theta@X[j]+bias) - y[j])
            theta-=learning_rate*gradient
    return bias,theta

def predict(
    theta, X, bias
):
    result= sigmoid(X@theta + bias)
    for i in range(len(result)):
        result[i]= result[i]>=0.5
    return np.array(result)

def compute_error(
    y_pred : np.ndarray, 
    y : np.ndarray
):
    n=np.sum(y_pred!=y)
    return n/len(y_pred)

if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the learning rate, you can use `args.learning_rate`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to formatted training data')
    parser.add_argument("validation_input", type=str, help='path to formatted validation data')
    parser.add_argument("test_input", type=str, help='path to formatted test data')
    parser.add_argument("train_out", type=str, help='file to write train predictions to')
    parser.add_argument("test_out", type=str, help='file to write test predictions to')
    parser.add_argument("metrics_out", type=str, help='file to write metrics to')
    parser.add_argument("num_epoch", type=int, 
                        help='number of epochs of stochastic gradient descent to run')
    parser.add_argument("learning_rate", type=float,
                        help='learning rate for stochastic gradient descent')
    args = parser.parse_args()

tr= np.loadtxt(args.train_input, delimiter='\t')
val= np.loadtxt(args.validation_input, delimiter='\t')
test= np.loadtxt(args.test_input, delimiter='\t')

def getY(data):
    result=[]
    for row in data:
        result.append(row[0])
    return np.array(result)

def getX(data):
    return np.delete(data, [0], 1)

y= getY(tr)
X= getX(tr)

# train model on training set
thing1,thing2=train(X, y, args.num_epoch, args.learning_rate)

testY= getY(test)
testX= getX(test)
 
trainOut=predict(thing2, X, thing1)
testOut= predict(thing2, testX, thing1)

trainError=compute_error(trainOut, y)
testError=compute_error(testOut, testY)

with open(args.train_out, "w") as f_out:
    for item in trainOut:
        f_out.write(str(int(item)) +"\n")

with open(args.test_out, "w") as f_out:
    for item in testOut:
        f_out.write(str(int(item))+"\n")

with open(args.metrics_out, "w") as f_out:
    f_out.write("error(train): "+ '{:.6f}'.format(trainError)+"\n")
    f_out.write("error(test): " + '{:.6f}'.format(testError)+"\n")