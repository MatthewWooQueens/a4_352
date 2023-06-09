import nn

class PerceptronModel(object):
    def __init__(self, dim):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dim` is the dimensionality of the data.
        For example, dim=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dim)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x_point):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x_point: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.get_weights(), x_point)

    def get_prediction(self, x_point):
        """
        Calculates the predicted class for a single data point `x_point`.

        Returns: -1 or 1
        """
        "*** YOUR CODE HERE ***"
        dot = nn.as_scalar(self.run(x_point))
        if dot >= 0.0:
            return 1
        else:
            return -1

    def train_model(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        noError = False
        while not noError:
            noError = True
            for x, y in dataset.iterate_once(1):
                pred = self.get_prediction(x)
                if pred != int(nn.as_scalar(y)):
                    self.get_weights().update(pred * -1, x)
                    noError = False

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.W1 = nn.Parameter(1,20)
        self.B1 = nn.Parameter(1,20)
        self.W2 = nn.Parameter(20,40)
        self.B2 = nn.Parameter(1,40)

        self.W3 = nn.Parameter(40,1)
        self.B3 = nn.Parameter(1,1)
        self.Learn = 0.001

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"

        x1 = nn.Linear(x, self.W1)
        x1b = nn.AddBias(x1, self.B1)
        xx = nn.ReLU(x1b)

        x2 = nn.Linear(xx, self.W2)
        x2b = nn.AddBias(x2, self.B2)
        xx2 = nn.ReLU(x2b)

        x3 = nn.Linear(xx2, self.W3)
        x3b = nn.AddBias(x3, self.B3)
        predicted_y = x3b
        return predicted_y
        #dont use dot product
        
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        pred = self.run(x)
        return nn.SquareLoss(pred,y)
        #USe squareloss
    def train_model(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        BestFit = False
        while not BestFit:
            count = 0
            totalLoss = 0
            for x, y in dataset.iterate_once(25):
                loss = self.get_loss(x,y)
                totalLoss += nn.as_scalar(loss)
                count+=1

                grad_W1,grad_b1,grad_W2,grad_b2,grad_W3,grad_b3 = nn.gradients([self.W1,self.B1,self.W2,self.B2,self.W3,self.B3],loss)
                self.W1.update(-self.Learn, grad_W1)
                self.B1.update(-self.Learn, grad_b1)
                self.W2.update(-self.Learn, grad_W2)
                self.B2.update(-self.Learn, grad_b2)
                self.W3.update(-self.Learn, grad_W3)
                self.B3.update(-self.Learn, grad_b3)

            if(totalLoss/count < 0.001):
                BestFit = True

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.W1 = nn.Parameter(784,150)
        self.B1 = nn.Parameter(1,150)
        self.W2 = nn.Parameter(150,50)
        self.B2 = nn.Parameter(1,50)
        self.W3 = nn.Parameter(50,50)
        self.B3 = nn.Parameter(1,50)
        self.W4 = nn.Parameter(50,10)
        self.B4 = nn.Parameter(1,10)
        self.Learn = -0.1

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        L1 = nn.ReLU(nn.AddBias(nn.Linear(x,self.W1),self.B1))
        L2 = nn.ReLU(nn.AddBias(nn.Linear(L1,self.W2),self.B2))
        L3 = nn.ReLU(nn.AddBias(nn.Linear(L2,self.W3),self.B3))
        pred = nn.AddBias(nn.Linear(L3,self.W4),self.B4)
        return pred

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        pred = self.run(x)
        return nn.SoftmaxLoss(pred, y)

    def train_model(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        BestFit = False
        epoch = 0
        while not BestFit:
            epoch+=1
            for x, y in dataset.iterate_once(75):
                loss = self.get_loss(x,y)
                grad_W1,grad_b1,grad_W2,grad_b2,grad_W3,grad_b3,grad_W4,grad_b4 = nn.gradients([self.W1,self.B1,self.W2,self.B2,self.W3,self.B3,self.W4,self.B4],loss)
                self.W1.update(self.Learn, grad_W1)
                self.B1.update(self.Learn, grad_b1)
                self.W2.update(self.Learn, grad_W2)
                self.B2.update(self.Learn, grad_b2)
                self.W3.update(self.Learn, grad_W3)
                self.B3.update(self.Learn, grad_b3)
                self.W4.update(self.Learn, grad_W4)
                self.B4.update(self.Learn, grad_b4)

            if(dataset.get_validation_accuracy() > 0.975):
                BestFit = True
        print("Epoch: " + str(epoch))


#10:15:45