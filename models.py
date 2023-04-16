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
        self.W1 = nn.Parameter(1,10) #1st layer
        self.b1 = nn.Parameter(1,10) 
        self.W2 = nn.Parameter(50,1) #3rd layer
        self.b2 = nn.Parameter(1,1)
        self.W3 = nn.Parameter(10,30) #2nd layer
        self.b3 = nn.Parameter(1,30)
        self.W4 = nn.Parameter(30,50) #4th layer
        self.b4 = nn.Parameter(1,50)
        self.Learn = -0.001

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        print(x)
        print("-------------------------------")
        xa = nn.Linear(x, self.W1)
        xb = nn.AddBias(xa, self.b1)
        xx = nn.ReLU(xb)
        
        xe = nn.Linear(xx, self.W3)
        xf = nn.AddBias(xe, self.b3)
        x4 = nn.ReLU(xf)

        xg = nn.Linear(x4, self.W4)
        xh = nn.AddBias(xg, self.b4)
        x5 = nn.ReLU(xh)

        xc = nn.Linear(x5, self.W2)
        xd = nn.AddBias(xc, self.b2)
        predicted_y = xd
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
            total_loss = 0
            total_batches = 0
            for x, y in dataset.iterate_once(25):
                total_batches += 1
                loss = self.get_loss(x,y)
                total_loss += nn.as_scalar(loss)
                params = [self.W1,self.b1,self.W2,self.b2,self.W3,self.b3,self.W4,self.b4]
                grad_W1,grad_b1,grad_W2,grad_b2,grad_W3,grad_b3,grad_W4,grad_b4 = nn.gradients(params,loss)
                self.W1.update(self.Learn, grad_W1)
                self.b1.update(self.Learn, grad_b1)
                self.W2.update(self.Learn, grad_W2)
                self.b2.update(self.Learn, grad_b2)
                self.W3.update(self.Learn, grad_W3)
                self.b3.update(self.Learn, grad_b3)
                self.W4.update(self.Learn, grad_W4)
                self.b4.update(self.Learn, grad_b4)
            avg_loss = total_loss / total_batches
            if avg_loss <= 0.02:
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

    def train_model(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

