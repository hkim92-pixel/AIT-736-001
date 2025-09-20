HW 4 - Comparing Machine Learning Approaches: Linear Models, Decision Trees, and SVMs

Machine learning has been one of the main topics in class, and I’ve been learning about different models that can be used depending on the type of data and problem. Three of the methods that stood out to me were linear models, decision trees, and support vector machines (SVMs). Each one has its own way of making predictions, and through readings and exercises, I’ve gotten to see where they work well and where they don’t.

Starting with linear models, these are probably the most straightforward to understand. They assume there’s a straight-line relationship between the features and the output. For example, linear regression tries to fit the best line through the data to predict continuous outcomes, while logistic regression is used for classification. What I liked about linear models is how interpretable they are. you can look at the coefficients and understand how each feature contributes to the prediction. They are also very fast to train, which makes them practical when you just need a simple baseline. On the downside, linear models aren’t great when the data has nonlinear relationships. In class, we worked on a taxi dataset where the goal was to predict fares from trip distances. The linear model captured the basic trend, but it didn’t really handle things like traffic delays or unusual routes. This showed me that while the model was simple and easy to explain, it missed some of the complexity in the real-world data.

Here is an example of the linear regression model.
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([6, 9, 12, 15, 20])  # pretend taxi fares

model = LinearRegression().fit(X, y)

plt.scatter(X, y, color="blue", label="Data")
plt.plot(X, model.predict(X), color="red", label="Linear Fit")
plt.title("Taxi Fare vs Trip Distance")
plt.xlabel("Trip Distance")
plt.ylabel("Fare")
plt.legend()
plt.savefig("linear_regression_taxi.png")
plt.show()

![Linear Regression](LinearRegression.png)

Decision trees, on the other hand, work in a very different way. Instead of fitting a line, they split the data based on questions like “is the feature greater than this value?” Each split makes the tree more specific until you end up at a prediction. I liked trees because they are intuitive. You can actually trace the path and explain why the model made a decision. They also don’t need much preprocessing and can handle both numbers and categories. But the main problem I saw is that they can easily overfit, especially when the tree is too deep. Small changes in the data can also change the whole structure of the tree. In one of our exercises with Random Forests, I saw how combining many trees helped with this problem. A single decision tree might memorize the data too much, but a random forest balances it out by averaging the predictions from many trees. That made the results more stable and accurate, which really highlighted why ensembles are often preferred over a single tree.

SVMs were probably the trickiest for me to wrap my head around at first. The main idea is that they try to separate classes by finding the best boundary that maximizes the margin between the closest points. What makes SVMs powerful is the “kernel trick,” which allows the algorithm to separate data that isn’t linearly separable by transforming it into higher dimensions. In class, I tested both linear and RBF kernels on a tabular dataset. The linear kernel worked okay, but the RBF kernel was able to capture more complex boundaries, which improved accuracy. The drawback I noticed is that SVMs take longer to train and are harder to interpret compared to trees or linear models. Tuning the hyperparameters like gamma also made a big difference and it wasn’t always clear what the best values should be.

Comparing the three methods side by side, I noticed each has its strengths and weaknesses. Linear models are simple and fast but can be too limited when the data isn’t linear. Decision trees are easy to explain and flexible, but they overfit easily. SVMs works very well especially when the data is high-dimensional or nonlinear, but it can be slow to run and harder to explain. In practice, I think the choice depends a lot on the situation. If I needed a quick answer, I’d start with a linear model. If I needed clear rules, like in financial institutions, a decision tree maybe would make more sense. If the data was complex and accuracy mattered most like scientific research, then I’d probably lean toward SVMs.

Overall, working with these three models gave me a good understanding of the tradeoffs in machine learning. I realized there isn’t a one fits all model for everything. It really depends on the underlying problem, the data, and what’s most important to solve. Seeing them actually applied to datasets like the taxi regression and the Random Forest vs. SVM comparisons helped me understand the strengths and limits of each method beyond just reading the theory.


