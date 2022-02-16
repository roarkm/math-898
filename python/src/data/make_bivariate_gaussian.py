import numpy as np

num_a = 50
class_a_means = [13, 5]
class_a_cov = [[5, .3], [.3, 2]]

class_b_means = [5, 13]
class_b_cov = [[5, .3], [.3, 2]]
num_b = 50

a = np.random.multivariate_normal(class_a_means,
                                  class_a_cov,
                                  num_a)
b = np.random.multivariate_normal(class_b_means,
                                  class_b_cov,
                                  num_b)

np.savetxt('data/a_bivariate_gaussian_normal.csv', a,
           fmt="%f", delimiter=',')
np.savetxt('data/b_bivariate_gaussian_normal.csv', b,
           fmt="%f", delimiter=',')
