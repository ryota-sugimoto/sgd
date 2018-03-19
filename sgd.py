import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class Optimizer:
  def optimize(self, threshold=10e-6, max_iteration=10e6):
    prev = [self.var]
    self.one_step()
    i = 0
    while np.linalg.norm(self.var-prev[-1]) > threshold and i <= max_iteration:
      prev.append(self.var)
      self.one_step()
      i += 1
    return np.array(prev)

class SGD(Optimizer):
  def __init__(self, df, init, eta=0.01):
    self.df = df
    self.var = init
    self.eta = eta
  
  def one_step(self):
    self.var = self.var - self.eta*self.df(*self.var)
    return self.var
  

class Moment(Optimizer):
  def __init__(self, df, init, eta=0.01, alpha=0.9):
    self.df = df
    self.var = init
    self.eta = eta
    self.alpha = alpha
    self.v = np.zeros_like(init)
  
  def one_step(self):
    self.v = self.alpha*self.v - self.eta*self.df(*self.var)
    self.var = self.var + self.v
    return self.var
 
class AdaGrad(Optimizer):
  def __init__(self, df, init, eta=0.01, e=10e-7):
    self.df = df
    self.var = init
    self.eta = eta
    self.e = e
    self.h = np.zeros_like(init)

  def one_step(self):
    grad = self.df(*self.var)
    self.h = self.h + grad**2
    self.var = self.var - self.eta*grad/(np.sqrt(self.h)+self.e)
    return self.var

class Adam(Optimizer):
  def __init__(self, df, init, eta=0.01, beta1=0.9, beta2=0.999, e=10e-8):
    self.df = df
    self.var = init
    self.eta = eta
    self.beta1 = beta1
    self.beta2 = beta2
    self.e = e
    self.m = np.zeros_like(init)
    self.v = np.zeros_like(init)
    self.t = 0
  
  def one_step(self):
    self.t += 1
    grad = self.df(*self.var)
    self.m = self.beta1*self.m + (1-self.beta1)*grad
    self.v = self.beta2*self.v + (1-self.beta2)*grad**2
    e_m = self.m/(1-self.beta1**self.t)
    e_v = self.v/(1-self.beta2**self.t)
    self.var = self.var - self.eta*e_m/(np.sqrt(e_v)+self.e)
    return self.var

if __name__ == '__main__':
  def f(x,y):
    return 1/20*x**2 + y**2

  def df(x,y):
    return np.array((1/10*x, 2*y))

  def sgd_plot(handle, Optimizer):
    x = np.arange(-10, 10.1, 0.1)
    y = np.arange(-10, 10.1, 0.1)
    X, Y = np.meshgrid(x,y)
    Z = np.vectorize(f)(X,Y)

    init = np.array((-7.0, -3.0))
    handle.contour(X,Y,Z,levels=[0.1*np.e**n for n in range(20)],
                   linewidths=0.1,
                   colors='black')
    num_experiments = 5
    colors = cm.jet(np.linspace(0,1,num_experiments))
    line_handles = []
    for i in range(num_experiments):
      eta = 0.8*0.5**i
      opt = Optimizer(df, init, eta)
      vars = opt.optimize()
      line_handle, = handle.plot(vars[:,0], vars[:,1], c=colors[i],
                        label='eta={}, steps={}'.format(eta,len(vars)))
      line_handles.append(line_handle)
    handle.legend(handles=line_handles)

  fig = plt.figure()
  fig.suptitle('Comparison of Gradient Decient Methods (threshold=10e-6)')
  plt1 = fig.add_subplot(221)
  plt1.set_title('SGD')
  sgd_plot(plt1, SGD)
  plt2 = fig.add_subplot(222)
  plt2.set_title('Moment')
  sgd_plot(plt2, Moment)
  plt3 = fig.add_subplot(223)
  plt3.set_title('AdaGrad (alpha=0.9)')
  sgd_plot(plt3, AdaGrad)
  plt4 = fig.add_subplot(224)
  plt4.set_title('Adam (beta1=0.9, beta2=0.999)')
  sgd_plot(plt4, Adam)
  plt.show()
