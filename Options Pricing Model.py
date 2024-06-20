import numpy as np

class OptionPricing:
    def __init__(self, S0,E,T,rf,sigma,iterations):
        #we dont need data points (N) as we will be directly calculating at maturity date , i.e., dt=T
        self.S0 = S0
        self.E = E
        self.T = T
        self.rf = rf
        self.sigma = sigma
        self.iterations = iterations

    def call_option_price(self):
        #2 columns, one with 0 and the other with payoffs (S-E) to calculate max(0,S-E)
        option_data=np.zeros([self.iterations,2])
        #as we are calculating directly at T with dt=T and no other data point in b/w, we will have only W(0)=0 and W(T)=N(0,T) i.e. only 1 value for each iteration, therefore only a row matrix in total
        rand=np.random.standard_normal([1,self.iterations])
        stock_price = self.S0 * np.exp(self.T * (self.rf - 0.5 * self.sigma ** 2) + self.sigma * np.cumsum(rand, axis=0)) # we could have also used self.sigma*np.sqrt(T)*rand, same thing.
        option_data[:, 1] = stock_price - self.E
        average = np.sum(np.amax(option_data, axis=1)) / float(self.iterations)
        return np.exp(-1.0 * self.rf * self.T) * average #discount factor (e^-rt) to calculate the present value

    def put_option_price(self):
        #2 columns, one with 0 and the other with payoffs (E-S) to calculate max(0,E-S)
        option_data=np.zeros([self.iterations,2])
        #as we are calculating directly at T with dt=T and no other data point in b/w, we will have only W(0)=0 and W(T)=N(0,T) i.e. only 1 value for each iteration, therefore only a row matrix in total
        rand=np.random.standard_normal([1,self.iterations])
        stock_price = self.S0 * np.exp(self.T * (self.rf - 0.5 * self.sigma ** 2) + self.sigma * np.cumsum(rand, axis=0)) # we could have also used self.sigma*np.sqrt(T)*rand, same thing.
        option_data[:, 1] = self.E - stock_price
        average = np.sum(np.amax(option_data, axis=1)) / float(self.iterations)
        return np.exp(-1.0 * self.rf * self.T) * average #discount factor (e^-rt) to calculate the present value


if __name__ == '__main__':
    option = OptionPricing(100, 100, 1, 0.05, 0.2, 10000)
    print('Value of the call option is $%.2f' % option.call_option_price())
    print('Value of the put option is $%.2f' % option.put_option_price())