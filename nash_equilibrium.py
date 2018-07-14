# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 23:55:15 2018

@author: vibhanshu.singh
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import linear_model
import random

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

class environment():
    def __init__(self, market_demand, price_coeff1, comp_coeff1, intercept1):
        #self.pkgs = packages()
        self.market_demand = market_demand        
        self.price_coeff1 = price_coeff1
        self.comp_coeff1 = comp_coeff1
        self.intercept1 = intercept1
        self.price_coeff2 = -comp_coeff1
        self.comp_coeff2 = -price_coeff1
        self.intercept2 = market_demand - intercept1


class pricing:
    def __init__(self, cost_price1, environment):        
        self.environment = environment
        self.market_demand = environment.market_demand
        self.cost_price1 = cost_price1
        self.price_coeff1 = environment.price_coeff1
        self.comp_coeff1 = environment.comp_coeff1
        self.intercept1 = environment.intercept1
        self.price_coeff2 = -environment.comp_coeff1
        self.comp_coeff2 = -environment.price_coeff1
        self.intercept2 = environment.market_demand - environment.intercept1
        self.selling_price1 = cost_price1
        self.selling_price2 = 0
        self.reaction_line1_coeff = []
        self.actual_X1 = pd.DataFrame()
        self.actual_y1 = pd.DataFrame()
        
        self.as_price_coeff1 = -random.uniform(0.01,0.1)
        self.as_comp_coeff1 = random.uniform(0.01,0.1)
        self.as_intercept1 = random.uniform(60, 80)
        self.as_cost_price2 = cost_price1
        #self.sales_vol1 = sales_volume(price_coeff1, comp_coeff1, intercept1, selling_price1, selling_price2)        
    
    def sales_volume(self, price_coeff, comp_coeff, intercept, selling_price, selling_price_comp):
        #print("sales_volume")
        return ((price_coeff*selling_price) + (comp_coeff*selling_price_comp) + intercept)
    
    def find_new_coeff(self, actual_X, actual_y):
        #print("find_new_coeff")
        clf = linear_model.LinearRegression()
        clf.fit(actual_X, actual_y)
        return (list(clf.coef_[0]) + [clf.intercept_[0]])
    
    def main3(self):
        #print("main3")
        #self.selling_price1 = self.cost_price1
        selling_price_list1 = []        
        
        D = (1-((self.as_comp_coeff1*(-self.as_price_coeff1))/(4*self.as_price_coeff1*(-self.as_comp_coeff1))))
        a1 = 0.5/D
        a2 = -(self.as_comp_coeff1/(4*(self.as_price_coeff1)))/D
        a3 = -((self.as_intercept1/(2*(self.as_price_coeff1))) - (((self.market_demand-self.as_intercept1)*self.as_comp_coeff1)/(4*(self.as_price_coeff1)*(-self.as_comp_coeff1))))/D
        self.selling_price1 = max(self.cost_price1, a1*self.cost_price1 + a2*self.as_cost_price2 + a3)        
        #print([self.selling_price1, self.selling_price2])
        self.actual_X1 = self.actual_X1.append(pd.DataFrame([self.selling_price1,self.selling_price2]).T)
        actual_sales1 = np.random.normal(self.sales_volume(self.price_coeff1, self.comp_coeff1, self.intercept1, self.selling_price1, self.selling_price2), 10, 1)[0]
        self.actual_y1 = self.actual_y1.append(pd.DataFrame([actual_sales1]))
        
        D = (1-((self.as_comp_coeff1*(-self.as_price_coeff1))/(4*self.as_price_coeff1*(-self.as_comp_coeff1))))
        b1 = -((-self.as_price_coeff1)/(4*(-self.as_comp_coeff1)*D))
        b2 = 0.5/D
        b3 = -(((self.market_demand - self.as_intercept1)/(2*(-self.as_comp_coeff1))) - (((self.as_intercept1)*(-self.as_price_coeff1))/(4*(self.as_price_coeff1)*(-self.as_comp_coeff1))))/D
        self.as_cost_price2 = (self.selling_price2 - b1*self.cost_price1 - b3)/b2
        
        selling_price_list1 = selling_price_list1 + [self.selling_price1]
        coefficients1 = self.find_new_coeff(self.actual_X1, self.actual_y1)
        if (coefficients1[0]==0):
            self.as_price_coeff1 = -random.uniform(0,1)
            self.as_comp_coeff1 = random.uniform(0,1)
        else:
            self.as_price_coeff1 = coefficients1[0]
            self.as_comp_coeff1 = coefficients1[1]
        self.as_intercept1 = coefficients1[2]    
        return (self.selling_price1)


def actual_optimal(cost_price1, cost_price2, market_demand, market_coeff, price_coeff1, intercept1, comp_coeff1):
    comp_coeff2 = (market_coeff/2)-price_coeff1
    price_coeff2 = (market_coeff/2)-comp_coeff1
    intercept2 = market_demand-intercept1
    A_row1_coeff1 = 1
    A_row1_coeff2 = (comp_coeff1)/(2*price_coeff1)
    B_row1 = (cost_price1/2) - (intercept1/(2*price_coeff1))
    A_row2_coeff1 = (comp_coeff2)/(2*price_coeff2)
    A_row2_coeff2 = 1
    B_row2 = (cost_price2/2) - (intercept2/(2*price_coeff2))
    
    
    A_matrix = np.matrix([[A_row1_coeff1, A_row1_coeff2],[A_row2_coeff1, A_row2_coeff2]])
    B_matrix = np.matrix([[B_row1], [B_row2]])
    result = np.dot(np.linalg.inv(A_matrix), B_matrix)
    return result

def main(cost_price1, cost_price2, market_demand, market_coeff, price_coeff1, intercept1, comp_coeff1, iterations):
    shop1_conditions = environment(market_demand = market_demand, price_coeff1 = price_coeff1, intercept1 = intercept1, comp_coeff1 = comp_coeff1)
    shop2_conditions = environment(market_demand = market_demand, price_coeff1 = (market_coeff/2)-comp_coeff1, intercept1 = market_demand-intercept1, comp_coeff1 = (market_coeff/2)-price_coeff1)
    shop1 = pricing(cost_price1 = cost_price1, environment=shop1_conditions)
    shop2 = pricing(cost_price1 = cost_price2, environment=shop2_conditions)
    actual_nash_equilibrium = actual_optimal(cost_price1, cost_price2, market_demand, market_coeff, price_coeff1, intercept1, comp_coeff1)
    selling_price_list1=[]
    selling_price_list2=[]
    for i in range(iterations):
        selling_price1 = shop1.main3()
        selling_price2 = shop2.main3()
        shop1.selling_price2 = selling_price2
        shop2.selling_price2 = selling_price1
        selling_price_list1+=[shop1.selling_price1]
        selling_price_list2+=[shop2.selling_price1]
        print([shop1.selling_price1, shop2.selling_price1])
    print("actual nash equilibrium:" + str(actual_nash_equilibrium))
    plt.scatter(selling_price_list1, selling_price_list2)
    plt.scatter(selling_price_list1[len(selling_price_list1)-1], selling_price_list2[len(selling_price_list2)-1], c = "black")
    plt.scatter([actual_nash_equilibrium[0]], [actual_nash_equilibrium[1]], c = "red")


main(1200, 1100, 300, -0.001, -0.002, 100, 0.001, iterations=500)


