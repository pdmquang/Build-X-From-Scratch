import pandas as pd
import numpy as np

class DecisionTree:
    def __init__(self, max_depth = 5, depth = 1):
        self.left = None
        self.right = None
        self.impurity_score = None
        self.depth = depth
        self.max_depth = max_depth
    
    def __calculate_impurity_score(self, data):
        '''
        data: Series
        value_counts(): count unique (classes: 0, 1)
        x/len(data): prob, no. of independent?? in each class / total 
        '''
        if data is None or data.empty:
            return 0
        p_i, _ = data.value_counts().apply(lambda x: x/len(data)).tolist()
        return p_i * (1 - p_i) * 2
    
    def __find_best_split_for_column(self, col):
        '''
        Within 1 feature, find best val that reduce impurity the most
        '''
        x = self.data[col] # Feature

        # pd.Series([Male, Female, Male, Male], name='A').unique()
        # >>> array([Male, Female])
        unique_values = x.unique()
        if len(unique_values) == 1:
            # only 1 row in each class (overfit)
            return None, None
        information_gain = None
        split = None

        for val in unique_values:
            left = x <= val # Boolean pandas.Series
            right = x > val # Boolean pandas.Series
            left_data = self.data[left]
            right_data = self.data[right]
            
            # Impurity = metric
            left_impurity = self.__calculate_impurity_score(left_data[self.target])
            right_impurity  = self.__calculate_impurity_score(right_data[self.target])

            # Information gain = How much Impurity is reduced
            score = self.__calculate_information_gain(left_count=len(left_data),
                                                      left_impurity=left_impurity,
                                                      right_count=len(right_data),
                                                      right_impurity=right_impurity)

            # Use the value in Feature, that yields the highest impurity reduction (aka Information gain)
            if information_gain is None or score > information_gain:
                information_gain = score
                split = val

        return information_gain, split

    def __calculate_information_gain(self, left_count, left_impurity, right_count, right_impurity):
        return self.impurity_score - ((left_count/len(self.data)) * left_impurity + (right_count/len(self.data)) * right_impurity)

    def __find_best_split(self):
        '''
        Select the best split out of all the splits that has highest reductions?

        split on independent (rows) <= 5
            any rows <= 5, Left
            else, right
        calculate impurity for left and right branch
        calculate information gain (impurity reduction)

        '''
        best_split = {}
        for col in self.independent:
            information_gain, split = self.__find_best_split_for_column(col)
            if split is None:
                continue
            if not best_split or best_split["information_gain"] < information_gain:
                best_split = {
                    "split": split,
                    "col": col,
                    "information_gain": information_gain
                }
        return best_split["split"], best_split["col"]

    def __create_branches(self):
        self.left = DecisionTree()
        self.right = DecisionTree()
        left_rows = self.data[self.data[self.split_feature] <= self.criteria]
        right_rows = self.data[self.data[self.split_feature] > self.criteria]
        self.left.fit(data=left_rows, target=self.target)
        self.right.fit(data=right_rows, target=self.target)

    def fit(self, data, target):
        if self.depth <= self.max_depth:
            print(f"procesing at Depth: {self.depth}")
        self.data = data
        self.target = target
        self.independent = self.data.columns.tolist() # all the features
        self.independent.remove(target) # remove the label column
        if self.depth <= self.max_depth:
            self.__validate_data() # ensure datatype 

            # What is the purpose of this line below???
            self.impurity_score = self.__calculate_impurity_score(self.data[self.target])
            self.criteria, self.split_feature, self.information_gain = self.__find_best_split()
            if self.criteria is not None and self.information_gain > 0:
                self.__create_branches()
        else:
            print("Stop splitting as Max depth reached")
    
    def predict(self, data):
        return np.array([self.__flow_data_thru_tree(row) for _, row in data.iterrows()])
        
    def __flow_data_thru_tree(self, row):
        # Recursive
        if self.is_leaf_node:
            return self.probability
        tree = self.left if row[self.split_feature] <= self.criteria else self.right # self.criteria = 5???
        return tree.__flow_data_thru_tree(row)
    
    # Getters
    @property
    def is_leaf_node(self):
        return self.left is None
    @property
    def probability(self):
        return self.data[self.target].value_counts.apply(lambda x: x/len(self.data))


if __name__ == "__main__":
    train = pd.read_csv
    test = pd.read_csv
    model = DecisionTree()
    model.fit(data=train, target = "Survived")
    predictions = model.predict(test)