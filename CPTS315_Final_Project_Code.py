import numpy as np
import pandas as pd
from efficient_apriori import apriori
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance
import matplotlib.pyplot as plt
from collections import Counter

def generate_frequent_itemsets(baskets, support=0.0005, confidence=0):
    itemsets, rules = apriori(baskets, min_support=0.0005, min_confidence=0)
    lhs = []
    rhs = []
    conf = []
    lhs_count = max(itemsets.keys()) #Maximum number of items on left hand side
    for i in range(1,lhs_count+1):
        rules_rhs = filter(lambda rule: len(rule.lhs) == i and len(rule.rhs) == 1, rules)
        for rule in sorted(rules_rhs, key=lambda rule: rule.lift):
            lhs.append(rule.lhs)
            rhs.append(rule.rhs[0])
            conf.append(float(rule.count_full/rule.count_lhs))
    frequent_items = pd.DataFrame(
    {'Item_1': lhs,
     'Item_2': rhs,
     'Confidence': conf
    })
    return frequent_items

def product_feature_set(total_orders):
    product_purchases = total_orders.groupby('product_id')['order_id'].count().to_frame('total_product_purchases').reset_index()
    product_reorder_ratio = total_orders.groupby('product_id')['reordered'].mean().to_frame('total_product_reorder_ratio').reset_index()
    product_cart_ranking = total_orders.groupby('product_id')['add_to_cart_order'].mean().to_frame('total_avg_product_cart_pos').reset_index()
    product_features = product_purchases.merge(product_reorder_ratio, on='product_id', how='left')
    product_features = product_features.merge(product_cart_ranking, on='product_id', how='left')
    return product_features

def user_feature_set(total_orders):
    user_purchases = total_orders.groupby('user_id')['order_number'].max().to_frame('user_purchases').reset_index()
    user_reorder_ratio = total_orders.groupby('user_id')['reordered'].mean().to_frame('user_reorder_ratio').reset_index()
    user_cart_size = total_orders.groupby(['user_id', 'order_id'])['add_to_cart_order'].max().reset_index(name='user_avg_order_size').groupby('user_id')['user_avg_order_size'].mean().reset_index()
    days_between_orders = total_orders.groupby(['user_id', 'order_id'])['days_since_prior_order'].max().reset_index(name='avg_days_between_orders').groupby('user_id')['avg_days_between_orders'].mean().reset_index()
    user_features = user_purchases.merge(user_reorder_ratio, on='user_id', how='left')
    user_features = user_features.merge(user_cart_size, on='user_id', how='left')
    user_features = user_features.merge(days_between_orders, on='user_id', how='left')
    return user_features

def user_product_feature_set(total_orders):
    user_product_purchases = total_orders.groupby(['user_id', 'product_id'])['order_number'].max().to_frame('user_product_purchases').reset_index()
    user_product_reorder_ratio = total_orders.groupby(['user_id', 'product_id'])['reordered'].mean().to_frame('user_product_reorder_ratio').reset_index()
    user_product_avg_cart_pos = total_orders.groupby(['user_id','product_id'])['add_to_cart_order'].mean().to_frame('user_avg_product_cart_pos').reset_index()
    user_product_features = user_product_purchases.merge(user_product_reorder_ratio, on=['user_id','product_id'], how='left')
    user_product_features = user_product_features.merge(user_product_avg_cart_pos, on=['user_id','product_id'], how='left')
    return user_product_features

def full_feature_set(product_features, user_features, user_product_features):
    feature_set = user_product_features.merge(user_features, on='user_id', how='outer')
    feature_set = feature_set.merge(product_features, on='product_id', how='outer')
    return feature_set

def train_test_split(orders, feature_set):
    order_types = orders[((orders.eval_set=='train') | (orders.eval_set=='test'))]
    order_types = order_types[['user_id', 'eval_set', 'order_id']]
    data = feature_set.merge(order_types, on='user_id', how='left')
    #training data
    train = data[data.eval_set=='train']
    train_data = train.merge(orders_train[['product_id', 'order_id', 'reordered']], on=['product_id', 'order_id'], how='left')
    train_data['reordered'] = train_data['reordered'].fillna(0)
    train_data = train_data.set_index(['user_id', 'product_id'])
    train_data = train_data.drop(['eval_set', 'order_id'], axis = 1)
    #testing data (for Kaggle evaluation)
    test_data = data[data.eval_set=='test']
    test_data = test_data.drop(['eval_set', 'order_id'], axis = 1)
    #test_data = test_data.set_index(['user_id', 'product_id'])
    #Model test/train split
    X_train, X_val, y_train, y_val = sklearn_train_test_split(
        train_data.drop(['reordered'], axis = 1), 
        train_data.reordered, 
        test_size=0.3,
        random_state=42)
    return X_train, X_val, y_train, y_val, test_data, train_data

def train_classifier(X_train, X_val, y_train, y_val):
    accuracy = {}
    #sklearn gradient booster
    #GBC = GradientBoostingClassifier()
    #GBC.fit(X_train, y_train)
    #GBC_y_pred = GBC.predict(X_val)
    #accuracy['GradientBoostingClassifier'] = accuracy_score(y_val, GBC_y_pred) #0.907136
    
    #Random forest
    #RF = RandomForestClassifier()
    #RF.fit(X_train, y_train)
    #RF_y_pred = RF.predict(X_val)
    #accuracy['RandomForest'] = accuracy_score(y_val, RF_y_pred) #0.907348
    
    #XGBoost
    XGB = XGBClassifier()
    XGB.fit(X_train, y_train)
    XGB_y_pred = XGB.predict(X_val)
    accuracy['XGBoost'] = accuracy_score(y_val, XGB_y_pred) #0.908406
    
    return XGB, accuracy

#get the predictions for every product for a user/order
def get_predictions(df, model, accuracy):
    predictions_dict = {}
    predictions = model.predict(df)
    i = 0
    for index, row in df.iterrows():
        product_id = index[1]
        predictions_dict[product_id] = predictions[i]
        i+=1
    #Saves the prediction as a weight (if reordered, save the accuracy (0.908) else 1-0.908)
    for key in predictions_dict:
        if predictions_dict[key] == 1:
            predictions_dict[key] = a
        else:
            predictions_dict[key] = (1-a)
    return predictions_dict

#For every product in the lhs of the frequent itemsets, multiply by the prediction weights
def calculate_purchase_probability(row):
    item1 = row["Item_1"]
    confidence = row["Confidence"]
    
    if len(item1) == 1:
        constant = predictions.get(item1[0], 0)
        purchase_probability = confidence * constant
    else:
        #Calculate average weight of the item1 basket if there is more than one item
        total_weight = 0
        for item in item1:
            weight = predictions.get(item, 0)
            total_weight += weight
        average_weight = total_weight / len(item1)
        purchase_probability = confidence * average_weight
    
    return purchase_probability

#Calculate purchase probability by multiplying confidence of items the order includes by the prediciton weights.
#Only returns items not purchased and order must include all items on lhs if more than one.
def get_recommendations(user, xgb, a):
    prediction_data = feature_set.loc[feature_set.user_id == user]
    prediction_data = prediction_data.set_index(['user_id', 'product_id'])

    global predictions
    predictions = get_predictions(prediction_data, xgb, a)

    recommendations = frequent_items.copy()
    recommendations["Purchase_Probability"] = recommendations.apply(calculate_purchase_probability,axis=1)
    recommendations = recommendations[recommendations['Item_1'].apply(lambda x: all(item in predictions.keys() for item in x))] #has all items in basket
    recommendations = recommendations[~recommendations['Item_2'].isin(predictions.keys())] #only new products
    recommendations = recommendations.sort_values("Purchase_Probability", ascending=False)
    return recommendations

products = pd.read_csv("instacart-market-basket-analysis/products.csv")
orders = pd.read_csv("instacart-market-basket-analysis/orders.csv")
orders_prior = pd.read_csv("instacart-market-basket-analysis/order_products__prior.csv")
orders_train = pd.read_csv("instacart-market-basket-analysis/order_products__train.csv")
aisles = pd.read_csv("instacart-market-basket-analysis/aisles.csv")
departments = pd.read_csv("instacart-market-basket-analysis/departments.csv")

total_orders = orders.merge(orders_prior, on='order_id', how='inner')

#Get order products list as product name
products_list = orders_prior[["order_id", "product_id"]]
products_list = products_list.merge(products, on='product_id', how='inner')
products_list = products_list.groupby('order_id')['product_name'].apply(list)
products_list = products_list.to_list()
#Get top 10 products as percentage of occurances in orders
product_counts = Counter(item for basket in products_list for item in basket)
total_products = sum(product_counts.values())
product_percentages = {item: count/total_products for item, count in product_counts.items()}
sorted_products = sorted(product_percentages.items(), key=lambda x: x[1], reverse=True)
sorted_products = sorted_products[:10]
#Plot the top 10 most purchased products
fig, ax = plt.subplots()
ax.bar([item[0] for item in sorted_products], [item[1] for item in sorted_products])
plt.xticks(rotation=90)
ax.set_xlabel('Product')
ax.set_ylabel('Percentage of Baskets')
plt.show()

#Get order products list as product number
products_list = orders_prior[["order_id", "product_id"]]
products_list = products_list.groupby('order_id')['product_id'].apply(list)
products_list = products_list.to_list()

frequent_items = generate_frequent_itemsets(products_list)

product_features = product_feature_set(total_orders)
user_features = user_feature_set(total_orders)
user_product_features = user_product_feature_set(total_orders)
feature_set = full_feature_set(product_features, user_features, user_product_features)
X_train, X_val, y_train, y_val, test_data, train_data = train_test_split(orders, feature_set)

XGB, accuracy = train_classifier(X_train, X_val, y_train, y_val)

#Plot the distribution of the average number of products reordered by users for both the actual and predicted data
pred = pd.Series(XGB_y_pred, name='Predicted')
data = pd.DataFrame(y_val).reset_index()
data = pd.concat([data, pred], axis=1)
counts = data.groupby('user_id').agg({'reordered': ['sum', 'count'], 'Predicted': ['sum', 'count']}).mean().to_frame()
counts.reset_index(inplace=True)
counts = counts.set_index(['level_0', 'level_1']).unstack()
counts.columns = counts.columns.swaplevel()
counts = counts.sort_index(level=0, axis=1)
counts.index = ['Predicted', 'Actual']
counts.columns = ['Not Reordered', 'Reordered']
counts.iloc[:,0] = counts.iloc[:,0] - counts.iloc[:,1]
counts = counts.round(2)
counts.plot(kind='bar',rot=0)

plot_importance(XGB)
plt.show()

a = accuracy['XGBoost']
frequent_items.sort_values("Confidence", ascending=False).head(5)

#Find an example with the highest purchase probability
users = feature_set['user_id'].unique().tolist()
max_prob = 0
user = 0
for i in range(0,1000):
    recommendations = get_recommendations(users[i], XGB, a)
    if not recommendations.empty:
        if (recommendations.iloc[0]['Purchase_Probability'] > max_prob):
            max_prob = recommendations.iloc[0]['Purchase_Probability']
            user = users[i]
recommendations = get_recommendations(user, XGB, a)

recommendations.head(5)
products.loc[products.product_id == 24852]