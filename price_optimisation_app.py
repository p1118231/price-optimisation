#!/usr/bin/env python
# coding: utf-8

# In[3]:


from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load('price_optimization_model.pkl')
product_data = pd.read_csv('enhanced_product_data.csv')  # Assuming this CSV has all necessary columns

@app.route('/predict/<int:product_id>', methods=['GET'])
def predict(product_id):
    product_features = product_data[product_data['ProductId'] == product_id]
    if product_features.empty:
        return jsonify({'error': 'Product not found'}), 404

    features = product_features[['SalesVolume', 'CompetitorPrice', 'CostOfGoods', 'SeasonalInfluence', 'MarketingSpend', 'CustomerDemographics']]
    prediction = model.predict(features)
    return jsonify({'predicted_price': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True, port=5000)


# In[ ]:




