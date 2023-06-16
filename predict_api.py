# # Data Modelling
# This is a notebook to experiment with the data modelling of the sales quantity data.
# This was done on a cloud instance so the file paths will be different if you are running this locally.
# Note that the dataset is also propiertary so it will not be included in this repository.

# # 1.Imports and Constants
# We will be using the following libraries:

# In[169]:


import numpy as np
import pandas as pd
import requests
import json
from google.cloud import aiplatform
import sys

from flask import Flask, jsonify, Response

app = Flask(__name__)

@app.route('/predict/<string:supplierName>', methods=['GET'])   # In[78]:
def predict(supplierName:str):
    WINDOW = 30
    
    
    # In[107]:
    
    
    api_url = "http://34.128.116.172:8080/api/send-supplier-data"
    
    body = {
        "supplierName": supplierName
    }
    response = requests.post(api_url, json=body)
    
    json_data = response.json()
    print(json_data)
    
    
    # In[111]:
    
    
    # Extract details
    details = json_data['details']
    
    # Create DataFrame
    data = pd.DataFrame(details)
    data = data.rename(columns={'kode_barang':'item_code', 'tanggal':'date','total_qty':'quantity'})
    data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')
    data['quantity'] = data['quantity'].astype(np.float64)
    
    # Add missing items with current date and 0 total_qty
    items = json_data['items']
    item_codes = [item['kode'] for item in items]
    missing_items = set(item_codes) - set(data['item_code'].unique())
    #get missing dates from the 'tanggal' column going back 30 days
    date_str = '2023-04-01'
    missing_dates = pd.date_range(end=pd.to_datetime(date_str), periods=WINDOW).strftime('%Y-%m-%d')
    #get dates for a single item
    reference_date = pd.to_datetime(data[data.item_code == data['item_code'].unique()[0]]['date'])
    reference_date = reference_date.dt.strftime('%Y-%m-%d')
    
    #remove dates that are already in the data
    missing_dates = set(missing_dates) - set(reference_date)
    print(len(missing_dates))
    #add missing dates for a single item
    missing_data = pd.DataFrame({'date':list(missing_dates), 'item_code':data['item_code'].unique()[0], 'quantity':0})
    #add missing items
    data = pd.concat([data, missing_data], ignore_index=True)
    #sort by date
    
    # Rearrange columns
    
    # Display DataFrame
    print(data)
    
    
    # In[112]:
    
    
    #extract date features from date column
    data['date'] = pd.to_datetime(data['date'])
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data['day_of_week'] = data['date'].dt.dayofweek
    data['day_of_year'] = data['date'].dt.dayofyear
    
    
    
    # We need to create a wide dataframe with each item code as a column and the quantity sold for each day as the values.
    
    # In[113]:
    
    
    #stack dataframe based on item_code
    data = data.groupby(['item_code','date','year','month','day','day_of_week','day_of_year'])['quantity'].sum().unstack(level=0)
    #turn each NaN value to 0
    data = data.sort_values('date')
    data.fillna(0, inplace=True)
    data.reset_index(inplace=True)
    print(data)
    
    
    # ## Prepare item code and dates
    # Since the dataset will use date and item code feature as input, to create an array of item code mapped to every date value
    
    # In[114]:
    
    
    #prepare the list of item codes
    
    items = np.array(data.columns[6:])
    total_items = items.shape[0]
    print(items.shape)
    
    
    # In[115]:
    
    
    # prepare the array of date_related features, since we will be windowing these features
    # we ignore the first few ones
    
    dates = np.array(data[['year','month','day','day_of_week','day_of_year']][WINDOW-1:])
    
    #normalize for cyclic feature
    
    dates = np.sin(dates) + np.cos(dates)
    total_dates = dates.shape[0]
    dates_feature = dates.shape[1]
    print(dates.shape)
    
    
    # Create numpy arrays for each repeated item and dates for later joining.
    
    # In[116]:
    
    
    repeated_items = items.repeat(total_dates)
    repeated_dates = dates.reshape(1,dates.shape[0],dates.shape[1]).repeat(total_items,axis=0).reshape(-1,dates_feature)
    
    print(repeated_items)
    print(repeated_dates)
    
    
    # ## Prepare the sales data to be windowed
    # We need to create windows of the sales data corresponding to the dates. This will be used as input and output for the data later on.
    
    # In[117]:
    
    
    #transpose the sales quantity so dates are columns
    sales = np.array(data[items].fillna(0)).T
    
    
    #create the windows
    windowed = np.lib.stride_tricks.sliding_window_view(sales, WINDOW, axis=-1).reshape(-1,WINDOW)
    print(f'Shape of windowed data {windowed.shape}')
    
    
    # In[138]:
    
    
    #convert the data to json format using the following structure
    #{instances:[
    #    {"sales_window":[windowed[0]], "item_code":[repeated_items[0]], "date_features":[repeated_dates[0]]},
    #    {"sales_window":[windowed[1]], "item_code":[repeated_items[1]], "date_features":[repeated_dates[1]]},...]}
    
    instances = []
    for i in range(windowed.shape[0]):
        instances.append({"sales_window":windowed[i].tolist(), "item_code":repeated_items[i], "date_features":repeated_dates[i].tolist()})
    
    data_json = {"instances":instances}
    print(data_json)
    
    
    
    # In[139]:
    
    
    #save the data to a json variable
    json_object = json.dumps(data_json, indent = 4)
    
    print(json_object)
    
    
    # In[157]:
    
    
    def endpoint_predict_sample(
            project: str, location: str, instances: list, endpoint: str
    ):
        aiplatform.init(project=project, location=location)
    
        endpoint = aiplatform.Endpoint(endpoint)
    
        prediction = endpoint.predict(instances=instances)
        print(prediction)
        return prediction
    
    
    predictions = endpoint_predict_sample(
        "1058401447829",
        "asia-southeast2",
        instances,
        "3449440655217000448"
    )
    
    
    # In[165]:
    
    
    denormalized_predictions = ((np.array(predictions[0])*3.997635572233167)+ 2.1712620248965555).flatten()
    
    
    # In[166]:
    
    
    final = zip(items,denormalized_predictions)
    final = [{'code':item[0],'qty':int(item[1])} for item in final]
    resp = Response(response=json.dumps(final), status=200, mimetype="text/plain")
    return resp


if __name__ == "__main__":
    app.run(port=int(sys.argv[1]),host='0.0.0.0')
# In[ ]:




