<h2 align="center">Wawasan</h2>

## Built using:
- [Tensorflow](https://www.tensorflow.org/)
- [Google Colab](https://colab.research.google.com/)
- [Google Cloud Platform](https://cloud.google.com/)

## [Models](ml_modelling/models)

### Supplier Schedule Predictor
**Input:**
- *supplier* (string): A string code of the supplier.
- *date features* (float): List of features for date (refer to notebook for more detail).

**Output:**
- *days* (int): Predicted days until supplier will come next.

This is the architecture:
<p align="center">
    <img src="images/supplier_schedule_architecture.png" alt="Prediction architecture">
</p>

### Sales Prediction Model
**Input:**
- *supplier* (string): A string code of the supplier.
- *date features* (float): List of feature for date (refer to notebook for more details).
- *sales_window* (float): List of sales of the last 30 days

**Output:**
  - *quantity* (int): Predicted quantity of the sales for the next day.

  This is the architecture:
  <p align="center">
  <img src="images/sales_prediction_architecture.png" alt="Sales architecture">
  </p>

## API
### Deploying

**Requires linux**
You need to use flask, so ensure you install the dependencies by running the following command in your shell:
'''
pip install -r requirements.txt
'''

You then just have to run the app by running:
'''
python predict_api.py
'''

The server should then start in port 5000.

### Making a prediction
SEND a get request to:
'''
http://hostname:5000/predict/supplier_name
'''
Where:
- *hostname* is the server that the API is on
- *supplier_name* is the name of suppliers (undisclosed)
