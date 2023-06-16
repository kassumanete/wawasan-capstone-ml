# Notebooks
These are the notebooks used for experimenting. The data is confidential but you can use the structure to reproduce the model using your own data.

## Supplier Schedule
The csv has the following column structure:
>- *tanggal* (string): Date the supplier came in the format %Y-%m-%d
>- *supplier* (string): Supplier code

## Sales Prediction
The csv has the following column structure:
>- *tanggal* (string): Date the purchase was made
>- *kode_barang* (string): Barcode of the item
>- *quantity* (int): Quantity of item sold on that day

# Models
You can also used the pretrained models in [model](model) folder by using the following command:
```
new_model = tf.keras.models.load_model('modelpath')
```
