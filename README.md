# MACHINE LEARNING HOUSE SALE PRICE PREDICTOR
For best user interaction, please use the [**Google Colab notebook**](https://colab.research.google.com/drive/16fxDjot_85PfMpMvJKpHJbPC4garb1Re?usp=sharing) as many features including Tensorboard and Plotly graph interactivity are not available unless using a notebook. 

## TABLE OF CONTENTS
1. [Descriptions](#DESCRIPTION)
2. [Model Overview](#MODEL-OVERVIEW)
    * [Neural Network Specifications](#NEURAL-NETWORK-SPECIFICATIONS)
    * [Model Metrics](#MODEL-METRICS)
3. [Contextual Regional Map](#CONTEXTUAL-REGIONAL-MAP)
4. [Visualizations Snapshots](#VISUALIZATION-SNAPSHOTS)
5. [Variable Descriptions](#VARIABLE-DESCRIPTIONS)
    * [Price Variable Breakdown](#PRICE-DESCRIPTION)
    * [Table of Variables](#TABLE-OF-VARIABLES)
6. [Resources](#RESOURCES)

## DESCRIPTION
* Applied Tensorflow/Keras Neural Network Regression capabilities to predict house sale prices in King County, Washington.
* Visualized historical sales on interactive Mapbox scatter maps, animated time series, and 3D plots using Plotly.
* Utilized Pandas and Seaborn on Kaggle CSVs for data cleaning, exploratory analysis, and feature engineering.

## MODEL OVERVIEW

### NEURAL NETWORK SPECIFICATIONS:
* Sequential Model with 1 input layer, 2 hidden layers, and 1 output layer.
* Rectified linear unit (ReLU) activation function used for all layers except output.
* Layers 1 to 3 contain 88 nodes with hidden layers having a 40% drop out rate.
* Compiled with adam optimizer evaluating by mean squared error. 
* Preprocessed using MinMaxScaler for predictor variables.
* Batch size of 128 (2^7) to balance compile time and overfitting reduction amount.
* Arbitraly large epochs of 10000 to be corrected by early stop callback.
* Tensorboard log recorded in callback.

### MODEL METRICS:
* Mean Absolute Error (MAE): 61,589.66
* Mean Squared Error (MSE): 7,542,717,726.97
* Root Mean Squared Error (RMSE): 86,848.82
* Explained Variance: 0.83
* Residual Variation: 0.17

## VISUALIZATION SNAPSHOTS
**Interactive Scatter Map of Prices (USD):**

![Price Mapbox](https://github.com/aidanandrucyk/Machine_Learning_Predictor_for_King_County_Home_Sale_Price/blob/master/img/price_map.png)

**Square Feet of Living Space Interactive Scatter Map for Houses Built in 1906:**
![Square Feet Mapbox](https://github.com/aidanandrucyk/Machine_Learning_Predictor_for_King_County_Home_Sale_Price/blob/master/img/sqrt_feet_map.png)

## CONTEXTUAL REGIONAL MAP
* Red section represents the City of Seattle.
* Orange section represents King County.

![Map Outline of King County](https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/King_County_Washington_Incorporated_and_Unincorporated_areas_Burien_Highlighted.svg/1200px-King_County_Washington_Incorporated_and_Unincorporated_areas_Burien_Highlighted.svg.png)

## VARIABLE DESCRIPTIONS
<p>This dataset contains house sale prices for King County, Washington. It includes homes sold between May 2014 and May 2015.</p>

### PRICE DESCRIPTION
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>Metric</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Mean</td>
      <td>4.580474e+09</td>
    </tr>
    <tr>
      <td>Std</td>
      <td>2.876736e+09</td>
    </tr>
    <tr>
      <td>Min</td>
      <td>1.000102e+06</td>
    </tr>
    <tr>
      <td>Q1</td>
      <td>2.123049e+09</td>
    </tr>
    <tr>
      <td>Median</td>
      <td>3.904930e+09</td>
    </tr>
    <tr>
      <td>Q3</td>
      <td>7.308900e+09</td>
    </tr>
    <tr>
      <td>Max</td>
      <td>9.900000e+09</td>
    </tr>
  </tbody>
</table>

### TABLE OF VARIABLES
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Column</th>
      <th>Variable</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>price</td>
      <td>Sale price of the house.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>bedrooms</td>
      <td>Number of bedrooms.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>bathrooms</td>
      <td>Number of bathrooms.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sqft__ving</td>
      <td>Size of living area in square feet.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>sqft_lot</td>
      <td>Size of the lot in square feet.</td>
    </tr>
    <tr>
      <th>5</th>
      <td>floors</td>
      <td>Number of floors.</td>
    </tr>
    <tr>
      <th>6</th>
      <td>waterfront</td>
      <td>‘1’ if the property has a waterfront, ‘0’ if not.</td>
    </tr>
    <tr>
      <th>7</th>
      <td>view</td>
      <td>An index from 0 to 4 of how good the view of the property was.</td>
    </tr>
    <tr>
      <th>8</th>
      <td>condition</td>
      <td>An index from 1 to 5 on the condition of the apartment.</td>
    </tr>
    <tr>
      <th>9</th>
      <td>grade</td>
      <td>An index from 1 to 13, where 1-3 falls short of building construction and design, 7 has an average level of construction and design, and 11-13 have a high quality level of construction and design.</td>
    </tr>
    <tr>
      <th>10</th>
      <td>sqft_above</td>
      <td>The square footage of the interior housing space that is above ground level.</td>
    </tr>
    <tr>
      <th>11</th>
      <td>sqft_basement </td>
      <td>The square footage of the interior housing space that is below ground level.</td>
    </tr>
    <tr>
      <th>12</th>
      <td>yr_built</td>
      <td>The year the house was initially built.</td>
    </tr>
    <tr>
      <th>13</th>
      <td>yr_renovated</td>
      <td>The year of the house’s last renovation.</td>
    </tr>
    <tr>
      <th>14</th>
      <td>zipcode</td>
      <td>What zipcode area the property is in.</td>
    </tr>
    <tr>
      <th>15</th>
      <td>lat</td>
      <td>Lattitude.</td>
    </tr>
    <tr>
      <th>16</th>
      <td>long</td>
      <td>Longitude.</td>
    </tr>
        <tr>
      <th>17</th>
      <td>sqft_living15</td>
      <td>The square footage of interior housing living space for the nearest 15 neighbors.</td>
    </tr>
        <tr>
      <th>18</th>
      <td>sqft_lot15</td>
      <td>The square footage of the land lots of the nearest 15 neighbors</td>
    </tr>
  </tbody>
</table>

---
----
## RESOURCES
- [Kaggle](https://www.kaggle.com/harlfoxem/housesalesprediction)

- [Data Source](https://geodacenter.github.io/data-and-lab//KingCounty-HouseSales2015/)
