

## Dataset Description

The dataset used for this assignment is the India Air Quality Data, obtained from Kaggle. From this dataset, the NO₂ (Nitrogen Dioxide) concentration feature is selected for analysis.

**Dataset:**

* India Air Quality Data

**Feature Used:**

* NO₂ concentration

**Source:**

* Kaggle

**Data Type:**

* Continuous numerical values

---

## Preprocessing Steps

Before applying the model, the data is cleaned and prepared:

* Non-numeric values are converted to NaN
* Missing values are removed to avoid inconsistencies
* The data is normalized before applying the transformation

These preprocessing steps ensure that the data is suitable for mathematical operations and statistical modeling.

---

## Roll Number Based Parameters

The parameters are defined as:

[
a_r = 0.05 \times (r \bmod 7)
]

[
b_r = 0.3 \times ((r \bmod 5) + 1)
]

For this assignment, my roll number is:

```
r = 102317026
```

Each student obtains a slightly different transformed dataset and learned probability density function.

---

## Methodology

### Data Normalization

To keep the NO₂ values within a stable and valid numerical range, normalization is applied using:

[
x_{norm} = \frac{x}{\max(|x|)}
]

---

### Non-Linear Transformation

Each normalized value is transformed using the roll-number-based function:

[
z = T_r(x) = x + a_r \cdot \arcsin(b_r \cdot x)
]

**Purpose of the Transformation:**

* Introduces non linearity into the dataset
* Changes the original distribution of NO₂ values
* Embeds roll-number dependency into the data

The transformed variable **z** is then used for probability density function learning.

---

## Parameter Estimation of Probability Density Function

After transformation, the data is assumed to follow a Gaussian-like probability density function, defined as:

[
p(z) = c \cdot e^{-\lambda (z - \mu)^2}
]

Where:

* μ (mean) represents the central value of the transformed data
* σ² (variance) represents the spread of the data
* ( \lambda = \frac{1}{2\sigma^2} )
* ( c = \sqrt{\frac{\lambda}{\pi}} ) ensures proper normalization

These parameters are computed directly from the transformed dataset.



## Results

### Estimated Parameters

The estimated parameters obtained after processing the data are shown below:

| Parameter                  | Description                          | Value                |
| -------------------------- | ------------------------------------ | -------------------- |
| Mean (μ)                   | Central tendency of transformed data | 0.029463039837684103 |
| λ (Lambda)                 | Inverse variance term                | 1120.7039391836472   |
| c (Normalization Constant) | Ensures total probability = 1        | 18.887327585639866   |

These values define the final learned probability density function for the transformed NO₂ data.

---

