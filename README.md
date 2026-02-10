# learning probability density function using gan

---

## objective
to learn an unknown probability density function of a transformed random variable using a generative adversarial network (gan), **without assuming any parametric distribution**.

---

## transformation
each sample x is transformed into z using the following function:

z = x + a · sin(b · x)

where:
- a = 0.5 × (r mod 7)
- b = 0.3 × ((r mod 5) + 1)
- r is the university roll number

this nonlinear transformation introduces multimodality into the data distribution.

---

## approach

### 1. data preprocessing
- extract no₂ values
- remove missing entries
- apply transformation x → z
- standardize z for stable gan training

### 2. gan architecture
the gan consists of two neural networks:

#### generator
- input: gaussian noise ~ n(0,1)
- output: synthetic z samples
- fully connected feedforward network with relu activations

#### discriminator
- input: real or fake z samples
- output: probability of being real
- trained using binary cross-entropy loss

---

## training details
- epochs: 3000  
- batch size: 128  
- optimizer: adam  
- loss function: binary cross entropy  

these values were chosen to ensure stable adversarial convergence and adequate mode coverage.

---

## pdf estimation
after training:
a large number of samples are generated from the generator
kernel density estimation (kde) is applied to approximate the learned pdf
the estimated pdf is compared with the histogram of real transformed samples

---

## observations

### mode coverage
the generator successfully captures the dominant modes of the transformed distribution
minor mode loss may occur during early training but stabilizes with more epochs

### training stability
loss values show oscillatory behavior, which is expected in gan training
convergence improves after sufficient iterations

### quality of learned distribution
the estimated pdf closely matches the empirical distribution
the model learns the density purely from samples without any parametric assumption

---

## how to run
1. place the dataset csv file in the project directory
2. update the roll number `r` in the code
3. run the script or notebook sequentially
4. observe the generated pdf and training loss plots

---

## dependencies
- python 3.x
- numpy
- pandas
- matplotlib
- scikit-learn
- pytorch

