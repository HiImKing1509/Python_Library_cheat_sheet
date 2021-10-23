# PYTHON LIBRARIES FOR MACHINE LEARNING
![alt tag](https://miro.medium.com/max/875/1*RIrPOCyMFwFC-XULbja3rw.png)

## Introduction to Python for Machine Learning
 > In the 21st century, most of the applications developed by companies are somehow built using Artificial Intelligence, Machine Learning, or Deep Learning that uses Python Machine Learning library. Usually, AI projects are distinct from conventional projects in the software industry. Variations in development approaches lie in the application framework, the necessary skills needed for the AI-based application, and the need for in-depth analysis.

Hence, one of the important factors involved in developing AI-based applications is the use of a suitable programming language. We should employ a programming language that is efficient in making the applications stable and extensible. For this, companies use the Python programming language as it offers a lot of libraries and packages for the development task, and hence, it is widely used for working on AI-based projects.

## Benefits of Using Python
Here are a few of the benefits of using Python:

1. **Simple and compatible**: Python provides a descriptive and interactive code. Although complicated algorithms and adaptable workflows are behind Artificial Intelligence and Machine Learning, the simplicity of Python Machine Learning library and framework, enables application developers to develop reliable systems.
2. **Platform-independent**: One aspect adding to the success of Python is that it is a language that is independent of the platform on which it is being operated. There are various platforms that support Python, such as Windows, macOS, and Linux. For the most commonly used software, Python language code can be used to build discrete executable programs. This ensures that Python programs can be quickly deployed, and we can use them without having a Python interpreter on operating systems.
3. **Large community**: According to a survey by Stack Overflow, it is one of the top 10 programming languages used by various software industries. Also, Python is one of the most searched programming languages than any other. It is considered the best language for Web Development as well. It has a large community of developers that can help the newbies starting with Python programming to learn and grow with experienced developers.
Now, as we have discussed Python and its benefits, let’s look at the top 10 Python libraries for Machine Learning.

## Best Python Libraries for Machine Learning

### 1. Numpy
> Website: https://numpy.org/
> 
> Github Repository: https://github.com/numpy/numpy
> 
> Developed By: Community Project (originally authored by Travis Oliphant)
> 
> Primary purpose: General Purpose Array Processing

![alt tag](https://intellipaat.com/blog/wp-content/uploads/2020/12/NumPy.jpg)

- NumPy is regarded as being one of the most widely used and best Python libraries for Machine Learning. Other libraries, such as TensorFlow and Keras, use NumPy to implement various operations on tensors.
- Created on the top of an older library Numeric, the Numpy is used for handling multi-dimensional data and intricate mathematical functions. Numpy is a fast computational library that can handle tasks and functions ranging from basic algebra to even Fourier transforms, random simulations, and shape manipulations. This library is written in C language, which gives it an edge over standard python built-in sequencing. Numpy arrays are better than pandas series in the term of indexing and Numpy works better if the number of records is less than 50k. The NumPy arrays are loaded into a single CPU which can cause slowness in processing over the new alternatives like Tensorflow, Dask, or JAX, but still, the learning of Numpy is very easy and it is one of the most popular libraries to enter into the Machine Learning world.

> Few of the points in favor of NumPy are:
>
>  * Support for mathematical and logical operations
> 
> * Shape manipulation
> 
> * Sorting and Selecting capabilities
> 
> * Discrete Fourier transformations
> 
> * Basic linear algebra and statistical operations
> 
> * Random simulations
> 
> * Support for n-dimensional arrays

| **Advantages** | **Disadvantages** |
| ---- | --- |
| Using NumPy, we can easily deal with multi-dimensional data | NumPy is highly dependent on non-Pythonic entities. It uses the functionalities of Cython and other libraries that use C/C++ | 
| The library helps in the matrix manipulation of the data and the operations such as transpose, reshape, and many more | Its high productivity comes at a price |
| NumPy enables enhanced performance and the management of garbage collection as it provides a dynamic data structure | The data types are hardware-native and not Python-native, so it costs heavily when we want to translate NumPy entities back to Python-equivalent entities and vice versa |
| It allows us to improve the performance of the Machine Learning model | |

### 2. Pandas
> Website: https://pandas.pydata.org/
> 
> Github Repository: https://github.com/pandas-dev/pandas
> 
> Developed By: Community Developed (Originally Authored by Wes McKinney)
> 
> Primary Purpose: Data Analysis and Manipulation

![alt tag](https://intellipaat.com/blog/wp-content/uploads/2020/12/Pandas-Python-Library-for-Machine-Learning.jpg)

- One of widely used Python’s Machine Learning library is Pandas. Pandas is the best Python library that is majorly used for data manipulation. It uses handy and descriptive data structures such as DataFrames to create programs for implementing functions. Developed on top of NumPy, it is a quick and easier-to-use library.
- Pandas provides data reading and writing capabilities using various sources such as Excel, HDFS, and many more. If you are planning on a use case for a real-world Machine Learning model, then, sooner or later, you would use Pandas for implementing it.

> Some of the great features of Pandas when it comes to handling data are:
> 
> * Dataset reshaping and pivoting
> 
> * Merging and joining of datasets
> 
> * Handling of missing data and data alignment
> 
> * Various indexing options such as Hierarchical axis indexing, Fancy indexing
> 
> * Data filtration options

| **Advantages** | **Disadvantages** |
| ---- | --- |
| It has descriptive, quick, and compliant data structures | It is based on Matplotlib, which means that an inexperienced programmer needs to be acquainted with both libraries to understand which one will be better to solve a specific business problem
| It supports operations such as grouping, integrating, iterating, re-indexing, and representing data | Pandas is much less suitable for quantitative modeling and n-dimensional arrays. In such scenarios, where we need to work on quantitative/statistical modeling, we can use Numpy or SciPy
| The Pandas library is very flexible for usage in association with other libraries
| It contains inherent data manipulation functionalities that can be implemented using minimal commands
| It can be implemented in a large variety of areas, especially related to business and education, due to its optimized performance



### 3. Matplotlib
> Website: https://matplotlib.org/
> 
> Github Repository: https://github.com/matplotlib/matplotlib
> 
> Developed By: Micheal Droettboom, Community
> 
> Primary purpose: Data Visualization

![alt tag](https://intellipaat.com/blog/wp-content/uploads/2020/12/Matplotlib-Python-Library-for-Machine-Learning.jpg)

- Matplotlib is a library used in Python for graphical representation to understand the data before moving it to data-processing and training it for Machine learning purposes. It uses python GUI toolkits to produce graphs and plots using object-oriented APIs. The Matplotlib also provides a MATLAB-like interface so that a user can do similar tasks as MATLAB. This library is free and open-source and has many extension interfaces that extend matplotlib API to various other libraries.

| **Advantages** | **Disadvantages** |
| ---- | --- |
| It helps produce plots that are configurable, powerful, and accurate | Matplotlib has a strong dependency on NumPy and other such libraries for the SciPy stack
| Matplotlib can be easily streamlined with Jupyter Notebook | It has a high learning curve as the use of Matplotlib takes quite a lot of knowledge and application from the learners’ end
| It supports GUI toolkits that include wxPython, Qt, and Tkinter | There can be confusion for developers because Matplotlib provides two distinct frameworks: Object-oriented and MATLAB
| Matplotlib is leveraged with a structure that can support Python as well as IPython shells | Matplotlib is a library used for data visualization. It is not suitable for data analysis. To get both data visualization and analysis, we will have to integrate it with other libraries


### 4. Scikit-Learn

> Website: https://scikit-learn.org/
> 
> Github Repository: https://github.com/scikit-learn/scikit-learn
> 
> Developed By: SkLearn.org 
> 
> Primary Purpose: Predictive Data Analysis and Data Modeling

![alt tag](https://intellipaat.com/blog/wp-content/uploads/2020/12/Scikit-Learn.jpg)

- Scikit-learn is mostly focused on various data modeling concepts like regression, classification, clustering, model selections, etc. The library is written on the top of Numpy, Scipy, and matplotlib. It is an open-source and commercially usable library that is also very easy to understand. It has easy integrability which other ML libraries like Numpy and Pandas for analysis and Plotly for plotting the data in a graphical format for visualization purposes. This library helps both in supervised as well as unsupervised learnings. 

> Scikit-learn comes with the support of various algorithms such as:
> 
> * Classification
> 
> * Regression
> 
> * Clustering
> 
> * Dimensionality Reduction
> 
> * Model Selection
> 
> * Preprocessing

| **Advantages** | **Disadvantages** |
| ---- | --- |
| The Scikit-Learn library has a go-to package that consists of all the methods for implementing the standard algorithms of Machine Learning | Scikit-Learn is not capable of employing categorical data to algorithms
| It has a simple and consistent interface that helps fit and transform the model over any dataset | It is heavily dependent on the SciPy stack
| It is the most suitable library for creating pipelines that help build a fast prototype | 
| It is also the best for the reliable deployment of Machine Learning models | 

### 5. TensorFlow
> Website: https://www.tensorflow.org/ 
> 
> GitHub Repository: https://github.com/tensorflow/tensorflow
> 
> Developed By: Google Brain Team
> 
> Primary Purpose: Deep Neural Networks!
>
![alt tag](https://intellipaat.com/blog/wp-content/uploads/2020/12/TensorFlow.jpg)

- From the Python Machine Learning library list, we have another most important one that is TensorFlow. It is one of the best open-source libraries used for building Machine Learning and Deep Learning models. It was created by Google’s research team for developing Google products. Eventually, it gained a lot of popularity, and it has proved to be a resourceful library for many business projects by now.
- TensorFlow has a powerful ecosystem of tools and resources for the community. Such kinds of toolsets enable engineers to perform research work on Machine Learning and Deep Learning to build efficient applications. Also, Google continues to add a variety of valuable features to TensorFlow to keep up with the pace of the highly competitive world. However, there are some advantages and disadvantages of using Tensorflow, and they are discussed below.

> Some of the essential areas in ML and DL where TensorFlow shines are:
> 
> * Handling deep neural networks
> 
> * Natural Language Processing
> 
> * Partial Differential Equation
> 
> * Abstraction capabilities
> 
> * Image, Text, and Speech recognition
> 
> * Effortless collaboration of ideas and code

| **Advantages** | **Disadvantages** |
| ---- | --- |
| The TensorFlow library helps us implement reinforcement learning | It runs considerably slower in comparison to those CPUs/GPUs that are using other frameworks
| We can straight away visualize Machine Learning models using TensorBoard, a tool in the TensorFlow library | The computational graphs in TensorFlow are slow when executed
| We can deploy the models built using TensorFlow on CPUs as well as GPUs | 

### 6. Keras

> Website: https://keras.io/
> 
> Github Repository: https://github.com/keras-team/keras
> 
> Developed By: various Developers, initially by Francois Chollet
> 
> Primary purpose: Focused on Neural Networks

![alt tag](https://intellipaat.com/blog/wp-content/uploads/2020/12/Keras.jpg)

- Keras is a widely used framework/library for fast and efficient experimentation related to deep neural networks. It is a standalone library comprehensively used for building ML/DL models that help engineers develop applications, such as Netflix, Uber, and many more.
- It is a user-friendly library designed to reduce the difficulty of developers in creating ML-based applications. Further, it provides multi-backend support that helps developers integrate models with a backend for providing the application with high stability

> Keras features several of the building blocks and tools necessary for creating a neural network such as:
>
> * Neural layers
> 
> * Activation and cost functions
> 
> * Objectives
> 
> * Batch normalization
>
> * Dropout
> 
> * Pooling

| **Advantages** | **Disadvantages** |
| ---- | --- |
| Keras is the best for research work and efficient prototyping | Keras is slow as it requires a computational graph before implementing an operation
| The Keras framework is portable | 
| It allows an easy representation of neural networks | 
| It is highly efficient for visualization and modeling | 

### 7. PyTorch

> Website: https://pytorch.org/
> 
> Github Repository: https://github.com/pytorch/pytorch
> 
> Developed By: Facebook AI Research lab (FAIR)
> 
> Primary purpose: Deep learning, Natural language Processing, and Computer Vision

![alt tag](https://intellipaat.com/blog/wp-content/uploads/2020/12/PyTorch-Python-Library-for-Machine-Learning.jpg)

- PyTorch is a framework that enables the execution of tensor computations. It helps create effective computational graphs and provides an extensive API for handling the errors of neural networks. Pytorch is a Facebook-developed ML library that is based on the Torch Library (an open-source ML library written in Lua Programming language). The project is written in Python Web Development, C++, and CUDA languages. Along with Python, PyTorch has extensions in both C and C++ languages. It is a competitor to Tensorflow as both of these libraries use tensors but it is easier to learn and has better integrability with Python. Although it supports NLP, but the main focus of the library is only on developing and training deep learning models only.

> The various modules PyTorch comes with, that help create and train neural networks:
> 
> * Tensors — torch.Tensor
> 
> * Optimizers — torch.optim module
> 
> * Neural Networks — nn module
> 
> * Autograd

| **Advantages** | **Disadvantages** |
| ---- | --- |
| Tensor computing with the ability for accelerated processing via Graphics Processing Units | The community for PyTorch is not extensive, and it lags to provide content for queries
| Easy to learn, use and integrate with the rest of the Python ecosystem | In comparison to other Python frameworks, PyTorch has lesser features in terms of providing visualizations and application debugging
| Support for neural networks built on a tape-based auto diff system |




## Reference source
1. https://www.zenesys.com/blog/top-10-python-libraries-for-machine-learning
2. https://intellipaat.com/blog/top-python-libraries-for-machine-learning/#no10
3. https://towardsdatascience.com/best-python-libraries-for-machine-learning-and-deep-learning-b0bd40c7e8c
