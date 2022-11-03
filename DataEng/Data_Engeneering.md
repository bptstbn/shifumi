Research Question along one of the following dimensions:
- __Descriptive:__ Overview of X
- __Relational:__ Compare X with Y
- __Causal:__ What effect does X have?


Research Question:
__combine uncombinable datasets-- hand detection or reduce complexity without loosing information or expand existing data by rotating, scaling and transforming__
__does heavy preprocessing(only black and white) ( removing the background (color, faces etc) and making hands only white) is superior to less prepocessing (greyscale) minimal preprcessing (only same size)__ 

Problems:
- how to get rid of the faces

What kind of images do we want to have at the end of the preprocessing
- simple black and white pictures of a hand -> get rid of color of human color
- same image sizes of course
- images from top/front/both ????


Data Engineering is the act of collecting, translating, and validating data for analysis.

what do we need to do:
- what kind of format do we want to use -- images ....
- what kind of quality does our data need to have


4 steps of data engineering:
- Data selection,
  - filter methods, wrapper methods, or embedded methods
  - dealing with unbalanced classes by applying over-sampling or under-sampling strategies
- data cleaning,
  - error detection and error correction steps
- feature engineering,
  - depends on the ml learning task at hand
- and data standardization 
  - unifying input data
  - data and input data transformation pipelines for data pre-processing and feature creation


Some things to consider:

- Principal Component analysis in order to compress images


!!TODO!! need to check which datasets are the same
!!TODO!! which resolution do these datasets have 
Datasets:
- cgi
  - [Rock Paper Scissors Dataset](https://www.kaggle.com/datasets/sanikamal/rock-paper-scissors-dataset) animated hands, white background, from above
  - [Rock Paper Scissors dataset](https://www.kaggle.com/datasets/glushko/rock-paper-scissors-dataset): animated hands, different colors, always from top
  - [rock_paper_scissors](https://www.tensorflow.org/datasets/catalog/rock_paper_scissors): animated hands from tensorflow, all form above
  - [lecture 1](https://public.roboflow.com/classification/rock-paper-scissors): animated hands, from top, different colors
  - [rock-paper-scissors](https://www.kaggle.com/datasets/frtgnn/rock-paper-scissor): hands from above
- real
  - [Rock-Paper-Scissors Images](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors): real hands, green background, only hands from above
  - [Rock, Paper, Scissors Dataset](https://www.kaggle.com/datasets/glushko/rock-paper-scissors-dataset): real humans with head in background rotated hands
-custom
  - should we add our custom dataset, and make some pictures
    - either from pictures or from video stream


Additional information:
- [little example](https://towardsdatascience.com/building-a-rock-paper-scissors-ai-using-tensorflow-and-opencv-d5fc44fc8222): explains how to use cv2 and python to build rock paper scissors
- [example in lab view](https://github.com/asjad99/Rock-Paper-Scissors-): has the image i would like to have in the end


What to do today:
- What kind of images do we want to have in the end
- What do wen want to have finished by next week
- decide who writes the preprocessor that creates black white image of a hand
- who combines the dataset and throws them into the same format/ what format to be used