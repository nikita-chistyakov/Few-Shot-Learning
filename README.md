
# Image classifier using Few Shot technique
## Overview
A web application to classify images based on their similarities using few shot (siamese model) and pretrained model VGG16 used to predict the image details.

## Use Case
To classify agriculture crops based on their similarities and to predict the crop details. Display results in a Web App using a web browser.

## Features
#### Pretrained model
Here user can to upload the crop image and in result field it will give the crop name, recommened soil type, and alternative soil type.
#### Siamese Model
Here user sould give one query image and 3 support image using sieamese model it find the similarities between the crop and gives the query image and similar image. So that we can check the details of the similar crop in pretrained model
#### Botanical AI assistant at your service 💬
Here user can get all those crop details from our chat bot support

## Technologies used

Web application [Streamlit-python]
Deep learning [tensorflow-Keras]
Computer vision [Siamese algorithm]

## Dataset 

We had taken data from kaggle, we used 7 classes of crops [cucumber, jute, maize, wheat, rice,mustard-oil, tobacoo-plant].    
The link is provided below:  https://www.kaggle.com/datasets/mdwaquarazam/agricultural-crops-image-classification 
 

## Model

Our first pretrained model we have used VGG16 and achived 75% of accuracy and tested the query images, then model saved and stored in drive.

Using the pretrained model we tunned our few short siamese model to get more accuracy and tested the similar image by giving support set images of 3 and query 1 image,moddel saved.






## Acknowledgements

 - Mentor: [Alaa Bakhti](https://github.com/bachtn)
 - Contributor: [Nikita Chistyakov](https://github.com/nikita-chistyakov)
 - Contributor: [Skander Ben brik](https://github.com/Skander79)
 - Contributor: [Ranjithavadivel] (https://github.com/Ranjithavadivel)



## Installation

- Clone the repository

```bash
  git clone https://github.com/Ranjithavadivel/DSA_actionLearning_Team7.git

```

-Download the model siamese_model.h5 and crop_classification_model_fine_tuned.h5 and place it inside the working folder 

[Drive link](https://drive.google.com/drive/folders/1SsqNUJ0blVwc3D76WwnOFH7pth5ydeJI?usp=sharing)

- Run the webapp 

```bash
  cd DSA_actionLearning_Team7
  streamlit run webapp.py
```

- Setup 

```bash
  pip install -r requirement.txt
```
    
