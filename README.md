# pneumonia-classification
🩺 Investigation of image processing techniques that increase the accuracy of a neural network implementation for the classification of pneumonia. 

![xray](/aes/xray.jpg)

## 📗 About

The purpose of this approach is dual; first, to build a neural network that can accurately distinguish between pneumonia and non-pneumonia from X-ray images, and, second, to explore the various image processing techniques that can lead to more robust models and, consequently, to more accurate results. *Spoiler: they do! 🎆* 

In the frame of this project, a very simple Convolutional Neural Network was constructed, utilizing tensorflow and keras. *Transfer learning* was also implemented for the initialization of the first layers of the neural network, taking the weights from a neural network model trained on **Imagenet**. The use of weights from a pre-trained model can provide critical advantages for both the performance of a neural network and its accuracy; essentially the first models capture general details and fine-tuning is a much better approach than randomly initializing them.

The classes that we will work with for this approach are normal/pneumonia, and we will not delve into the sub-classes of the pneumonia class (viral/bacterial). The data utilized for this project/walkthrough could not be uploaded to this GitHub repository due to storage space issues. However, it is readily available as a Kaggle dataset, [here](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), which is where it was originally downloaded from.

## 👟 Walkthrough

A project walkthrough with comments and observations can be found in this [Python notebook](/src/PneumoniaClassification.ipynb). The auxiliary functions utilized throughout this project are also separately given in an [auxiliary script](/src/PneumoniaClassification.aux_arc.py).

The weights from the pre-trained model trained on Imagenet can be found in  the [aux directory](/aux/).

The final models are also supplied with this repository, and can be found in the [out directory](/out/).

### 🎆 So, how is this different from other approaches?

Contrary to existing approaches that usually focus on more sophisticated NN architectures or on transfer learning (although we also do that), the project's approach is more data-centric. It leverages on image processing techniques that reshape the input and, consequently, "help" the model perform better. In the frame of this project, 3 different combinations of image processing techniques are compared, and we observe an increase in accuracy from ~75% to 100%!
