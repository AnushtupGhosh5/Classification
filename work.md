## This is a research project

We are using val loss, for the this research to choose the best model, 
this research is no image classification, and we are classifing skin cancer.
Datasets for this research are mainly ISIC16, ISIC17, ISIC18, MILK10k
OUr novel method which we are Targetting now is EDF, Expert Disagreement Fusion, 
Only one base model will be used like Resnet, Efficientnet, Densebet, Mobilenet, Convnext, etc, and this method will be applied on of the best performing base models to reach near SOTA.

# The problem

The accuracy of the basemodels are not currently the best for ISIC16, ISIC17, ISIC18, MILK10k, so our fusion method wont be that much effective, its quite less than the SOTA, 
I have tried a lot of preprocessing methods taking inspiration from the SEFFNET_Codes notebooks, but still could not increase the accuracy of the basemodels, it did improve but still it lacks behind a lot. 

## Dont's

Dont change the criteria for saving the base model, it has to be the validation loss.
The split for test set is official, we cant change that.

## OBjective for now

we need to increase the accuracy and other metrics of the model adn reach near sota, do whatever you feel that can be done without violating the ethics and guidelines of research. Lets first increase the accuracy of the basemodels , later we will use our fusion techniques to improve
