# Rock-Scissors-Paper-ML-CSD-2023
# Rock-Scissors-Paper Agent
This is the final project of Machine Learning Course (CSD - AUTH)

# Goal
The goal of the project is to create an intelligent agent that will learn to play the popular Rock-Scissors-Paper game. Particularly, a photo of Rock, Scissors or Paper will be presented to the agent, and then it has to predict the winning symbol. That is:
- Rock is beaten by Paper
- Scissors are beaten by Rock
- Paper is beaten by Scissors  
# Game Labels
- Paper --> 0
- Rock --> 1
- Scissors --> 2
# Dataset and Preprocessing (In Colab env)
The dataset utilized in this project is [Rock-Paper-Scissors Images](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors) from Kaggle site.  
![](https://github.com/nikifori/Rock-Scissors-Paper-ML-CSD-2023/blob/main/artifacts/rock_scissors_paper_images_sample.png)  
  
Dataset Information:  

> **CONTENTS**: The dataset contains a total of 2188 images corresponding to the 'Rock' (726 images), 'Paper' (710 images) and 'Scissors' (752 images) hand gestures of the Rock-Paper-Scissors game. All image are taken on a green background with relatively consistent ligithing and white balance.   
> **FORMAT**: All images are RGB images of 300 pixels wide by 200 pixels high in .png format. The images are separated in three sub-folders named 'rock', 'paper' and 'scissors' according to their respective class.

The dataset is saved in Dropbox as a zip file. Code snippets to carry out the below processes were written:
* Download zip file
* Extract zip file
* Check the number of instances per class (number of photos in each folder)
* Create new workspace directory structure so as to adhere to [ImageFolder](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html) format
* Split images to Train, Test and Validation sets retaining the propotion of all classes equal to all sets  

The script is deterministic so each time it is executed, the results and the splits are identical.

# How is the game going to be played ?
1. Select an image from the test set randomly
2. Apply to the image:
	* vertical flip with probability, P_v = 0.5
	* horizontal flip with probability, P_h = 0.5
	* add Gaussian noise with μ = 0 and σ = pixels_max_value * 0.05, pixels_max_value = 1 or 255
3. Custom Agent receives the image
4. Custom Agent predicts optimal gesture (determenistic based on Custom Agent prediction)
5. Calculate reward
	* If win --> +1
	* if draw --> +0
	* if loss --> -1
6. Accumulate reward
7. Print final accumulated reward

# Test Baseline models
In this section I load the data using ImageFolder class. Then, I decided to augment both Train and Validation sets using the listed transforms below:
* simple_transform:
	* Grayscale
	* Resize to (75, 50)
	* ToTensor --> scale inputs to (0,1)
* simple_vertical_mainly_transform --> 
	* Grayscale
	* Resize to (75, 50)
	* ToTensor --> scale inputs to (0,1)
	* VerticalFlip to all images
	* HorizontalFlip with 0.5 probability
	* Add Gaussian noise as in (2) in "How is the game going to be played ?"
* simple_horizontal_mainly_transform --> 
	* Grayscale
	* Resize to (75, 50)
	* ToTensor --> scale inputs to (0,1)
	* VerticalFlip with 0.5 probability
	* HorizontalFlip to all images
	* Add Gaussian noise as in (2) in "How is the game going to be played ?"  

Then, for each dataset (Train, Validation) I combined the datasets with the aforesaid "transforms" and convert them to numpy arrays (flattened).

Afterwards, I trained 3 Baseline models on the Train set:
* A Support Vector Machine Classifier (sklearn)
* A Random Forest Classifier (sklearn)
* A k-Nearest Neighbors Classifier (sklearn)

For all the models we maintain the default parameters and the Accuracy on the Validation set is shown below in the table.  
| Model           | Accuracy |
|-----------------|----------|
| SVC             | 0.908    |
| Random Forest   | 0.912    |
| KNN             | 0.882    |

Comment: Not bad for Baseline models at all.  
  
Finally, I tested those Baseline models on Test set for 1000 rounds using as data the Test set that models have never seen before. This procedure is based on "How is the game going to be played ?" section. Thus, I calculated the cummulative reward at the end of all rounds for each model. During the game loop, each time, I choose randomly a sample from the test set with replacement because Test set contains about ~450 instances. Obviously, due to the utilization of "How is the game going to be played ?" random transforms, it is not frequent to obtain the exact same datum. The cummulative results are presented below:  
| Model           | Cummulative Reward|
|-----------------|----------|
| SVC             | 893/1000		|
| Random Forest   | 895/1000		|
| KNN             | 880/1000		|  

![](https://github.com/nikifori/Rock-Scissors-Paper-ML-CSD-2023/blob/main/artifacts/baseline_models_cummulative_rewards_test.png)  

# Create Custom Agent using as backbone ViT-base/16
In this section, ImageFolder was also exploited for loading the data and the same data augmentation approach was followed as the former section.  
  
I took advantage of [ViT](https://huggingface.co/google/vit-base-patch16-224) because it has demonstated exceptional results on downstream image tasks. I added on top of it a Linear, a ReLU, a Dropout and a final Linear layer as it is depicted in the architecture underneath. Snow flake reflects to the parameters that are frozen and flame to the trainable ones.  

![](https://github.com/nikifori/Rock-Scissors-Paper-ML-CSD-2023/blob/main/artifacts/vit_model_architecture.png)  

I trained for 30 epochs utilizing cross-entropy loss and Adam optimizer on a T4 GPU in Google Colab, while logging metrics on [Weights & Biases](https://wandb.ai/site).  
  
![](https://github.com/nikifori/Rock-Scissors-Paper-ML-CSD-2023/blob/main/artifacts/training_metrics_wandb.png)    
  
Based on the metrics, I selected the model at the 17th epoch, because it scored Accuracy score of 1.0 on the Validation set while having the smaller validation loss.  
  
Ultimately, I tested the selected Custom Agent on the test set for 100 games of 1000 turns. The average cummulative reward was **998.76/1000**.  

  **Pretty much perfect!**

# Test Custom Agent on web images
| Image | Image | Image |
|:-----:|:-----:|:-----:|
| <img src="https://github.com/nikifori/Rock-Scissors-Paper-ML-CSD-2023/assets/57923121/d00fa170-16bb-4d74-aa6f-218e0b1cded6" width="100" height="100"> | <img src="https://github.com/nikifori/Rock-Scissors-Paper-ML-CSD-2023/assets/57923121/91dca13b-2679-4c27-a6a3-91cab8b3072b" width="100" height="100"> | <img src="https://github.com/nikifori/Rock-Scissors-Paper-ML-CSD-2023/assets/57923121/78802400-1dc2-4926-844b-9578224c8499" width="100" height="100"> |
| Prediction: Paper | Prediction: Rock | Prediction: Scissors |



  



