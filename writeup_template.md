# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./WriteUp_Images/nvidia_cnn_architecture.png "Model Visualization"
[image2]: ./WriteUp_Images/ModelBash.png "model visu"
[image3]: ./WriteUp_Images/DataSet3Epoch4.png "traiing Save"
[image4]: ./WriteUp_Images/steeringAngles.png "Recovery Image"
[image5]: ./WriteUp_Images/placeholder_small.png "Recovery Image"
[image6]: ./WriteUp_Images/placeholder_small.png "Model Architecture"
[image7]: ./WriteUp_Images/placeholder_small.png "Flipped Image"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. Different training sets are used to variate the input of the model. Therefore, different input files are used in the code.

### Model Architecture and Training Strategy

#### 1. Datasets

The model used in this excersise uses data from the provided siumulator. We ended up with 3 different datasets for training:
1. Udacity's dataset on track 1 
2. Manually created dataset on track 1
3. Manually created dataset where the car was driven the other way around the track and some "recovery" modes where performed 

All collected data sets are combined to one big data package and one csv file with the measurements.

#### 2. Dataset Exploration

The measurements of the steering angles are very inconsistant, since the keyboard is used as an input. The model acts very incinsistant to such kind of data. As also seen in the image below, we have a huge amount of measurements with a "0" degree angle. Therefore, the model is overtrained to drive straigth.

![alt text][image4]

#### 3. Dataset Split

From the collected data set, a validation set is split up to validate the trained model. The "train_test_split" function is used for that purpous. The set is split into parts of 80% training data and 20% validation data.

```
d_train, d_valid = train_test_split(driving_log, test_size=0.2, random_state=42)
```
This gives us 10510 data points for training and 2628 data points for validation. The number of points in the two sets will be increased later on with image processing.

#### 4. Camera And Steering Wheel Angle Calibration

As known from the image capturing, the cameras on the side of the car collect a different point of few from the scene. Therefore, the steering angle for those measurements have to be addapted.

![alt text][image4]

First of all, we add a correction value to images captured by either left or right cameras:
* for the left camera we want the car to steer to the right (positive offset)
* for the right camera we want the car to steer to the left (negative offset)

```
 if i==1:
    measurement = measurement + 0.25
 elif i==2:
    measurement = measurement - 0.25
```

#### 5. Image Horizontal Flip

To increase the amount of training and validation data without driving many times around the track, we use a image flip on every defined training image. This gives us 6 times the data we hade before. We are flipping the images from the center camera and also the images from the side cameras. This leads to a total amound of 63060 data points for training.


#### 6. Keras Image Generator

We are generating new and agmented images on the fly as we train the model. This gives us some advantages in the starage for the model. We are not loading all the data at once in the memory. The mentioned image flip happens in the data generator.

```
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            for batch_sample in batch_samples:
                # Import also the images on the side
                for i in range(3):
                    name = './Data3/IMG/' + batch_sample[i].split('/')[-1]
                    image = cv2.imread(name)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    measurement = float(batch_sample[3])
                    # i == 1 second row in data package - will be corrected with +0.25
                    if i==1:
                        measurement = measurement + 0.25
                    # i == 2 third row in data package - will be corrected with -0.25
                    elif i==2:
                        measurement = measurement - 0.25
                    
                    # Append all images in pipeline
                    images.append(image)
                    measurements.append(measurement)

            augmented_images, augmented_measurements = [], []
            # Flip the images to double the data number
            for image, measurement in zip(images, measurements):
                    augmented_images.append(image)
                    augmented_measurements.append(measurement)
                    augmented_images.append(cv2.flip(image, 1))
                    augmented_measurements.append(measurement * -1.0)
                
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            
            yield shuffle(X_train, y_train)       
```

## Model

The initially used model is a NVIDIA research model with few modifications. The structure of the model is the following:

![alt text][image1]

### Model Tweaks

However, the model is slightly addopted.
* A crop of the image is added. Since we know that we do not need the top of the images for training. This could confuse the model
* The size of the images is reshaped to 66x200
* The images are normalized

## Model Architecture

The full architecture of the model is as follows:

![alt text][image6]

Some dropout functions are used to prevent overfitting of the model.

## Activations And Regularization

For all layers, the activation function is the ReLu function. ELU activation was also tried but ReLu gives the best results for training. 

## Training And Results

We trained the model using [Adam](https://www.quora.com/Can-you-explain-basic-intuition-behind-ADAM-a-method-for-stochastic-optimization) as the optimizer and a learning rate of 0.001. After much, tweaking of parameters, and experimentation of multiple models, we ended up with one that is able power our virtual car to drive autonomously on both tracks. 


![Car Drives Autonomously On Track 1](media/nvidia_model_track1_drives_well.gif)

We can see how the vehicle effectively manages to drive down a steep slope on track 2.

![Car Drives Autonomously On Track 2](media/nvidia_model_track_2_15MPH_10seconds.gif)


We also show what the front camera sees when driving autonomously on track 2. We can see how the car tries to stick to the lane and not go in the middle, as we ourselves strived to drive on only one side of the road during our data collection phase. This shows the model has indeed learned to stay within its lane.

![Front Camera Vision On Track 2](media/track_2_center_camera.gif)

# Conclusion

We have shown that it is possible to create a model that reliably predicts steering wheel angles for a vehicle using a deep neural network and a plethora of data augmentation techniques. While we have obtained encouraging results, we would like in the future to explore the following:
* Take into account speed and throttle in the model
* Get the car to drive faster than 15-20MPH
* Experiment with models based VGG/ResNets/Inception via transfer learning
* Use Recurrent Neural Networks like in this [paper](http://cs231n.stanford.edu/reports/2017/pdfs/626.pdf) from people using the Udacity dataset
* Experiment with Reinforcement Learning

As can be seen, there are many areas we could explore to push this project further and obtain even more convincing results.

From a personal perspective, I have tremendously enjoyed this project, the hardest so far, as it enabled me to gain more practical experience of hyperparameter tweaking, data augmentation, and dataset balancing among other important concepts. I feel my intuition of neural network architectures has deepened as well. 

# Acknowledgements

I would also like to thank to thank my Udacity mentor Dylan for his support and sound advice, as well as the Udacity students before my cohort who explained how they approached this project via blog posts.
