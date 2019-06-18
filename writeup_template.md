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
[image6]: ./WriteUp_Images/placeholder_small.png "Normal Image"
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


#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 





#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.


Over the recent years, and more particularly since the success of the [Darpa Grand Challenge](https://en.wikipedia.org/wiki/DARPA_Grand_Challenge) competitions a decade ago, the race towards the development of fully autonomous vehicle has accelerated tremendously. Many components make up an autonomous vehicle, and some of its most critical ones are the sensors and AI software that powers it. Moreover, with the increase in computational capabilities, we are now able to train complex and deep neural networks that are able to _learn_ crucial details, visual and beyond, and become the _brain_ of the car, instructing the vehicle on the next decisions to make.

In this writeup, we are going to cover how we can get train a deep learning model to predict steering wheel angles and help a virtual vehicle drive itself in a simulator. The model is created using Keras, relying on Tensorflow as the backend. This is project 3 of Term 1 of the [Udacity Self-Driving Car Engineer Nanodegree](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013). 



## Darken Image

Since some parts of our tracks are much darker, due to shadows or otherwise, we also darken a proportion of our images by multiplying all RGB color channels by a scalar randomly picked from a range:

```
def change_image_brightness_rgb(img, s_low=0.2, s_high=0.75):
    """
    Changes the image brightness by multiplying all RGB values by the same scalacar in [s_low, s_high).
    Returns the brightness adjusted image in RGB format.
    """
    img = img.astype(np.float32)
    s = np.random.uniform(s_low, s_high)
    img[:,:,:] *= s
    np.clip(img, 0, 255)
    return  img.astype(np.uint8)
```

![Original And Darkened Images](media/original_and_darkened_images.png)

## Random Shadow

Since we sometimes have patches of the track covered by a shadow, we also have to train our model to recognise them and not be spooked by them.

```
def add_random_shadow(img, w_low=0.6, w_high=0.85):
    """
    Overlays supplied image with a random shadow polygon
    The weight range (i.e. darkness) of the shadow can be configured via the interval [w_low, w_high)
    """
    cols, rows = (img.shape[0], img.shape[1])
    
    top_y = np.random.random_sample() * rows
    bottom_y = np.random.random_sample() * rows
    bottom_y_right = bottom_y + np.random.random_sample() * (rows - bottom_y)
    top_y_right = top_y + np.random.random_sample() * (rows - top_y)
    if np.random.random_sample() <= 0.5:
        bottom_y_right = bottom_y - np.random.random_sample() * (bottom_y)
        top_y_right = top_y - np.random.random_sample() * (top_y)

    
    poly = np.asarray([[ [top_y,0], [bottom_y, cols], [bottom_y_right, cols], [top_y_right,0]]], dtype=np.int32)
        
    mask_weight = np.random.uniform(w_low, w_high)
    origin_weight = 1 - mask_weight
    
    mask = np.copy(img).astype(np.int32)
    cv2.fillPoly(mask, poly, (0, 0, 0))
    #masked_image = cv2.bitwise_and(img, mask)
    
    return cv2.addWeighted(img.astype(np.int32), origin_weight, mask, mask_weight, 0).astype(np.uint8)
```

![Original And Shadowed Images](media/original_and_shadowed_images.png)

## Shift Image Left/Right/Up/Down

To combat the high number of neutral angles, and provide more variety to the dataset, we apply random shifts to the image, and add a given offset to the steering angle for every pixel shifted laterally. In our case we empirically settled on adding (or subtracting) 0.0035 for every pixel shifted to the left or right. Shifting the image up/down should cause the model to believe it is on the upward/downward slope. From experimentation, we believe that **this is possibily the most important augmentation** to get the car to drive properly.

```
# Read more about it here: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
def translate_image(img, st_angle, low_x_range, high_x_range, low_y_range, high_y_range, delta_st_angle_per_px):
    """
    Shifts the image right, left, up or down. 
    When performing a lateral shift, a delta proportional to the pixel shifts is added to the current steering angle 
    """
    rows, cols = (img.shape[0], img.shape[1])
    translation_x = np.random.randint(low_x_range, high_x_range) 
    translation_y = np.random.randint(low_y_range, high_y_range) 
    
    st_angle += translation_x * delta_st_angle_per_px

    translation_matrix = np.float32([[1, 0, translation_x],[0, 1, translation_y]])
    img = cv2.warpAffine(img, translation_matrix, (cols, rows))
    
    return img, st_angle 
```

![Original And Shifted Images](media/original_and_shifted_images.png)

## Image Augmentation Pipeline

Our image augmentation function is straightforward: each supplied image goes through a series of augmentations, each occuring with a probability p between 0 and 1. All the actually code of augmenting the image is delegated to the appropriate augmenter function.

```
def augment_image(img, st_angle, p=1.0):
    """
    Augment a given image, by applying a series of transformations, with a probability p.
    The steering angle may also be modified.
    Returns the tuple (augmented_image, new_steering_angle)
    """
    aug_img = img
    
    if np.random.random_sample() <= p: 
        aug_img = fliph_image(aug_img)
        st_angle = -st_angle
     
    if np.random.random_sample() <= p:
        aug_img = change_image_brightness_rgb(aug_img)
    
    if np.random.random_sample() <= p: 
        aug_img = add_random_shadow(aug_img, w_low=0.45)
            
    if np.random.random_sample() <= p:
        aug_img, st_angle = translate_image(aug_img, st_angle, -60, 61, -20, 21, 0.35/100.0)
            
    return aug_img, st_angle
```

## Keras Image Generator

Since we are generating new and augmented images _on the fly_ as we train the model, we create a Keras generator to produce new images at each batch:

```
def generate_images(df, target_dimensions, img_types, st_column, st_angle_calibrations, batch_size=100, shuffle=True, 
                    data_aug_pct=0.8, aug_likelihood=0.5, st_angle_threshold=0.05, neutral_drop_pct=0.25):
    """
    Generates images whose paths and steering angle are stored in the supplied dataframe object df
    Returns the tuple (batch,steering_angles)
    """
    # e.g. 160x320x3 for target_dimensions
    batch = np.zeros((batch_size, target_dimensions[0],  target_dimensions[1],  target_dimensions[2]), dtype=np.float32)
    steering_angles = np.zeros(batch_size)
    df_len = len(df)
    
    while True:
        k = 0
        while k < batch_size:            
            idx = np.random.randint(0, df_len)       

            for img_t, st_calib in zip(img_types, st_angle_calibrations):
                if k >= batch_size:
                    break
                                
                row = df.iloc[idx]
                st_angle = row[st_column]            
                
                # Drop neutral-ish steering angle images with some probability
                if abs(st_angle) < st_angle_threshold and np.random.random_sample() <= neutral_drop_pct :
                    continue
                    
                st_angle += st_calib                                                                
                img_type_path = row[img_t]  
                img = read_img(img_type_path)                
                
                # Resize image
                    
                img, st_angle = augment_image(img, st_angle, p=aug_likelihood) if np.random.random_sample() <= data_aug_pct else (img, st_angle)
                batch[k] = img
                steering_angles[k] = st_angle
                k += 1
            
        yield batch, np.clip(steering_angles, -1, 1)            
```

Note that we have the ability to drop a proportion of neutral angles, as well as keeping (i.e. not augmenting) a proportion of images at each batch. 

The following shows a small portion of augmted images from a batch:

![Augmented Images From Batch](media/augmented_images_in_batch.png)

Moreover, the accompanying histogram of steering angles of those augmented images shows much more balance:

![Steering Angles In Augmented Dataset](media/distribution_of_steering_angles_in_batch.png)

# Model

We initially tried a variant of the VGG architecture, with less layers and no transfer learning, and ultimately settled on the architecture used by the researchers at NVIDIA as it gave us the best results:

![NVIDIA Neural Net Architecture](media/nvidia_cnn_architecture.png)

## Model Tweaks

However, we added some slight tweaks to the model:
* We crop the top of the images so as to exclude the horizon (it not not play a role in *immediately* determining the steering angle)
* We resize the images to 66x200 as one the early layers to take advantage of the GPU
* We apply [BatchNormalization](https://www.quora.com/Why-does-batch-normalization-help) after each activation function for faster convergence.

## Model Architecture

The full architecture of the model is as follows:

* Input image is 160x320 (height x width format)
* Image is **vertically cropped** at the top, by removing half of the height (80 pixels), resulting in an image of 80x320
* Cropped image is normalized, to make sure the mean of our pixel distribution is 0
* Cropped image is *resized* to 66x200, using Tensorflow's [*tf.image.resize_images*](https://www.tensorflow.org/api_docs/python/tf/image/resize_images)
* We apply a series of 3 of 5x5 convolutional layers, using a stride of 2x2. Each convolutional layer is followed by a BatchNormalization operation to improve convergence. The respective depth of each layer is 24, 36 and 48 as we go deeper into the network
* We apply a 2 consecutive 3x3 convolutional layers, with a depth of 64. Each convolutional layer is immediately followed by a BatchNormalization layer
* We flatten the input at this stage and enter the fully connected phase
* We  apply a series of fully connected layers, of gradually decreasing sizes: 1164, 200, 50 and 10
* The output layer is obviously of size 1, since we predict only one variable, the steering wheel angle.

## Activations And Regularization

The activation function used across all layers, bar the last one, is [ReLU](https://stats.stackexchange.com/questions/226923/why-do-we-use-relu-in-neural-networks-and-how-do-we-use-it). We tried [ELU](https://www.quora.com/How-does-ELU-activation-function-help-convergence-and-whats-its-advantages-over-ReLU-or-sigmoid-or-tanh-function) as well but got better results with ReLU + BatchNormalization. We use the [Mean Squared Error](https://en.wikipedia.org/wiki/Mean_squared_error) activation for the output layer since this is a regression problem, not a classification one.

As stated in the previous section, we employed BatchNormalization to hasten convergence. We did try some degree of [Dropout](https://www.quora.com/What-does-a-dropout-in-neural-networks-mean) but did not find any noticeable difference. We believe the fact that we are generating new images at every batch and discarding some of the neutral angle images help in reducing overfitting. Moreover, we did not apply any [MaxPool](http://cs231n.github.io/convolutional-networks/#pool) operation to our NVIDIA network (although we tried on the VGG inspired one) as it would have required significant changes in the architecture since we would have reduced dimensionality much earlier. Moreover, we did have the time to experiment with L2 regularisation, but plan to try it in the future. 

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
