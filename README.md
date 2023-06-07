# **CLIP Age Predictor 📸⏳**

Welcome to the **CLIP Age Predictor**! 👋 This Python-based project is a reimagination of a prior implementation found at [replicate.com/andreasjansson/clip-age-predictor](https://replicate.com/andreasjansson/clip-age-predictor). Due to some compatibility issues, we took the baton and brought it to the finish line, ensuring it's up-to-date with the latest libraries and frameworks. 👩‍💻

We no longer get the error:<br>
"""<br>
😵 *Uh oh! This model can't be run on Replicate because it was built with a version of Cog that is no longer supported. Consider opening an issue on the model's GitHub repository to see if it can be updated to use a recent version of Cog. If you need any help, please hop into our Discord channel or email us about it.*<br>
"""

## What Does This Code Do? 🧐

In essence, our code uses OpenAI's CLIP model 🚀 to predict a person's age based on a provided image, creating a novel and fun AI-powered age guessing game!

Here's the breakdown:

1. **Setting up Prompts:** We first set up 'prompts', sentences like `"this person is {age} years old"`. We have 99 of these, ranging from 1 to 99 years old.

2. **Model Preparation:** We load the CLIP model and preprocess these prompts, converting them into a format (a tensor of encoded text) that the model understands.

3. **Image Processing:** The input image is also preprocessed to ensure that the model can extract meaningful features from it.

4. **Cosine Similarity Computation:** The core of our age prediction lies in computing the cosine similarity between the processed image and each of our age prompts. 

Cosine similarity measures the cosine of the angle between two vectors. This value can range from -1 to 1, indicating complete oppositeness and perfect similarity, respectively. We're exploiting the fact that when vectors are similar (meaning they point in roughly the same direction), their cosine similarity approaches 1.

Applying this to our scenario, we're comparing the 'direction' or 'meaning' captured by our image with the 'directions' or 'meanings' of our age prompts. If the image aligns more closely with the vector for "this person is 25 years old" than any other prompt, we get a high cosine similarity, indicating that the person in the image is likely around 25 years old!

## Under the Hood: CLIP and Contrastive Losses 🧠🔧

CLIP (Contrastive Language–Image Pretraining) is a unique model created by OpenAI, adept at understanding and associating images and text. Instead of training the model using traditional loss functions, CLIP employs 'contrastive losses', an innovative approach that significantly boosts the model's generalizability.

But what exactly is a contrastive loss?

![Contrastive Pretraining](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*tz_yyNvvna59tYDoqD4CUg.png)

Think of contrastive losses as a guiding hand that steers the model during training. They encourage the model to pull together related items (positive pairs like an image and its corresponding text) while pushing apart unrelated ones (negative pairs like an image and a random text). This is done by minimizing the distance between positive pairs and maximizing the distance between negative pairs in the embedding space.

Unlike classic ImageNet-style loss functions that treat each classification problem independently, contrastive losses teach the model a more comprehensive understanding of the data. The model isn't just looking at individual images or texts; it's learning to perceive the broader relationships between text and image data. This capacity to grasp context, draw analogies, and comprehend abstract concepts makes CLIP an ideal candidate for tasks like age prediction.

By training CLIP with a vast array of internet text–image pairs, the model has learned to create accurate mappings between images and a large space of textual descriptions, making it a powerful tool for tasks that require understanding of images in relation to text prompts. 

In our age predictor, we leverage this ability of CLIP to gauge the 'similarity' between an image and our age-related text prompts. This forms the core of our model's age prediction mechanism, displaying the strength and versatility of contrastive learning methods in practical applications.

## The Art and Science of Predicting Age 🎨⚗️

Predicting age in this manner might seem a bit 'hacky', but it's a testament to the flexibility of the CLIP model. CLIP is not explicitly trained to guess ages, yet it manages to perform this task by leveraging its understanding of the associations between images and prompts.

Remember, its predictions are educated guesses based on its training and the prompts provided, so the quality of input can influence the accuracy of its predictions. 📈🎯

## Building with Cog 🛠️

Before you start, make sure you have [Cog](https://github.com/replicate/cog) installed. Cog streamlines the process of building and deploying models, provided you have a CUDA-enabled GPU compatible with PyTorch > 1.8. 

Just run the following command in your terminal:
```zsh
cog build
```
This command will start the building process. 

## Executing a Prediction 🏃‍♀️

Executing a prediction is straightforward. After successfully building with Cog, you can run the predictor with the following command:
```
cog predict -i image=@image.jpg
```
For example, with an input image of a person, the script provides an age prediction like so:
```zsh
Running prediction...
19
```

**I'm actually 22 in the picture, so nice guess!**

That's the model's best guess at the person's age in the image. Remember, it's an estimate, not an absolute fact. Enjoy the magic of artificial intelligence, brought to life by the wonders of CLIP and the ingenious use of contrastive losses!