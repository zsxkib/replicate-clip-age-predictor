import clip
import torch
import numpy as np
from PIL import Image
from typing import Any, Union
from replicate import BasePredictor, Input, Path


AGES = list(range(1, 100))


class Predictor(BasePredictor):
    def setup(self) -> None:
        """
        Initializes the setup for the prediction model.

        This method is called to setup the model and prepare the features for age prediction.
        It prepares the prompts for all ages and encodes them into features.

        Examples
        --------
        >>> p = Predictor()
        >>> p.setup()
        """

        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device, jit=False)
        texts: list = [f"this person is {age} years old" for age in AGES]
        prompts = clip.tokenize(texts).to(self.device)
        with torch.no_grad():
            self.prompt_features = self.model.encode_text(prompts)
        self.prompt_features /= self.prompt_features.norm(dim=-1, keepdim=True)

    def compute_similarity(self, image_features: torch.Tensor, prompt_features: torch.Tensor) -> np.ndarray:
        """
        Compute similarity between image features and prompt features.

        This method calculates the similarity between the image features and the encoded age prompt features.

        Parameters
        ----------
        image_features : torch.Tensor
            Features encoded from the image.
        prompt_features : torch.Tensor
            Features encoded from age prompts.

        Returns
        -------
        np.ndarray
            The similarity score between the image features and each age prompt feature.

        Examples
        --------
        >>> p = Predictor()
        >>> p.setup()
        >>> img_features = ...
        >>> similarity = p.compute_similarity(img_features, p.prompt_features)
        """

        return (100.0 * image_features @ prompt_features.T).softmax(dim=-1).detach().cpu().numpy()

    def predict(self, image: Path = Input(description="Input image of the person's age we'd like to predict")) -> str:
        """
        Predict the age of the person in the provided image.

        This method is called to make predictions based on the provided image.
        It processes the image, encodes it into features, and computes similarity with age prompts to predict the age.

        Parameters
        ----------
        image : Path
            The path of the image for which age prediction is to be done.

        Returns
        -------
        str
            The predicted age of the person in the image.

        Examples
        --------
        >>> p = Predictor()
        >>> p.setup()
        >>> p.predict("person.jpg")
        '25'
        """

        pil_image = Image.open(image)
        with torch.no_grad():
            image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            image_features = self.model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = self.compute_similarity(image_features, self.prompt_features)
        age = AGES[np.argmax(similarity[0])]
        return f"{age}"
