# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
import argparse
import ast
import logging
import os
import torch
import json
import PIL
from PIL import Image
import requests
import clip
import numpy as np
import io


torch.set_grad_enabled(False)
JSON_CONTENT_TYPE = "application/json"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info("Loading model")


def model_fn(model_dir):
    logger.info(f"inside model_fn, model_dir= {model_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device Type: {}".format(device))
    # clip_model, clip_preprocessor = clip.load("ViT-B/32", device="cpu")
    clip_model, clip_preprocessor = torch.load(
        "/opt/ml/model/model.pth", map_location=torch.device("cpu")
    )
    clip_model.requires_grad_(False)
    logger.info("Model loaded with Clip")
    return clip_model, clip_preprocessor


def predict_fn(data, model):
    with torch.no_grad():
        try:
            logger.info(f"Got input Data: {data}")
            thumbnail_url = data.get("thumbnail_url")
            logger.info(f"thumbnail_url: {thumbnail_url}")
            parsed_image = Image.open(requests.get(thumbnail_url, stream=True).raw)
            logger.info(f"parsed_image: {parsed_image}")
            img_pre = model[1](parsed_image).unsqueeze(0)
            logger.info(f"img_pre: {img_pre}")
            logger.info(f"img_pre.shape: {img_pre.shape}")
            img_features = model[0].encode_image(img_pre).cpu().numpy()
        except PIL.UnidentifiedImageError:
            logger.info("Image not found")
            img_features = np.zeros((1, 512))
    return img_features


def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):
    logger.info(f"serialized_input_data object: {serialized_input_data}")
    if content_type == JSON_CONTENT_TYPE:
        input_data = json.loads(serialized_input_data)
        logger.info(f"input_data object: {input_data}")
        return input_data
    else:
        raise Exception("Requested unsupported ContentType in Accept: " + content_type)
        return


def output_fn(prediction, content_type):
    logger.info(f"prediction object before: {prediction}, type: {type(prediction)}")
    if prediction is None:
        return "No Image Found"
    prediction = prediction.round(4).tolist()
    prediction_result = json.dumps(prediction)
    logger.info(f"prediction_result object: {prediction_result}")
    return prediction_result
