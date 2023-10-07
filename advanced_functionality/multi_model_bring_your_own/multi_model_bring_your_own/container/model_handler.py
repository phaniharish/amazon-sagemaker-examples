"""
ModelHandler defines an example model handler for load and inference requests for MXNet CPU models
"""
import torch
from collections import namedtuple
from PIL import Image
import requests
import clip
import numpy as np
import io


class ModelHandler(object):
    """
    A sample Model handler implementation.
    """

    def __init__(self):
        self.initialized = False
        self.clip_model = None
        self.preprocessor = None
        self.shapes = None

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self.initialized = True
        # properties = context.system_properties
        # # Contains the url parameter passed to the load request
        # model_dir = properties.get("model_dir")
        # gpu_id = properties.get("gpu_id")
        
        self.clip_model, self.clip_preprocessor = clip.load("ViT-B/32", device="cpu")
        self.clip_model.requires_grad_(False)

    def preprocess(self, request):
        """
        Transform raw input into model input data.
        :param request: list of raw requests
        :return: list of preprocessed model input data
        """
        # Take the input data and pre-process it make it inference ready

        tensor_list = []
        for idx, data in enumerate(request):
            image = data.get("body")
            print(image.decode())
            parsed_image = Image.open(requests.get(image.decode(), stream=True).raw)
            # print(parsed_image)
            img_pre = self.clip_preprocessor(
                    parsed_image
                ).unsqueeze(0)
            # print(img_pre)
            print(img_pre.shape)
        tensor_list.append(img_pre)
        return tensor_list

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data list
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        with torch.no_grad():
            image_features = self.clip_model.encode_image(torch.cat(model_input)).numpy()
            expanded_image_features = np.zeros((1024))
            expanded_image_features[:512] = image_features[0]
        return expanded_image_features

    def postprocess(self, inference_output):
        """
        Return predict result in as list.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        return [np.array2string(inference_output, separator=",", precision=4)]
        # prob = np.squeeze(inference_output)
        # a = np.argsort(prob)[::-1]
        # return [["probability=%f, class=%s" % (prob[i], self.labels[i]) for i in a[0:5]]]

    def handle(self, data, context):
        """
        Call preprocess, inference and post-process functions
        :param data: input data
        :param context: mms context
        """

        model_input = self.preprocess(data)
        model_out = self.inference(model_input)
        print(model_out.shape)
        result = self.postprocess(model_out)
        return result


_service = ModelHandler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)
