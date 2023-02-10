import os
import requests
import base64
import json
import subprocess

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_single_tag_keys

image_cache_dir = os.path.join(os.path.dirname(__file__), 'image-cache')
os.makedirs(image_cache_dir, exist_ok=True)


class ImageClassifierAPI(LabelStudioMLBase):

    def __init__(self, freeze_extractor=False, **kwargs):
        super(ImageClassifierAPI, self).__init__(**kwargs)
        self.from_name, self.to_name, self.value, self.classes = get_single_tag_keys(
            self.parsed_label_config, 'Choices', 'Image')

    def predict(self, tasks, **kwargs):
        image_urls = [task['data'][self.value] for task in tasks]
        subprocess.check_call(f"aws s3 cp {image_urls[0]} {image_cache_dir}", shell=True)
        subprocess.check_call(f"ls -al {image_cache_dir}", shell=True)
        file_name = image_urls[0].split("/")[-1]

        print(f"{image_urls=}")
        image_path = os.path.join(image_cache_dir, file_name)
        print(f"{image_path=}")

        url = "https://5hbrm6xu63.execute-api.ap-south-1.amazonaws.com"
        
        with open(image_path, "rb") as f:
            ext = image_path.split('.')[-1]
            prefix = f'data:image/{ext};base64,'
            base64_data = prefix + base64.b64encode(f.read()).decode('utf-8')

        payload = json.dumps({
                    "body": [base64_data]
                })
        headers = {"content-type": "application/json"}

        response = requests.request("POST", url, json=payload, headers=headers)

        print(response.json())
        prediction = response.json()[0]
        sorted_prediction = sorted(prediction.items(), key=lambda x:x[1])
        predicted_label = sorted_prediction[-1][0]
        predicted_score = sorted_prediction[-1][1]


        predictions = []
        # prediction result for the single task
        result = [{
            'from_name': self.from_name,
            'to_name': self.to_name,
            'type': 'choices',
            'value': {'choices': [predicted_label]}
        }]

        # expand predictions with their scores for all tasks
        predictions.append({'result': result, 'score': float(predicted_score)})

        return predictions