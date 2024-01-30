import boto3
import json

# Use Amazon Rekognition for label detection in photos
def detect_labels(photo):
    client = boto3.client('rekognition')

    with open(photo, 'rb') as image:
        response = client.detection(Image={'Bytes': image.read()})

    return response['Labels']


# create a function to save the labels to a json file
def save_labels(labels):
    with open('labels.json', 'w') as f:
        json.dump(labels, f)


if __name__ == "__main__":
    photo = 'IMG_0543.jpg'
    labels = detect_labels(photo)
    # print(labels)
    print(len(labels))
    print(labels[0]['Name'])
    print(labels[0]['Confidence'])
    print(labels[0]['Instances'])
    print(labels[0]['Parents'])
    print(labels[0]['Categories'])

    save_labels(labels)