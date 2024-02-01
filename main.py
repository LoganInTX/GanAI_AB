import boto3
import json


# Use Amazon DynamoDB to retrieve a user's preferences
def get_user_preferences(user_id) -> dict:
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('GenAI_AB')

    response = table.get_item(Key={'username': user_id})

    return response['Item'] if 'Item' in response else {}


# User Amazon Bedrock and the Anthropic Claude V2 model to generate the text for the user's profile bio
def generate_bio_text(prefs) -> str:
    bedrock = boto3.client('bedrock-runtime')   

    body = {'prompt': f'Human: Create a bio for a dating profile for the Tinder app using first person tone of voice. The bio is for a man named Logan who is {prefs["age"]} years old who is seeking a woman. The man has the following interests: {prefs["interests"]}. He has the following hobbies: {prefs["hobbies"]}. He is looking for the following in a potential match: {prefs["others"]}. Assistant:',
        'maxTokens': 300,
        'temperature': 0.3  # Scale between 0 and 1 where smaller value means more deterministic and large value means more creative
    }
    print(body['prompt'], '\n\n')

    response = bedrock.invoke_model(
        modelId='ai21.j2-mid-v1',
        body = json.dumps(body)        
    )
    response_body = json.loads(response.get("body").read())
    response_text = response_body["completions"][0]["data"]["text"]
    return response_text


# Use Amazon Comprehend to do toxicity detection
def detect_toxicity(text) -> str:
    client = boto3.client('comprehend')

    response = client.detect_toxic_content(TextSegments=text, LanguageCode='en')

    return response


# Use Amazon DynamoDB to update a user's profile bio
def update_user_profile(user_id, bio) -> None:
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('GenAI_AB')

    response = table.update_item(
        Key={'username': user_id},
        UpdateExpression='SET bio = :bio',
        ExpressionAttributeValues={':bio': bio}
    )

    return response

if __name__ == "__main__":

    # Get User Profile Preferences
    username = 'logan'
    prefs = get_user_preferences(username)
    print(prefs, '\n\n')

    # Use Bedrock to Generate Bio
    bio = generate_bio_text(prefs).strip()
    print(bio, '\n\n')

    # Check for Toxicity in Bio
    bio_text = [{'Text': bio}]
    toxicity = detect_toxicity(bio_text)
    print(toxicity, '\n\n')

    for label in toxicity['ResultList'][0]['Labels']:
        if label['Score'] > 0.2:
            print(f'The bio is toxic based on label: {label["Name"]}')
            break
    
    # Update User Profile with New Bio
    response = update_user_profile(username, bio)
    print(response)
