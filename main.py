from flask import Flask, request, jsonify
from kandinsky3 import get_T2I_pipeline
from imgurpython import ImgurClient
import os

app = Flask(__name__)

# Initialize the text-to-image pipeline
t2i_pipe = get_T2I_pipeline('cuda', fp16=True)

# Imgur API credentials
client_id = 'your_client_id'
client_secret = 'your_client_secret'
access_token = 'your_access_token'
refresh_token = 'your_refresh_token'

client = ImgurClient(client_id, client_secret, access_token, refresh_token)


def generate_image(prompt):
    # Generate image based on the provided prompt
    image = t2i_pipe(prompt)
    return image


def upload_to_imgur(image_path):
    # Upload image to Imgur and return the URL
    response = client.upload_from_path(image_path, anon=True)
    return response['link']


@app.route('/imagine', methods=['POST'])
def imagine():
    try:
        # Get the prompt from the request
        prompt = request.form.get('prompt')

        # Generate the image
        image = generate_image(prompt)

        # Save the image locally
        image_path = 'generated_image.png'
        image.save(image_path)

        # Upload the image to Imgur
        imgur_url = upload_to_imgur(image_path)

        # Remove the local image file
        os.remove(image_path)

        return jsonify({'imgur_url': imgur_url})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
