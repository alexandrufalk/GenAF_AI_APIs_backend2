from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import tensorflow as tf
from keras_cv.src.models import StableDiffusion
from PIL import Image
import io
import numpy as np



app = Flask(__name__)
CORS(app)

model = StableDiffusion(img_height=512, img_width=512, jit_compile=True)



@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        # Get the prompt from the request
        data = request.json
        prompt = data.get('prompt', '')
        print("prompt", prompt)

        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        # Generate image from the prompt
        
        images = model.text_to_image(prompt=prompt, batch_size=1)
        print("Image generated",images)

        # Convert the generated image (numpy array) to a Pillow image
        generated_img = Image.fromarray((images[0] * 255).astype(np.uint8))

        # Save the image to a BytesIO object to send over the response
        img_io = io.BytesIO()
        generated_img.save(img_io, 'PNG')
        img_io.seek(0)

        # Send the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        print(f"Exception encountered: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5010)  # Change the port to 5010