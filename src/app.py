from flask import Flask, Response, jsonify
from flask import request, abort
from redis import Redis
from rq import Queue
from src.worker import process_image, from_redis
from src.constants import IMAGE_HEIGHT, IMAGE_WIDTH
import numpy as np
import datetime
import base64

app = Flask(__name__)

cache = Redis(host='localhost', port=6379, db=0)
queue = Queue(connection=cache)


@app.route('/api/v1.0/image', methods=['POST'])
def main():
    if not request.json:
        abort(400)
    date_time = request.json['date_time']

    image = np.array(request.json['image'], dtype='uint8')
    queue.enqueue(process_image, args=(date_time, image))
    return jsonify({'status': 'Success'}), 201


@app.route('/api/v1.0/annotated-image', methods=['GET'])
def annotated_image():
    if cache.exists('latest-image'):
        image = from_redis(cache, 'latest-image')
        image = image.tolist()
    else:
        image = np.zeros((300, 300, 3)).astype('uint8').tolist()
    return jsonify({'image': image})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
