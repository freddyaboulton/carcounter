from flask import Flask
from flask import request, abort
from redis import Redis
from rq import Queue
from worker import process_image
from constants import IMAGE_HEIGHT, IMAGE_WIDTH
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
    return 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
