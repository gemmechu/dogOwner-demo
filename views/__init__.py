from PIL import Image
from flask import Blueprint, render_template, request, jsonify
from torch_mtcnn import detect_faces

from util import is_same, ModelLoaded

base = Blueprint('base', __name__)
THRESHOLD = 1.2


@base.route('/')
def index():
    return render_template('index.html')


@base.route('/predict', methods=['post'])
def predict():
    files = request.files
    imgPerson = Image.open(files.get('imgPerson')).convert('RGB')
    imgA = Image.open(files.get('imgA')).convert('RGB')
    imgB = Image.open(files.get('imgB')).convert('RGB')
    imgC = Image.open(files.get('imgC')).convert('RGB')

    distanceA, similarA = is_same(imgPerson, imgA, THRESHOLD)
    distanceB, similarB = is_same(imgPerson, imgB, THRESHOLD)
    distanceC, similarC = is_same(imgPerson, imgC, THRESHOLD)

    distances = [round(distanceA.item(),2),round(distanceB.item(),2),round(distanceC.item(),2)]

    model_acc = ModelLoaded.acc
    return jsonify(dog_a=similarA.item(),
                   score_a=distances[0],
                   dog_b =similarB.item(),
                   score_b=distances[1],
                   dog_c=similarC.item(),
                   score_c=distances[2],
                   model_acc=model_acc,
                   threshold=THRESHOLD)
