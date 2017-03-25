from flask import Flask
from flask_frozen import Freezer

app = Flask(__name__)
app.config.from_object('config')
freezer = Freezer(app)