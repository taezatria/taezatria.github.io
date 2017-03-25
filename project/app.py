# -*- coding: utf-8 -*-

from flask import Flask
from flask_frozen import Freezer

app = Flask(__name__)
app.config.from_pyfile('settings.py')
freezer = Freezer(app)