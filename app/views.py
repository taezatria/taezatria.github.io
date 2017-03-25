from . import app
from flask import Flask,render_template,redirect,url_for,request
from werkzeug import secure_filename
import os

@app.route('/',methods=['POST','GET'])
def main():
	if request.method == 'POST':
		f = request.files['file']
		filename = secure_filename(f.filename)
		f.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
		s = 'uploaded'
	else:
		s = ''
	return render_template('index.html',success = s)