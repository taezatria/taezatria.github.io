from flask import render_template
from werkzeug import secure_filename
from app import app
import os

@app.route('/')
def home():
	if request.method == 'POST':
		f = request.files['file']
		filename = secure_filename(f.filename)
		f.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
		s = 'Uploaded Successfully'
	else:
		s = ''
	return render_template('upload.html',success = s)