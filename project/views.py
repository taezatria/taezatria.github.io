from flask import render_template
from app import app, pages

@app.route('/')
def home():
#	posts = [page for page in pages if 'date' in page.meta]
	# Sort pages by date
#	sorted_posts = sorted(posts, reverse=True, key=lambda page: page.meta['date'])
	return render_template('index.html')