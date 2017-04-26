.PHONY: render data

data:
	echo "No soup for you!"
	exit 1

classified.p: data
	python3 classifier.py

render_test: classified.p
	python3 Proto2.py test_video.mp4

render_project: classified.p
	python3 Proto2.py project_video.mp4