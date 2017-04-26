.PHONY: render

data:
	echo "No soup for you!"
	exit 1

classified.p: data
	python3 classifier.py

render_test: classified.p
	python3 vehicledetector.py test_video.mp4

render_project: classified.p
	python3 vehicledetector.py project_video.mp4