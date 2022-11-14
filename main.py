import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import scipy.cluster
import sklearn.cluster
import numpy
from PIL import Image
import binascii

UPLOAD_FOLDER = 'static/img'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


#function which take image and find dominant colors e.g. n_clusters = 10
def dominant_colors(image):  # PIL image input

    image = image.resize((150, 150))      # optional, to reduce time
    ar = numpy.asarray(image)
    shape = ar.shape
    ar = ar.reshape(numpy.product(shape[:2]), shape[2]).astype(float)

    kmeans = sklearn.cluster.MiniBatchKMeans(
        n_clusters=10,
        init="k-means++",
        max_iter=20,
        random_state=1000
    ).fit(ar)
    codes = kmeans.cluster_centers_

    vecs, _dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
    counts, _bins = numpy.histogram(vecs, len(codes))    # count occurrences

    colors = []
    hex_data = []
    for index in numpy.argsort(counts)[::-1]:
        #rgb data
        colors.append(tuple([int(code) for code in codes[index]]))
        #hex code data
        hex_data.append(f"#{binascii.hexlify(bytearray(int(c) for c in codes[index])).decode('ascii')}")
    print(colors)
    return hex_data


#function to check correct extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    #take route of uploaded file and manipulate
    data = f"img/{request.args.get('filename')}"
    try:
        im = Image.open(f"static/{data}")
    except FileNotFoundError:
        #if image doesn't exist, put codes as none and pass the first step when programs runs
        codes = None
        pass
    else:
        #call function route of image exist
        codes = dominant_colors(im)
        print(codes)
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('upload_file',
                                    filename=filename))
    return render_template("index.html", photo=data, codes=codes)


if __name__ == "__main__":
    app.run(debug=True)