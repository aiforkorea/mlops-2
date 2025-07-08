from flask import render_template
#from flask_login import login_required, current_user
from apps.main import main

@main.route("/")
def index():
#    if current_user.is_authenticated:
#        return render_template("main/index.html", username=current_user.username)
    return render_template("main/index.html")
