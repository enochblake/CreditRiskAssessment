from flask import Flask, request, render_template_string
from inference import build_model, run_ve, run_bp, run_lw  # Make sure these functions exist!

app = Flask(__name__)
model = build_model()

FORM_HTML = """
<!doctype html>
<title>Credit Risk Inference</title>
<h2>Credit Risk Bayesian Network Inference</h2>
<form method=post>
  <label>Income Level:</label>
  <select name="Income_Level">
    <option>Low</option><option>Medium</option><option>High</option>
  </select><br><br>
  <label>Experience Level:</label>
  <select name="Experience_Level">
    <option>Junior</option><option>Mid</option><option>Senior</option>
  </select><br><br>
  <label>House Ownership:</label>
  <select name="House_Ownership">
    <option>Yes</option><option>No</option><option>Unknown</option>
  </select><br><br>
  <label>Car Ownership:</label>
  <select name="Car_Ownership">
    <option>Yes</option><option>No</option><option>Unknown</option>
  </select><br><br>
  <label>Algorithm:</label>
  <select name="algorithm">
    <option value="ve">Variable Elimination</option>
    <option value="bp">Belief Propagation</option>
    <option value="lw">Likelihood Weighting</option>
  </select><br><br>
  <input type=submit value=Infer>
</form>
{% if results %}
  <h3>Inference Results:</h3>
  <ul>
  {% for state, prob in results.items() %}
    <li>{{ state }}: {{ prob }}</li>
  {% endfor %}
  </ul>
{% endif %}
"""

@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    if request.method == "POST":
        evidence = {
            "Income_Level": request.form["Income_Level"],
            "Experience_Level": request.form["Experience_Level"],
            "House_Ownership": request.form["House_Ownership"],
            "Car_Ownership": request.form["Car_Ownership"],
        }
        algorithm = request.form["algorithm"]

        if algorithm == "ve":
            query = run_ve(model, evidence)
            results = {state: f"{prob:.4f}" for state, prob in zip(query.state_names["Risk_Flag"], query.values)}
        elif algorithm == "bp":
            query = run_bp(model, evidence)
            results = {state: f"{prob:.4f}" for state, prob in zip(query.state_names["Risk_Flag"], query.values)}
        else:
            probs, _ = run_lw(model, evidence)
            results = {state: f"{prob:.4f}" for state, prob in probs.items()}

    return render_template_string(FORM_HTML, results=results)

if __name__ == "__main__":
    app.run(debug=True)
