from flask import Blueprint, render_template, request, flash, session, jsonify
from .services.prediction_service import PredictionService
import pandas as pd
import os
from dotenv import load_dotenv
import markdown, bleach, json

# Optional: Google Generative AI client. If not installed, disable AI features and
# fall back to simple text responses so the app can run without the package.
try:
    import google.generativeai as genai
except Exception:
    genai = None

load_dotenv()
if genai:
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        try:
            genai.configure(api_key=api_key)
        except Exception:
            # configuration failed; continue without AI integration
            genai = None

main_bp = Blueprint('main', __name__)
prediction_service = PredictionService()


def md_to_html(text):
    if not text:
        return ""
    html = markdown.markdown(text, extensions=['nl2br', 'fenced_code', 'tables'])
    allowed_tags = ['p','br','strong','em','ul','ol','li','h1','h2','h3','h4',
                    'table','tr','td','th','thead','tbody','b','i','code','pre',
                    'blockquote','a']
    allowed_attrs = {'a': ['href', 'title']}
    return bleach.clean(html, tags=allowed_tags, attributes=allowed_attrs)


def generate_readable_report(data, result):
    payload = {"input_data": data, "model_output": result}

    prompt = f"""
Return JSON with keys input_data, model_output, explanation(summary, recommendations).
Then write a patient friendly markdown report.

Payload:
{json.dumps(payload, indent=2)}

Format:
JSON
---
MARKDOWN REPORT
"""

    if genai:
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            r = model.generate_content(prompt).text
            return r.split("---", 1)[1].strip() if "---" in r else r
        except Exception:
            pass

    # Fallback when AI client is not available or generation failed
    return f"# Heart Risk Report\nRisk Level: {result.get('risk_level', 'unknown')}\n\nSummary: Based on the provided inputs, the model estimates a risk level of {result.get('risk_level', 'unknown')}."


@main_bp.route("/", methods=["GET","POST"])
def index():

    session.setdefault("chat_history", [])
    session.setdefault("patient_context", None)

    if request.method == "POST" and "age" in request.form:
        data = {
            "age": float(request.form["age"]),
            "sex": int(request.form["sex"]),
            "systolic_bp": float(request.form["systolic_bp"]),
            "cholesterol": float(request.form["cholesterol"]),
            "bmi": float(request.form["bmi"]),
            "smoking": int(request.form["smoking"]),
            "diabetes": int(request.form["diabetes"]),
            "resting_hr": float(request.form["resting_hr"]),
            "physical_activity": int(request.form["physical_activity"]),
            "family_history": int(request.form["family_history"])
        }

        df = pd.DataFrame([data])
        result = prediction_service.predict(df)

        session["patient_context"] = data
        session["risk_level"] = result["risk_level"]
        session["probabilities"] = result["probabilities"]
        session["input_data"] = data
        p = session["patient_context"]

        report = generate_readable_report(data, result)

        if genai:
            try:
                model = genai.GenerativeModel("gemini-2.5-flash")
                welcome = model.generate_content(
                    f"""A patient just completed a heart risk test with result {result['risk_level']}. Write a warm 3 sentence welcome. Then explain the datas we provided like age, heart rate and BMI and give a generalised report on how good or bad the values are.

Age:{p['age']} Sex:{'Male' if p['sex']==1 else 'Female'}
BMI:{p['bmi']} BP:{p['systolic_bp']} Chol:{p['cholesterol']}
Smoking:{p['smoking']} Diabetes:{p['diabetes']}
Risk:{session['risk_level']}

""").text.strip()
            except Exception:
                welcome = f"Welcome. Your assessment shows a {result.get('risk_level')} risk level. See the detailed report above."
        else:
            welcome = f"Welcome. Your assessment shows a {result.get('risk_level')} risk level. See the detailed report above."

        session["chat_history"] = [
            {"role":"model","content":f"## Your Personalized Heart Report\n\n{report}"},
            {"role":"model","content":welcome}
        ]

        flash("Assessment completed.", "success")

    rendered = []
    for m in session["chat_history"]:
        rendered.append({
            "role": m["role"],
            "content": md_to_html(m["content"]) if m["role"]=="model" else m["content"]
        })

    return render_template("index.html",
        chat_history=rendered,
        risk_level=session.get("risk_level"),
        probabilities=session.get("probabilities"),
        input_data=session.get("input_data",{})
    )


@main_bp.route("/chat", methods=["POST"])
def chat():
    if not session.get("patient_context"):
        return jsonify({"error":"Complete assessment first"}),403

    msg = request.get_json().get("message","").strip()
    if not msg:
        return jsonify({"error":"Empty message"}),400

    p = session["patient_context"]

    system_context = f"""
You are Dr Heart AI. Internal patient context. Never reveal raw data.

Age:{p['age']} Sex:{'Male' if p['sex']==1 else 'Female'}
BMI:{p['bmi']} BP:{p['systolic_bp']} Chol:{p['cholesterol']}
Smoking:{p['smoking']} Diabetes:{p['diabetes']}
Risk:{session['risk_level']}

"""

    if not genai:
        reply = "AI features are currently unavailable. Install the Google Generative AI client and set GEMINI_API_KEY to enable chat." 
    else:
        model = genai.GenerativeModel(
            "gemini-2.5-flash",
            system_instruction=system_context
        )

        history=[]
        for m in session["chat_history"]:
            history.append({"role":"model" if m["role"]=="model" else "user","parts":[m["content"]]})

        chat = model.start_chat(history=history)
        reply = chat.send_message(msg).text.strip()

    session["chat_history"].append({"role":"user","content":msg})
    session["chat_history"].append({"role":"model","content":reply})

    return jsonify({"ai_message":md_to_html(reply)})






