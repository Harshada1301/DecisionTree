"""
Loan Approval Predictor - Decision Tree + Flask (single file, no templates folder needed).
Put this file, requirements.txt and your model file in the SAME folder.
Render start command:  gunicorn app:app
"""
import glob
import os
import pickle

import pandas as pd
from flask import Flask, Response, jsonify, request

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load model (uses Decisiontree.pkl; falls back to the first .pkl found,
# because Linux file names are case-sensitive)
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE, "Decisiontree.pkl")
if not os.path.exists(MODEL_FILE):
    found = glob.glob(os.path.join(BASE, "*.pkl")) + glob.glob(os.path.join(BASE, "*.PKL"))
    MODEL_FILE = found[0]
with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

# Exact training column order
FEATURES = [
    "person_age", "person_gender", "person_education", "person_income",
    "person_emp_exp", "person_home_ownership", "loan_amnt", "loan_intent",
    "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
    "credit_score", "previous_loan_defaults_on_file",
]

# ---------------------------------------------------------------------------
# Category codes (alphabetical LabelEncoder order). If your training used a
# different encoding, edit the numbers in the OPTIONS dict below.
# ---------------------------------------------------------------------------
OPTIONS = {
    "gender": {"Female": 0, "Male": 1},
    "education": {"Associate": 0, "Bachelor": 1, "Doctorate": 2, "High School": 3, "Master": 4},
    "home": {"Mortgage": 0, "Other": 1, "Own": 2, "Rent": 3},
    "intent": {"Debt Consolidation": 0, "Education": 1, "Home Improvement": 2,
               "Medical": 3, "Personal": 4, "Venture": 5},
    "defaults": {"No": 0, "Yes": 1},
}

APPROVED_LABELS = {"1", "1.0", "yes", "approved", "true", "y"}


@app.route("/")
def home():
    import json
    html = PAGE.replace("__OPTIONS__", json.dumps(OPTIONS))
    return Response(html, mimetype="text/html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        d = request.get_json(force=True)
        income = float(d["income"])
        loan = float(d["loan_amnt"])
        pct = float(d.get("loan_percent_income") or (loan / income if income else 0))
        vals = [
            float(d["age"]), float(d["gender"]), float(d["education"]), income,
            float(d["emp_exp"]), float(d["home"]), loan, float(d["intent"]),
            float(d["int_rate"]), pct, float(d["cred_hist"]),
            float(d["credit_score"]), float(d["defaults"]),
        ]
        row = pd.DataFrame([vals], columns=FEATURES)
        pred = str(model.predict(row)[0])
        approved = pred.strip().lower() in APPROVED_LABELS
        confidence = None
        if hasattr(model, "predict_proba"):
            confidence = round(float(max(model.predict_proba(row)[0])) * 100, 1)
        return jsonify(ok=True, approved=approved, label=pred, confidence=confidence)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 400


# ---------------------------------------------------------------------------
# Front-end (HTML + CSS + JS)
# ---------------------------------------------------------------------------
PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LoanSense AI · Instant Loan Decision</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
:root{--bg:#050816;--ink:#eaf0ff;--mute:#8a95b5;--c1:#00e5ff;--c2:#7c4dff;--c3:#ff4ecd;--ok:#2dffa7;--bad:#ff5c7a;
--glass:rgba(255,255,255,.055);--line:rgba(255,255,255,.13)}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--ink);min-height:100vh;overflow-x:hidden}
/* animated aurora */
.aurora{position:fixed;inset:0;z-index:0;overflow:hidden;pointer-events:none}
.aurora i{position:absolute;border-radius:50%;filter:blur(90px);opacity:.5;animation:drift 18s ease-in-out infinite}
.aurora i:nth-child(1){width:520px;height:520px;background:var(--c2);top:-160px;left:-120px}
.aurora i:nth-child(2){width:460px;height:460px;background:var(--c1);bottom:-160px;right:-100px;animation-delay:-6s}
.aurora i:nth-child(3){width:380px;height:380px;background:var(--c3);top:35%;left:55%;animation-delay:-12s;opacity:.32}
@keyframes drift{0%,100%{transform:translate(0,0) scale(1)}33%{transform:translate(70px,-50px) scale(1.15)}66%{transform:translate(-50px,60px) scale(.92)}}
body::after{content:"";position:fixed;inset:0;z-index:0;pointer-events:none;opacity:.07;
background-image:linear-gradient(#fff 1px,transparent 1px),linear-gradient(90deg,#fff 1px,transparent 1px);background-size:48px 48px;
-webkit-mask-image:radial-gradient(circle at 50% 30%,#000,transparent 70%);mask-image:radial-gradient(circle at 50% 30%,#000,transparent 70%)}
#fx{position:fixed;inset:0;z-index:5;pointer-events:none}

.wrap{position:relative;z-index:1;max-width:1100px;margin:0 auto;padding:42px 20px 60px}
header{text-align:center;margin-bottom:34px}
.logo{display:inline-flex;align-items:center;gap:10px;font-family:'Space Grotesk';font-weight:700;letter-spacing:.5px;font-size:15px;
padding:8px 18px;border-radius:99px;border:1px solid var(--line);background:var(--glass)}
.logo b{width:10px;height:10px;border-radius:50%;background:var(--ok);box-shadow:0 0 14px var(--ok);animation:pl 1.8s infinite}
@keyframes pl{50%{transform:scale(1.5);opacity:.5}}
h1{font-family:'Space Grotesk';font-size:clamp(36px,6.5vw,68px);line-height:1.05;margin:20px 0 12px;letter-spacing:-1.5px}
h1 span{background:linear-gradient(100deg,var(--c1),var(--c2),var(--c3),var(--c1));background-size:300% auto;-webkit-background-clip:text;background-clip:text;color:transparent;animation:sh 7s linear infinite}
@keyframes sh{to{background-position:300% center}}
header p{color:var(--mute);max-width:560px;margin:auto;line-height:1.6;font-weight:300}

.grid{display:grid;grid-template-columns:1.5fr .9fr;gap:24px;align-items:start}
@media(max-width:920px){.grid{grid-template-columns:1fr}}
.card{background:var(--glass);border:1px solid var(--line);border-radius:26px;padding:28px;backdrop-filter:blur(22px);-webkit-backdrop-filter:blur(22px);
box-shadow:0 30px 70px rgba(0,0,0,.5),inset 0 1px 0 rgba(255,255,255,.12);animation:up .9s cubic-bezier(.2,.8,.2,1) both;position:relative;overflow:hidden}
.card.r{animation-delay:.15s}
@keyframes up{from{opacity:0;transform:translateY(40px)}to{opacity:1;transform:none}}
/* glowing border sweep */
.card::before{content:"";position:absolute;inset:-1px;border-radius:26px;padding:1px;background:conic-gradient(from var(--a,0deg),transparent 0 70%,var(--c1),var(--c3),transparent);
-webkit-mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0);-webkit-mask-composite:xor;mask-composite:exclude;animation:spin 6s linear infinite;pointer-events:none;opacity:.8}
@property --a{syntax:'<angle>';initial-value:0deg;inherits:false}
@keyframes spin{to{--a:360deg}}

.sec{font-family:'Space Grotesk';font-size:13px;letter-spacing:2px;text-transform:uppercase;color:var(--c1);margin:4px 0 14px;display:flex;align-items:center;gap:10px}
.sec::after{content:"";flex:1;height:1px;background:linear-gradient(90deg,var(--line),transparent)}
.sec:not(:first-child){margin-top:26px}
.row{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:560px){.row{grid-template-columns:1fr}}
.f{display:flex;flex-direction:column;gap:7px}
.f.full{grid-column:1/-1}
label{font-size:11.5px;letter-spacing:1px;text-transform:uppercase;color:var(--mute);font-weight:500;display:flex;justify-content:space-between}
label b{color:var(--ink);font-family:'Space Grotesk';font-size:15px;letter-spacing:0;text-transform:none}
input[type=number],select{width:100%;padding:13px 14px;border-radius:13px;border:1px solid var(--line);background:rgba(255,255,255,.06);color:var(--ink);font:inherit;font-size:15px;outline:none;transition:.3s;appearance:none;-webkit-appearance:none}
select{background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8'%3E%3Cpath d='M1 1l5 5 5-5' stroke='%238a95b5' fill='none' stroke-width='2'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 14px center;padding-right:36px}
select option{color:#111}
input:focus,select:focus{border-color:var(--c1);box-shadow:0 0 0 4px rgba(0,229,255,.16);background:rgba(255,255,255,.1);transform:translateY(-2px)}
input[type=range]{-webkit-appearance:none;appearance:none;width:100%;height:6px;border-radius:9px;outline:none;cursor:pointer;background:linear-gradient(90deg,var(--c1),var(--c2) var(--p,50%),rgba(255,255,255,.12) var(--p,50%))}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:22px;height:22px;border-radius:50%;background:#fff;border:4px solid var(--c2);box-shadow:0 0 18px var(--c2);transition:.2s}
input[type=range]::-webkit-slider-thumb:hover{transform:scale(1.25)}
input[type=range]::-moz-range-thumb{width:16px;height:16px;border-radius:50%;background:#fff;border:4px solid var(--c2)}
.auto{padding:13px 14px;border-radius:13px;border:1px dashed var(--line);color:var(--c1);font-family:'Space Grotesk';font-weight:600;font-size:16px;background:rgba(0,229,255,.06)}
.btn{margin-top:28px;width:100%;padding:18px;border:0;border-radius:16px;cursor:pointer;position:relative;overflow:hidden;font:inherit;font-weight:600;font-size:16px;letter-spacing:1px;color:#fff;
background:linear-gradient(100deg,var(--c2),var(--c3),var(--c1),var(--c2));background-size:300% auto;box-shadow:0 14px 36px rgba(124,77,255,.5);transition:.4s;animation:sh 6s linear infinite}
.btn:hover{transform:translateY(-3px);box-shadow:0 20px 46px rgba(255,78,205,.5)}
.btn::after{content:"";position:absolute;top:0;left:-70%;width:45%;height:100%;background:linear-gradient(100deg,transparent,rgba(255,255,255,.55),transparent);transform:skewX(-20deg);animation:sw 3s infinite}
@keyframes sw{to{left:150%}}
.btn.busy{pointer-events:none;opacity:.75}

/* result */
.res{position:sticky;top:22px;text-align:center;min-height:430px;display:flex;flex-direction:column;align-items:center;justify-content:center}
.gauge{position:relative;width:200px;height:200px;margin-bottom:16px}
.gauge svg{width:100%;height:100%;transform:rotate(-90deg)}
.gauge .t{stroke:rgba(255,255,255,.09)}
.gauge .v{stroke:url(#gr);stroke-linecap:round;stroke-dasharray:565;stroke-dashoffset:565;transition:stroke-dashoffset 1.5s cubic-bezier(.2,.8,.2,1),stroke .4s;filter:drop-shadow(0 0 10px var(--c1))}
.gauge .mid{position:absolute;inset:0;display:grid;place-items:center;font-size:60px;animation:bob 3.2s ease-in-out infinite}
@keyframes bob{50%{transform:translateY(-8px)}}
.tag{font-size:11.5px;letter-spacing:2px;text-transform:uppercase;color:var(--mute)}
.verdict{font-family:'Space Grotesk';font-weight:700;font-size:clamp(32px,5vw,44px);margin:8px 0}
.verdict.ok{color:var(--ok);text-shadow:0 0 28px rgba(45,255,167,.6)}
.verdict.no{color:var(--bad);text-shadow:0 0 28px rgba(255,92,122,.6)}
.hint{color:var(--mute);font-size:14px;line-height:1.55;max-width:250px;font-weight:300}
.pop{animation:pp .7s cubic-bezier(.2,1.4,.4,1)}
@keyframes pp{from{opacity:0;transform:scale(.7)}to{opacity:1;transform:none}}
.shake{animation:shk .6s}
@keyframes shk{20%,60%{transform:translateX(-8px)}40%,80%{transform:translateX(8px)}}
.chips{display:flex;flex-wrap:wrap;gap:10px;justify-content:center;margin-top:18px}
.chip{padding:9px 14px;border-radius:12px;background:rgba(255,255,255,.06);border:1px solid var(--line);font-size:13px}
.chip small{display:block;font-size:10px;letter-spacing:1px;text-transform:uppercase;color:var(--mute);margin-bottom:2px}
.err{background:rgba(255,92,122,.12);border:1px solid rgba(255,92,122,.4);color:#ffd0d8;padding:12px 14px;border-radius:12px;font-size:14px;margin-top:14px}
footer{text-align:center;color:var(--mute);font-size:12px;margin-top:36px;opacity:.7}
</style>
</head>
<body>
<div class="aurora"><i></i><i></i><i></i></div>
<canvas id="fx"></canvas>

<div class="wrap">
<header>
  <span class="logo"><b></b>LoanSense AI · Decision Tree</span>
  <h1>Get your loan <span>decision in seconds</span></h1>
  <p>Fill in the applicant profile and our trained model will predict whether the loan gets approved.</p>
</header>

<div class="grid">
<section class="card">
<form id="form" autocomplete="off">
  <div class="sec">Personal</div>
  <div class="row">
    <div class="f"><label>Age <b id="o_age">30</b></label><input type="range" id="age" min="18" max="80" value="30"></div>
    <div class="f"><label>Gender</label><select id="gender"></select></div>
    <div class="f"><label>Education</label><select id="education"></select></div>
    <div class="f"><label>Work experience (yrs) <b id="o_exp">5</b></label><input type="range" id="emp_exp" min="0" max="50" value="5"></div>
  </div>

  <div class="sec">Finances</div>
  <div class="row">
    <div class="f"><label>Annual income</label><input type="number" id="income" min="0" value="60000"></div>
    <div class="f"><label>Home ownership</label><select id="home"></select></div>
    <div class="f"><label>Credit score <b id="o_cs">700</b></label><input type="range" id="credit_score" min="300" max="850" value="700"></div>
    <div class="f"><label>Credit history (yrs) <b id="o_ch">5</b></label><input type="range" id="cred_hist" min="0" max="40" value="5"></div>
    <div class="f full"><label>Previous loan defaults on file</label><select id="defaults"></select></div>
  </div>

  <div class="sec">Loan request</div>
  <div class="row">
    <div class="f"><label>Loan amount</label><input type="number" id="loan_amnt" min="0" value="10000"></div>
    <div class="f"><label>Loan purpose</label><select id="intent"></select></div>
    <div class="f"><label>Interest rate % <b id="o_ir">11.0</b></label><input type="range" id="int_rate" min="5" max="25" step="0.1" value="11"></div>
    <div class="f"><label>Loan % of income</label><div class="auto" id="pct">16.7%</div></div>
  </div>
  <button class="btn" id="btn" type="submit">⚡ Check loan decision</button>
</form>
</section>

<aside class="card r res" id="res">
  <div class="gauge">
    <svg viewBox="0 0 200 200"><defs><linearGradient id="gr" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#00e5ff"/><stop offset="1" stop-color="#ff4ecd"/></linearGradient></defs>
    <circle class="t" cx="100" cy="100" r="90" fill="none" stroke-width="10"/><circle class="v" id="arc" cx="100" cy="100" r="90" fill="none" stroke-width="10"/></svg>
    <div class="mid" id="emoji">🏦</div>
  </div>
  <div class="tag">Model decision</div>
  <div class="verdict" id="verdict">Awaiting input</div>
  <p class="hint" id="hint">Set the details and press <b>Check loan decision</b>.</p>
  <div class="chips" id="chips"></div>
</aside>
</div>
<footer>Built with Flask &amp; scikit-learn · Predictions are model outputs, not financial advice</footer>
</div>

<script>
const OPT=__OPTIONS__, $=id=>document.getElementById(id);
const sel={gender:'gender',education:'education',home:'home',intent:'intent',defaults:'defaults'};
for(const k in sel){$(k).innerHTML=Object.entries(OPT[k]).map(([n,v])=>`<option value="${v}">${n}</option>`).join('')}

const outs={age:'o_age',emp_exp:'o_exp',credit_score:'o_cs',cred_hist:'o_ch',int_rate:'o_ir'};
function paint(el){el.style.setProperty('--p',(el.value-el.min)/(el.max-el.min)*100+'%');if(outs[el.id])$(outs[el.id]).textContent=el.id==='int_rate'?Number(el.value).toFixed(1):el.value}
document.querySelectorAll('input[type=range]').forEach(el=>{paint(el);el.addEventListener('input',()=>paint(el))});
function pct(){const i=+$('income').value,l=+$('loan_amnt').value;const p=i>0?l/i:0;$('pct').textContent=(p*100).toFixed(1)+'%';return p}
['income','loan_amnt'].forEach(id=>$(id).addEventListener('input',pct));pct();

/* confetti / sparks */
const cv=$('fx'),cx=cv.getContext('2d');let parts=[];
function size(){cv.width=innerWidth;cv.height=innerHeight}size();addEventListener('resize',size);
function burst(good){
  const cols=good?['#2dffa7','#00e5ff','#7c4dff','#ffd700','#ff4ecd']:['#ff5c7a','#ff8a5c','#8a95b5'];
  const r=$('res').getBoundingClientRect(),ox=r.left+r.width/2,oy=r.top+130;
  for(let i=0;i<(good?160:60);i++)parts.push({x:ox,y:oy,vx:(Math.random()-.5)*(good?16:9),vy:Math.random()*-(good?15:6)-2,g:.35,s:Math.random()*7+3,c:cols[i%cols.length],l:1,r:Math.random()*6});
}
(function loop(){cx.clearRect(0,0,cv.width,cv.height);
  parts=parts.filter(p=>p.l>0);parts.forEach(p=>{p.x+=p.vx;p.y+=p.vy;p.vy+=p.g;p.l-=.012;p.r+=.2;cx.save();cx.globalAlpha=Math.max(p.l,0);cx.translate(p.x,p.y);cx.rotate(p.r);cx.fillStyle=p.c;cx.fillRect(-p.s/2,-p.s/2,p.s,p.s*.6);cx.restore()});
  requestAnimationFrame(loop)})();

$('form').addEventListener('submit',async e=>{
  e.preventDefault();const b=$('btn');b.classList.add('busy');b.textContent='⏳ Analysing...';
  const body={age:$('age').value,gender:$('gender').value,education:$('education').value,income:$('income').value,emp_exp:$('emp_exp').value,
    home:$('home').value,loan_amnt:$('loan_amnt').value,intent:$('intent').value,int_rate:$('int_rate').value,loan_percent_income:pct(),
    cred_hist:$('cred_hist').value,credit_score:$('credit_score').value,defaults:$('defaults').value};
  const old=document.querySelector('.err');if(old)old.remove();
  try{
    const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await r.json();if(!d.ok)throw new Error(d.error);
    const v=$('verdict');v.textContent=d.approved?'Approved ✓':'Declined ✕';v.className='verdict pop '+(d.approved?'ok':'no');
    $('emoji').textContent=d.approved?'🎉':'🛑';
    $('hint').textContent=d.approved?'Great news! This profile is predicted to qualify for the loan.':'This profile is predicted not to qualify under the current model.';
    const c=d.confidence;
    $('chips').innerHTML=`<div class="chip pop"><small>Loan</small>${Number(body.loan_amnt).toLocaleString()}</div><div class="chip pop"><small>Score</small>${body.credit_score}</div>`+(c!=null?`<div class="chip pop"><small>Confidence</small>${c}%</div>`:'');
    const arc=$('arc');arc.style.strokeDashoffset=565;
    setTimeout(()=>arc.style.strokeDashoffset=565*(1-(d.approved?1:.25)*(c?c/100:1)),40);
    burst(d.approved);if(!d.approved){$('res').classList.remove('shake');void $('res').offsetWidth;$('res').classList.add('shake')}
    if(innerWidth<920)$('res').scrollIntoView({behavior:'smooth',block:'center'});
  }catch(err){
    $('res').insertAdjacentHTML('beforeend',`<div class="err">⚠️ ${err.message}</div>`);
  }
  b.classList.remove('busy');b.textContent='⚡ Check loan decision';
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
