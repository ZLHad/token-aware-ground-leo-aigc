"""VQA soft accuracy with the standard EvalAI answer normalization (compact port)."""
import re
CONTRACTIONS = {"dont":"don't","doesnt":"doesn't","isnt":"isn't","arent":"aren't","cant":"can't","couldnt":"couldn't",
    "wont":"won't","wouldnt":"wouldn't","didnt":"didn't","hasnt":"hasn't","havent":"haven't","im":"i'm","ive":"i've",
    "youre":"you're","theyre":"they're","thats":"that's","whats":"what's","wheres":"where's","whos":"who's","hes":"he's","shes":"she's"}
NUM = {"none":"0","zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5","six":"6","seven":"7","eight":"8","nine":"9","ten":"10"}
ARTICLES = {"a","an","the"}
PUNCT = [';', r"/", '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-', '>', '<', '@', '`', ',', '?', '!']
_period_strip = re.compile(r"(?!<=\d)(\.)(?!\d)"); _comma_strip = re.compile(r"(\d)(,)(\d)")

def normalize(ans: str) -> str:
    ans = ans.replace("\n", " ").replace("\t", " ").strip().lower()
    out = ans
    for p in PUNCT:
        if (p + " " in ans or " " + p in ans) or (re.search(_comma_strip, ans) is not None): out = out.replace(p, "")
        else: out = out.replace(p, " ")
    out = _period_strip.sub("", out)
    words = []
    for w in out.split():
        w = NUM.get(w, w)
        if w in ARTICLES: continue
        words.append(CONTRACTIONS.get(w, w))
    return " ".join(words).strip()

def soft_accuracy(pred: str, gts: list) -> float:
    """VQA accuracy: mean over leave-one-out of min(#match/3, 1)."""
    p = normalize(pred); g = [normalize(x) for x in gts]
    if len(g) == 0: return 0.0
    accs = []
    for i in range(len(g)):
        others = g[:i] + g[i+1:]
        accs.append(min(sum(1 for x in others if x == p) / 3.0, 1.0))
    return sum(accs) / len(accs)

def _lev(a, b):
    if a == b: return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1): cur.append(min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + (ca != cb)))
        prev = cur
    return prev[-1]

def anls(pred: str, gts: list, tau: float = 0.5) -> float:
    """DocVQA ANLS: 1 - NL(pred, gt) if below threshold tau, else 0; max over ground truths."""
    p = pred.strip().lower(); best = 0.0
    for g in gts:
        g = g.strip().lower(); L = max(len(p), len(g))
        if L == 0: continue
        nl = _lev(p, g) / L; best = max(best, 1 - nl if nl < tau else 0.0)
    return best

def relaxed_accuracy(pred: str, gt: str) -> float:
    """ChartQA relaxed accuracy: numeric within 5% relative error, else normalized exact match."""
    def num(x):
        x = x.strip().replace(",", "").replace("%", "").replace("$", "")
        try: return float(x)
        except ValueError: return None
    a, b = num(pred), num(gt)
    if a is not None and b is not None:
        return 1.0 if (abs(a - b) <= 0.05 * abs(b) if b != 0 else abs(a) < 1e-9) else 0.0
    return 1.0 if normalize(pred) == normalize(gt) else 0.0

def score(source: str, pred: str, gts: list) -> float:
    if source == "docvqa": return anls(pred, gts)
    if source == "chartqa": return relaxed_accuracy(pred, gts[0])
    return soft_accuracy(pred, gts)

if __name__ == "__main__":
    print("anls", anls("University of California", ["university of california, san diego", "university of california"]))
    print("relaxed", relaxed_accuracy("14", "14"), relaxed_accuracy("0.6", "0.57"), relaxed_accuracy("0.7", "0.57"))
    print(soft_accuracy("Down", ["down","down","at table","down","down","down","down","skateboard","down","down"]))
    print(soft_accuracy("2", ["two","2","2","two","3","2","2","2","two","2"]))
