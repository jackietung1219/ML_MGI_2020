def data_set(a, b, c, d, e, f):
    global X, y, v, theta, h, r
    X = a
    y = b
    v = c
    theta = d
    h = e
    r = f

def parameter(k, l, m):
    global cv, scoring, train_sizes
    cv = k
    scoring = l
    train_sizes = m

def learned_detail(n, o, p, q, s, t, u, w, xs, ys):
    global train_sizes_h, train_score_h, test_score_h, train_score_h_mean, test_score_h_mean, train_sizes_r, train_score_r, test_score_r, train_score_r_mean, test_score_r_mean
    train_sizes_h = n
    train_score_h = o
    test_score_h = p
    train_score_h_mean = q
    test_score_h_mean = s
    train_sizes_r = t
    train_score_r = u
    test_score_r = w
    train_score_r_mean = xs
    test_score_r_mean = ys

def models(cs, ds):
    global model_h, model_r
    model_h = cs
    model_r = ds

def modelname(es, fs):
    global model_h_name, model_r_name, model_h_path, model_r_path
    model_h_name = es
    model_h_path = str('./_train_model/' + str(model_h_name) + '.joblib')
    model_r_name = fs
    model_r_path = str('./_train_model/' + str(model_r_name) + '.joblib')