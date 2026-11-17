"""Funciones de variables del Parcial 3 (competencia Kaggle nuevo / usado).
Las usan las notebooks 2_entrenar_modelo_final y 3_predecir_test (y la 1a para cargar los datos).

Todo lo que usa el target (target encoding, modelos lineales de texto) se calcula dentro de cada fold:
para las filas de validación sólo se usan etiquetas de las filas de entrenamiento de ese fold."""
import os, pickle, re, time
import numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

def cargar(path):
    df = pd.read_json(path, lines=True)
    return pd.json_normalize(df.to_dict('records'), max_level=3)

largo = lambda v: len(v) if isinstance(v, list) else 0
CATS = ['listing_type_id', 'buying_mode', 'shipping.mode', 'status', 'seller_address.state.name']
RE_USADO = r'\busad[oa]s?\b|\bimpecable\b|excelente estado'
RE_NUEVO = r'\bnuev[oa]s?\b|\bsellad[oa]s?\b'

def norm_txt(s):
    return (s.fillna('').str.lower().str.normalize('NFKD')
             .str.encode('ascii', 'ignore').str.decode('ascii'))

def features(d, flags_titulo=True):
    X = pd.DataFrame(index=d.index)
    X['log_price'] = np.log1p(d.price)
    X['usd'] = (d.currency_id == 'USD').astype(int)
    for c in ['initial_quantity', 'sold_quantity', 'available_quantity']:
        X['log_' + c] = np.log1p(d[c])
    X['vendido_algo'] = (d.sold_quantity > 0).astype(int)
    for c in ['accepts_mercadopago', 'automatic_relist', 'shipping.free_shipping', 'shipping.local_pick_up']:
        X[c] = d[c].astype(int)
    for c in ['pictures', 'variations', 'attributes', 'tags', 'non_mercado_pago_payment_methods', 'descriptions', 'sub_status']:
        X['n_' + c] = d[c].map(largo)
    pagos = d.non_mercado_pago_payment_methods.map(lambda l: {m['description'] for m in l})
    for p in ['Efectivo', 'Transferencia bancaria', 'Tarjeta de crédito', 'Acordar con el comprador', 'Giro postal']:
        X['pago_' + p.split()[0].lower()] = pagos.map(lambda s: p in s).astype(int)
    for t in ['dragged_bids_and_visits', 'good_quality_thumbnail', 'dragged_visits', 'free_relist']:
        X['tag_' + t] = d.tags.map(lambda l: t in l).astype(int)
    X['warranty_null'] = d.warranty.isna().astype(int)
    X['tiene_video'] = d.video_id.notna().astype(int)
    X['tiene_parent'] = d.parent_item_id.notna().astype(int)
    X['tiene_official_store'] = d.official_store_id.notna().astype(int)
    w = norm_txt(d.warranty)
    X['warranty_sin'] = w.str.contains('sin gar').astype(int)
    X['warranty_meses'] = w.str.contains(r'mes|ano').astype(int)
    X['warranty_reputacion'] = w.str.contains(r'reputacion|calificacion').astype(int)
    t = norm_txt(d.title)
    X['titulo_n_palabras'] = t.str.split().str.len()
    if flags_titulo:
        X['titulo_flag_usado'] = t.str.contains(RE_USADO).astype(int)
        X['titulo_flag_nuevo'] = t.str.contains(RE_NUEVO).astype(int)
    X['dias_publicacion'] = (d.stop_time - d.start_time).dt.days
    X['dias_hasta_update'] = (pd.to_datetime(d.last_updated) - pd.to_datetime(d.date_created)).dt.days
    X['seller_id_num'] = d.seller_id  # id numérico del vendedor (anonimizado: no tiene orden temporal)
    for c in CATS:
        X[c] = d[c].astype('category')
    return X

def add_freq(Xtr, Xte, dtr, dte, cols=('seller_id', 'category_id')):
    for c in cols:
        vc = pd.concat([dtr[c], dte[c]]).value_counts()
        Xtr['freq_' + c] = dtr[c].map(vc).values
        Xte['freq_' + c] = dte[c].map(vc).values
    return Xtr, Xte

def target_encode(tr_keys, y_tr, ap_keys, k=5, alpha=10, seed=42):
    """TE suavizado. Para train: out-of-fold (k folds). Para 'ap' (validación/test): mapping con todo train."""
    from sklearn.model_selection import KFold
    prior = y_tr.mean()
    def mapping(keys, yy):
        g = pd.DataFrame({'k': keys.values, 'y': yy.values}).groupby('k').y.agg(['sum', 'count'])
        return (g['sum'] + alpha * prior) / (g['count'] + alpha)
    oof = np.full(len(tr_keys), prior)
    for a, b in KFold(k, shuffle=True, random_state=seed).split(tr_keys):
        m = mapping(tr_keys.iloc[a], y_tr.iloc[a])
        oof[b] = tr_keys.iloc[b].map(m).fillna(prior).values
    ap = ap_keys.map(mapping(tr_keys, y_tr)).fillna(prior).values
    return oof, ap


# ---------------------------------------------------------------------------
# Features v2 (precio, fotos, agregados de vendedor/categoría, texto)
# ---------------------------------------------------------------------------

def extra_features(tr, te):
    """Features nuevas calculadas sobre train+test juntos (no usan el target)."""
    a = pd.concat([tr, te], ignore_index=True)
    n_tr = len(tr)
    F = pd.DataFrame(index=a.index)
    p = a.price
    # --- precio
    F['precio_decimales'] = (p % 1 != 0).astype(int)
    F['precio_redondo_100'] = (p % 100 == 0).astype(int)
    F['precio_redondo_1000'] = (p % 1000 == 0).astype(int)
    F['precio_termina_9'] = (p.astype(int) % 10 == 9).astype(int)
    F['precio_ars'] = np.log1p(np.where(a.currency_id == 'USD', p * 9.5, p))
    # --- fotos: fecha de subida (id = "..._MMYYYY") y tamaño
    def fechas(l):
        out = []
        for x in l:
            s = x['id'].rsplit('_', 1)[-1]
            if len(s) == 6 and s.isdigit(): out.append(int(s[2:]) * 12 + int(s[:2]))
        return out
    fd = a.pictures.map(fechas)
    creado = pd.to_datetime(a.date_created)
    m_creado = creado.dt.year * 12 + creado.dt.month
    F['foto_meses_antes_min'] = m_creado - fd.map(lambda l: min(l) if l else np.nan)
    F['foto_meses_antes_max'] = m_creado - fd.map(lambda l: max(l) if l else np.nan)
    def area(l):
        try: w, h = l[0]['max_size'].split('x'); return int(w) * int(h)
        except Exception: return np.nan
    F['foto_area'] = np.log1p(a.pictures.map(area))
    def vertical(l):
        try: w, h = l[0]['max_size'].split('x'); return int(int(h) > int(w))
        except Exception: return 0
    F['foto_vertical'] = a.pictures.map(vertical)
    # --- título
    t = a.title.fillna('')
    F['titulo_len'] = t.str.len()
    F['titulo_pct_mayus'] = t.str.count(r'[A-Z]') / F.titulo_len.clip(lower=1)
    F['titulo_n_digitos'] = t.str.count(r'\d')
    # --- fechas
    st = a.start_time
    F['hora'] = st.dt.hour; F['dia_semana'] = st.dt.dayofweek
    F['mes_creado'] = m_creado - m_creado.min()
    # --- variaciones
    F['var_stock'] = np.log1p(a.variations.map(lambda l: sum(v.get('available_quantity', 0) for v in l)))
    # --- agregados por vendedor (sobre train+test)
    a['_lp'] = np.log1p(a.price); a['_lq'] = np.log1p(a.initial_quantity)
    a['_free'] = (a.listing_type_id == 'free').astype(int); a['_relist'] = a.automatic_relist.astype(int)
    a['_fship'] = a['shipping.free_shipping'].astype(int); a['_npic'] = a.pictures.map(len)
    a['_tusado'] = norm_txt(a.title).str.contains(RE_USADO).astype(int)
    a['_wnull'] = a.warranty.isna().astype(int); a['_me2'] = (a['shipping.mode'] == 'me2').astype(int)
    g = a.groupby('seller_id')
    for c in ['_lp', '_lq', '_free', '_relist', '_fship', '_npic', '_tusado', '_wnull', '_me2']:
        F['seller_mean' + c] = g[c].transform('mean')
    F['seller_std_lp'] = g['_lp'].transform('std')
    F['seller_n_categorias'] = g['category_id'].transform('nunique')
    # --- precio relativo a su categoría
    gc = a.groupby('category_id')['_lp']
    F['cat_mediana_lp'] = gc.transform('median')
    F['precio_vs_cat'] = a['_lp'] - F['cat_mediana_lp']
    F['precio_rank_cat'] = gc.rank(pct=True)
    return F.iloc[:n_tr].reset_index(drop=True), F.iloc[n_tr:].reset_index(drop=True)


def textos(df):
    return norm_txt(df.title), norm_txt(df.warranty)

def tfidf_fit_transform(tr_txt, ap_txts):
    vw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200000)
    vc = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=300000)
    vg = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    t, w = tr_txt
    A = hstack([vw.fit_transform(t), vc.fit_transform(t), vg.fit_transform(w)]).tocsr()
    outs = [hstack([vw.transform(t2), vc.transform(t2), vg.transform(w2)]).tocsr() for t2, w2 in ap_txts]
    return A, outs

def lr_text_oof(txt_tr, y_tr, txt_ap, C=4.0, k=5, seed=42):
    """Prob. de 'usado' según el texto (título + garantía) con TF-IDF + regresión logística.
    Train: out-of-fold (k folds); 'ap': modelo entrenado con todo train."""
    t, w = txt_tr
    oof = np.zeros(len(t))
    for a, b in StratifiedKFold(k, shuffle=True, random_state=seed).split(t, y_tr):
        A, (B,) = tfidf_fit_transform((t.iloc[a], w.iloc[a]), [(t.iloc[b], w.iloc[b])])
        oof[b] = LogisticRegression(C=C, max_iter=2000).fit(A, y_tr.iloc[a]).predict_proba(B)[:, 1]
    A, (B,) = tfidf_fit_transform((t, w), [txt_ap])
    ap = LogisticRegression(C=C, max_iter=2000).fit(A, y_tr).predict_proba(B)[:, 1]
    return oof, ap


# ---------------------------------------------------------------------------
# Carga, folds y guardado
# ---------------------------------------------------------------------------

SEED = 42
TE_COLS = ['seller_id', 'category_id']

def cargar_todo(train_path='train_data.jsonlines', test_path='test_data.jsonlines', cache='cache_features_v2.pkl'):
    """Carga train/test, arma las features (con flags de título) y las cachea en disco.
    Devuelve dict con X, Xt (features), y, ids de test y las columnas crudas para TE / categóricas."""
    if cache and os.path.exists(cache):
        with open(cache, 'rb') as f:
            return pickle.load(f)
    train, test = cargar(train_path), cargar(test_path)
    y = (train.condition == 'used').astype(int)
    X, Xt = features(train, True), features(test, True)
    X, Xt = add_freq(X, Xt, train, test)
    E, Et = extra_features(train, test)
    X = pd.concat([X.reset_index(drop=True), E], axis=1); Xt = pd.concat([Xt.reset_index(drop=True), Et], axis=1)
    d = dict(txt_tr=textos(train), txt_te=textos(test), X=X, Xt=Xt, y=y, test_id=test['id'].values,
             raw_tr=train[TE_COLS].astype(str), raw_te=test[TE_COLS].astype(str))
    if cache:
        with open(cache, 'wb') as f:
            pickle.dump(d, f)
    return d

def armar_folds(d, n_splits=5, te=True, raw_cats=False, texto=True):
    """Folds estratificados (mismos en todas las notebooks). Por fold:
    - te=True: agrega target encoding OOF de seller_id / category_id (calculado sólo con la parte de train del fold)
    - raw_cats=True: agrega seller_id / category_id crudos como string (para CatBoost)."""
    X, y = d['X'], d['y']
    TXT = features_texto(d) if texto else None
    folds = []
    for a, b in StratifiedKFold(n_splits, shuffle=True, random_state=SEED).split(X, y):
        Xa, Xb = X.iloc[a].copy(), X.iloc[b].copy()
        for c in TE_COLS:
            if te:
                Xa['te_' + c], Xb['te_' + c] = target_encode(d['raw_tr'][c].iloc[a], y.iloc[a], d['raw_tr'][c].iloc[b])
            if raw_cats:
                Xa['cat_' + c], Xb['cat_' + c] = d['raw_tr'][c].iloc[a].values, d['raw_tr'][c].iloc[b].values
        if texto:
            Xa['txt_lr'], Xb['txt_lr'] = TXT['folds'][len(folds)]
        folds.append((a, b, Xa, y.iloc[a], Xb, y.iloc[b]))
    return folds

def armar_full(d, te=True, raw_cats=False, texto=True):
    """Train completo + test para el modelo final (TE: OOF en train, mapping completo en test)."""
    X, Xt = d['X'].copy(), d['Xt'].copy()
    for c in TE_COLS:
        if te:
            X['te_' + c], Xt['te_' + c] = target_encode(d['raw_tr'][c], d['y'], d['raw_te'][c])
        if raw_cats:
            X['cat_' + c], Xt['cat_' + c] = d['raw_tr'][c].values, d['raw_te'][c].values
    if texto:
        X['txt_lr'], Xt['txt_lr'] = features_texto(d)['full']
    for c in CATS:   # mismas categorías (y mismos códigos) en train y test, sin importar la versión de XGBoost
        if str(X[c].dtype) == 'category':
            Xt[c] = pd.Categorical(Xt[c].astype(str), categories=X[c].cat.categories)
    return X, Xt

def guardar_submission(d, proba, path, umbral=0.5):
    sub = pd.DataFrame({'ID': np.arange(1, len(proba) + 1),  # Kaggle: ID = n° de fila del test, desde 1
                        'condition': np.where(proba >= umbral, 'used', 'new')})
    sub.to_csv(path, index=False)
    return sub


def features_texto(d, cache='cache_texto.pkl'):
    """Feature 'txt_lr' = prob. de usado según TF-IDF(título + garantía) + regresión logística (stacking).
    Anidado para no filtrar: para cada fold externo, OOF interno sobre su parte de train y modelo completo para su validación.
    'full': OOF sobre todo train + predicción para test (modelo final). Tarda ~15 min la primera vez; después lee el cache."""
    if cache and os.path.exists(cache):
        with open(cache, 'rb') as f: return pickle.load(f)
    (tt, tw), y = d['txt_tr'], d['y']
    out = {'folds': []}
    for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(d['X'], y):
        out['folds'].append(lr_text_oof((tt.iloc[a], tw.iloc[a]), y.iloc[a], (tt.iloc[b], tw.iloc[b])))
    out['full'] = lr_text_oof((tt, tw), y, d['txt_te'])
    if cache:
        with open(cache, 'wb') as f: pickle.dump(out, f)
    return out


# ---------------------------------------------------------------------------
# Features v3: modelo lineal 'ancho', target encodings combinados, foto ajena
# ---------------------------------------------------------------------------

def wide_inputs(df):
    t = norm_txt(df.title); w = norm_txt(df.warranty)
    attr = df.attributes.map(lambda l: ' '.join(f"{x.get('id','')}_{x.get('value_name','')}" for x in l)).str.lower()
    c = pd.DataFrame({'lt': df.listing_type_id, 'bm': df.buying_mode, 'sm': df['shipping.mode'], 'st': df.status,
        'seller': df.seller_id.astype(str), 'cat': df.category_id, 'state': df['seller_address.state.name'],
        'cur': df.currency_id, 'relist': df.automatic_relist.astype(str), 'fs': df['shipping.free_shipping'].astype(str),
        'q': np.floor(np.log1p(df.initial_quantity) * 1.5).astype(int).astype(str),
        'sold': np.floor(np.log1p(df.sold_quantity) * 1.5).astype(int).astype(str),
        'p': np.floor(np.log1p(df.price) * 2).astype(int).astype(str),
        'npic': df.pictures.map(len).clip(upper=8).astype(str), 'nvar': df.variations.map(len).clip(upper=5).astype(str),
        'pay': df.non_mercado_pago_payment_methods.map(lambda l: '|'.join(sorted(m['description'] for m in l))),
        'wnull': df.warranty.isna().astype(str)})
    c['lt_q'] = c['lt'] + c['q']; c['cat_lt'] = c['cat'] + c['lt']; c['seller_lt'] = c['seller'] + c['lt']
    return pd.DataFrame({'t': t.values, 'w': w.values, 'attr': attr.values}), c.reset_index(drop=True)

def _mats(Tf, Cf, aps):
    vw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200000)
    vc = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=300000)
    vg = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    va = TfidfVectorizer(token_pattern=r'\S+', min_df=2)
    oh = OneHotEncoder(handle_unknown='ignore', min_frequency=2)
    A = hstack([vw.fit_transform(Tf.t), vc.fit_transform(Tf.t), vg.fit_transform(Tf.w), va.fit_transform(Tf.attr), oh.fit_transform(Cf)]).tocsr()
    return A, [hstack([vw.transform(T.t), vc.transform(T.t), vg.transform(T.w), va.transform(T.attr), oh.transform(C)]).tocsr() for T, C in aps]

def wide_oof(T, C, y, Tap, Cap, Cval=1.0, k=5, seed=42):
    """Modelo lineal 'ancho' (TF-IDF título/garantía/atributos + one-hot de categóricas, vendedor, categoría y bins numéricos).
    Train: OOF k-fold; ap: modelo con todo."""
    y = pd.Series(np.asarray(y)); oof = np.zeros(len(y))
    for a, b in StratifiedKFold(k, shuffle=True, random_state=seed).split(T, y):
        A, (B,) = _mats(T.iloc[a], C.iloc[a], [(T.iloc[b], C.iloc[b])])
        oof[b] = LogisticRegression(C=Cval, max_iter=3000, solver='liblinear').fit(A, y.iloc[a]).predict_proba(B)[:, 1]
    A, (B,) = _mats(T, C, [(Tap, Cap)])
    return oof, LogisticRegression(C=Cval, max_iter=3000, solver='liblinear').fit(A, y).predict_proba(B)[:, 1]


def cargar_raw(train_path='train_data.jsonlines', test_path='test_data.jsonlines', cache='cache_raw.pkl'):
    """DataFrames crudos (aplanados) de train y test, cacheados."""
    if cache and os.path.exists(cache):
        with open(cache, 'rb') as f: return pickle.load(f)
    r = (cargar(train_path), cargar(test_path))
    if cache:
        with open(cache, 'wb') as f: pickle.dump(r, f)
    return r

def _keys_v3(df):
    t = norm_txt(df.title); s = df.seller_id.astype(str)
    return {'te_seller_lt': s + '|' + df.listing_type_id, 'te_cat_lt': df.category_id + '|' + df.listing_type_id,
            'te_seller_cat': s + '|' + df.category_id, 'te_cat_bm': df.category_id + '|' + df.buying_mode, 'te_titulo': t}

def _foto_ajena(df):
    """1 si la primera foto pertenece a otra publicación (id de la foto != id del ítem)."""
    pid = df.pictures.map(lambda l: l[0]['id'].split('-')[1].split('_')[0] if l and '-' in l[0]['id'] else '')
    return ((pid != df.id) & (pid != '')).astype(int).values

def features_wide(d, raw, cache='cache_wide.pkl'):
    """'wide_lr': prob. de usado de una regresión logística sobre TF-IDF (título, garantía, atributos) + one-hot de
    categóricas (vendedor, categoría, bins numéricos). Stacking anidado igual que txt_lr. ~20 min la 1ra vez."""
    if cache and os.path.exists(cache):
        with open(cache, 'rb') as f: return pickle.load(f)
    tr, te = raw; y = d['y'].values
    T, C = wide_inputs(tr); Tt, Ct = wide_inputs(te)
    out = {'folds': []}
    for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(d['X'], y):
        out['folds'].append(wide_oof(T.iloc[a].reset_index(drop=True), C.iloc[a].reset_index(drop=True), y[a],
                                     T.iloc[b].reset_index(drop=True), C.iloc[b].reset_index(drop=True)))
    out['full'] = wide_oof(T, C, y, Tt, Ct)
    if cache:
        with open(cache, 'wb') as f: pickle.dump(out, f)
    return out

def armar_folds_v3(d, raw, te=True, raw_cats=False):
    """Folds v2 + wide_lr + target encodings combinados + foto_ajena (todo calculado dentro de cada fold)."""
    W = features_wide(d, raw); K = _keys_v3(raw[0]); FA = _foto_ajena(raw[0])
    out = []
    for i, (a, b, Xa, ya, Xb, yb) in enumerate(armar_folds(d, te=te, raw_cats=raw_cats)):
        Xa['wide_lr'], Xb['wide_lr'] = W['folds'][i]
        for k in K: Xa[k], Xb[k] = target_encode(K[k].iloc[a], ya, K[k].iloc[b])
        Xa['foto_ajena'], Xb['foto_ajena'] = FA[a], FA[b]
        out.append((a, b, Xa, ya, Xb, yb))
    return out

def armar_full_v3(d, raw, te=True, raw_cats=False):
    W = features_wide(d, raw); K, Kt = _keys_v3(raw[0]), _keys_v3(raw[1])
    X, Xt = armar_full(d, te=te, raw_cats=raw_cats)
    X['wide_lr'], Xt['wide_lr'] = W['full']
    for k in K: X[k], Xt[k] = target_encode(K[k], d['y'], Kt[k])
    X['foto_ajena'], Xt['foto_ajena'] = _foto_ajena(raw[0]), _foto_ajena(raw[1])
    return X, Xt


# ---------------------------------------------------------------------------
# Features de fotos (metadata: proporciones, tamaños, formato de ids)
# ---------------------------------------------------------------------------

def _wh(s):
    try: w, h = s.split('x'); return int(w), int(h)
    except Exception: return np.nan, np.nan

def fotos_features(df):
    """Metadata de las fotos: proporciones (cuadrada = foto de producto), variedad de tamaños, formato/numeración de ids."""
    F = pd.DataFrame(index=range(len(df)))
    P = df.pictures.reset_index(drop=True)
    sizes = P.map(lambda l: [_wh(p['size']) for p in l])
    maxs = P.map(lambda l: [_wh(p['max_size']) for p in l])
    def ratios(lst): return [w / h for w, h in lst if h and h == h]
    r = sizes.map(ratios)
    F['foto1_ratio'] = r.map(lambda x: x[0] if x else np.nan)
    F['foto1_cuadrada'] = F.foto1_ratio.between(0.98, 1.02).astype(int)
    F['fotos_pct_cuadradas'] = r.map(lambda x: np.mean([0.98 <= v <= 1.02 for v in x]) if x else np.nan)
    F['fotos_pct_43'] = r.map(lambda x: np.mean([abs(v - 4/3) < .02 or abs(v - 3/4) < .02 for v in x]) if x else np.nan)
    F['fotos_n_tamanos'] = P.map(lambda l: len({p['size'] for p in l}))
    F['fotos_n_maxsize'] = P.map(lambda l: len({p['max_size'] for p in l}))
    F['fotos_maxarea_media'] = maxs.map(lambda l: np.log1p(np.mean([w * h for w, h in l if w == w])) if l else np.nan)
    F['fotos_maxarea_min'] = maxs.map(lambda l: np.log1p(min([w * h for w, h in l if w == w], default=np.nan)) if l else np.nan)
    # ids de fotos: "PREFIJO-MLAxxxx_SUFIJO"
    def parse(pid):
        m = re.match(r'(\d+)-MLA\d+_(\d+)', pid or '')
        return (int(m.group(1)), len(m.group(2))) if m else (np.nan, np.nan)
    ids = P.map(lambda l: [parse(p['id']) for p in l])
    F['foto_prefijo_log'] = ids.map(lambda l: np.log1p(l[0][0]) if l else np.nan)
    F['foto_prefijo_min_log'] = ids.map(lambda l: np.log1p(np.nanmin([a for a, b in l])) if l and any(a == a for a, b in l) else np.nan)
    F['foto_sufijo_len'] = ids.map(lambda l: l[0][1] if l else np.nan)
    F['fotos_pct_formato_viejo'] = ids.map(lambda l: np.mean([b < 6 for a, b in l if b == b]) if l else np.nan)
    th = df.thumbnail.reset_index(drop=True).str.extract(r'/(\d+)-MLA\d+_(\d+)-')
    F['thumb_prefijo_log'] = np.log1p(th[0].astype(float)); F['thumb_sufijo_len'] = th[1].str.len()
    F['thumb_distinto_foto1'] = (df.thumbnail.reset_index(drop=True).str.extract(r'/(\d+-MLA\d+_\d+)-I')[0]
                                 != P.map(lambda l: l[0]['id'] if l else None)).astype(int)
    # pagos: combinación exacta de ids (frecuencia) y variaciones con SKU
    F['var_con_sku'] = df.variations.reset_index(drop=True).map(lambda l: int(any(v.get('seller_custom_field') for v in l)))
    F['free_methods'] = df['shipping.free_methods'].reset_index(drop=True).map(lambda l: len(l) if isinstance(l, list) else 0)
    return F


def cargar_texto_nn(nombre):
    """Predicciones out-of-fold + test de un modelo de lenguaje (bert / beto). Busca en preds/ y en la carpeta actual."""
    for carpeta in ['preds', '.']:
        if os.path.exists(f'{carpeta}/oof_{nombre}.npy'):
            return np.load(f'{carpeta}/oof_{nombre}.npy'), np.load(f'{carpeta}/test_{nombre}.npy')
    raise FileNotFoundError(f'No encuentro oof_{nombre}.npy (correr las notebooks 1a / 1b)')

def _logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5); return np.log(p / (1 - p))

def armar_final(d, raw, te=True, raw_cats=False, base='v3'):
    """Set de features final = v3 (o v2) + fotos + logit(prob. BERT) + logit(prob. BETO). Devuelve (folds, X, Xt)."""
    FF, FFt = fotos_features(raw[0]), fotos_features(raw[1])
    nn = {k: cargar_texto_nn(k) for k in ['bert', 'beto']}
    def add(X, idx, test=False):
        X = pd.concat([X.reset_index(drop=True), (FFt if test else FF.iloc[idx]).reset_index(drop=True)], axis=1)
        for k, (o, t) in nn.items(): X[k] = _logit(t if test else o[idx])
        return X
    if base == 'v3':
        folds = armar_folds_v3(d, raw, te=te, raw_cats=raw_cats); X, Xt = armar_full_v3(d, raw, te=te, raw_cats=raw_cats)
    else:
        folds = armar_folds(d, te=te, raw_cats=raw_cats); X, Xt = armar_full(d, te=te, raw_cats=raw_cats)
    folds = [(a, b, add(Xa, a), ya.reset_index(drop=True), add(Xb, b), yb.reset_index(drop=True)) for a, b, Xa, ya, Xb, yb in folds]
    allidx = np.arange(len(d['y']))
    return folds, add(X, allidx), add(Xt, None, test=True)


# ---------------------------------------------------------------------------
# Tiempos exactos (en segundos) y ritmo de publicación del vendedor
# ---------------------------------------------------------------------------

def tiempos_features(tr, te):
    """Tiempos EXACTOS (en segundos) y ritmo de publicación del vendedor. Train+test juntos (no usa el target)."""
    a = pd.concat([tr, te], ignore_index=True); n = len(tr)
    dc = pd.to_datetime(a.date_created).dt.tz_localize(None); lu = pd.to_datetime(a.last_updated).dt.tz_localize(None)
    st, sp = a.start_time, a.stop_time
    T = pd.DataFrame(index=a.index)
    dur = (sp - st).dt.total_seconds()
    T['dur_s'] = dur
    T['dur_resto_hora_s'] = dur - np.round(dur / 3600) * 3600          # segundos que sobran/faltan respecto de la hora exacta
    T['dur_resto_dia_s'] = dur - np.round(dur / 86400) * 86400
    T['dur_60d_mas_segundos'] = ((dur > 5184000) & (dur <= 5184000 + 60)).astype(int)
    T['start_menos_created_s'] = (st - dc).dt.total_seconds()
    T['upd_menos_created_s'] = (lu - dc).dt.total_seconds()
    T['upd_menos_start_s'] = (lu - st).dt.total_seconds()
    T['upd_menos_created_log'] = np.log1p(T.upd_menos_created_s.clip(lower=0))
    T['seg_creado'] = dc.dt.second; T['min_creado'] = dc.dt.minute
    T['hora_update'] = lu.dt.hour
    ts = dc.astype('int64') // 10**9; a['_t'] = ts
    o = a.sort_values(['seller_id', '_t'])
    T.loc[o.index, 'seg_desde_anterior_vendedor'] = np.log1p(o.groupby('seller_id')._t.diff().clip(lower=0))
    T.loc[o.index, 'seg_hasta_siguiente_vendedor'] = np.log1p((-o.groupby('seller_id')._t.diff(-1)).clip(lower=0))
    hora = a.seller_id.astype(str) + dc.dt.strftime('%Y%m%d%H')
    T['pubs_vendedor_misma_hora'] = hora.map(hora.value_counts())
    return T.iloc[:n].reset_index(drop=True), T.iloc[n:].reset_index(drop=True)


def armar_final_t(d, raw, te=True, raw_cats=False, base='v3'):
    """Set final + 14 variables de tiempo exacto."""
    T, Tt = tiempos_features(*raw)
    F, X, Xt = armar_final(d, raw, te=te, raw_cats=raw_cats, base=base)
    F = [(a, b, pd.concat([Xa, T.iloc[a].reset_index(drop=True)], axis=1), ya, pd.concat([Xb, T.iloc[b].reset_index(drop=True)], axis=1), yb)
         for a, b, Xa, ya, Xb, yb in F]
    return F, pd.concat([X, T], axis=1), pd.concat([Xt, Tt], axis=1)


# ---------------------------------------------------------------------------
# Vecinos de título: precio y atributos de las publicaciones del 'mismo producto'
# ---------------------------------------------------------------------------

def vecinos_features(tr, te, k=10, chunk=800, cache='cache_vecinos.pkl'):
    """Para cada publicación, sus k títulos más parecidos en train+test (TF-IDF coseno, sin ella misma y sin el mismo vendedor).
    Compara precio y atributos con esos 'mismos productos'. No usa el target."""
    import os, pickle
    if cache and os.path.exists(cache): return pickle.load(open(cache, 'rb'))
    a = pd.concat([tr, te], ignore_index=True); n = len(tr)
    V = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, dtype=np.float32).fit_transform(norm_txt(a.title))
    lp = np.log1p(np.where(a.currency_id == 'USD', a.price * 9.5, a.price)).astype(np.float32)
    seller = a.seller_id.values
    dur = (a.stop_time - a.start_time).dt.total_seconds().values
    attr = {'free': (a.listing_type_id == 'free').values, 'garantia': a.warranty.notna().values,
            'flujo60s': ((dur > 5184000) & (dur <= 5184060)), 'stock1': (a.initial_quantity == 1).values,
            'envio_gratis': a['shipping.free_shipping'].astype(bool).values, 'relist': a.automatic_relist.astype(bool).values}
    N = len(a); out = {c: np.full(N, np.nan, np.float32) for c in ['vec_sim1', 'vec_sim_med', 'vec_lp_med', 'vec_lp_min', 'vec_n_sim50'] + [f'vec_pct_{c}' for c in attr]}
    t0 = time.time()
    for s in range(0, N, chunk):
        S = (V[s:s+chunk] @ V.T).toarray()
        rows = np.arange(s, min(s + chunk, N))
        S[np.arange(len(rows)), rows] = -1                                   # sin sí misma
        S[seller[rows][:, None] == seller[None, :]] = -1                     # sin el mismo vendedor (otro producto "independiente")
        idx = np.argpartition(-S, k, axis=1)[:, :k]; sim = np.take_along_axis(S, idx, 1)
        o = np.argsort(-sim, 1); idx = np.take_along_axis(idx, o, 1); sim = np.take_along_axis(sim, o, 1)
        ok = sim > 0.3; w = np.where(ok, sim, 0)
        out['vec_sim1'][rows] = sim[:, 0]; out['vec_sim_med'][rows] = sim.mean(1)
        out['vec_n_sim50'][rows] = (S > 0.5).sum(1)
        L = lp[idx]; Lm = np.where(ok, L, np.nan)
        with np.errstate(all='ignore'):
            out['vec_lp_med'][rows] = np.nanmedian(Lm, 1); out['vec_lp_min'][rows] = np.nanmin(Lm, 1)
            for c, v in attr.items():
                out[f'vec_pct_{c}'][rows] = (w * v[idx]).sum(1) / np.where(w.sum(1) > 0, w.sum(1), np.nan)
        if s % 20000 == 0: print(f'  vecinos {s}/{N} ({time.time()-t0:.0f}s)', flush=True)
    F = pd.DataFrame(out)
    F['vec_precio_vs_mismo_prod'] = lp - F.vec_lp_med          # <0: más barato que el mismo producto en otras publicaciones
    F['vec_precio_vs_min'] = lp - F.vec_lp_min
    r = (F.iloc[:n].reset_index(drop=True), F.iloc[n:].reset_index(drop=True))
    if cache: pickle.dump(r, open(cache, 'wb'))
    return r


def armar_final_tv(d, raw, te=True, raw_cats=False, base='v3'):
    """Set final + tiempos exactos + vecinos de título: el set de variables del modelo final (132; la notebook 2 le suma XLM-R)."""
    Vtr, Vte = vecinos_features(*raw)
    F, X, Xt = armar_final_t(d, raw, te=te, raw_cats=raw_cats, base=base)
    F = [(a, b, pd.concat([Xa, Vtr.iloc[a].reset_index(drop=True)], axis=1), ya, pd.concat([Xb, Vtr.iloc[b].reset_index(drop=True)], axis=1), yb)
         for a, b, Xa, ya, Xb, yb in F]
    return F, pd.concat([X, Vtr], axis=1), pd.concat([Xt, Vte], axis=1)
