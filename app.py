"""
app.py v5 — ORS Primary + OSRM Fallback (No Azure)
====================================================
Routing chain: ORS → OSRM → Haversine

ORS advantages over Azure:
  - avoid_polygons (GeoJSON) for Safest route
  - preference=fastest/shortest for different route types
  - Free: 2000 req/day, no credit card

OSRM advantages:
  - No key needed at all
  - Unlimited (fair use)
  - Very fast
  - But no avoid areas support
"""

import os, json, logging, requests, time, csv
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timezone
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from math import radians, sin, cos, sqrt, atan2
from threading import Lock

app = Flask(__name__, static_folder="static")
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Load Model + Data ──
MODEL_PATH = os.environ.get("MODEL_PATH", "model/random_forest_model.pkl")
FLOOD_CSV = os.environ.get("FLOOD_CSV", "data/Desa_Rawan_Banjir.csv")
SHELTER_CSV = os.environ.get("SHELTER_CSV", "data/Tempat_Evakuasi_Final__1_.csv")

try: model = joblib.load(MODEL_PATH); logger.info(f"Model: {MODEL_PATH}")
except: model = None; logger.error("Model failed")

try:
    df_flood = pd.read_csv(FLOOD_CSV); df_flood.columns = df_flood.columns.str.strip()
    df_shelters = pd.read_csv(SHELTER_CSV); df_shelters.columns = df_shelters.columns.str.strip()
except: df_flood, df_shelters = pd.DataFrame(), pd.DataFrame()

# ── API Keys ──
ORS_KEY = os.environ.get("ORS_API_KEY", "")
# No key needed for OSRM

# ── Constants ──
THRESHOLDS = {"rr_critical":63,"rr_severe":39,"rr_caution":19,"rh_critical":86,"rh_severe":84,"rh_caution":83}
DELAY_FACTORS = {5:5.0,4:2.5,3:1.5,2:1.2,1:1.0}
RISK_LABELS = {1:"Safe",2:"Moderate",3:"Caution",4:"Severe",5:"Critical"}
ROUTE_LABELS = {0:"Fastest",1:"Safest",2:"Balanced"}
ROAD_FACTOR = 1.4; AVG_SPEED = 40

KECAMATAN = [
    {"adm4":"33.01.01.2001","nama":"Dayeuhluhur","lat":-7.218,"lon":108.405},
    {"adm4":"33.01.02.2001","nama":"Wanareja","lat":-7.333,"lon":108.687},
    {"adm4":"33.01.03.2001","nama":"Majenang","lat":-7.300,"lon":108.760},
    {"adm4":"33.01.04.2001","nama":"Cimanggu","lat":-7.245,"lon":108.895},
    {"adm4":"33.01.05.2001","nama":"Karangpucung","lat":-7.336,"lon":108.834},
    {"adm4":"33.01.06.2001","nama":"Cipari","lat":-7.395,"lon":108.727},
    {"adm4":"33.01.07.2001","nama":"Sidareja","lat":-7.470,"lon":108.821},
    {"adm4":"33.01.08.2001","nama":"Kedungreja","lat":-7.548,"lon":108.775},
    {"adm4":"33.01.09.2001","nama":"Patimuan","lat":-7.578,"lon":108.777},
    {"adm4":"33.01.10.2001","nama":"Gandrungmangu","lat":-7.530,"lon":108.700},
    {"adm4":"33.01.11.2001","nama":"Bantarsari","lat":-7.570,"lon":108.890},
    {"adm4":"33.01.12.2001","nama":"Kawunganten","lat":-7.598,"lon":108.930},
    {"adm4":"33.01.13.2001","nama":"Kampung Laut","lat":-7.660,"lon":108.820},
    {"adm4":"33.01.14.2001","nama":"Jeruklegi","lat":-7.598,"lon":109.010},
    {"adm4":"33.01.15.2001","nama":"Kesugihan","lat":-7.630,"lon":109.060},
    {"adm4":"33.01.16.2001","nama":"Adipala","lat":-7.680,"lon":109.080},
    {"adm4":"33.01.17.2001","nama":"Maos","lat":-7.620,"lon":109.120},
    {"adm4":"33.01.18.2001","nama":"Sampang","lat":-7.580,"lon":109.145},
    {"adm4":"33.01.19.2001","nama":"Kroya","lat":-7.630,"lon":109.245},
    {"adm4":"33.01.20.2001","nama":"Binangun","lat":-7.610,"lon":109.310},
    {"adm4":"33.01.21.2001","nama":"Nusawungu","lat":-7.660,"lon":109.365},
    {"adm4":"33.01.22.2001","nama":"Cilacap Selatan","lat":-7.740,"lon":109.010},
    {"adm4":"33.01.23.2001","nama":"Cilacap Tengah","lat":-7.720,"lon":109.015},
    {"adm4":"33.01.24.2001","nama":"Cilacap Utara","lat":-7.690,"lon":109.005},
]

# ── Helpers ──
def haversine_km(lat1,lon1,lat2,lon2):
    R=6371.0; la1,lo1,la2,lo2=map(radians,[lat1,lon1,lat2,lon2])
    dlat,dlon=la2-la1,lo2-lo1; a=sin(dlat/2)**2+cos(la1)*cos(la2)*sin(dlon/2)**2
    return R*2*atan2(sqrt(a),sqrt(1-a))

def assign_risk(rr,rh):
    th=THRESHOLDS
    if rr>=8888 or rr>th["rr_critical"]: return 5
    if rr>th["rr_severe"] or rh>=th["rh_critical"]: return 4
    if rr>th["rr_caution"] or rh>=th["rh_severe"]: return 3
    if rr>0 or rh>=70: return 2
    return 1

def find_nearest_kecamatan(lat,lon):
    return min(KECAMATAN, key=lambda k: haversine_km(lat,lon,k["lat"],k["lon"]))

def find_nearest_shelters(lat,lon,n=3):
    d=[(sh,haversine_km(lat,lon,sh['Latitude'],sh['Longitude'])) for _,sh in df_shelters.iterrows()]
    d.sort(key=lambda x:x[1]); return d[:n]

def find_nearby_flood_points(lat,lon,radius_km=5.0):
    nearby=[]
    for _,fp in df_flood.iterrows():
        d=haversine_km(lat,lon,fp['Latitude'],fp['Longitude'])
        if 0.1<d<=radius_km: nearby.append({"lat":fp['Latitude'],"lon":fp['Longitude'],"dist":round(d,2),"nama":fp['Nama']})
    nearby.sort(key=lambda x:x['dist']); return nearby[:10]

def _make_polyline(coords_lonlat):
    """Convert [[lon,lat],...] to [[lat,lon],...] for Leaflet."""
    return [[c[1],c[0]] for c in coords_lonlat]


# =================================================================
# ROUTING: ORS → OSRM → Haversine
# =================================================================
_cache = {}

# ── ORS (OpenRouteService) ──
ORS_URL = "https://api.openrouteservice.org/v2/directions/driving-car"

def _ors_route(olat,olon,dlat,dlon, preference="fastest", avoid_polygons=None):
    """Call ORS Directions API. Returns parsed route or None."""
    if not ORS_KEY: return None
    
    body = {
        "coordinates": [[olon,olat],[dlon,dlat]],
        "preference": preference,
        "geometry": True,
        "instructions": False,
    }
    if avoid_polygons:
        body["options"] = {"avoid_polygons": avoid_polygons}
    
    try:
        resp = requests.post(ORS_URL, json=body, timeout=10,
                             headers={"Authorization": ORS_KEY, "Content-Type":"application/json"})
        if resp.status_code == 200:
            data = resp.json()
            routes = data.get("routes", [])
            if routes:
                r = routes[0]
                summary = r.get("summary", {})
                # ORS returns encoded polyline by default — decode it
                geom = r.get("geometry", "")
                if isinstance(geom, str):
                    coords = _decode_polyline(geom)
                elif isinstance(geom, dict):
                    coords = geom.get("coordinates", [])
                else:
                    coords = []
                
                return {
                    "distance_km": round(summary.get("distance",0)/1000, 2),
                    "travel_time_min": round(summary.get("duration",0)/60, 1),
                    "polyline": _make_polyline(coords),
                    "source": f"ORS_{preference}",
                }
        elif resp.status_code == 429:
            logger.warning("ORS rate limit hit")
        else:
            logger.debug(f"ORS {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        logger.debug(f"ORS error: {e}")
    return None


def _decode_polyline(encoded):
    """Decode Google-style encoded polyline to [[lon,lat],...]."""
    coords = []; idx = 0; lat = 0; lng = 0
    while idx < len(encoded):
        for var in ['lat', 'lng']:
            shift = 0; result = 0
            while True:
                b = ord(encoded[idx]) - 63; idx += 1
                result |= (b & 0x1f) << shift; shift += 5
                if b < 0x20: break
            dlat_or_dlng = ~(result >> 1) if (result & 1) else (result >> 1)
            if var == 'lat': lat += dlat_or_dlng
            else: lng += dlat_or_dlng
        coords.append([lng/1e5, lat/1e5])
    return coords


# ── OSRM (Open Source Routing Machine) ──
OSRM_URL = "https://router.project-osrm.org/route/v1/driving"

def _osrm_route(olat,olon,dlat,dlon, alternatives=False):
    """Call OSRM demo server. No key needed. Returns parsed route or None."""
    try:
        params = "overview=full&geometries=geojson"
        if alternatives: params += "&alternatives=true"
        url = f"{OSRM_URL}/{olon},{olat};{dlon},{dlat}?{params}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("code") == "Ok" and data.get("routes"):
                results = []
                for r in data["routes"]:
                    coords = r["geometry"]["coordinates"]  # [[lon,lat],...]
                    results.append({
                        "distance_km": round(r["distance"]/1000, 2),
                        "travel_time_min": round(r["duration"]/60, 1),
                        "polyline": _make_polyline(coords),
                        "source": "OSRM",
                    })
                return results if alternatives else results[0]
    except Exception as e:
        logger.debug(f"OSRM error: {e}")
    return None


# ── Haversine fallback ──
def _haversine_route(olat,olon,dlat,dlon, multiplier=1.0):
    h = haversine_km(olat,olon,dlat,dlon) * ROAD_FACTOR * multiplier
    return {
        "distance_km": round(h,2),
        "travel_time_min": round(h/AVG_SPEED*60, 1),
        "polyline": [[olat,olon],[dlat,dlon]],
        "source": "haversine",
    }


# ── Public routing functions (chain: ORS → OSRM → Haversine) ──

def get_fastest_route(olat,olon,dlat,dlon):
    ck = f"fast:{olat:.5f},{olon:.5f}:{dlat:.5f},{dlon:.5f}"
    if ck in _cache: return _cache[ck]
    
    # Try ORS fastest
    r = _ors_route(olat,olon,dlat,dlon, preference="fastest")
    if r: _cache[ck]=r; return r
    
    # Try OSRM
    r = _osrm_route(olat,olon,dlat,dlon)
    if r: _cache[ck]=r; return r
    
    # Haversine
    r = _haversine_route(olat,olon,dlat,dlon)
    _cache[ck]=r; return r


def get_safest_route(olat,olon,dlat,dlon, flood_points):
    ck = f"safe:{olat:.5f},{olon:.5f}:{dlat:.5f},{dlon:.5f}:{len(flood_points)}"
    if ck in _cache: return _cache[ck]
    
    # Try ORS with avoid_polygons (BEST — actual flood zone avoidance)
    if ORS_KEY and flood_points:
        avoid_coords = []
        for fp in flood_points[:5]:
            lat,lon = fp["lat"],fp["lon"]
            r = 0.008  # ~0.9km box
            avoid_coords.append([
                [lon-r,lat-r],[lon+r,lat-r],[lon+r,lat+r],[lon-r,lat+r],[lon-r,lat-r]
            ])
        avoid_poly = {"type":"MultiPolygon","coordinates":[[c] for c in avoid_coords]}
        
        r = _ors_route(olat,olon,dlat,dlon, preference="recommended", avoid_polygons=avoid_poly)
        if r:
            r["source"] = "ORS_safest_avoid"
            r["avoided_zones"] = len(avoid_coords)
            _cache[ck]=r; return r
    
    # Try ORS recommended (different from fastest, slightly safer)
    r = _ors_route(olat,olon,dlat,dlon, preference="recommended")
    if r:
        r["source"] = "ORS_recommended"
        _cache[ck]=r; return r
    
    # Try OSRM alternatives (pick longest = most detour)
    alts = _osrm_route(olat,olon,dlat,dlon, alternatives=True)
    if alts and len(alts) > 1:
        safest = max(alts, key=lambda x: x["distance_km"])
        safest["source"] = "OSRM_alt_longest"
        _cache[ck]=safest; return safest
    
    # Haversine detour (+40%)
    r = _haversine_route(olat,olon,dlat,dlon, multiplier=1.4)
    r["source"] = "haversine_detour"
    _cache[ck]=r; return r


def get_balanced_route(olat,olon,dlat,dlon):
    ck = f"bal:{olat:.5f},{olon:.5f}:{dlat:.5f},{dlon:.5f}"
    if ck in _cache: return _cache[ck]
    
    # Try ORS shortest (different path from fastest)
    r = _ors_route(olat,olon,dlat,dlon, preference="shortest")
    if r: _cache[ck]=r; return r
    
    # Try OSRM alternatives (pick middle distance)
    alts = _osrm_route(olat,olon,dlat,dlon, alternatives=True)
    if alts and len(alts) >= 2:
        alts.sort(key=lambda x: x["distance_km"])
        mid = alts[len(alts)//2]
        mid["source"] = "OSRM_alt_mid"
        _cache[ck]=mid; return mid
    
    # OSRM single route (still better than haversine)
    r = _osrm_route(olat,olon,dlat,dlon)
    if r:
        r["source"] = "OSRM_balanced"
        _cache[ck]=r; return r
    
    # Haversine moderate (+15%)
    r = _haversine_route(olat,olon,dlat,dlon, multiplier=1.15)
    r["source"] = "haversine_balanced"
    _cache[ck]=r; return r


# =================================================================
# BMKG Weather
# =================================================================
BMKG_URL = "https://api.bmkg.go.id/publik/prakiraan-cuaca"

def fetch_bmkg_weather(adm4):
    try:
        resp = requests.get(BMKG_URL,params={"adm4":adm4},timeout=10,
                            headers={"User-Agent":"FloodEvac/5.0"})
        if resp.status_code!=200: return None
        bmkg=resp.json(); data_list=bmkg.get("data",[])
        if not data_list: return None

        all_fc=[]
        for g in data_list[0].get("cuaca",[]):
            if isinstance(g,list): all_fc.extend(g)
            elif isinstance(g,dict): all_fc.append(g)
        if not all_fc: return None
        all_fc.sort(key=lambda x:x.get("datetime",""))
        cur=all_fc[0]; tp24=round(sum(f.get("tp",0) or 0 for f in all_fc[:8]),1)
        lok=bmkg.get("lokasi",data_list[0].get("lokasi",{}))
        return {
            "source":"BMKG","adm4":adm4,"kecamatan":lok.get("kecamatan",""),
            "rainfall_24h":tp24,"humidity_now":cur.get("hu",80),
            "temperature":cur.get("t",0),"wind_speed":cur.get("ws",0),
            "weather_desc":cur.get("weather_desc",""),"weather_image":cur.get("image",""),
            "forecast_24h":[{"local_datetime":f.get("local_datetime",""),
                "weather_desc":f.get("weather_desc",""),"temperature":f.get("t",0),
                "humidity":f.get("hu",0),"rainfall_3h":f.get("tp",0),
                "wind_speed":f.get("ws",0),"image":f.get("image","")} for f in all_fc[:8]],
            "timestamp":datetime.now(timezone.utc).isoformat()}
    except Exception as e:
        logger.warning(f"BMKG error: {e}"); return None

def simulated_weather():
    import random; random.seed(int(time.time())%10000)
    tp=round(random.uniform(0,40),1) if random.random()<0.4 else 0
    return {"source":"simulated","rainfall_24h":tp,"humidity_now":round(random.uniform(70,98)),
            "temperature":round(random.uniform(24,32)),"wind_speed":round(random.uniform(1,12)),
            "weather_desc":"Hujan Ringan" if tp>0 else "Berawan","weather_image":"",
            "forecast_24h":[],"timestamp":datetime.now(timezone.utc).isoformat()}


# =================================================================
# Prediction Logging
# =================================================================
LOG_PATH="prediction_log.csv"
_log_lock=Lock()
def log_pred(row):
    with _log_lock:
        exists=os.path.exists(LOG_PATH)
        with open(LOG_PATH,"a",newline="") as f:
            w=csv.DictWriter(f,fieldnames=["timestamp","village","vlat","vlon","shelter","slat","slon",
                "route_type","distance_km","travel_time_min","route_source",
                "rainfall_24h","humidity","risk_level","rf_prediction",
                "rf_prob_fastest","rf_prob_safest","rf_prob_balanced","weather_source"])
            if not exists: w.writeheader()
            w.writerow(row)


# =================================================================
# MAIN ENDPOINT
# =================================================================
@app.route("/api/analyze-village", methods=["POST"])
def analyze_village():
    if model is None: return jsonify({"error":"Model not loaded"}),503
    data=request.get_json()
    lat,lon=float(data.get("lat",0)),float(data.get("lon",0))
    nama=data.get("nama","Unknown")

    # 1. Weather
    kec=find_nearest_kecamatan(lat,lon)
    weather=fetch_bmkg_weather(kec["adm4"])
    if weather is None:
        weather=simulated_weather(); weather["kecamatan"]=kec["nama"]; weather["adm4"]=kec["adm4"]
    rr,rh=weather["rainfall_24h"],weather["humidity_now"]
    risk=assign_risk(rr,rh); delay=DELAY_FACTORS[risk]
    weather["risk_level"]=risk; weather["risk_label"]=RISK_LABELS[risk]; weather["delay_factor"]=delay

    # 2. Nearest shelter + alternatives
    nearest=find_nearest_shelters(lat,lon,n=3)
    shelter,_=nearest[0]
    slat,slon=float(shelter['Latitude']),float(shelter['Longitude'])

    # 3. Nearby flood points
    floods=find_nearby_flood_points(lat,lon,radius_km=5.0)

    # 4. Three different routes to SAME shelter
    r_fast=get_fastest_route(lat,lon,slat,slon)
    r_safe=get_safest_route(lat,lon,slat,slon,floods)
    r_bal=get_balanced_route(lat,lon,slat,slon)

    # 5. RF validates
    routes_out=[]
    for rtype,rdata,label_idx in [("Fastest",r_fast,0),("Safest",r_safe,1),("Balanced",r_bal,2)]:
        adj_time=round(rdata["travel_time_min"]*delay,1)
        X=np.array([[rdata["distance_km"],adj_time,rr,rh,float(risk)]])
        pred=int(model.predict(X)[0]); proba=model.predict_proba(X)[0]
        
        warning=None
        if rdata["distance_km"]<2 and rtype=="Fastest":
            warning="Rute pendek — mungkin melewati jalan kecil"

        routes_out.append({
            "type":rtype,"distance_km":rdata["distance_km"],"travel_time_min":adj_time,
            "travel_time_raw":rdata["travel_time_min"],"polyline":rdata["polyline"],
            "route_source":rdata["source"],"rf_prediction":ROUTE_LABELS[pred],
            "rf_agrees":ROUTE_LABELS[pred]==rtype,
            "probability":round(float(proba[label_idx]),4),
            "all_probabilities":{ROUTE_LABELS[i]:round(float(p),4) for i,p in enumerate(proba)},
            "road_warning":warning,"avoided_zones":rdata.get("avoided_zones",0)})

        log_pred({"timestamp":datetime.now(timezone.utc).isoformat(),
            "village":nama,"vlat":lat,"vlon":lon,"shelter":shelter["Nama"],"slat":slat,"slon":slon,
            "route_type":rtype,"distance_km":rdata["distance_km"],"travel_time_min":adj_time,
            "route_source":rdata["source"],"rainfall_24h":rr,"humidity":rh,"risk_level":risk,
            "rf_prediction":ROUTE_LABELS[pred],"rf_prob_fastest":round(float(proba[0]),4),
            "rf_prob_safest":round(float(proba[1]),4),"rf_prob_balanced":round(float(proba[2]),4),
            "weather_source":weather.get("source","")})

    # 6. Alternative shelters
    alternatives=[]
    for alt_sh,_ in nearest[1:3]:
        alt_lat,alt_lon=float(alt_sh['Latitude']),float(alt_sh['Longitude'])
        alt_r=get_fastest_route(lat,lon,alt_lat,alt_lon)
        adj_t=round(alt_r["travel_time_min"]*delay,1)
        X=np.array([[alt_r["distance_km"],adj_t,rr,rh,float(risk)]])
        p=int(model.predict(X)[0]); pr=model.predict_proba(X)[0]
        alternatives.append({"shelter_name":alt_sh["Nama"],"shelter_lat":alt_lat,"shelter_lon":alt_lon,
            "distance_km":alt_r["distance_km"],"travel_time_min":adj_t,
            "polyline":alt_r["polyline"],"route_source":alt_r["source"],
            "rf_prediction":ROUTE_LABELS[p]})

    return jsonify({
        "village":{"nama":nama,"lat":lat,"lon":lon,"kecamatan":kec["nama"]},
        "shelter":{"nama":shelter["Nama"],"lat":slat,"lon":slon},
        "weather":weather,"risk_level":risk,"risk_label":RISK_LABELS[risk],
        "nearby_flood_points":floods[:5],"routes":routes_out,"alternatives":alternatives,
        "routing_provider":"ORS" if ORS_KEY else "OSRM",
        "timestamp":datetime.now(timezone.utc).isoformat()})


# ── Other endpoints ──
@app.route("/api/weather")
def api_weather():
    adm4=request.args.get("adm4","33.01.22.2001")
    w=fetch_bmkg_weather(adm4)
    if w is None: w=simulated_weather(); w["adm4"]=adm4
    r=assign_risk(w["rainfall_24h"],w["humidity_now"])
    w.update({"risk_level":r,"risk_label":RISK_LABELS[r],"delay_factor":DELAY_FACTORS[r]})
    return jsonify(w)

@app.route("/api/flood-points")
def api_fp():
    return jsonify(df_flood.to_dict(orient="records")) if not df_flood.empty else (jsonify({"error":"No data"}),503)

@app.route("/api/shelters")
def api_sh():
    return jsonify(df_shelters.to_dict(orient="records")) if not df_shelters.empty else (jsonify({"error":"No data"}),503)

@app.route("/api/prediction-log")
def api_log():
    if not os.path.exists(LOG_PATH): return jsonify({"rows":0})
    df=pd.read_csv(LOG_PATH)
    if request.args.get("format")=="csv":
        from flask import Response
        return Response(df.to_csv(index=False),mimetype="text/csv",
                        headers={"Content-Disposition":"attachment;filename=prediction_log.csv"})
    return jsonify({"rows":len(df),"latest":df.tail(5).to_dict(orient="records"),
                    "retrain_ready":len(df)>=1000})

@app.route("/api/health")
def api_health():
    log_n=0
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f: log_n=max(0,sum(1 for _ in f)-1)
    return jsonify({"status":"ok","model":model is not None,
        "flood_pts":len(df_flood),"shelters":len(df_shelters),
        "ors_key":bool(ORS_KEY),"osrm":"available",
        "routing_chain":"ORS → OSRM → Haversine",
        "prediction_log":log_n})

@app.route("/")
def index(): return send_from_directory("static","index.html")

if __name__=="__main__":
    app.run(host="0.0.0.0",port=int(os.environ.get("PORT",5000)),debug=True)
