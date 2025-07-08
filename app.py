import streamlit as st import pandas as pd import numpy as np from math import radians, sin, cos, sqrt, asin, ceil import pydeck as pdk

--- Utility Functions ---

@st.cache_data def haversine_array(lats, lons): """ Vectorized detection of distances (m) between consecutive lat/lon points. """ lat = np.radians(lats) lon = np.radians(lons) dlat = lat[1:] - lat[:-1] dlon = lon[1:] - lon[:-1] a = np.sin(dlat/2)*2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlon/2)*2 c = 2 * np.arcsin(np.sqrt(a)) return np.concatenate(([0.0], c * 6371000))

def interpolate_path(lats, lons, dists, target): """ Linear interpolation of lat/lon at a given cumulative distance. """ for j in range(len(dists)-1): if dists[j] <= target <= dists[j+1]: frac = (target - dists[j]) / (dists[j+1] - dists[j]) lat = lats[j] + frac * (lats[j+1] - lats[j]) lon = lons[j] + frac * (lons[j+1] - lons[j]) return lat, lon return lats[-1], lons[-1]

--- App UI ---

st.set_page_config(page_title="Slot-based Signal Analysis", layout="wide") st.title("📊 Slot-based Signal Summary & Map")

Logs

if 'logs' not in st.session_state: st.session_state['logs'] = [] def log(msg, level='INFO'): st.session_state['logs'].append((level, msg)) st.sidebar.header("Logs") st.sidebar.code("\n".join(f"[{l}] {m}" for l,m in st.session_state['logs'])) if st.sidebar.button("🗑 Clear Logs"): st.session_state['logs']=[]

Inputs

uploaded = st.file_uploader("Upload CSV with TIME, LATITUDE, LONGITUDE, RSRP, BTS ID, OPERATOR", type='csv') slot_len = st.number_input("Slot length (m)", 0.1, 1000.0, 5.0) thresh = st.number_input("Signal threshold (dBm)", -200.0, 0.0, -124.0)

if uploaded: try: # 1. Read and sort df0 = pd.read_csv(uploaded) log(f"Loaded {len(df0)} rows", 'INFO') if 'TIME' in df0.columns: def parse_time(t): parts = str(t).split(':') if len(parts)==2: return int(parts[0])*60 + float(parts[1]) if len(parts)==3: return int(parts[0])*3600 + int(parts[1])*60 + float(parts[2]) return float(t) df0['time_sec'] = df0['TIME'].apply(parse_time) df0 = df0.sort_values('time_sec').reset_index(drop=True) log("Sorted by TIME", 'INFO') else: log("No TIME column; using file order", 'WARNING')

# 2. Filter
    ops = df0['OPERATOR'].dropna().unique().tolist()
    op_choice = st.selectbox("Filter by OPERATOR", ['--all--'] + ops)
    if op_choice != '--all--':
        before = len(df0)
        df0 = df0[df0['OPERATOR']==op_choice]
        log(f"Operator filter: {before}->{len(df0)} rows", 'INFO')
    drop_ids = [b for b in df0['BTS ID'].unique()
                if (df0[df0['BTS ID']==b]['RSRP']<thresh).all()]
    if drop_ids:
        before2 = len(df0)
        df0 = df0[~df0['BTS ID'].isin(drop_ids)]
        log(f"Dropped {len(drop_ids)} BTS IDs ({before2}->{len(df0)} rows)", 'INFO')

    # 3. Prepare samples
    df = df0.rename(columns={'LATITUDE':'Lat','LONGITUDE':'Long','RSRP':'Signal Strength','BTS ID':'BTS'})
    df = df[['Lat','Long','Signal Strength','BTS']].dropna().reset_index(drop=True)
    st.success(f"Prepared {len(df)} samples")

    # 4. Compute cumulative distances vectorized
    lats = df['Lat'].to_numpy(); lons = df['Long'].to_numpy()
    deltas = haversine_array(lats, lons)
    dists = np.cumsum(deltas)
    total = dists[-1]
    st.info(f"Total path length: {total:.1f} m ({total/1000:.2f} km)")
    if total <= 0:
        st.error("Path length zero; cannot slot.")
        st.stop()

    # 5. Determine number of slots and boundaries
    n_slots = int(ceil(total / slot_len))
    log(f"Path {total:.1f}m → {n_slots} slots @ {slot_len}m", 'INFO')
    boundaries = np.linspace(0, total, n_slots+1)

    # 6. Assign samples to slots vectorized
    slot_ids = np.minimum((dists // slot_len).astype(int)+1, n_slots)
    df['Slot'] = slot_ids

    # 7. Aggregate counts via groupby
    grp = df.groupby('Slot')['Signal Strength']
    num_samples = grp.count().reindex(range(1, n_slots+1), fill_value=0).to_numpy()
    num_good    = grp.apply(lambda x: (x>=thresh).sum()).reindex(range(1, n_slots+1), fill_value=0).to_numpy()
    num_bad     = num_samples - num_good

    # 8. Compute boundaries lat/lon by interpolation
    start_pts = np.array([interpolate_path(lats, lons, dists, bd) for bd in boundaries[:-1]])
    end_pts   = np.array([interpolate_path(lats, lons, dists, bd) for bd in boundaries[1: ]])

    # 9. Build summary DataFrame
    summary = pd.DataFrame({
        'Slot_ID': np.arange(1, n_slots+1),
        'start_lat': start_pts[:,0], 'start_lon': start_pts[:,1],
        'end_lat':   end_pts[:,0],   'end_lon':   end_pts[:,1],
        'num_samples': num_samples,
        'num_good':    num_good,
        'num_bad':     num_bad
    })
    # preliminary status
    summary['Status'] = np.where(
        summary['num_samples']==0, 'ZERO',
        np.where(summary['num_good']>0, 'GOOD', 'BAD')
    )

            # --- Enforce first and last slot boundaries ---
    # First slot start = actual first sample coordinates
    summary.at[0, 'start_lat'] = lats[0]
    summary.at[0, 'start_lon'] = lons[0]
    # Last slot end = actual last sample coordinates
    summary.at[n_slots-1, 'end_lat'] = lats[-1]
    summary.at[n_slots-1, 'end_lon'] = lons[-1]

    # 10. Collapse ZERO runs → SKIP/BAD Collapse ZERO runs → SKIP/BAD
    status = summary['Status'].to_numpy()
    i = 0
    while i < n_slots:
        if status[i]=='ZERO':
            j = i
            while j+1<n_slots and status[j+1]=='ZERO': j+=1
            if j==i:
                status[i] = 'SKIP'
            else:
                status[i:j+1] = 'BAD'
            i = j+1
        else:
            i += 1
    summary['Status'] = status

    # 11. Coverage Metrics (single display)
    skip = (summary['Status']=='SKIP').sum()
    counted = n_slots - skip
    good_slots = (summary['Status']=='GOOD').sum()
    coverage = (good_slots / counted * 100) if counted else 0
    st.subheader("Coverage")
    st.write({
        'Total slots': n_slots,
        'Skip slots':  skip,
        'Counted':     counted,
        'Good slots':  good_slots,
        'Coverage (%)': f"{coverage:.2f}%"
    })

    # 12. Show summary table Show summary CSV
    st.subheader("Slot Summary Table")
    st.dataframe(summary)

            # 13. Map visualization with PathLayer for cleaner segments
    summary['color'] = summary['Status'].map({
        'GOOD': [0,255,0],
        'BAD': [255,0,0],
        'SKIP': [0,0,255]
    })
    # build path coordinates for each slot
    summary['path'] = summary.apply(
        lambda r: [[r['start_lon'], r['start_lat']], [r['end_lon'], r['end_lat']]], axis=1
    )
    # create a PathLayer
    layer = pdk.Layer(
        "PathLayer",
        data=summary,
        get_path="path",
        get_color="color",
        get_width=10,
        width_min_pixels=4,
        pickable=False
    )
    # center map
    center_lat = summary['start_lat'].mean()
    center_lon = summary['start_lon'].mean()
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=14,
            pitch=0
        ),
        map_style='mapbox://styles/mapbox/light-v9'
    )
    st.subheader("Slot Map")
    st.pydeck_chart(deck)

    # 14. Download CSV
    csv = summary.to_csv(index=False).encode('utf-8')
    st.download_button("Download Slot Summary CSV", csv, "slot_summary.csv", "text/csv")

except Exception as e:
    log(str(e), 'ERROR')