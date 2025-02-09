import os.path
import random
import streamlit as st
import pandas


m3u_filepaths_file = 'playlists/streamlit.m3u8'
ESSENTIA_ANALYSIS_PATH = 'results.csv'


def load_essentia_analysis():
    return pandas.read_csv(ESSENTIA_ANALYSIS_PATH, index_col=0)

def filter_analysis(df, style_select, bpm_range,  key_select, scale_select, arousal_select, valence_select, instrumental_select=None, danceable_select=None):
    filtered_df = df[df['style'].isin(style_select)]
    filtered_df = filtered_df[(filtered_df['bpm'] >= bpm_range[0]) & (filtered_df['bpm'] <= bpm_range[1])]
    filtered_df = filtered_df[filtered_df['key_temperley'].isin(key_select)]
    filtered_df = filtered_df[filtered_df['scale_temperley'].isin(scale_select)]
    filtered_df = filtered_df[(filtered_df['arousal'] >= arousal_select[0]) & (filtered_df['arousal'] <= arousal_select[1])]
    filtered_df = filtered_df[(filtered_df['valence'] >= valence_select[0]) & (filtered_df['valence'] <= valence_select[1])]
    if instrumental_select is not None:
        filtered_df = filtered_df[filtered_df['instrumental'].fillna(False) == instrumental_select]
    if danceable_select is not None:
        filtered_df = filtered_df[filtered_df['danceability'].fillna(False) == danceable_select]
    return filtered_df

st.set_page_config(
    page_title="Playlist Generator",
    page_icon="🎵",
    layout
    ="wide"
)


st.write('# Playlist Generator')
audio_analysis = load_essentia_analysis()

style_counts = audio_analysis['style'].value_counts().reset_index()
style_possible = audio_analysis['style'].unique()
style_counts.columns = ['Style', 'Count']

keys_possible = audio_analysis['key_temperley'].unique()
scales_possible = audio_analysis['scale_temperley'].unique()


with st.sidebar:
    st.write('# Filters')

    style_select = st.multiselect('Select by style activations:', style_possible, [])
    selected_styles = style_select if style_select else style_possible

    bpm_range = st.slider('Select by BPM:', min_value=0, max_value=200, value=(0, 200))

    key_select = st.multiselect('Select by key:', keys_possible, [])
    selected_keys = key_select if key_select else keys_possible

    scale_select = st.multiselect('Select by scale:', scales_possible, [])
    selected_scales = scale_select if scale_select else scales_possible

    instrumental_select = st.radio(
        'Select track type:',
        options=[None, 'instrumental', 'voice'],
        format_func=lambda x: 'All' if x is None else ('Instrumental' if x == 'instrumental' else 'Vocal')
    )

    danceable_select = st.radio(
        'Select danceable tracks:',
        options=[None, 'danceable', 'not_danceable'],
        format_func=lambda x: 'All' if x is None else ('Danceable' if x == 'danceable' else 'Not danceable')
    )

    arousal_select = st.slider('Select by arousal:', min_value=0.0, max_value=10.0, value=(0.0, 10.0))
    valence_select = st.slider('Select by valence:', min_value=0.0, max_value=10.0, value=(0.0, 10.0))


display_columns = ['style', 'bpm', 'key_temperley', 'scale_temperley', 'loudness', 'instrumental', 'danceability', 'arousal', 'valence']

filtered_df = filter_analysis(
    audio_analysis, 
    selected_styles, 
    bpm_range, 
    selected_keys, 
    selected_scales, 
    arousal_select,
    valence_select,
    instrumental_select, 
    danceable_select
    )

st.dataframe(filtered_df[display_columns], use_container_width=True)

st.write('### 🔀 Generate playlist')
col1, col2, col3, col4 = st.columns([1,1,1, 1])
col1.write("Maximum number of tracks (0 for all):")
max_tracks = col2.number_input('max tracks', min_value=0, value=0, label_visibility='collapsed')
shuffle = col3.checkbox('Random shuffle')
run = col4.button('RUN')

if run:
    st.write('## 🔊 Results')
    mp3s = list(filtered_df.path)

    if max_tracks:
        mp3s = mp3s[:max_tracks]
        st.write('Using top', len(mp3s), 'tracks from the results.')

    if shuffle:
        random.shuffle(mp3s)
        st.write('Applied random shuffle.')

    with open(m3u_filepaths_file, 'w') as f:
        f.write('\n'.join(mp3s))
        st.write(f'Stored M3U playlist (local filepaths) to `{m3u_filepaths_file}`.')

    st.write('Audio previews for the first 10 results:')
    for mp3 in mp3s[:10]:
        st.audio(mp3, format="audio/mp3", start_time=0)

