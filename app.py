import os.path
import random
import streamlit as st
import pandas
import numpy as np
import json


m3u_filepaths_file = 'playlists/'
ESSENTIA_ANALYSIS_PATH = 'results.csv'


def load_essentia_analysis():
    return pandas.read_csv(ESSENTIA_ANALYSIS_PATH, index_col=0)

def filter_analysis(df, style_select, bpm_range,  key_select, scale_select, arousal_select, valence_select, instrumental_select=None, danceable_select=None):
    filtered_df = df[df['style'].isin(style_select)]
    filtered_df = filtered_df[(filtered_df['bpm'] >= bpm_range[0]) & (filtered_df['bpm'] <= bpm_range[1])]
    filtered_df = filtered_df[filtered_df['key_krumhansl'].isin(key_select)]
    filtered_df = filtered_df[filtered_df['scale_krumhansl'].isin(scale_select)]
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

keys_possible = audio_analysis['key_krumhansl'].unique()
scales_possible = audio_analysis['scale_krumhansl'].unique()


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


filtered_df['Select'] = False

display_columns = ['Select', 'style', 'bpm', 'key_krumhansl', 'scale_krumhansl', 'loudness', 'instrumental', 'danceability', 'arousal', 'valence']

# Display interactive dataframe with selection
selected_rows = st.data_editor(
    filtered_df[display_columns],
    use_container_width=True,
    num_rows="dynamic",
    hide_index=True,  # Hide the empty left column
    column_config={
        "Select": st.column_config.CheckboxColumn(
            "Select",
            help="Select track to preview",
            default=False,
        )
    }
)

# Show preview for selected track
selected_indices = []

if "Select" in selected_rows:
    selected_indices = [i for i, x in enumerate(selected_rows["Select"]) if x]
    if selected_indices:
        selected_tracks = filtered_df.iloc[selected_indices]
        for i, selected_track in selected_tracks.iterrows():
            st.audio(selected_track['path'], format="audio/mp3", start_time=0)



st.write('### Save filter playlist')
col0, col1, col2, col3, col4, col5, col6 = st.columns([2,1,1,1,1,1,1])
col1.write("Maximum number of tracks (0 for all):")
max_tracks = col2.number_input('max tracks', min_value=0, value=0, label_visibility='collapsed')
shuffle = col3.checkbox('Random shuffle')
name = col4.text_input('Playlist name', 'My Playlist', label_visibility='collapsed')
save = col5.button('SAVE PLAYLIST')

if save:
    st.write('saving playlist...')

    mp3s = list(filtered_df.path)

    playlist_path = os.path.join(m3u_filepaths_file, f'{name}.m3u')

    if max_tracks:
        mp3s = mp3s[:max_tracks]
        st.write('Using top', len(mp3s), 'tracks from the results.')

    if shuffle:
        random.shuffle(mp3s)
        st.write('Applied random shuffle.')

    with open(playlist_path, 'w') as f:
        f.write('\n'.join(mp3s))
        st.write(f'Stored M3U playlist (local filepaths) to `{playlist_path}`')

st.write('### Search for similar tracks')

col0, col1, col2= st.columns([1, 1, 1])

col0.write('Select a track to search for similar tracks')
selected_track = col1.selectbox('Select track:', filtered_df.iloc[selected_indices]['path'], label_visibility='collapsed')
search = col2.button('SEARCH')


if search:
    track_embedding_discogs = json.loads(filtered_df[filtered_df['path'] == selected_track].iloc[0]['discogs_embeddings_mean'])
    track_embedding_discogs = np.array(track_embedding_discogs)

    track_embeddig_musicnn = json.loads(filtered_df[filtered_df['path'] == selected_track].iloc[0]['musicnn_embeddings_mean'])
    track_embeddig_musicnn = np.array(track_embeddig_musicnn)

    audio_analysis['discogs_similarity'] = audio_analysis['discogs_embeddings_mean'].apply(
        lambda x: np.dot(np.array(json.loads(x)), track_embedding_discogs)
    )
    audio_analysis['musicnn_similarity'] = audio_analysis['musicnn_embeddings_mean'].apply(
        lambda x: np.dot(np.array(json.loads(x)), track_embeddig_musicnn)
    )
    
    filtered_df_similar_discogs = audio_analysis.sort_values('discogs_similarity', ascending=False)
    filtered_df_similar_musicnn = audio_analysis.sort_values('musicnn_similarity', ascending=False)

    st.write('## 🔊 SIMILAR TRACKS')
    
    col1, col2 = st.columns([1, 1])
    col1.write('Discogs Similarity')
    col2.write('Musicnn Similarity')
    col1.dataframe(filtered_df_similar_discogs[['path', 'discogs_similarity']].head(10), use_container_width=True)
    col2.dataframe(filtered_df_similar_musicnn[['path', 'musicnn_similarity']].head(10), use_container_width=True)

    for i, selected_track in filtered_df_similar_discogs.head(10).iterrows():
        col1.audio(selected_track['path'], format="audio/mp3", start_time=0)

    for i, selected_track in filtered_df_similar_musicnn.head(10).iterrows():
        col2.audio(selected_track['path'], format="audio/mp3", start_time=0)

