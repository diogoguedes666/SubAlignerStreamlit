import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import io
import os

# Set the page configuration with the favicon parameter
st.set_page_config(page_title="SubSync", page_icon=":speaker:")

# Apply custom light mode CSS
st.markdown(
    """
    <style>
    :root {
        --background-color: #ffffff !important;
        --text-color: #000000 !important;
        --primary-color: #f63366 !important; /* same color as the title */
        --selection-background-color: #1C3D73 !important; /* blue background color for selection */
        --selection-text-color: white !important; /* white text color for selection */
    }
    html, body, .stApp, .reportview-container, .main, .block-container, .css-18e3th9, .css-1r6slb0, .css-1cpxqw2, .css-1d391kg, .css-1d391kg h1, .css-1d391kg h3, .css-1cpxqw2, .css-1cpxqw2 h1, .css-1cpxqw2 h3, .stFileUpload {
        background-color: var(--background-color) !important;
        color: var(--text-color) !important;
    }
    .stApp header, .stApp [data-testid="stSidebar"], .stApp [data-testid="stToolbar"] {
        background-color: var(--background-color) !important;
        color: var(--text-color) !important;
    }
    .stButton button {
        background-color: var(--primary-color) !important;
        color: white !important;
    }
    .stFileUploadDropzone {
        background-color: var(--background-color) !important;
    }
    .stFileUploadLabel, .stFileUploadLabel span {
        background-color: var(--background-color) !important;
        color: var(--text-color) !important;
    }
    .selection-style {
        background-color: var(--selection-background-color) !important;
        color: var(--selection-text-color) !important;
        padding: 2px 4px; /* add some padding for better visual */
    }
    .center {
        display: flex;
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

image_url = "https://i.postimg.cc/9FTzQjqf/monsterlogo.png"
st.markdown(
    f'<div class="center"><img src="{image_url}" style="width: 25%; margin-left: -20px;" /></div>',
    unsafe_allow_html=True
)

st.logo("https://i.postimg.cc/9FTzQjqf/monsterlogo.png")

def calculate_average_amplitude(sum_amplitude_before, sum_amplitude_after):
    # Calculate the average amplitude from both traces
    avg_amplitude_before = np.mean(sum_amplitude_before)
    avg_amplitude_after = np.mean(sum_amplitude_after)
    return (avg_amplitude_before + avg_amplitude_after) / 2

# Preprocess function to handle invalid values
def preprocess_transfer_function(df):
    df['Frequency'] = pd.to_numeric(df['Frequency'], errors='coerce')
    df['Magnitude (dB)'] = pd.to_numeric(df['Magnitude (dB)'], errors='coerce')
    df['Phase (degrees)'] = pd.to_numeric(df['Phase (degrees)'], errors='coerce')
    df['Coherence'] = pd.to_numeric(df['Coherence'], errors='coerce')
    df.dropna(subset=['Frequency', 'Magnitude (dB)', 'Phase (degrees)', 'Coherence'], inplace=True)
    return df

def unwrap_phase(phase):
    return np.unwrap(np.radians(phase))



def find_initial_crossover(df1, df2, freq_range=(20, 20000)):
    common_frequencies = np.intersect1d(df1['Frequency'].values, df2['Frequency'].values)
    df1_common = df1[df1['Frequency'].isin(common_frequencies)]
    df2_common = df2[df2['Frequency'].isin(common_frequencies)]

    if df1_common.empty or df2_common.empty:
        return None

    df1_common = df1_common.set_index('Frequency')
    df2_common = df2_common.set_index('Frequency')
    df1_common = df1_common.loc[(df1_common.index >= freq_range[0]) & (df1_common.index <= freq_range[1])]
    df2_common = df2_common.loc[(df2_common.index >= freq_range[0]) & (df2_common.index <= freq_range[1])]

    gain_diff = np.abs(df1_common['Magnitude (dB)'] - df2_common['Magnitude (dB)']).rolling(window=5, min_periods=1).mean()
    return gain_diff.idxmin()

def format_hovertext(frequency, amplitude, trace_name):
    if frequency >= 1000:
        freq_label = f"{frequency/1000:.1f} kHz"
    else:
        freq_label = f"{frequency:.1f} Hz"
    return f"{trace_name}<br>Frequency: {freq_label}<br>Amplitude: {amplitude:.1f} dB"

def calculate_best_delay(df1, df2, crossover_freq_hz, amp_diff_dB):
    amp_diff_linear = 10 ** (amp_diff_dB / 20)
    crossover_amp1 = np.interp(crossover_freq_hz, df1['Frequency'], df1['Magnitude (dB)'])
    crossover_amp2 = np.interp(crossover_freq_hz, df2['Frequency'], df2['Magnitude (dB)'])

    lower_freq = crossover_freq_hz
    upper_freq = crossover_freq_hz

    while lower_freq >= df1['Frequency'].min() and abs(crossover_amp1 - np.interp(lower_freq, df1['Frequency'], df1['Magnitude (dB)'])) <= amp_diff_dB:
        lower_freq -= 1
    while upper_freq <= df1['Frequency'].max() and abs(crossover_amp1 - np.interp(upper_freq, df1['Frequency'], df1['Magnitude (dB)'])) <= amp_diff_dB:
        upper_freq += 1

    common_frequencies = np.intersect1d(df1['Frequency'].values, df2['Frequency'].values)
    df1_range = df1[df1['Frequency'].isin(common_frequencies)]
    df2_range = df2[df2['Frequency'].isin(common_frequencies)]

    if df1_range.empty or df2_range.empty:
        return None, None

    phase1_unwrapped = unwrap_phase(np.interp(common_frequencies, df1_range['Frequency'], df1_range['Phase (degrees)']))
    phase2_unwrapped = unwrap_phase(np.interp(common_frequencies, df2_range['Frequency'], df2_range['Phase (degrees)']))

    crossover_phase_diff = phase1_unwrapped - phase2_unwrapped
    initial_phase_diff = (crossover_phase_diff + np.pi) % (2 * np.pi) - np.pi

    best_delay = 0
    best_polarity = 'normal'
    min_error = float('inf')

    for polarity in ['normal', 'invert']:
        if polarity == 'invert':
            phase2_unwrapped = (phase2_unwrapped + np.pi) % (2 * np.pi)

        delays = np.linspace(-20, 20, 400)
        for delay in delays:
            adjusted_phase2 = phase2_unwrapped - 2 * np.pi * common_frequencies * delay / 1000
            phase_diff = (phase1_unwrapped - adjusted_phase2 + np.pi) % (2 * np.pi) - np.pi
            total_error = np.sum(phase_diff ** 2)

            if total_error < min_error:
                min_error = total_error
                best_delay = delay
                best_polarity = polarity

    return best_delay, best_polarity

def plot_transfer_function(df_list, coherence_tolerance, amp_diff_dB, crossover_freqs, x_range=None, y_range=None):
    fig = make_subplots(rows=1, cols=1)
    
    frequency = df_list[0]['Frequency'].values
    sum_complex_before = np.zeros(len(frequency), dtype=np.complex128)
    sum_complex_after = np.zeros(len(frequency), dtype=np.complex128)

    # Calculate the amplitudes of the input files
    amplitudes_before = []
    amplitudes_after = []

    for idx, df in enumerate(df_list):
        df_filtered = df[df['Coherence'] >= coherence_tolerance]
        amplitude_linear = 10 ** (df_filtered['Magnitude (dB)'] / 20)
        phase_radians = unwrap_phase(df_filtered['Phase (degrees)'])
        complex_response = amplitude_linear * np.exp(1j * phase_radians)

        # Compute average amplitude from the input files
        if idx == 0:
            amplitudes_before.extend(amplitude_linear)
        else:
            amplitudes_after.extend(amplitude_linear)

        sum_complex_before += np.interp(frequency, df_filtered['Frequency'], complex_response, left=0, right=0)

        trace_name = f'Trace {idx + 1}'
        hovertext = [format_hovertext(f, a, trace_name) for f, a in zip(df_filtered['Frequency'], 20 * np.log10(amplitude_linear))]
        fig.add_trace(go.Scatter(x=df_filtered['Frequency'], y=20 * np.log10(amplitude_linear), mode='lines', name=trace_name, text=hovertext, hoverinfo='text'), row=1, col=1)

        for crossover_freq in crossover_freqs:
            crossover_amp = np.interp(crossover_freq, df_filtered['Frequency'], 20 * np.log10(amplitude_linear))
            fig.add_trace(go.Scatter(x=[crossover_freq], y=[crossover_amp], mode='markers', marker=dict(color='red', size=10), showlegend=False, hoverinfo='skip'))

    sum_amplitude_before = 20 * np.log10(np.abs(sum_complex_before))
    hovertext = [format_hovertext(f, a, 'Sum Before Alignment') for f, a in zip(frequency, sum_amplitude_before)]
    fig.add_trace(go.Scatter(x=frequency, y=sum_amplitude_before, mode='lines', name='Sum Before Alignment', line=dict(color='gray', dash='dash'), text=hovertext, hoverinfo='text'), row=1, col=1)

    for idx, df in enumerate(df_list):
        df_filtered = df[df['Coherence'] >= coherence_tolerance]
        amplitude_linear = 10 ** (df_filtered['Magnitude (dB)'] / 20)
        phase_radians = unwrap_phase(df_filtered['Phase (degrees)'])
        complex_response = amplitude_linear * np.exp(1j * phase_radians)

        if idx > 0:
            delay_ms, polarity = calculate_best_delay(df_list[0], df, crossover_freqs[idx-1], amp_diff_dB)
            if polarity == 'invert':
                df_filtered['Phase (degrees)'] = (df_filtered['Phase (degrees)'] + 180) % 360
            df_filtered['Phase (degrees)'] = np.degrees(unwrap_phase(df_filtered['Phase (degrees)']) - 2 * np.pi * df_filtered['Frequency'] * delay_ms / 1000)

        amplitude_linear = 10 ** (df_filtered['Magnitude (dB)'] / 20)
        phase_radians = unwrap_phase(df_filtered['Phase (degrees)'])
        complex_response = amplitude_linear * np.exp(1j * phase_radians)

        sum_complex_after += np.interp(frequency, df_filtered['Frequency'], complex_response, left=0, right=0)

    sum_amplitude_after = 20 * np.log10(np.abs(sum_complex_after))
    hovertext = [format_hovertext(f, a, 'Sum After Alignment') for f, a in zip(frequency, sum_amplitude_after)]
    fig.add_trace(go.Scatter(x=frequency, y=sum_amplitude_after, mode='lines', name='Sum After Alignment', line=dict(color='red', dash='dash'), text=hovertext, hoverinfo='text'), row=1, col=1)

    # Calculate average amplitude from the input files
    avg_amplitude_before = np.mean(20 * np.log10(np.abs(np.array(amplitudes_before))))
    avg_amplitude_after = np.mean(20 * np.log10(np.abs(np.array(amplitudes_after))))
    avg_amplitude = (avg_amplitude_before + avg_amplitude_after) / 2

    # Define vertical range for the plot
    y_min = -20
    y_max = 20

    # Add shaded region for the isolation zone
    isolation_zone_start = crossover_freq - amp_diff_dB
    isolation_zone_end = crossover_freq + amp_diff_dB
    fig.add_shape(
        go.layout.Shape(
            type='rect',
            x0=isolation_zone_start,
            x1=isolation_zone_end,
            y0=-30,  
            y1=30,  
            fillcolor='rgba(0, 100, 255, 0.2)',
            line=dict(color='rgba(0, 100, 255, 0.5)', width=2),
            name='Isolation Zone'
        )
    )

    # Add a dummy trace for the legend entry
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color='rgba(0, 100, 255, 0.2)'),
        showlegend=True,
        name='Isolation Zone',
        legendgroup='Isolation Zone'
    ))

    fig.update_layout(xaxis_type="log", xaxis=dict(title='Frequency (Hz)'), yaxis=dict(title='Magnitude (dB)'), title='Magnitude Response')
    fig.update_xaxes(range=[np.log10(20), np.log10(20000)])
    if x_range:
        fig.update_xaxes(range=x_range)
    fig.update_yaxes(range=[y_min, y_max])

    return fig





def generate_aligned_sum_file(frequency, sum_amplitude_after, sum_phase_after, df_list, coherence_tolerance, inputfilename1, inputfilename2):
    output = io.StringIO()
    trace_name = inputfilename1 + '+' + inputfilename2
    output.write('\t' + trace_name + '\n')
    output.write('\t'.join(['Frequency (Hz)', 'Magnitude (dB)', 'Phase (degrees)', 'Coherence']) + '\n')

    coherences = np.zeros_like(frequency)
    coherence_counts = np.zeros_like(frequency)

    for df in df_list:
        df_filtered = df[df['Coherence'] >= coherence_tolerance]
        interp_coherence = np.interp(frequency, df_filtered['Frequency'], df_filtered['Coherence'], left=0, right=0)
        coherences += interp_coherence
        coherence_counts += (interp_coherence > 0).astype(int)

    average_coherences = np.divide(coherences, coherence_counts, out=np.zeros_like(coherences), where=coherence_counts != 0)

    decimal_places = len(str(frequency[0]).split('.')[-1])

    for i, (f, mag, phase, coherence) in enumerate(zip(frequency, sum_amplitude_after, sum_phase_after, average_coherences)):
        if np.isinf(mag):
            left_idx = max(i - 1, 0)
            right_idx = min(i + 1, len(frequency) - 1)
            if np.isinf(sum_amplitude_after[left_idx]) and np.isinf(sum_amplitude_after[right_idx]):
                mag = -120
            else:
                valid_neighbors = [val for val in [sum_amplitude_after[left_idx], sum_amplitude_after[right_idx]] if not np.isinf(val)]
                avg_mag = np.mean(valid_neighbors)
                mag = avg_mag
        f_formatted = "{:.{}f}".format(f, decimal_places)
        output.write(f'{f_formatted}\t{mag:.2f}\t{phase:.2f}\t{coherence:.2f}\n')

    return output.getvalue()

def main():
    st.markdown(
        f"""
        <style>
        .reportview-container {{
            background: url('data:image/gif;base64,R0lGODlhAQABAIAAAAUEBAAAACwAAAAAAQABAAACAkQBADs=') center center no-repeat;
            background-size: cover;
        }}
        h1, h4 {{
            text-align: center;
            color: #f63366;
            line-height: 0.4; /* Adjust the line height as needed */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h1>SubSync</h1>", unsafe_allow_html=True)
    st.markdown("<h4>Subwoofer Delay Calculator</h4>", unsafe_allow_html=True)

    st.markdown("Follow [monsterDSP](https://instagram.com/monsterdsp)")

    show_help = st.button("HELP", key="help_button", help="Show information")

    if show_help:
        st.markdown("""
            <div class="help-dialog">
                <h2 class="selection-style">How to use <span style="color: var(--primary-color);">SubSync</span> by monsterDSP:</h2>
                <ol>
                    <li>Measure subwoofer and PA in Smaart, and save two different files</li>
                    <li>Save measurements as ASCII files</li>
                    <li>Load subwoofer trace and PA trace in <span style="color: var(--primary-color);">SubSync</span></li>
                    <li>Adjust crossover, coherence threshold and dB isolation zone as needed to reach desired sum in crossover region</li>
                    <li>Get delay and polarity results</li>
                    <li>Apply calculated delay and polarity to subwoofer or PA (in case it’s negative)</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)



    categories = ['SUB', 'PA']
    file_dict = {category: None for category in categories}
    df_list = []

    for idx, category in enumerate(categories):
        file_dict[category] = st.file_uploader(f"Upload {category} Transfer Function ASCII File", type=['csv', 'txt'])

        if file_dict[category] is not None:
            try:
                df = pd.read_csv(file_dict[category], delimiter='\t')
                df.columns = ['Frequency', 'Magnitude (dB)', 'Phase (degrees)', 'Coherence']
                df = preprocess_transfer_function(df)

                if not df.empty:
                    df_list.append(df)

            except Exception as e:
                st.error(f"Error occurred while reading file: {e}")

    if len(df_list) == 2:
        initial_crossover_freq = find_initial_crossover(df_list[0], df_list[1])
        if initial_crossover_freq is None:
            st.error("Could not determine initial crossover frequency.")
            return

        if "crossover_freq" not in st.session_state:
            st.session_state.crossover_freq = initial_crossover_freq
        if "coherence_tolerance" not in st.session_state:
            st.session_state.coherence_tolerance = 0.65
        if "amp_diff_dB" not in st.session_state:
            st.session_state.amp_diff_dB = 24
            

        crossover_freq = st.session_state.crossover_freq
        coherence_tolerance = st.session_state.coherence_tolerance
        amp_diff_dB = st.session_state.amp_diff_dB

        st.markdown(f"**Detected Crossover Frequency**: {initial_crossover_freq:.1f} Hz")

        # Implement nudge buttons for the crossover frequency
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col1:
            if st.button("-1 Hz"):
                st.session_state.crossover_freq = max(20, st.session_state.crossover_freq - 1)
        with col3:
            if st.button("+1 Hz"):
                st.session_state.crossover_freq = min(500, st.session_state.crossover_freq + 1)
        with col2:
            crossover_freq = st.slider("Set Crossover Frequency (Hz)", min_value=20, max_value=500, value=int(st.session_state.crossover_freq), step=1)
            st.session_state.crossover_freq = crossover_freq


        # Coherence tolerance slider
        coherence_tolerance = st.slider("Coherence Tolerance", min_value=0.25, max_value=1.0, value=st.session_state.coherence_tolerance, step=0.01)
        st.session_state.coherence_tolerance = coherence_tolerance

        # Isolation zone range slider
        amp_diff_dB = st.slider("Isolation Zone Range (dB difference between curves)", min_value=1, max_value=30, value=int(st.session_state.amp_diff_dB), step=1)
        st.session_state.amp_diff_dB = amp_diff_dB

        crossover_freqs = [crossover_freq]

        st.header('Alignment Information')

        total_delay_ms = 0
        polarity_adjustments = []

        for df in df_list[1:]:
            delay_ms, polarity = calculate_best_delay(df_list[0], df, crossover_freq, amp_diff_dB)
            if delay_ms is not None:
                total_delay_ms += delay_ms
                polarity_adjustments.append(polarity)

        if abs(total_delay_ms) < 0.1:
            displayed_delay = 0.0
        else:
            displayed_delay = round(abs(total_delay_ms), 1)

        st.markdown(f"**Total Subwoofer Delay for Alignment**: {displayed_delay:.1f} ms")
        st.markdown(f"**Subwoofer Polarity Adjustments**: {' '.join(polarity_adjustments)}")

        fig = plot_transfer_function(df_list, coherence_tolerance, amp_diff_dB, crossover_freqs)
        st.plotly_chart(fig, use_container_width=True)

        sum_complex_after = np.zeros(len(df_list[0]), dtype=np.complex128)
        frequency = df_list[0]['Frequency'].values

        for idx, df in enumerate(df_list):
            df_filtered = df[df['Coherence'] >= coherence_tolerance]
            amplitude_linear = 10 ** (df_filtered['Magnitude (dB)'] / 20)
            phase_radians = unwrap_phase(df_filtered['Phase (degrees)'])
            complex_response = amplitude_linear * np.exp(1j * phase_radians)

            sum_complex_after += np.interp(frequency, df_filtered['Frequency'], complex_response, left=0, right=0)

        sum_amplitude_after = 20 * np.log10(np.abs(sum_complex_after))
        sum_phase_after = np.degrees(np.angle(sum_complex_after))

        inputfilename1 = os.path.splitext(file_dict['SUB'].name)[0]
        inputfilename2 = os.path.splitext(file_dict['PA'].name)[0]
        aligned_sum_data = generate_aligned_sum_file(frequency, sum_amplitude_after, sum_phase_after, df_list, coherence_tolerance, inputfilename1, inputfilename2)

        output_filename = f"{inputfilename1}_{inputfilename2}_aligned_sum.txt"

        # Hide the download button
        # st.download_button(
        #     label="Download Aligned Sum ASCII File",
        #     data=aligned_sum_data,
        #     file_name=output_filename,
        #     mime='text/plain'
        # )

if __name__ == "__main__":
    main()
