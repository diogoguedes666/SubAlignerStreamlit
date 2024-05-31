import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import os

st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url('data:image/gif;base64,R0lGODlhAQABAIAAAAUEBAAAACwAAAAAAQABAAACAkQBADs=') center center no-repeat;
        background-size: cover;
    }}
    .center {{
        display: flex;
        justify-content: center;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

image_url = "https://i.postimg.cc/9FTzQjqf/monsterlogo.png"
st.markdown(
    f'<div class="center"><img src="{image_url}" style="width: 25%; margin-left: -20px;" /></div>',
    unsafe_allow_html=True
)


def preprocess_transfer_function(df):
    df['Frequency'] = pd.to_numeric(df['Frequency'], errors='coerce')
    df['Magnitude (dB)'] = pd.to_numeric(df['Magnitude (dB)'], errors='coerce')
    df['Phase (degrees)'] = pd.to_numeric(df['Phase (degrees)'], errors='coerce')
    df['Coherence'] = pd.to_numeric(df['Coherence'], errors='coerce')
    df.dropna(subset=['Frequency', 'Magnitude (dB)', 'Phase (degrees)', 'Coherence'], inplace=True)
    return df

def unwrap_phase(phase):
    return np.unwrap(np.radians(phase))

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

    df1_range = df1[(df1['Frequency'] >= lower_freq) & (df1['Frequency'] <= upper_freq)]
    df2_range = df2[(df2['Frequency'] >= lower_freq) & (df2['Frequency'] <= upper_freq)]

    if df1_range.empty or df2_range.empty:
        return None, None

    phase1_unwrapped = unwrap_phase(df1_range['Phase (degrees)'])
    phase2_unwrapped = unwrap_phase(df2_range['Phase (degrees)'])

    freqs = df1_range['Frequency'].values

    crossover_phase_diff = np.interp(crossover_freq_hz, df1_range['Frequency'], phase1_unwrapped) - \
                           np.interp(crossover_freq_hz, df2_range['Frequency'], phase2_unwrapped)
    initial_phase_diff = (crossover_phase_diff + np.pi) % (2 * np.pi) - np.pi

    best_delay = 0
    best_polarity = 'normal'
    min_error = float('inf')

    for polarity in ['normal', 'invert']:
        if polarity == 'invert':
            phase2_unwrapped = unwrap_phase((df2_range['Phase (degrees)'] + 180) % 360)
        
        delays = np.linspace(-20, 20, 400)
        for delay in delays:
            adjusted_phase2 = phase2_unwrapped - 2 * np.pi * freqs * delay / 1000
            phase_diff = (phase1_unwrapped - adjusted_phase2 + np.pi) % (2 * np.pi) - np.pi
            total_error = np.sum(phase_diff ** 2)
            
            if total_error < min_error:
                min_error = total_error
                best_delay = delay
                best_polarity = polarity

    return best_delay, best_polarity

def plot_transfer_function(df_list, crossover_freqs, coherence_tolerance, amp_diff_dB):
    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(10, 12))

    sum_complex_before = np.zeros(len(df_list[0]), dtype=np.complex128)
    sum_complex_after = np.zeros(len(df_list[0]), dtype=np.complex128)
    frequency = df_list[0]['Frequency'].values

    for idx, df in enumerate(df_list):
        df_filtered = df[df['Coherence'] >= coherence_tolerance]

        if df_filtered.empty:
            continue

        amplitude_linear_filtered = 10 ** (df_filtered['Magnitude (dB)'] / 20)
        phase_radians_filtered = unwrap_phase(df_filtered['Phase (degrees)'])
        complex_response_filtered = amplitude_linear_filtered * np.exp(1j * phase_radians_filtered)

        sum_complex_before += np.interp(frequency, df_filtered['Frequency'], complex_response_filtered, left=0, right=0)

        if idx > 0:
            delay_ms, polarity = calculate_best_delay(df_list[0], df, crossover_freqs[idx-1], amp_diff_dB)
            if polarity == 'invert':
                df_filtered['Phase (degrees)'] = (df_filtered['Phase (degrees)'] + 180) % 360
            df_filtered['Phase (degrees)'] = np.degrees(unwrap_phase(df_filtered['Phase (degrees)']) - 2 * np.pi * df_filtered['Frequency'] * delay_ms / 1000)

        amplitude_linear = 10 ** (df_filtered['Magnitude (dB)'] / 20)
        phase_radians = unwrap_phase(df_filtered['Phase (degrees)'])
        complex_response = amplitude_linear * np.exp(1j * phase_radians)

        sum_complex_after += np.interp(frequency, df_filtered['Frequency'], complex_response, left=0, right=0)
        ax_mag.plot(df_filtered['Frequency'], 20 * np.log10(amplitude_linear_filtered), label=f'Trace {idx + 1}')
        ax_phase.plot(df_filtered['Frequency'], df_filtered['Phase (degrees)'], label=f'Trace {idx + 1}')

    sum_amplitude_before = 20 * np.log10(np.abs(sum_complex_before))
    sum_amplitude_after = 20 * np.log10(np.abs(sum_complex_after))
    sum_phase_after = np.degrees(np.angle(sum_complex_after))

    ax_mag.plot(frequency, sum_amplitude_before, 'gray', linestyle='--', label='Sum Before Alignment')
    ax_mag.plot(frequency, sum_amplitude_after, 'r--', label='Sum After Alignment')
    ax_phase.plot(frequency, sum_phase_after, 'r--', label='Sum After Alignment')

    for crossover_freq in crossover_freqs:
        ax_mag.axvline(x=crossover_freq, color='r', linestyle='--', label=f'Crossover Frequency: {crossover_freq} Hz')
        ax_phase.axvline(x=crossover_freq, color='r', linestyle='--', label=f'Crossover Frequency: {crossover_freq} Hz')

    ax_mag.set_xscale('log')
    ax_mag.set_xlim(left=20)
    x_ticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    ax_mag.set_xticks(x_ticks)
    ax_mag.set_xticklabels([str(tick) for tick in x_ticks])
    ax_mag.set_xlabel('Frequency (Hz)')
    ax_mag.set_ylabel('Magnitude (dB)')
    ax_mag.set_title('Magnitude Response')
    ax_mag.grid(True)
    ax_mag.legend()

    ax_phase.set_xscale('log')
    ax_phase.set_xlim(left=20)
    ax_phase.set_xticks(x_ticks)
    ax_phase.set_xticklabels([str(tick) for tick in x_ticks])
    ax_phase.set_ylim([-180, 180])
    ax_phase.axhline(y=0, color='k', linestyle='--')
    ax_phase.set_xlabel('Frequency (Hz)')
    ax_phase.set_ylabel('Phase (degrees)')
    ax_phase.set_title('Phase Response')
    ax_phase.grid(True)
    ax_phase.legend()

    st.pyplot(fig)

def generate_aligned_sum_file(frequency, sum_amplitude_after, sum_phase_after, df_list, coherence_tolerance, inputfilename1, inputfilename2):
    output = io.StringIO()
    trace_name = inputfilename1 + '+' + inputfilename2
    output.write('\t' + trace_name + '\n')
    output.write('\t'.join(['Frequency (Hz)', 'Magnitude (dB)', 'Phase (degrees)', 'Coherence']) + '\n')

    # Calculate the average coherence for each frequency
    coherences = np.zeros_like(frequency)
    coherence_counts = np.zeros_like(frequency)

    for df in df_list:
        df_filtered = df[df['Coherence'] >= coherence_tolerance]
        interp_coherence = np.interp(frequency, df_filtered['Frequency'], df_filtered['Coherence'], left=0, right=0)
        coherences += interp_coherence
        coherence_counts += (interp_coherence > 0).astype(int)

    average_coherences = np.divide(coherences, coherence_counts, out=np.zeros_like(coherences), where=coherence_counts != 0)

    # Determine the number of decimal places in the frequency values
    decimal_places = len(str(frequency[0]).split('.')[-1])

    for i, (f, mag, phase, coherence) in enumerate(zip(frequency, sum_amplitude_after, sum_phase_after, average_coherences)):
        # Handle -inf values in magnitude column
        if np.isinf(mag):
            # Find the indices of neighboring frequencies
            left_idx = max(i - 1, 0)
            right_idx = min(i + 1, len(frequency) - 1)
            # Check if both neighboring values are -inf
            if np.isinf(sum_amplitude_after[left_idx]) and np.isinf(sum_amplitude_after[right_idx]):
                mag = -120  # Replace -inf with a default value (e.g., -120 dB)
            else:
                # Calculate the average magnitude of neighboring frequencies
                valid_neighbors = [val for val in [sum_amplitude_after[left_idx], sum_amplitude_after[right_idx]] if not np.isinf(val)]
                avg_mag = np.mean(valid_neighbors)
                mag = avg_mag
        # Format frequency with the same number of decimal places as the input files
        f_formatted = "{:.{}f}".format(f, decimal_places)
        output.write(f'{f_formatted}\t{mag:.2f}\t{phase:.2f}\t{coherence:.2f}\n')

    return output.getvalue()

def main():
    st.markdown("<h1 style='text-align: center; color: #f63366;'>SubSync</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #f63366;'>Subwoofer Delay Calculator</h3>", unsafe_allow_html=True)

    st.markdown("Follow [monsterDSP](https://instagram.com/monsterdsp)")

    categories = ['SUB', 'PA']
    file_dict = {category: None for category in categories}
    df_list = []
    crossover_freqs = []

    for idx, category in enumerate(categories):
        file_dict[category] = st.file_uploader(f"Upload {category} Transfer Function File", type=['csv', 'txt'])

        if file_dict[category] is not None:
            try:
                df = pd.read_csv(file_dict[category], delimiter='\t')
                df.columns = ['Frequency', 'Magnitude (dB)', 'Phase (degrees)', 'Coherence']
                df = preprocess_transfer_function(df)

                if not df.empty:
                    df_list.append(df)
                    if idx > 0:
                        crossover_text = f"Enter Acoustic Crossover Frequency between {categories[idx-1].capitalize()} and {category.capitalize()} (Hz)"
                        crossover_freq = st.slider(crossover_text, min_value=20, max_value=500, value=120, step=1)
                        crossover_freqs.append(crossover_freq)

            except Exception as e:
                st.error(f"Error occurred while reading file: {e}")

    if file_dict['SUB'] is not None and file_dict['PA'] is not None:
        coherence_tolerance = st.slider("Coherence Tolerance", min_value=0.25, max_value=1.0, value=0.5, step=0.01)
        amp_diff_dB = st.slider("Isolation Zone Range (dB difference between curves)", min_value=1, max_value=60, value=18, step=1)

        if df_list and len(crossover_freqs) == len(df_list) - 1:
            st.header('Alignment Information')

            total_delay_ms = 0
            polarity_adjustments = []

            for df, crossover_freq in zip(df_list[1:], crossover_freqs):
                delay_ms, polarity = calculate_best_delay(df_list[0], df, crossover_freq, amp_diff_dB)
                if delay_ms is not None:
                    total_delay_ms += delay_ms
                    polarity_adjustments.append(polarity)

            if abs(total_delay_ms) < 0.1:
                displayed_delay = 0.0
            else:
                displayed_delay = round(abs(total_delay_ms), 1)

            st.markdown(f"**Total Delay for Alignment**: {displayed_delay:.1f} ms")
            st.markdown(f"**Polarity Adjustments**: {' '.join(polarity_adjustments)}")

            plot_transfer_function(df_list, crossover_freqs, coherence_tolerance, amp_diff_dB)

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

            st.download_button(
                label="Download Aligned Sum ASCII File",
                data=aligned_sum_data,
                file_name=output_filename,
                mime='text/plain'
            )


if __name__ == "__main__":
    main()